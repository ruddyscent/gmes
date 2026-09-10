"""Exact timestamp and descriptor contracts independent of filesystem epochs."""

import copy
import json
import os
import tarfile
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest

from benchmarks import issue123_completion as completion
from benchmarks import issue123_operations as operations
from benchmarks import issue123_privacy as privacy
from benchmarks import issue123_publication as publication
from tests import test_issue123_privacy as fixtures


def _validate(view):
    return privacy._validate_private_sdist_raw_first(
        view, (), limits=privacy._default_private_sdist_validation_limits()
    )


def test_real_unlinked_descriptor_preserves_caller_ownership_and_offset(tmp_path):
    path = tmp_path / "source.tar.gz"
    raw = fixtures.TestIssue123PrivacyScanner()._private_sdist()
    path.write_bytes(raw)
    fd = os.open(path, os.O_RDONLY)
    try:
        path.unlink()
        if os.fstat(fd).st_nlink != 0:
            pytest.skip("filesystem does not expose zero-link open regular files")
        os.lseek(fd, 7, os.SEEK_SET)
        with privacy._retain_private_sdist_fd(fd) as view:
            result = _validate(view)
            assert result.archive_size == len(raw)
            assert os.lseek(fd, 0, os.SEEK_CUR) == 7
            os.close(fd)
            fd = None
            assert _validate(view) == result
            duplicate = view.fd
        with pytest.raises(OSError):
            os.fstat(duplicate)
        assert not path.exists()
        assert "mtime" not in result.__dataclass_fields__
        assert "nlink" not in result.__dataclass_fields__
    finally:
        if fd is not None:
            os.close(fd)


@pytest.mark.parametrize("change", ("unlink", "mutate"))
def test_retained_descriptor_change_still_fails(tmp_path, change):
    path = tmp_path / "source.tar.gz"
    path.write_bytes(fixtures.TestIssue123PrivacyScanner()._private_sdist())
    fd = os.open(path, os.O_RDWR)
    try:
        with privacy._retain_private_sdist_fd(fd) as view:
            if change == "unlink":
                path.unlink()
            else:
                os.ftruncate(fd, 1)
            with pytest.raises(privacy._PrivateSdistValidationError) as caught:
                _validate(view)
            assert caught.value.token is privacy._PrivateSdistFailure.SOURCE_CHANGED
    finally:
        os.close(fd)


@pytest.mark.parametrize(
    "stamp",
    (-(1 << 63), -1, 0, (1 << 63) - 1),
    ids=("minimum", "pre-epoch", "epoch", "maximum"),
)
def test_signed_identity_boundaries_serialization_and_comparison(stamp):
    identity = privacy._PrivateSdistIdentity(0, 1, 0o100600, 0, 1, stamp, stamp)
    assert privacy._private_sdist_identity_is_valid(identity)
    raw = privacy.binding_canonical_json_bytes({"mtime_ns": stamp, "ctime_ns": stamp})
    restored = json.loads(raw)
    assert type(restored["mtime_ns"]) is int
    assert restored["mtime_ns"] == stamp
    assert str(stamp).encode("ascii") in raw
    view = privacy._PrivateSdistReadView(0, identity, privacy._PRIVATE_SDIST_VIEW_SEAL)
    assert privacy._private_sdist_source_is_valid(view)
    metadata = SimpleNamespace(
        st_dev=0,
        st_ino=1,
        st_mode=0o100600,
        st_nlink=0,
        st_size=1,
        st_mtime_ns=stamp,
        st_ctime_ns=stamp,
    )
    with mock.patch.object(privacy.os, "fstat", return_value=metadata):
        assert privacy._private_sdist_source_matches(view) is None
        metadata.st_mtime_ns = stamp + (1 if stamp < 0 else -1)
        assert (
            privacy._private_sdist_source_matches(view)
            is privacy._PrivateSdistFailure.SOURCE_CHANGED
        )
    assert "mtime_ns" not in repr(identity)
    assert not privacy._private_sdist_source_is_valid(replace(view, _seal=object()))


class _IntSubclass(int):
    pass


INVALID_STAT_TIMESTAMPS = (
    pytest.param(False, id="bool"),
    pytest.param(_IntSubclass(0), id="int-subclass"),
    pytest.param(-(1 << 63) - 1, id="underflow"),
    pytest.param(1 << 63, id="overflow"),
)


def _metadata_with_timestamp(metadata, timestamp):
    return SimpleNamespace(
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_mode=metadata.st_mode,
        st_nlink=getattr(metadata, "st_nlink", 1),
        st_size=metadata.st_size,
        st_mtime_ns=timestamp,
        st_ctime_ns=getattr(metadata, "st_ctime_ns", 0),
    )


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("capture", "recheck"))
def test_completion_bounded_read_rejects_invalid_stat_timestamp(
    tmp_path, timestamp, phase
):
    path = tmp_path / "bounded.json"
    raw = b"{}\n"
    path.write_bytes(raw)
    metadata = path.stat()
    valid = _metadata_with_timestamp(metadata, 0)
    invalid = _metadata_with_timestamp(metadata, timestamp)
    observations = [invalid] if phase == "capture" else [valid, invalid]
    with (
        mock.patch.object(completion.os, "fstat", side_effect=observations),
        pytest.raises(completion.EvidenceError),
    ):
        completion._read_opened_regular_file(
            path,
            "bounded fixture",
            max_bytes=len(raw),
            expected_size=len(raw),
        )


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("initial", "post-read"))
def test_retained_artifact_invalid_timestamp_closes_descriptor(
    tmp_path, timestamp, phase
):
    path = tmp_path / "retained.tar.gz"
    raw = b"retained-bytes"
    path.write_bytes(raw)
    metadata = path.stat()
    valid = _metadata_with_timestamp(metadata, 0)
    invalid = _metadata_with_timestamp(metadata, timestamp)
    lease = completion._RetainedArtifactLease(
        path,
        completion._artifact_descriptor_identity(valid),
        len(raw),
        completion._sha256(raw),
    )
    observations = [invalid] if phase == "initial" else [valid, invalid]
    opened = []
    real_open = os.open

    def record_open(*args, **kwargs):
        descriptor = real_open(*args, **kwargs)
        opened.append(descriptor)
        return descriptor

    with (
        mock.patch.object(completion.os, "open", side_effect=record_open),
        mock.patch.object(completion.os, "fstat", side_effect=observations),
        pytest.raises(
            completion.EvidenceError,
            match="macOS sdist source is invalid",
        ) as caught,
    ):
        lease.__enter__()
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert lease._fd == -1
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("capture", "descriptor-recheck", "named-recheck"))
def test_private_file_read_rejects_invalid_stat_timestamp(tmp_path, timestamp, phase):
    path = tmp_path / "private.json"
    raw = b"{}\n"
    path.write_bytes(raw)
    metadata = path.stat()
    valid = _metadata_with_timestamp(metadata, 0)
    invalid = _metadata_with_timestamp(metadata, timestamp)
    descriptor_metadata = (
        [invalid]
        if phase == "capture"
        else [valid, invalid if phase == "descriptor-recheck" else valid]
    )
    named = invalid if phase == "named-recheck" else valid
    with (
        mock.patch.object(privacy, "_lexical_path_without_symlinks", return_value=path),
        mock.patch.object(privacy.os, "fstat", side_effect=descriptor_metadata),
        mock.patch.object(type(path), "lstat", return_value=named),
        pytest.raises(privacy.PrivacyError, match="identity or byte bound differs"),
    ):
        privacy._private_file_bytes(path, "private fixture", maximum=len(raw))


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("capture", "recheck"))
def test_operations_bounded_read_rejects_invalid_stat_timestamp(timestamp, phase):
    resolved = mock.Mock()
    supplied = mock.Mock()
    supplied.resolve.return_value = resolved
    metadata = SimpleNamespace(
        st_dev=1,
        st_ino=2,
        st_mode=0o100600,
        st_size=1,
        st_mtime_ns=0,
    )
    invalid = _metadata_with_timestamp(metadata, timestamp)
    resolved.stat.side_effect = [invalid] if phase == "capture" else [metadata, invalid]
    resolved.read_bytes.return_value = b"x"
    resolved.is_file.return_value = True
    with pytest.raises(operations.EvidenceError, match="file identity"):
        operations._bounded_file_bytes(supplied, "bounded fixture", 1)


def _baseline_lease(tmp_path):
    root = tmp_path / "baseline"
    root.mkdir()
    path = root / "asset.json"
    raw = b"baseline\n"
    path.write_bytes(raw)
    path.chmod(0o600)
    root.chmod(0o700)
    root_descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    descriptor = os.open(path, os.O_RDONLY)
    root_metadata = os.fstat(root_descriptor)
    metadata = _metadata_with_timestamp(os.fstat(descriptor), 0)
    expectation = operations.BaselineAssetExpectation(
        ordinal=0,
        thread_mode="one",
        name=path.name,
        publication_url="https://example.invalid/asset.json",
        size_bytes=len(raw),
        sha256=operations.hashlib.sha256(raw).hexdigest(),
    )
    asset = operations._BaselineAssetLease(
        expectation,
        descriptor,
        operations._file_identity_with_mode(metadata, "fixture"),
    )
    lease = operations.BaselineAuthorityLease(
        mock.Mock(),
        root,
        root_descriptor,
        (root_metadata.st_dev, root_metadata.st_ino, root_metadata.st_mode),
        (asset,),
        {},
    )
    return lease, metadata


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("retained", "reopened"))
def test_operations_baseline_rejects_invalid_stat_timestamp(tmp_path, timestamp, phase):
    lease, valid = _baseline_lease(tmp_path)
    invalid = _metadata_with_timestamp(valid, timestamp)
    real_fstat = os.fstat

    def observed(descriptor):
        if descriptor == lease._root_descriptor:
            return real_fstat(descriptor)
        if descriptor == lease._assets[0].descriptor:
            return invalid if phase == "retained" else valid
        return invalid

    try:
        with (
            mock.patch.object(operations.os, "fstat", side_effect=observed),
            pytest.raises(operations.EvidenceError, match="retained baseline asset"),
        ):
            lease.require_unchanged()
    finally:
        lease.close()


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("descriptor", "named"))
def test_operations_receipt_capture_rejects_invalid_stat_timestamp(
    tmp_path, timestamp, phase
):
    path = tmp_path / "receipt.json"
    raw = b"receipt\n"
    path.write_bytes(raw)
    path.chmod(0o600)
    actual = path.stat()
    invalid = _metadata_with_timestamp(actual, timestamp)
    fstat_value = invalid if phase == "descriptor" else actual
    named_value = invalid if phase == "named" else actual
    with (
        mock.patch.object(operations.os, "fstat", return_value=fstat_value),
        mock.patch.object(type(path), "lstat", return_value=named_value),
        pytest.raises(operations.EvidenceError, match="durable bytes differ"),
    ):
        operations._retain_live_receipt(path, raw)


@pytest.mark.parametrize("timestamp", INVALID_STAT_TIMESTAMPS)
@pytest.mark.parametrize("phase", ("descriptor", "named"))
def test_operations_receipt_recheck_rejects_invalid_stat_timestamp(
    tmp_path, timestamp, phase
):
    path = tmp_path / "receipt.json"
    raw = b"receipt\n"
    path.write_bytes(raw)
    path.chmod(0o600)
    receipt = operations._retain_live_receipt(path, raw)
    actual = path.stat()
    valid = _metadata_with_timestamp(actual, 0)
    invalid = _metadata_with_timestamp(actual, timestamp)
    receipt.identity = operations._file_identity_with_mode(valid, "fixture")
    descriptor_value = invalid if phase == "descriptor" else valid
    named_value = invalid if phase == "named" else valid
    try:
        with (
            mock.patch.object(operations.os, "fstat", return_value=descriptor_value),
            mock.patch.object(type(path), "lstat", return_value=named_value),
            pytest.raises(operations.EvidenceError, match="retained live receipt"),
        ):
            receipt.require_unchanged()
    finally:
        receipt.close()


@pytest.mark.parametrize("field", ("mtime_ns", "ctime_ns"))
@pytest.mark.parametrize(
    "value",
    (-(1 << 63) - 1, 1 << 63, True, _IntSubclass(-1)),
    ids=("underflow", "overflow", "bool", "int-subclass"),
)
def test_private_timestamp_invalid_values(field, value):
    identity = privacy._PrivateSdistIdentity(0, 1, 0o100600, 0, 1, -1, -1)
    assert not privacy._private_sdist_identity_is_valid(
        replace(identity, **{field: value})
    )


@pytest.mark.parametrize("stamp", (-(1 << 63), -1, 0, (1 << 63) - 1))
def test_completion_identity_producers_keep_exact_signed_values(stamp):
    metadata = SimpleNamespace(
        st_dev=0,
        st_ino=1,
        st_mode=0o100600,
        st_size=1,
        st_mtime_ns=stamp,
        st_ctime_ns=stamp,
    )
    assert completion._file_identity(metadata) == (0, 1, 1, stamp)
    assert completion._artifact_descriptor_identity(metadata).ctime_ns == stamp
    assert completion._retained_file_identity(metadata)[-1] == stamp
    for value in (-(1 << 63) - 1, 1 << 63, True, _IntSubclass(-1)):
        metadata.st_mtime_ns = value
        with pytest.raises(completion.EvidenceError):
            completion._file_identity(metadata)


def _bindings(stamp):
    return {
        role: [
            {
                "case": "case",
                "path": f"{role}/case.npz",
                "sha256": "a" * 64,
                "size_bytes": 1,
                "media_type": completion.MEDIA_TYPE_NPZ,
                "payload_identity": (0, 1, 1, stamp),
            }
        ]
        for role in ("reference", "candidate")
    }


@pytest.mark.parametrize("stamp", (-(1 << 63), -1, 0, (1 << 63) - 1))
def test_completion_signed_identity_contract(stamp):
    value = _bindings(stamp)
    manifest = {"correctness": [{"name": "case"}], "physical_checks": []}
    assert (
        completion._ordered_correctness_archive_bindings(value, manifest, "fixture")
        == value
    )
    for index, invalid in (
        (0, -1),
        (1, -1),
        (2, -1),
        (3, -(1 << 63) - 1),
        (3, 1 << 63),
        (3, True),
        (3, _IntSubclass(-1)),
    ):
        changed = copy.deepcopy(value)
        identity = list(changed["reference"][0]["payload_identity"])
        identity[index] = invalid
        changed["reference"][0]["payload_identity"] = tuple(identity)
        with pytest.raises(completion.EvidenceError):
            completion._ordered_correctness_archive_bindings(
                changed, manifest, "fixture"
            )


def test_real_pre_epoch_mtime_private_sdist_completion_path(tmp_path):
    path = tmp_path / "source.tar.gz"
    raw = fixtures.TestIssue123PrivacyScanner()._private_sdist()
    path.write_bytes(raw)
    requested = -1_000_000_001
    try:
        os.utime(path, ns=(requested, requested))
    except OSError, OverflowError:
        pytest.skip("filesystem cannot set pre-epoch timestamps")
    if path.stat().st_mtime_ns >= 0:
        pytest.skip("filesystem clamps pre-epoch timestamps")
    candidate = {"candidate_git_commit": "a" * 40}
    descriptor = {
        "path": path.name,
        "size_bytes": len(raw),
        "sha256": privacy.hashlib.sha256(raw).hexdigest(),
        "media_type": completion.MEDIA_TYPE_GZIP,
        "candidate_evidence": candidate,
    }
    artifact, inventory = completion.ArtifactReader(
        tmp_path, candidate
    ).load_private_sdist(descriptor, "fixture")
    assert artifact.raw == raw
    assert inventory.archive_sha256 == descriptor["sha256"]
    assert not any("time" in field for field in inventory.__dataclass_fields__)


@pytest.mark.parametrize(
    "stamp", ("-0.000000001", "-1.25", "-9223372036854775808", "9223372036854775807")
)
def test_pax_exact_signed_decimal_range(stamp):
    scanner = fixtures.TestIssue123PrivacyScanner()
    result = scanner._validate_private_sdist(
        scanner._private_sdist(pax_items=(("mtime", stamp),))
    )
    assert result.logical_member_count == 1
    public_tar = fixtures._physical_tar_bytes(
        [
            fixtures._pax_helper(tarfile.XHDTYPE, (("mtime", stamp),)),
            fixtures._ordinary_tar_record(),
        ]
    )
    privacy.scan_payload("packages/timestamps.tar", public_tar)
    with pytest.raises(privacy.PrivacyError):
        privacy.scan_payload(
            "packages/timestamps.tar", public_tar, forbidden_values=(stamp,)
        )
    with pytest.raises(privacy._PrivateSdistValidationError):
        scanner._validate_private_sdist(
            scanner._private_sdist(pax_items=(("mtime", stamp),)), openings=(stamp,)
        )


@pytest.mark.parametrize(
    "stamp",
    (
        "-9223372036854775808.000000001",
        "9223372036854775807.000000001",
        "-1e2",
        "-0.0000000001",
    ),
)
def test_pax_rejects_fractional_overflow_and_noncanonical_numbers(stamp):
    with pytest.raises(privacy.PrivacyError, match="invalid tar timestamp"):
        privacy._scan_tar_metadata_fields({"mtime": stamp}, "fixture", ())


def test_pre_epoch_trace_projection_has_identical_public_bytes():
    policy, original = fixtures._fixture()
    shifted = copy.deepcopy(original)
    for scope in shifted["scopes"]:
        for trace in scope["traces"]:
            document = json.loads(trace["trace_bytes"])
            for event in document["traceEvents"]:
                if "ts" in event:
                    event["ts"] -= 2_000_000_000_000
            trace["trace_bytes"] = fixtures._json_bytes(document)
    salt = bytes(range(32))
    expected = privacy.project_publication(
        original, policy, private_openings=privacy.PrivateOpenings(salt)
    )
    actual = privacy.project_publication(
        shifted, policy, private_openings=privacy.PrivateOpenings(salt)
    )
    assert privacy.canonical_json_bytes(actual) == privacy.canonical_json_bytes(
        expected
    )
    assets = publication.build_publication_assets(
        actual, expected_policy=policy, expected_bindings=policy["bindings"]
    )
    ledger = {
        role: {
            "name": name,
            "size_bytes": len(assets[name]),
            "sha256": privacy.hashlib.sha256(assets[name]).hexdigest(),
        }
        for role, name in publication.ASSET_ORDER
    }
    publication.validate_publication_assets(
        assets,
        expected_policy=policy,
        expected_bindings=policy["bindings"],
        expected_assets=ledger,
    )
    for raw in assets.values():
        assert b"mtime_ns" not in raw and b"ctime_ns" not in raw
        assert b"-300000000000" not in raw


@pytest.mark.parametrize(
    "module", (operations, completion), ids=("operations", "completion")
)
def test_pre_epoch_iso_chronology_is_timezone_aware_and_exact(module):
    first = "1969-12-31T23:59:59.000001Z"
    equivalent = "1970-01-01T00:59:59.000001+01:00"
    epoch = "1970-01-01T00:00:00Z"
    assert module._timestamp(first, "fixture") == module._timestamp(
        equivalent, "fixture"
    )
    assert module._timestamp(first, "fixture") < module._timestamp(epoch, "fixture")
    restored = json.loads(json.dumps({"observed_at": first}))
    assert (
        module._timestamp(restored["observed_at"], "fixture").isoformat()
        == "1969-12-31T23:59:59.000001+00:00"
    )
    with pytest.raises(module.EvidenceError):
        module._timestamp("1969-12-31T23:59:59", "fixture")
    if module is operations:
        with pytest.raises(operations.EvidenceError):
            operations._creation_update_window(
                {"created_at": epoch, "updated_at": first}, "fixture"
            )
