"""Local filesystem contracts; no network, credentials, or production authority."""

import ctypes
import errno
import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from benchmarks import issue123_completion as completion
from benchmarks import issue123_directory as directory
from benchmarks import issue123_operations as operations
from benchmarks import issue123_publication as publication


def test_complete_directory_publication(tmp_path):
    output = tmp_path / "parent" / "output"
    with directory.StagedDirectory(output, create_parents=True) as stage:
        stage.write("with.dot/nested/data", b"payload")
        stage.write("index.json", b"{}\n")
        assert not output.exists()
        assert stage.read("with.dot/nested/data") == b"payload"
        retained = list(stage._fds)
        stage.publish()
        assert stage.state == "committed"
        assert output.joinpath("with.dot/nested/data").read_bytes() == b"payload"
    assert sorted(path.name for path in output.parent.iterdir()) == ["output"]
    for fd in retained:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("kind", ("file", "directory", "symlink"))
def test_native_collision_preserves_both_entries(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload").write_bytes(b"complete")
    destination = tmp_path / "destination"
    if kind == "file":
        destination.write_bytes(b"existing")
    elif kind == "directory":
        destination.mkdir()
    else:
        destination.symlink_to("missing")
    identity = destination.lstat()
    fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with pytest.raises(OSError):
            directory.rename_directory_exclusive(fd, "source", "destination")
    finally:
        os.close(fd)
    assert destination.lstat() == identity
    assert (source / "payload").read_bytes() == b"complete"


@pytest.mark.parametrize(
    "system,symbol,flag", (("Linux", "renameat2", 1), ("Darwin", "renameatx_np", 4))
)
def test_native_abi_and_errno(system, symbol, flag):
    function = mock.Mock(return_value=-1)
    library = mock.Mock(**{symbol: function})
    with (
        mock.patch.object(directory.platform, "system", return_value=system),
        mock.patch.object(directory.ctypes, "CDLL", return_value=library) as load,
        mock.patch.object(directory.ctypes, "get_errno", return_value=errno.EXDEV),
        pytest.raises(OSError) as caught,
    ):
        directory.rename_directory_exclusive(12, "stage", "final")
    load.assert_called_once_with(None, use_errno=True)
    function.assert_called_once_with(12, b"stage", 12, b"final", flag)
    assert function.argtypes == [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    assert function.restype == ctypes.c_int
    assert caught.value.errno == errno.EXDEV
    assert "stage" not in str(caught.value)


@pytest.mark.parametrize(
    "name", ("", ".", "..", "/absolute", "nested/leaf", "nul\x00suffix")
)
def test_native_rejects_non_leaf_before_loading_library(name):
    with mock.patch.object(directory, "_native_rename") as load:
        with pytest.raises(directory.DirectoryPublicationError):
            directory.rename_directory_exclusive(0, name, "final")
    load.assert_not_called()


@pytest.mark.parametrize(
    "code", (errno.EXDEV, errno.ENOSYS, errno.EINVAL, errno.ENOTSUP, errno.EACCES)
)
def test_rename_failures_never_fall_back_and_clean_owned_stage(tmp_path, code):
    output = tmp_path / "output"
    primary = OSError(code, "private path canary")
    with (
        mock.patch.object(directory, "rename_directory_exclusive", side_effect=primary),
        pytest.raises(OSError) as caught,
    ):
        with directory.StagedDirectory(output) as stage:
            stage.write("data", b"payload")
            stage.publish()
    assert caught.value is primary
    assert primary.directory_publication_state == "not-committed"
    assert primary.directory_cleanup_codes == ()
    assert list(tmp_path.iterdir()) == []


def test_missing_native_symbol_fails_before_creating_parent(tmp_path):
    with (
        mock.patch.object(directory.ctypes, "CDLL", return_value=object()),
        pytest.raises(directory.DirectoryPublicationError),
    ):
        with directory.StagedDirectory(
            tmp_path / "new" / "output", create_parents=True
        ):
            pytest.fail("stage unexpectedly opened")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("phase", ("write", "publish", "after-publish"))
def test_parent_displacement_keeps_io_on_retained_directory(tmp_path, phase):
    parent = tmp_path / "parent"
    parent.mkdir()
    displaced = tmp_path / "displaced"
    output = parent / "output"
    native = directory.rename_directory_exclusive

    def displace():
        parent.rename(displaced)
        parent.mkdir()
        (parent / "sentinel").write_bytes(b"foreign")

    def publish_then_displace(*args):
        native(*args)
        displace()

    with pytest.raises(directory.DirectoryPublicationError) as caught:
        with directory.StagedDirectory(output) as stage:
            stage.write("first", b"owned")
            if phase == "after-publish":
                with mock.patch.object(
                    directory,
                    "rename_directory_exclusive",
                    side_effect=publish_then_displace,
                ):
                    stage.publish()
            else:
                displace()
                if phase == "write":
                    stage.write("second", b"owned")
                else:
                    stage.publish()
    assert (parent / "sentinel").read_bytes() == b"foreign"
    assert sorted(p.name for p in parent.iterdir()) == ["sentinel"]
    if phase == "after-publish":
        assert caught.value.directory_publication_state == "committed"
        assert (displaced / "output" / "first").read_bytes() == b"owned"
    else:
        assert caught.value.directory_publication_state == "not-committed"
        assert list(displaced.iterdir()) == []


def test_source_substitution_is_preserved_and_reported(tmp_path):
    moved = tmp_path / "moved"
    with pytest.raises(directory.DirectoryPublicationError) as caught:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            stage.write("data", b"owned")
            source = tmp_path / stage._stage_name
            source.rename(moved)
            source.mkdir()
            (source / "sentinel").write_bytes(b"foreign")
            stage.publish()
    assert (source / "sentinel").read_bytes() == b"foreign"
    assert not (tmp_path / "output").exists()
    assert "directory-cleanup-incomplete" in caught.value.directory_cleanup_codes


@pytest.mark.parametrize("observation", ("committed", "ambiguous"))
def test_failure_after_native_commit_does_not_remove_output(tmp_path, observation):
    native = directory.rename_directory_exclusive
    primary = OSError(errno.EIO, "private canary")
    with pytest.raises(OSError) as caught:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            stage.write("data", b"complete")

            def commit_then_fail(*args):
                native(*args)
                if observation == "ambiguous":
                    # An unavailable post-syscall observation cannot prove rollback.
                    stage._reconcile_rename = lambda **_kwargs: None
                raise primary

            with mock.patch.object(
                directory, "rename_directory_exclusive", side_effect=commit_then_fail
            ):
                stage.publish()
    assert caught.value is primary
    assert primary.directory_publication_state == observation
    assert (tmp_path / "output" / "data").read_bytes() == b"complete"


@pytest.mark.parametrize("operation", ("unlink", "rmdir", "stat", "close"))
def test_cleanup_failures_preserve_primary_and_emit_fixed_diagnostics(
    tmp_path, capsys, monkeypatch, operation
):
    primary = RuntimeError("private-primary-canary")
    real = getattr(directory.os, operation)

    def fail(*args, **kwargs):
        if operation == "close":
            real(*args, **kwargs)  # Do not leak a descriptor in the test itself.
        raise OSError(errno.EIO, "private-cleanup-canary")

    with pytest.raises(RuntimeError) as caught:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            stage.write("data", b"owned")
            monkeypatch.setattr(directory.os, operation, fail)
            raise primary
    monkeypatch.undo()
    assert caught.value is primary
    assert primary.directory_cleanup_codes
    directory.report_diagnostics(primary)
    captured = capsys.readouterr()
    assert "private" not in captured.err
    assert "issue123-directory-not-committed" in captured.err


@pytest.mark.parametrize(
    "module,command",
    ((operations, "capture"), (completion, "assemble"), (publication, "prepare")),
)
def test_cli_preserves_fixed_cleanup_and_commit_diagnostics(module, command, capsys):
    primary = ValueError("private canary")
    directory.annotate_failure(
        primary, "partial-commit", ("directory-cleanup-incomplete",)
    )
    with mock.patch.object(module, "_main", side_effect=primary):
        assert module.main([command]) == 2
    stderr = capsys.readouterr().err
    assert "private" not in stderr
    assert "issue123-directory-partial-commit\n" in stderr
    assert "issue123-directory-cleanup-incomplete\n" in stderr


@pytest.mark.parametrize("alias", ("tmp", "var"))
def test_darwin_canonical_system_aliases(alias):
    if directory.platform.system() != "Darwin":
        pytest.skip("requires real Darwin root aliases")
    import tempfile

    with tempfile.TemporaryDirectory(
        dir="/tmp" if alias == "tmp" else "/var/tmp"
    ) as root:
        lexical = Path(root) / "output"
        with directory.StagedDirectory(lexical) as stage:
            stage.write("data", b"portable")
            stage.publish()
        assert lexical.joinpath("data").read_bytes() == b"portable"


def test_arbitrary_parent_symlink_is_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(directory.DirectoryPublicationError):
        with directory.StagedDirectory(alias / "output"):
            pytest.fail("symlink parent accepted")
    assert list(real.iterdir()) == []


@pytest.fixture
def publication_arguments(tmp_path, monkeypatch):
    from benchmarks import issue123_privacy as privacy
    from tests import test_issue123_privacy as fixtures

    policy, private = fixtures._fixture()
    for policy_scope, private_scope in zip(
        policy["scopes"], private["scopes"], strict=True
    ):
        policy_scope["correctness"] = []
        private_scope["correctness"] = []
    specification, runtime_paths, document = fixtures._production_source_fixture(
        tmp_path, policy, private
    )
    completion_index = fixtures._completion_bundle_fixture(
        tmp_path / "bundle", tmp_path, document
    )
    bindings = {}
    for target, record in zip(
        privacy._expected_source_targets(policy), document["sources"], strict=True
    ):
        key = privacy._target_key(target)
        bindings[key] = privacy.EvaluatorTargetBinding(
            target_key=key,
            primary=privacy.RoleSelector(record["completion_role"], record["selector"]),
        )
    # The established synthetic profile is confined to this test's finalizer.
    monkeypatch.setattr(privacy, "CODE_OWNED_LITERAL_TARGET_BINDINGS", bindings)
    policy_raw = publication.canonical_json_bytes(policy)
    policy_path = tmp_path / "policy.json"
    policy_path.write_bytes(policy_raw)
    private_parent = tmp_path / "private"
    private_parent.mkdir(mode=0o700)
    return dict(
        source_specification=specification,
        completion_index=completion_index,
        policy_path=policy_path,
        policy_sha256=directory.hashlib.sha256(policy_raw).hexdigest(),
        runtime_receipt_paths=runtime_paths,
        asset_output_directory=tmp_path / "assets",
        private_openings_output=private_parent / "openings.json",
        salt=bytes(range(32)),
    )


@pytest.mark.parametrize(
    "sidecar_committed", (False, True), ids=("partial-commit", "both-committed")
)
def test_publication_multi_output_failure_keeps_complete_assets(
    publication_arguments, sidecar_committed
):
    real_commit = publication._commit_private_authority_file

    def fail_sidecar(*args, **kwargs):
        if sidecar_committed:
            real_commit(*args, **kwargs)
            raise publication.PublicationCommitError(
                "private authority verification failed"
            )
        raise publication.PublicationError("private authority commit failed")

    with (
        mock.patch.object(
            publication, "_commit_private_authority_file", side_effect=fail_sidecar
        ),
        pytest.raises(publication.PublicationError) as caught,
    ):
        publication.prepare_publication(**publication_arguments)
    assert caught.value.directory_publication_state == (
        "committed" if sidecar_committed else "partial-commit"
    )
    assets = publication_arguments["asset_output_directory"]
    assert {path.name for path in assets.iterdir()} == {
        name for _, name in publication.ASSET_ORDER
    }
    assert (
        publication_arguments["private_openings_output"].exists() is sidecar_committed
    )
    assert not list(assets.parent.glob(".assets.*"))


def test_publication_exdev_has_no_sidecar_or_partial_assets(publication_arguments):
    with (
        mock.patch.object(
            directory,
            "rename_directory_exclusive",
            side_effect=OSError(errno.EXDEV, "private canary"),
        ),
        mock.patch.object(publication, "_commit_private_authority_file") as commit,
        pytest.raises(publication.PublicationError) as caught,
    ):
        publication.prepare_publication(**publication_arguments)
    commit.assert_not_called()
    assert caught.value.directory_publication_state == "not-committed"
    assert caught.value.__context__ is None
    assert "canary" not in str(caught.value)
    assert not publication_arguments["asset_output_directory"].exists()
    assert not publication_arguments["private_openings_output"].exists()


def test_completion_assembly_uses_exclusive_adapter_and_preserves_failure(tmp_path):
    from tests.test_issue123_bundle import _Issue123BundleFixture

    fixture = _Issue123BundleFixture()
    fixture.initialize(tmp_path)
    primary = OSError(errno.EXDEV, "private canary")
    with (
        mock.patch.object(directory, "rename_directory_exclusive", side_effect=primary),
        pytest.raises(OSError) as caught,
    ):
        fixture.assemble()
    assert caught.value is primary
    assert primary.directory_publication_state == "not-committed"
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.*"))


def test_post_commit_close_failure_is_explicit(tmp_path, monkeypatch):
    real_close = directory.os.close
    with pytest.raises(directory.DirectoryPublicationError) as caught:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            stage.write("data", b"complete")
            stage.publish()

            def close_then_fail(fd):
                real_close(fd)
                raise OSError(errno.EIO, "private close canary")

            monkeypatch.setattr(directory.os, "close", close_then_fail)
    monkeypatch.undo()
    assert caught.value.directory_publication_state == "committed"
    assert caught.value.directory_cleanup_codes == ("directory-close-failed",)
    assert (tmp_path / "output" / "data").read_bytes() == b"complete"


def test_uncertain_native_failure_preserves_stage_for_inspection(tmp_path):
    primary = OSError(errno.EIO, "private canary")
    with pytest.raises(OSError) as caught:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            stage.write("data", b"complete")
            name = stage._stage_name
            with mock.patch.object(
                directory, "rename_directory_exclusive", side_effect=primary
            ):
                stage.publish()
    assert caught.value is primary
    assert primary.directory_publication_state == "ambiguous"
    assert "directory-cleanup-incomplete" in primary.directory_cleanup_codes
    assert (tmp_path / name / "data").read_bytes() == b"complete"


def test_first_file_fstat_failure_cannot_delete_unidentified_entry(
    tmp_path, monkeypatch
):
    primary = OSError(errno.EIO, "private canary")
    real_fstat = directory.os.fstat
    retained = []

    def fail_file(fd):
        metadata = real_fstat(fd)
        if stat.S_ISREG(metadata.st_mode):
            retained.append(fd)
            raise primary
        return metadata

    with pytest.raises(OSError) as caught:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            monkeypatch.setattr(directory.os, "fstat", fail_file)
            stage.write("data", b"owned")
    monkeypatch.undo()
    assert caught.value is primary
    assert "directory-cleanup-incomplete" in primary.directory_cleanup_codes
    for fd in retained:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_default_file_mode_preserves_restrictive_umask(tmp_path):
    previous = os.umask(0o077)
    try:
        with directory.StagedDirectory(tmp_path / "output") as stage:
            stage.write("private", b"private")
            stage.write("public", b"public", mode=0o644)
            stage.publish()
    finally:
        os.umask(previous)
    assert stat.S_IMODE((tmp_path / "output" / "private").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "output" / "public").stat().st_mode) == 0o644
