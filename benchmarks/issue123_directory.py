"""Descriptor-relative staging for issue #123 directory publications.

No-replace is a kernel guarantee on supported filesystems. Identity checks do
not isolate a process from other writers with the same UID; see benchmarks/README.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import platform
import secrets
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class DirectoryPublicationError(ValueError):
    """A directory could not be published and verified exclusively."""


_STATES = frozenset({"not-committed", "committed", "ambiguous", "partial-commit"})
_CLEANUP_CODES = frozenset({"directory-cleanup-incomplete", "directory-close-failed"})


def annotate_failure(error: BaseException, state: str, codes=()) -> None:
    """Attach only fixed diagnostics, without replacing the primary failure."""
    if state not in _STATES:
        raise ValueError("invalid directory publication state")
    error.directory_publication_state = state
    previous = getattr(error, "directory_cleanup_codes", ())
    error.directory_cleanup_codes = tuple(
        sorted(set(previous).union(codes).intersection(_CLEANUP_CODES))
    )
    for token in (
        f"issue123-directory-{state}",
        *(f"issue123-{code}" for code in error.directory_cleanup_codes),
    ):
        if token not in getattr(error, "__notes__", ()):
            error.add_note(token)


def copy_diagnostics(source: BaseException, target: BaseException) -> None:
    state = getattr(source, "directory_publication_state", None)
    if state in _STATES:
        annotate_failure(target, state, getattr(source, "directory_cleanup_codes", ()))


def report_diagnostics(error: BaseException) -> None:
    state = getattr(error, "directory_publication_state", None)
    if state in _STATES:
        print(f"issue123-directory-{state}", file=sys.stderr)
    for code in getattr(error, "directory_cleanup_codes", ()):
        if code in _CLEANUP_CODES:
            print(f"issue123-{code}", file=sys.stderr)


def _leaf(value: str) -> bytes:
    if (
        type(value) is not str
        or value in {"", ".", ".."}
        or "/" in value
        or "\x00" in value
    ):
        raise DirectoryPublicationError("directory publication leaf is invalid")
    return os.fsencode(value)


def _native_rename():
    system = platform.system()
    if system == "Linux":
        symbol, flag = "renameat2", 1  # RENAME_NOREPLACE
    elif system == "Darwin":
        symbol, flag = "renameatx_np", 4  # RENAME_EXCL
    else:
        raise DirectoryPublicationError("exclusive directory rename is unsupported")
    try:
        function = getattr(ctypes.CDLL(None, use_errno=True), symbol)
    except AttributeError, OSError:
        raise DirectoryPublicationError(
            "exclusive directory rename is unsupported"
        ) from None
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    return function, flag


def rename_directory_exclusive(parent_fd: int, source: str, destination: str) -> None:
    """Rename siblings atomically; never emulate exclusivity by checking a path."""
    if type(parent_fd) is not int or not 0 <= parent_fd < (1 << 31):
        raise DirectoryPublicationError("directory publication descriptor is invalid")
    source_bytes, destination_bytes = _leaf(source), _leaf(destination)
    if source == destination:
        raise DirectoryPublicationError("directory publication leaves must differ")
    function, flag = _native_rename()
    if function(parent_fd, source_bytes, parent_fd, destination_bytes, flag) != 0:
        code = ctypes.get_errno()
        # No paths, raw errno text, fallback rename, or cross-device copy.
        raise OSError(code, "exclusive directory rename failed")


def _identity(metadata):
    return metadata.st_dev, metadata.st_ino, metadata.st_mode


def _file_identity(metadata):
    identity = (
        *_identity(metadata),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )
    if (
        not all(type(value) is int for value in identity)
        or any(value < 0 for value in identity[:5])
        or any(not -(1 << 63) <= value < (1 << 63) for value in identity[5:])
    ):
        raise DirectoryPublicationError("directory file identity is invalid")
    return identity


def _named(parent_fd, name):
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


@dataclass(repr=False)
class _Directory:
    fd: int
    parent_fd: int
    name: str
    identity: tuple[int, int, int]


class StagedDirectory:
    """Own a complete staged tree until explicit publish(), then close handles."""

    def __init__(self, output: Path, *, create_parents: bool = False):
        self.output = Path(output)
        self.create_parents = create_parents
        self.state = "not-committed"
        self._fds = []
        self._parents = []
        self._directories = {}
        self._files = {}
        self._stage_name = None
        self._parent_fd = -1
        self._entered = False

    def __repr__(self):
        return "<StagedDirectory redacted>"

    def _open_directory(self, parent_fd, name):
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        failure = False
        try:
            fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError:
            failure = True
        if failure:
            raise DirectoryPublicationError(
                "directory parent is unavailable or uses a symlink"
            ) from None
        self._fds.append(fd)
        identity = _identity(os.fstat(fd))
        named = _named(parent_fd, name)
        if (
            named is None
            or _identity(named) != identity
            or not stat.S_ISDIR(identity[2])
        ):
            raise DirectoryPublicationError("directory identity differs")
        return _Directory(fd, parent_fd, name, identity)

    def __enter__(self):
        if self._entered:
            raise DirectoryPublicationError("directory stage cannot be reused")
        self._entered = True
        try:
            required = {os.open, os.stat, os.mkdir, os.unlink, os.rmdir}
            if (
                not required.issubset(os.supports_dir_fd)
                or os.listdir not in os.supports_fd
                or os.stat not in os.supports_follow_symlinks
                or not all(
                    hasattr(os, flag)
                    for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
                )
            ):
                raise DirectoryPublicationError("directory staging is unsupported")
            _native_rename()  # Resolve the ABI before creating any output.
            absolute = self.output.absolute()
            _leaf(absolute.name)
            if ".." in absolute.parts:
                raise DirectoryPublicationError("directory parent is not canonical")
            # Only the established Darwin root aliases are supported. Reopen
            # their canonical components below; never follow arbitrary links.
            if platform.system() == "Darwin" and absolute.parts[1] in {"tmp", "var"}:
                alias = Path("/") / absolute.parts[1]
                if alias.is_symlink():
                    target = Path("/private") / absolute.parts[1]
                    if alias.resolve(strict=True) != target:
                        raise DirectoryPublicationError("directory alias differs")
                    absolute = target.joinpath(*absolute.parts[2:])
            self.output = absolute
            root_fd = os.open(
                absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
            )
            self._fds.append(root_fd)
            parent_fd = root_fd
            for name in absolute.parts[1:-1]:
                if self.create_parents:
                    try:
                        os.mkdir(name, dir_fd=parent_fd)
                    except FileExistsError:
                        pass
                directory = self._open_directory(parent_fd, name)
                self._parents.append(directory)
                parent_fd = directory.fd
            self._parent_fd = parent_fd
            self._verify_parents()
            if _named(parent_fd, absolute.name) is not None:
                raise DirectoryPublicationError("directory output already exists")
            name = f".{absolute.name}.{secrets.token_hex(16)}"
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            self._stage_name = name
            self._directories[""] = self._open_directory(parent_fd, name)
            return self
        except BaseException as error:
            self.__exit__(type(error), error, error.__traceback__)
            raise

    @staticmethod
    def _verify_directory(directory):
        named = _named(directory.parent_fd, directory.name)
        if (
            _identity(os.fstat(directory.fd)) != directory.identity
            or named is None
            or _identity(named) != directory.identity
        ):
            raise DirectoryPublicationError("directory identity differs")

    def _verify_parents(self):
        for directory in self._parents:
            self._verify_directory(directory)

    def write(self, relative: str, raw: bytes, mode: int | None = None):
        parts = relative.split("/")
        for name in parts:
            _leaf(name)
        if self.state != "not-committed" or type(raw) is not bytes:
            raise DirectoryPublicationError("directory write is invalid")
        self._verify_parents()
        directory = self._directories[""]
        self._verify_directory(directory)
        for index, name in enumerate(parts[:-1]):
            key = "/".join(parts[: index + 1])
            if key not in self._directories:
                os.mkdir(name, dir_fd=directory.fd)
                self._directories[key] = self._open_directory(directory.fd, name)
            directory = self._directories[key]
            self._verify_directory(directory)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(
            parts[-1], flags, 0o666 if mode is None else mode, dir_fd=directory.fd
        )
        self._fds.append(fd)
        # Retain each created handle even when its first metadata/read/write fails.
        identity = _file_identity(os.fstat(fd))
        self._files[relative] = (directory.fd, parts[-1], fd, identity, None)
        if mode is not None:
            os.fchmod(fd, mode)
        remaining = memoryview(raw)
        while remaining:
            count = os.write(fd, remaining)
            if count <= 0:
                raise DirectoryPublicationError("directory write made no progress")
            remaining = remaining[count:]
        os.fsync(fd)
        self._files[relative] = (
            directory.fd,
            parts[-1],
            fd,
            _file_identity(os.fstat(fd)),
            hashlib.sha256(raw).digest(),
        )
        if self.read(relative) != raw:
            raise DirectoryPublicationError("directory staged bytes differ")

    def read(self, relative: str) -> bytes:
        parent_fd, name, fd, identity, digest = self._files[relative]
        named = _named(parent_fd, name)
        if (
            named is None
            or _file_identity(named) != identity
            or _file_identity(os.fstat(fd)) != identity
            or not stat.S_ISREG(identity[2])
            or digest is None
        ):
            raise DirectoryPublicationError("directory staged file identity differs")
        chunks = []
        offset = 0
        while offset < identity[4]:
            chunk = os.pread(fd, min(1024 * 1024, identity[4] - offset), offset)
            if not chunk:
                raise DirectoryPublicationError("directory staged bytes differ")
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        if (
            os.pread(fd, 1, offset)
            or hashlib.sha256(raw).digest() != digest
            or _file_identity(os.fstat(fd)) != identity
        ):
            raise DirectoryPublicationError("directory staged bytes differ")
        return raw

    def verify(self):
        self._verify_parents()
        for key, directory in self._directories.items():
            self._verify_directory(directory)
            expected = {
                PurePosixPath(path).name
                for path in (*self._directories, *self._files)
                if path
                and (
                    ""
                    if PurePosixPath(path).parent == PurePosixPath(".")
                    else str(PurePosixPath(path).parent)
                )
                == key
            }
            if set(os.listdir(directory.fd)) != expected:
                raise DirectoryPublicationError("directory staged closure differs")
        for relative in self._files:
            self.read(relative)
        self._verify_parents()
        for directory in self._directories.values():
            self._verify_directory(directory)

    def publish(self):
        if self.state != "not-committed":
            raise DirectoryPublicationError("directory publication cannot be repeated")
        self.verify()
        stage = self._directories[""]
        if _named(self._parent_fd, self.output.name) is not None:
            raise DirectoryPublicationError("directory output appeared during assembly")
        for directory in reversed(list(self._directories.values())):
            os.fsync(directory.fd)
        self._verify_parents()
        self._verify_directory(stage)
        self.state = "ambiguous"
        try:
            rename_directory_exclusive(self._parent_fd, stage.name, self.output.name)
        except BaseException as error:
            # An interrupted call or I/O error can have an uncertain outcome.
            # Only definite pre-commit errors may establish non-publication.
            uncommitted = isinstance(error, DirectoryPublicationError) or (
                isinstance(error, OSError)
                and error.errno
                in {
                    errno.EEXIST,
                    errno.ENOTEMPTY,
                    errno.EXDEV,
                    errno.ENOSYS,
                    errno.EINVAL,
                    errno.ENOTSUP,
                    errno.EACCES,
                    errno.EPERM,
                    errno.EROFS,
                    errno.ENOENT,
                    errno.ENOTDIR,
                    errno.EBADF,
                }
            )
            self._reconcile_rename(allow_uncommitted=uncommitted)
            raise
        self._reconcile_rename()
        if self.state != "committed":
            raise DirectoryPublicationError("directory publication is ambiguous")
        self.verify()
        os.fsync(self._parent_fd)
        self._verify_parents()
        self._verify_directory(stage)

    def _reconcile_rename(self, *, allow_uncommitted=False):
        stage = self._directories[""]
        try:
            source = _named(self._parent_fd, self._stage_name)
            destination = _named(self._parent_fd, self.output.name)
            if (
                destination is not None
                and _identity(destination) == stage.identity
                and source is None
            ):
                self.state = "committed"
                stage.name = self.output.name
            elif (
                allow_uncommitted
                and source is not None
                and _identity(source) == stage.identity
                and (destination is None or _identity(destination) != stage.identity)
            ):
                self.state = "not-committed"
        except OSError:
            pass  # A failed observation cannot establish non-publication.

    def __exit__(self, exc_type, error, traceback):
        codes = set()
        if self.state == "not-committed" and self._stage_name is not None:
            # Never recurse through a name supplied by a competing writer. Only
            # remove ledger-owned entries via their retained parent descriptors.
            for parent_fd, name, fd, identity, _digest in reversed(
                list(self._files.values())
            ):
                try:
                    named = _named(parent_fd, name)
                    if named is not None:
                        if (
                            _identity(named)[:2] != identity[:2]
                            or _identity(os.fstat(fd))[:2] != identity[:2]
                        ):
                            raise DirectoryPublicationError("cleanup identity differs")
                        os.unlink(name, dir_fd=parent_fd)
                except OSError, DirectoryPublicationError:
                    codes.add("directory-cleanup-incomplete")
            for directory in reversed(list(self._directories.values())):
                try:
                    self._verify_directory(directory)
                    os.rmdir(directory.name, dir_fd=directory.parent_fd)
                except OSError, DirectoryPublicationError:
                    codes.add("directory-cleanup-incomplete")
            if "" not in self._directories:
                codes.add("directory-cleanup-incomplete")
        elif self.state == "ambiguous":
            codes.add("directory-cleanup-incomplete")
        while self._fds:
            fd = self._fds.pop()
            try:
                os.close(fd)
            except OSError:
                codes.add("directory-close-failed")
        if error is not None:
            annotate_failure(error, self.state, codes)
        elif codes:
            failure = DirectoryPublicationError("directory cleanup failed")
            annotate_failure(failure, self.state, codes)
            raise failure
        return False
