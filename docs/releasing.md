# Releasing GMES

GMES releases are built from a version tag in clean GitHub-hosted runners.
The workflow transfers the resulting source distribution and wheels through
GitHub Actions artifacts, publishes those exact files to PyPI with OpenID
Connect (OIDC), and attaches them to the matching GitHub Release. Never upload
local `dist/` contents or a rebuilt copy of an artifact.

## Supported artifacts for 0.10.0

The release contains exactly three files:

- `gmes-0.10.0.tar.gz`
- a CPython 3.14 Linux x86_64 `manylinux_2_34` wheel
- a CPython 3.14 macOS arm64 wheel targeting macOS 11

Windows and macOS x86_64 are intentionally unsupported for 0.10.0. Add either
platform only in a separate change that builds and tests its wheel on a native
runner.

## One-time publishing setup

The `gmes` JSON endpoint on PyPI returned 404 when the release pipeline was
prepared on 2026-08-22, so no public project existed at that time. Recheck the
name before release and make sure it is either still available or controlled
by the GMES maintainer account.

Configure a pending Trusted Publisher on PyPI when the project does not yet
exist, or add a publisher to the existing project with these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `gmes` |
| GitHub owner | `ruddyscent` |
| Repository | `gmes` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The GitHub repository must also have an environment named `pypi`. Restrict it
to version tags matching `v*` and require a maintainer approval. The workflow
grants `id-token: write` only to the PyPI publishing job; no PyPI password or
API token is stored in GitHub.

## Prepare and validate a release

1. Update `VERSION`, the release heading and date in `CHANGELOG.md`, and the
   supported-platform table in `README.md` in a release-preparation pull
   request.
2. Confirm that the release workflow's expected wheel platforms match the
   documentation. All third-party Actions must remain pinned to immutable
   commit SHAs. Keep `build-constraints.txt` synchronized with
   `[tool.uv].build-constraint-dependencies` so uv and cibuildwheel use the
   same native build dependencies.
3. Run the local checks:

   ```sh
   uv python install 3.14
   uv sync --locked --extra hdf5
   uv run --no-sync python -m isort --check-only gmes examples tests utils setup.py
   uv run --no-sync python -m black --check gmes examples tests utils setup.py
   uv run --no-sync python -m mypy
   uv run --no-sync python -m mypy.stubtest --mypy-config-file pyproject.toml gmes.constant gmes.pw_material
   uv run --no-sync python -m pylint $(git ls-files 'gmes/*.py') setup.py
   uv run --no-sync python -m unittest discover -v
   uv build
   ```

4. Confirm that the preparation pull request's `Release` workflow succeeds.
   It builds and validates artifacts but cannot publish them. Once the workflow
   exists on the default branch, maintainers may also run it manually against
   a preparation branch for the same artifact-only validation.
5. Squash-merge the preparation pull request after the required CI and CodeQL
   checks pass. Confirm that local `master` is at the merged commit.

## Publish the release

1. Create the signed tag from the verified `master` commit and push only that
   tag:

   ```sh
   git tag -s v0.10.0 -m "GMES 0.10.0"
   git push origin v0.10.0
   ```

2. The tag starts the `Release` workflow. It checks that the tag, `VERSION`,
   package metadata, changelog heading, and GitHub Release title use the same
   version.
3. Wait for both full test jobs, the isolated sdist build, both cibuildwheel
   jobs, `twine check`, archive-content checks, clean-install smoke tests,
   auditwheel, and delocate to succeed.
4. Approve the protected `pypi` environment deployment. The publishing job
   contains no checkout or build step and fails on duplicate files.
5. Wait for the clean PyPI wheel installation check. The workflow then creates
   `GMES 0.10.0` as the GitHub Release and attaches the same three verified
   files.
6. Verify the PyPI project page, GitHub Release, and a clean user installation:

   ```sh
   python3.14 -m venv /tmp/gmes-release-check
   /tmp/gmes-release-check/bin/python -m pip install --only-binary=:all: "gmes==0.10.0"
   /tmp/gmes-release-check/bin/python -c "import gmes, gmes.material, gmes.pygeom"
   ```

If publication partially succeeds, do not enable `skip-existing` and do not
replace published files. Diagnose the failed downstream job and rerun only the
failed GitHub Actions jobs. If an uploaded distribution itself is wrong, make
a new version rather than overwriting it.
