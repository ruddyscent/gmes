# Releasing GMES

GMES releases are built on clean GitHub-hosted runners from a version tag. The
release workflow validates and transfers the exact archives through GitHub
Actions artifacts, publishes those same files with OpenID Connect (OIDC), and
attaches them to the matching GitHub Release. Never upload local `dist/`
contents or rebuild an artifact after validation.

## Artifact contract

The pure-package release contains exactly two files:

- `gmes-<version>-py3-none-any.whl`
- `gmes-<version>.tar.gz`

The wheel is `Root-Is-Purelib: true` and must contain the canonical
`gmes/constant.py`, `gmes/constant.pyi`, `gmes/py.typed`, and supported Torch
modules. Archive validation rejects retired GMES native binaries, generated
proxies, `src/` members, traversal paths, links, duplicate members, and
unexpected artifact names. A universal wheel does not by itself prove CPU,
CUDA, or two-GPU support; those require their respective installed-artifact
gates.

## Trusted installed-artifact cutover evidence

CPU, single-CUDA, and two-CUDA/NCCL evidence is a #124 pre-merge acceptance
concern. Run it on a trusted dedicated machine after binding a clean candidate
checkout to the candidate commit; a descriptive label alone does not bind a
commit. The machine need not be a GitHub self-hosted runner. Run from a
directory outside both checkouts, and retain the evidence outside either one:

```sh
test "$(git -C "$CANDIDATE" rev-parse HEAD)" = "$CANDIDATE_SHA"
test -z "$(git -C "$CANDIDATE" status --porcelain=v1 --untracked-files=all)"
uv venv --clear --python 3.14 "$ENV"
(
  cd "$CANDIDATE"
  UV_PROJECT_ENVIRONMENT="$ENV" uv sync --locked --no-install-project \
    --extra torch-cu130 --extra hdf5
)
"$ENV/bin/python" -m ensurepip
"$ENV/bin/python" -m pip install --no-deps \
  --constraint "$CANDIDATE/build-constraints.txt" setuptools==84.0.0 wheel==0.48.0
ARCHIVE_SHA256="$(shasum -a 256 "$ARCHIVE" | awk '{print $1}')"
INSTALLER=(--no-deps --no-index --force-reinstall)
case "$ARCHIVE" in *.tar.gz) INSTALLER+=(--no-build-isolation) ;; esac
"$ENV/bin/python" -m pip install "${INSTALLER[@]}" \
  "gmes @ file://${ARCHIVE}#sha256=${ARCHIVE_SHA256}"
cd "$RUN_DIRECTORY"
"$ENV/bin/python" -I "$CANDIDATE/benchmarks/package_cutover.py" \
  --candidate-label "${CANDIDATE_SHA}-cuda2" --archive "$ARCHIVE" \
  --forbidden-root "$CANDIDATE" --forbidden-root "$CONTROLLER_CHECKOUT" \
  --device cuda:0 --required-device-count 2 --evidence-dir "$EVIDENCE_DIRECTORY"
```

Use `torch-cpu`, `--device cpu`, and `--required-device-count 0` for CPU;
use a selected CUDA extra and `--required-device-count 1` for single-device
CUDA. The two-device helper launches `torchrun --nproc_per_node=2`, requires
NCCL and two visible devices, validates installed module origins before and
after the smoke, and records command, archive digest, stdout, and stderr. A
resource failure remains a failure; do not replace it with a skipped result or
claim this local evidence as publication authority.

## One-time publishing setup

Configure a pending Trusted Publisher on PyPI when the project does not yet
exist, or add a publisher to the existing project with these exact values:

| Field | Value |
| --- | --- |
| PyPI project | `gmes` |
| GitHub owner | `ruddyscent` |
| Repository | `gmes` |
| Workflow | `release.yml` |
| Environment | `pypi` |

The GitHub repository must have a protected `pypi` environment restricted to
version tags matching `v*` and requiring maintainer approval. Only the PyPI
publishing job receives `id-token: write`; no PyPI password or API token is
stored in GitHub.

## Prepare and validate a release

1. Update `VERSION`, the release heading/date in `CHANGELOG.md`, and any
   supported-runtime statement in `README.md` in a preparation pull request.
2. Keep third-party Actions pinned to immutable commit SHAs. Keep
   `build-constraints.txt` synchronized with the pure build constraints in
   `pyproject.toml`; do not reintroduce native build requirements.
3. Run the local checks:

   ```sh
   uv python install 3.14
   uv sync --locked --extra torch-cpu --extra hdf5
   uv run --no-sync python -m isort --check-only gmes examples tests utils setup.py
   uv run --no-sync python -m black --check gmes examples tests utils setup.py
   uv run --no-sync python -m mypy
   uv run --no-sync python -m pylint $(git ls-files 'gmes/*.py') setup.py
   uv run --no-sync python -m unittest discover -v
   uv lock --check
   uv build
   ```

4. Confirm the release workflow validates one universal wheel and one sdist,
   including archive contents, digest binding, and clean installed CPU checks
   outside the checkout. It must reuse those validated archives for publishing
   and release attachment.
5. Do not infer missing Linux/macOS CPU or trusted single-/two-GPU evidence
   from metadata, an artifact shape, or a skipped job. Those remain fail-closed
   gates for final acceptance.
6. Squash-merge only after the required CPU status checks and CodeQL pass.

## Publish the release

1. Create the signed tag from verified `master` and push only that tag:

   ```sh
   git tag -s v0.10.0 -m "GMES 0.10.0"
   git push origin v0.10.0
   ```

2. The tag workflow verifies the tag, package metadata, changelog heading,
   and GitHub Release title describe the same version.
3. Approve the protected `pypi` environment only after the verified archive
   set and installed-artifact checks succeed. The publishing job has no build
   step and rejects duplicate files; never enable `skip-existing`.
4. The GitHub Release attaches the exact verified wheel and sdist. If an
   uploaded distribution is wrong, publish a new version rather than replacing
   files or bypassing duplicate rejection.

No publication is authorized by this document; follow the protected workflow
and repository review policy.
