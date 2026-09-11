# Contributing to GMES

GMES targets Python 3.14 or newer and the supported solver is the pure-Python
Torch runtime. Python 2, the legacy solver/API, native GMES extensions, SWIG,
and OpenMP rebuild modes are out of scope.

## Development setup

Install [uv](https://docs.astral.sh/uv/), then use the locked CPU environment:

```sh
uv python install 3.14
uv sync --locked --extra torch-cpu --extra hdf5
uv run --no-sync python -m pytest -v
uv build
```

No compiler, SWIG, Cython, OpenMP runtime, or MPI installation is required to
build GMES. The `dev` dependencies are installed by default as a PEP 735
group. Select exactly one Torch extra (`torch-cpu`, `torch-cu126`, or
`torch-cu130`); CUDA extras are Linux deployment choices, not evidence that a
GPU gate ran. `plot` and `hdf5` remain independent optional extras.

`uv sync --locked` consumes the committed lockfile. Update dependencies only
in a dedicated change with `uv lock --upgrade`, then verify the lock and the
affected tests. The pure build constraints cover setuptools and wheel; do not
restore a native build dependency or cache key. `tests/test_packaging.py` and
`tests/test_release.py` check the universal artifact contract.

## Codex development settings

The project defaults to GPT-6 Astra (`gpt-6-astra`) with `low` reasoning effort
in [`.codex/config.toml`](.codex/config.toml). These settings configure the
development assistant; GMES does not require an OpenAI API integration or key.
For complex issue or pull request management, select `high` when warranted
as described in [`AGENTS.md`](AGENTS.md).

Codex loads project configuration only for trusted projects. The precedence
among these settings is command-line overrides, project configuration,
the selected profile, then user-level defaults;
see the [official configuration documentation](https://learn.chatgpt.com/docs/config-file/config-basic).
Before evaluating a task, check its effective model and reasoning effort in
the session's model controls (or `/status` in the CLI). Record those settings
with the revision, task, and verification results. Reading this file alone
does not establish an existing task's active settings or account access, and
editing project configuration does not switch an already-running task.

For a temporary CLI override, use the previously recorded values:

```sh
codex -m <previous-model-id> -c 'model_reasoning_effort="<previous-effort>"'
```

To roll back the project defaults, restore the previous model settings
or remove these two project keys to inherit the next applicable profile or
user defaults, and update the
project model guidance in `AGENTS.md` accordingly. Start a new session and
verify its effective settings after either change. Compare representative
tasks before claiming any improvement in quality, latency, or usage.

## Verification

Run the narrowest relevant tests, then the complete supported CPU suite:

```sh
uv run --no-sync python -m pytest -v tests/test_torch_fdtd.py
uv run --no-sync python -m isort --check-only gmes examples tests benchmarks utils setup.py
uv run --no-sync python -m black --check gmes examples tests benchmarks utils setup.py
uv run --no-sync python -m mypy
uv run --no-sync python -m pylint gmes setup.py
uv run --no-sync python -m pytest -v
uv build
```

Numerical changes need deterministic regression coverage. Configure device,
dtype, CPU threads, and `compile_policy` explicitly; account for compilation
warmup before reporting performance. Checkpoints, probes, host snapshots, and
plotting are explicit observation/output boundaries. Do not use large examples
as routine tests.

New tests use pytest-native functions or plain `Test*` classes with direct
assertions. Use `pytest.mark.parametrize` with stable semantic IDs for
independent cases, but retain intentional stateful sequences. Focused fixtures
and finalizers must restore patches, temporary resources, and runtime state on
failure. Future Taflove reproduction tests follow these rules and must state
their device, dtype, thread, and compile-policy requirements.

The required protected status names remain `Python 3.14 / ubuntu-latest` and
`Python 3.14 / macos-latest`; CodeQL must also complete. Trusted single- and
two-GPU installed-artifact gates are separate fail-closed release work and
must not be represented by an untrusted PR runner or a skipped CUDA test.

Historical OpenMP benchmark data is retained in
[`docs/openmp-benchmark.md`](docs/openmp-benchmark.md); it is not a current
tuning or build procedure. Production distributions are created only by the
tag-triggered release workflow. Follow [`docs/releasing.md`](docs/releasing.md)
and never publish or reuse files from a local `dist/` directory.

## Change scope and compatibility

- Keep public API changes explicit and document their migration impact.
- Preserve numerical behavior unless the change is intentional and covered by tests.
- Keep Torch device, dtype, compilation, checkpoint, and source contracts explicit.
- Update the README or examples whenever a user-facing workflow changes.

## Commit messages

Use Conventional Commit titles in the form `<type>(<scope>): <summary>` or
`<type>: <summary>`. Keep the title imperative and add a body explaining the
compatibility or numerical implications and verification performed.
