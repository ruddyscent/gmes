"""Collective failure-contract probes for the two-GPU runner."""

from __future__ import annotations

import argparse
import json

import torch
import torch.distributed as dist

import gmes


def _runtime(launch, precision="float32"):
    return gmes.TorchRuntimeConfig(
        device=f"cuda:{launch.local_rank}",
        precision=precision,
        cpu_threads=1,
        launch=launch,
    )


def _simulation(launch, **kwargs):
    return gmes.TorchDistributedSimulation(
        space=gmes.Cartesian((4, 4, 4), 2),
        geometry=[gmes.DefaultMedium(gmes.Dielectric())],
        runtime=_runtime(launch, kwargs.pop("precision", "float32")),
        split_axis=0,
        **kwargs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "strict-peer",
            "dtype-mismatch",
            "checkpoint-mismatch",
            "rank-failure",
        ),
    )
    args = parser.parse_args()
    launch = gmes.distributed_launch_from_environment()
    caught = None
    simulation = None
    try:
        if args.mode == "strict-peer":
            simulation = _simulation(launch, require_peer_access=True)
        elif args.mode == "dtype-mismatch":
            simulation = _simulation(
                launch,
                precision="float32" if launch.rank == 0 else "float64",
            )
        else:
            simulation = _simulation(launch)
            if args.mode == "checkpoint-mismatch":
                checkpoint = simulation.checkpoint()
                if launch.rank == 0:
                    checkpoint["plan_identity"] = "mismatch"
                simulation.load_checkpoint(checkpoint)
            else:
                if launch.rank == 0:

                    def fail(*_args, **_kwargs):
                        raise RuntimeError("injected rank-local failure")

                    simulation.local._electric = fail
                simulation.advance(1)
    except Exception as error:  # The failure itself is the expected evidence.
        caught = f"{type(error).__name__}: {error}"
        if args.mode == "rank-failure":
            raise

    passed = caught is not None
    if dist.is_initialized() and args.mode != "rank-failure":
        flag = torch.tensor(
            [int(passed)],
            device=f"cuda:{launch.local_rank}",
            dtype=torch.int32,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        passed = bool(int(flag.cpu()))
    if launch.rank == 0:
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "passed": passed,
                    "rank0_error": caught,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if simulation is not None and dist.is_initialized():
        simulation.close()
    elif dist.is_initialized():
        dist.destroy_process_group()
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
