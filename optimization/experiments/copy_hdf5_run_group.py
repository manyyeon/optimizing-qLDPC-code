#!/usr/bin/env python3
"""Copy or move one HDF5 run group from one result file to another."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def copy_run_group(
    src_file: Path,
    dst_file: Path,
    code_name: str,
    run_name: str,
    dst_run_name: str | None = None,
    overwrite: bool = False,
    delete_source: bool = False,
):
    dst_run_name = dst_run_name or run_name

    if src_file.resolve() == dst_file.resolve():
        raise ValueError("Source and destination files must be different.")

    with h5py.File(src_file, "a" if delete_source else "r") as src, h5py.File(dst_file, "a") as dst:
        if code_name not in src:
            raise KeyError(f"{code_name!r} not found in source file. Available: {list(src.keys())}")

        src_code_grp = src[code_name]

        if run_name not in src_code_grp:
            raise KeyError(
                f"{run_name!r} not found under {code_name!r}. "
                f"Available runs: {list(src_code_grp.keys())}"
            )

        src_run_grp = src_code_grp[run_name]
        dst_code_grp = dst.require_group(code_name)

        if dst_run_name in dst_code_grp:
            if overwrite:
                print(f"Deleting existing destination group: /{code_name}/{dst_run_name}")
                del dst_code_grp[dst_run_name]
            else:
                raise KeyError(
                    f"Destination group /{code_name}/{dst_run_name} already exists. "
                    "Use --overwrite to replace it."
                )

        print(f"Copying:")
        print(f"  from: {src_file}:/{code_name}/{run_name}")
        print(f"  to:   {dst_file}:/{code_name}/{dst_run_name}")

        src.copy(src_run_grp, dst_code_grp, name=dst_run_name)

        # Quick verification
        copied_grp = dst_code_grp[dst_run_name]
        print("Copied group keys:", list(copied_grp.keys()))
        print("Copied attrs:", dict(copied_grp.attrs))

        if delete_source:
            print(f"Deleting source group: /{code_name}/{run_name}")
            del src_code_grp[run_name]

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", required=True, type=Path)
    parser.add_argument("--dst-file", required=True, type=Path)
    parser.add_argument("--code", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dst-run-name", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="If set, delete the run group from the source file after copying.",
    )
    args = parser.parse_args()

    copy_run_group(
        src_file=args.src_file,
        dst_file=args.dst_file,
        code_name=args.code,
        run_name=args.run_name,
        dst_run_name=args.dst_run_name,
        overwrite=args.overwrite,
        delete_source=args.delete_source,
    )


if __name__ == "__main__":
    main()