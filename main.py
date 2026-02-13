#!/usr/bin/env python3
"""
COLMAP ↔ PLY Tool
=================

Workflow:
  1. **export**  — Extract a .ply point cloud from a COLMAP text model.
  2. *(manual)*  — Filter the .ply in CloudCompare (or any other tool).
  3. **update**  — Reconcile the filtered .ply back into a new COLMAP model.

Usage examples
--------------

  # Step 1: export PLY from COLMAP
  python main.py export --colmap_dir ./sparse/0 --output cloud.ply

  # Step 2 (external): open cloud.ply in CloudCompare, filter, save as
  #   cloud_filtered.ply

  # Step 3: produce an updated COLMAP model
  python main.py update --colmap_dir ./sparse/0 \\
                        --filtered_ply cloud_filtered.ply \\
                        --output_dir ./sparse/0_filtered
"""

from __future__ import annotations

import argparse
import sys

from colmap_ply_tool.converter import export_ply, update_model


def cmd_export(args: argparse.Namespace) -> None:
    n = export_ply(args.colmap_dir, args.output)
    print(f"Exported {n} points to {args.output}")


def cmd_update(args: argparse.Namespace) -> None:
    stats = update_model(
        colmap_dir=args.colmap_dir,
        filtered_ply=args.filtered_ply,
        output_dir=args.output_dir,
        tol=args.tol,
    )
    print()
    print("=== Update summary ===")
    print(f"  Original 3-D points : {stats['original_points']}")
    print(f"  Surviving 3-D points: {stats['surviving_points']}")
    print(f"  Removed 3-D points  : {stats['removed_points']}")
    print(f"  2-D observations invalidated: {stats['observations_invalidated']}")
    print(f"  Images               : {stats['images_count']}")
    print(f"  Cameras              : {stats['cameras_count']}")
    print(f"\nUpdated model written to: {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert between COLMAP text models and PLY point clouds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- export -------------------------------------------------------
    p_export = subparsers.add_parser(
        "export",
        help="Export COLMAP points3D to a .ply file.",
    )
    p_export.add_argument(
        "--colmap_dir", required=True,
        help="Directory containing cameras.txt, images.txt, points3D.txt",
    )
    p_export.add_argument(
        "--output", required=True,
        help="Output .ply file path.",
    )
    p_export.set_defaults(func=cmd_export)

    # --- update -------------------------------------------------------
    p_update = subparsers.add_parser(
        "update",
        help="Update a COLMAP model from a filtered .ply file.",
    )
    p_update.add_argument(
        "--colmap_dir", required=True,
        help="Directory of the *original* COLMAP model.",
    )
    p_update.add_argument(
        "--filtered_ply", required=True,
        help="Path to the filtered .ply (output of CloudCompare).",
    )
    p_update.add_argument(
        "--output_dir", required=True,
        help="Directory where the updated COLMAP model will be saved.",
    )
    p_update.add_argument(
        "--tol", type=float, default=1e-6,
        help="Coordinate matching tolerance (used only when the PLY lacks "
             "the point3d_id field). Default: 1e-6.",
    )
    p_update.set_defaults(func=cmd_update)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
