#!/usr/bin/env python3
"""
End-to-end test for the COLMAP ↔ PLY workflow.

Creates a synthetic COLMAP model, exports to PLY, simulates CloudCompare
filtering (removes some points), then updates the COLMAP model and
verifies consistency.
"""

import os
import shutil
import tempfile

from colmap_ply_tool.colmap_io import (
    load_colmap_model, parse_cameras, parse_images, parse_points3d,
)
from colmap_ply_tool.converter import export_ply, update_model
from colmap_ply_tool.ply_io import read_ply, write_ply


def create_synthetic_colmap(model_dir: str) -> None:
    """Write a minimal but realistic COLMAP text model."""
    os.makedirs(model_dir, exist_ok=True)

    # --- cameras.txt ---
    with open(os.path.join(model_dir, "cameras.txt"), "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        f.write("1 PINHOLE 1920 1080 1500.0 1500.0 960.0 540.0\n")

    # --- images.txt ---
    # 2 images, each observes some subset of 5 3-D points.
    with open(os.path.join(model_dir, "images.txt"), "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write("# Number of images: 2\n")
        # Image 1 observes points 1,2,3,4 (point2d_idx 0..3)
        f.write("1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 img001.jpg\n")
        f.write("100.0 200.0 1 300.0 400.0 2 500.0 600.0 3 700.0 800.0 4\n")
        # Image 2 observes points 2,3,5 and one unassociated (-1)
        f.write("2 0.9 0.1 0.0 0.0 1.0 0.0 0.0 1 img002.jpg\n")
        f.write("150.0 250.0 2 350.0 450.0 3 550.0 650.0 5 750.0 850.0 -1\n")

    # --- points3D.txt ---
    with open(os.path.join(model_dir, "points3D.txt"), "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write("# Number of points: 5\n")
        f.write("1 1.0 2.0 3.0 255 0 0 0.1 1 0\n")
        f.write("2 4.0 5.0 6.0 0 255 0 0.2 1 1 2 0\n")
        f.write("3 7.0 8.0 9.0 0 0 255 0.3 1 2 2 1\n")
        f.write("4 10.0 11.0 12.0 128 128 0 0.4 1 3\n")
        f.write("5 13.0 14.0 15.0 0 128 128 0.5 2 2\n")


def simulate_cloudcompare_filter(input_ply: str, output_ply: str,
                                  remove_ids: set) -> None:
    """Simulate CloudCompare by reading a PLY and writing it back without
    certain points.  Preserves the point3d_id scalar field."""
    vertices = read_ply(input_ply)
    kept = [v for v in vertices if v.point3d_id not in remove_ids]

    # Re-write as ASCII PLY (same format CloudCompare would produce)
    with open(output_ply, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(kept)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int point3d_id\n")
        f.write("end_header\n")
        for v in kept:
            f.write(f"{v.x} {v.y} {v.z} {v.r} {v.g} {v.b} {v.point3d_id}\n")


def simulate_cloudcompare_filter_no_id(input_ply: str,
                                        output_ply: str,
                                        remove_ids: set) -> None:
    """Same as above but drops the point3d_id field — tests the
    coordinate-matching fallback."""
    vertices = read_ply(input_ply)
    kept = [v for v in vertices if v.point3d_id not in remove_ids]

    with open(output_ply, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(kept)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for v in kept:
            f.write(f"{v.x} {v.y} {v.z} {v.r} {v.g} {v.b}\n")


def run_test(use_id_field: bool) -> None:
    tag = "with point3d_id" if use_id_field else "coordinate fallback"
    print(f"\n{'='*60}")
    print(f"  TEST: {tag}")
    print(f"{'='*60}\n")

    tmpdir = tempfile.mkdtemp(prefix="colmap_ply_test_")
    try:
        model_dir = os.path.join(tmpdir, "model")
        create_synthetic_colmap(model_dir)

        # --- Step 1: export ---
        ply_path = os.path.join(tmpdir, "cloud.ply")
        n = export_ply(model_dir, ply_path)
        print(f"[export] {n} points written to {ply_path}")
        assert n == 5, f"Expected 5 points, got {n}"

        # Read back and verify
        verts = read_ply(ply_path)
        assert len(verts) == 5
        ids_exported = {v.point3d_id for v in verts}
        assert ids_exported == {1, 2, 3, 4, 5}
        print("[export] PLY read-back OK")

        # --- Step 2: simulate filtering (remove points 1 and 4) ---
        filtered_ply = os.path.join(tmpdir, "cloud_filtered.ply")
        remove = {1, 4}
        if use_id_field:
            simulate_cloudcompare_filter(ply_path, filtered_ply, remove)
        else:
            simulate_cloudcompare_filter_no_id(ply_path, filtered_ply, remove)
        print(f"[filter] Removed points {remove}")

        # --- Step 3: update model ---
        updated_dir = os.path.join(tmpdir, "model_updated")
        stats = update_model(model_dir, filtered_ply, updated_dir)
        print(f"[update] stats = {stats}")

        assert stats["original_points"] == 5
        assert stats["surviving_points"] == 3
        assert stats["removed_points"] == 2

        # Verify updated files
        cams, _ = parse_cameras(
            os.path.join(updated_dir, "cameras.txt"))
        assert len(cams) == 1, "Cameras should be unchanged"

        imgs, _ = parse_images(
            os.path.join(updated_dir, "images.txt"))
        assert len(imgs) == 2, "Both images should still exist"

        pts, _ = parse_points3d(
            os.path.join(updated_dir, "points3D.txt"))
        assert set(pts.keys()) == {2, 3, 5}, (
            f"Expected surviving points {{2,3,5}}, got {set(pts.keys())}")

        # Image 1: points2d originally (1,2,3,4) → now (-1,2,3,-1)
        img1 = imgs[1]
        img1_ids = [p.point3d_id for p in img1.points2d]
        assert img1_ids == [-1, 2, 3, -1], (
            f"Image 1 observations wrong: {img1_ids}")

        # Image 2: points2d originally (2,3,5,-1) → unchanged
        img2 = imgs[2]
        img2_ids = [p.point3d_id for p in img2.points2d]
        assert img2_ids == [2, 3, 5, -1], (
            f"Image 2 observations wrong: {img2_ids}")

        # Check tracks are consistent
        for pid, pt in pts.items():
            for (track_img_id, track_pt2d_idx) in pt.track:
                img = imgs[track_img_id]
                assert track_pt2d_idx < len(img.points2d)
                assert img.points2d[track_pt2d_idx].point3d_id == pid, (
                    f"Track inconsistency: point {pid} track references "
                    f"image {track_img_id}[{track_pt2d_idx}] but that "
                    f"observation points to "
                    f"{img.points2d[track_pt2d_idx].point3d_id}")

        print(f"\n  ALL CHECKS PASSED ({tag})\n")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    run_test(use_id_field=True)
    run_test(use_id_field=False)
    print("\n" + "="*60)
    print("  ALL TESTS PASSED")
    print("="*60)
