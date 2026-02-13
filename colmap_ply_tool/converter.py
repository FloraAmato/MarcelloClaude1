"""
High-level conversion logic:

1. export_ply   – COLMAP model  →  PLY point cloud
2. update_model – filtered PLY + original COLMAP model  →  updated COLMAP model
"""

from __future__ import annotations

import copy
import math
import os
from typing import Dict, List, Optional, Set, Tuple

from .colmap_io import (Camera, Image, Point2D, Point3D,
                        load_colmap_model, save_colmap_model)
from .ply_io import PlyVertex, read_ply, write_ply


# ---------------------------------------------------------------------------
# Step 1: export
# ---------------------------------------------------------------------------

def export_ply(colmap_dir: str, output_ply: str) -> int:
    """Read a COLMAP text model and write a PLY file.

    Returns the number of exported vertices.
    """
    _, _, _, _, points3d, _ = load_colmap_model(colmap_dir)
    write_ply(output_ply, points3d)
    return len(points3d)


# ---------------------------------------------------------------------------
# Step 2: identify surviving point IDs from the filtered PLY
# ---------------------------------------------------------------------------

def _match_by_id(vertices: List[PlyVertex]) -> Optional[Set[int]]:
    """Try to extract surviving point3d_ids directly from PLY vertices."""
    ids: Set[int] = set()
    for v in vertices:
        if v.point3d_id is None:
            return None        # at least one vertex lacks the field
        ids.add(v.point3d_id)
    return ids


def _match_by_coords(vertices: List[PlyVertex],
                     points3d: Dict[int, Point3D],
                     tol: float = 1e-6) -> Set[int]:
    """Fall-back: match filtered PLY vertices to COLMAP points by coords.

    Uses a spatial hash grid for efficiency (O(N+M) expected time).
    """

    def _key(x: float, y: float, z: float) -> Tuple[int, int, int]:
        return (round(x / tol), round(y / tol), round(z / tol))

    # Build a grid map: hash-key → list of point3d_ids
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    for pt in points3d.values():
        k = _key(pt.x, pt.y, pt.z)
        grid.setdefault(k, []).append(pt.point3d_id)

    surviving: Set[int] = set()
    unmatched = 0

    for v in vertices:
        k = _key(v.x, v.y, v.z)
        # Search the key and immediate neighbours (handles rounding edge cases)
        found = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    neighbour = (k[0] + dx, k[1] + dy, k[2] + dz)
                    for pid in grid.get(neighbour, []):
                        pt = points3d[pid]
                        if (abs(pt.x - v.x) < tol and
                                abs(pt.y - v.y) < tol and
                                abs(pt.z - v.z) < tol):
                            surviving.add(pid)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        if not found:
            unmatched += 1

    if unmatched > 0:
        print(f"[warning] {unmatched} filtered PLY vertices could not be "
              f"matched to any original COLMAP point (tolerance={tol}).")

    return surviving


def identify_surviving_ids(vertices: List[PlyVertex],
                           points3d: Dict[int, Point3D],
                           tol: float = 1e-6) -> Set[int]:
    """Return the set of COLMAP point3d_ids that survived filtering.

    Strategy:
      1. If every vertex carries a *point3d_id* scalar  →  use it directly.
      2. Otherwise fall back to coordinate matching with *tol* tolerance.
    """
    ids = _match_by_id(vertices)
    if ids is not None:
        print(f"[info] Matching by point3d_id field — "
              f"{len(ids)} surviving points.")
        return ids

    print("[info] point3d_id field not found in PLY; "
          "falling back to coordinate matching …")
    ids = _match_by_coords(vertices, points3d, tol)
    print(f"[info] Coordinate matching complete — "
          f"{len(ids)} surviving points.")
    return ids


# ---------------------------------------------------------------------------
# Step 3: reconcile COLMAP model
# ---------------------------------------------------------------------------

def update_model(colmap_dir: str,
                 filtered_ply: str,
                 output_dir: str,
                 tol: float = 1e-6) -> dict:
    """Read original COLMAP model + filtered PLY, write updated model.

    Updates applied:
      • points3D.txt — only surviving points are kept.
      • images.txt   — 2-D observations that referenced a removed 3-D point
                        have their point3d_id reset to -1.
      • cameras.txt  — copied unchanged (camera intrinsics are unaffected).

    Returns a summary dict with statistics.
    """
    (cameras, cam_comments,
     images, img_comments,
     points3d, pts_comments) = load_colmap_model(colmap_dir)

    vertices = read_ply(filtered_ply)
    surviving_ids = identify_surviving_ids(vertices, points3d, tol=tol)

    removed_ids: Set[int] = set(points3d.keys()) - surviving_ids

    # --- update points3D ---
    new_points3d: Dict[int, Point3D] = {
        pid: pt for pid, pt in points3d.items() if pid in surviving_ids
    }

    # --- update images (2-D observations) ---
    observations_removed = 0
    new_images: Dict[int, Image] = {}
    for img_id, img in images.items():
        new_pts2d: List[Point2D] = []
        for p2d in img.points2d:
            if p2d.point3d_id in removed_ids:
                new_pts2d.append(Point2D(x=p2d.x, y=p2d.y, point3d_id=-1))
                observations_removed += 1
            else:
                new_pts2d.append(p2d)
        new_img = copy.copy(img)
        new_img.points2d = new_pts2d
        new_images[img_id] = new_img

    # --- update track lists for surviving points ---
    # Remove track entries whose referenced 2-D observation was invalidated.
    for pid, pt in new_points3d.items():
        new_track: List[Tuple[int, int]] = []
        for (track_img_id, track_pt2d_idx) in pt.track:
            img = new_images.get(track_img_id)
            if img is None:
                continue
            if track_pt2d_idx < len(img.points2d):
                if img.points2d[track_pt2d_idx].point3d_id == pid:
                    new_track.append((track_img_id, track_pt2d_idx))
        pt.track = new_track

    # --- save ---
    save_colmap_model(output_dir,
                      cameras, cam_comments,
                      new_images, img_comments,
                      new_points3d, pts_comments)

    stats = {
        "original_points": len(points3d),
        "surviving_points": len(new_points3d),
        "removed_points": len(removed_ids),
        "observations_invalidated": observations_removed,
        "images_count": len(new_images),
        "cameras_count": len(cameras),
    }
    return stats
