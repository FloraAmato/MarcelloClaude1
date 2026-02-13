"""
Parsing and writing of COLMAP text-format files:
  cameras.txt, images.txt, points3D.txt

Reference:
  https://colmap.github.io/format.html#text-format
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: List[float]


@dataclass
class Point2D:
    x: float
    y: float
    point3d_id: int          # -1 means no 3D association


@dataclass
class Image:
    image_id: int
    qw: float
    qx: float
    qy: float
    qz: float
    tx: float
    ty: float
    tz: float
    camera_id: int
    name: str
    points2d: List[Point2D] = field(default_factory=list)


@dataclass
class Point3D:
    point3d_id: int
    x: float
    y: float
    z: float
    r: int
    g: int
    b: int
    error: float
    track: List[Tuple[int, int]] = field(default_factory=list)   # (image_id, point2d_idx)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_cameras(filepath: str) -> Tuple[Dict[int, Camera], List[str]]:
    """Return {camera_id: Camera} and list of header comment lines."""
    cameras: Dict[int, Camera] = {}
    comments: List[str] = []
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                comments.append(line.rstrip("\n"))
                continue
            parts = stripped.split()
            cam_id = int(parts[0])
            cameras[cam_id] = Camera(
                camera_id=cam_id,
                model=parts[1],
                width=int(parts[2]),
                height=int(parts[3]),
                params=[float(p) for p in parts[4:]],
            )
    return cameras, comments


def parse_images(filepath: str) -> Tuple[Dict[int, Image], List[str]]:
    """Return {image_id: Image} and list of header comment lines.

    images.txt uses **two** lines per image:
      line 1: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
      line 2: POINTS2D[] as (X, Y, POINT3D_ID) ...
    """
    images: Dict[int, Image] = {}
    comments: List[str] = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == "" or stripped.startswith("#"):
            comments.append(lines[i].rstrip("\n"))
            i += 1
            continue

        # --- first line: pose ---
        parts = stripped.split()
        image_id = int(parts[0])
        qw, qx, qy, qz = (float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]))
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        # --- second line: 2-D observations ---
        points2d: List[Point2D] = []
        i += 1
        if i < len(lines):
            pts_line = lines[i].strip()
            if pts_line and not pts_line.startswith("#"):
                pts_parts = pts_line.split()
                for j in range(0, len(pts_parts), 3):
                    points2d.append(Point2D(
                        x=float(pts_parts[j]),
                        y=float(pts_parts[j + 1]),
                        point3d_id=int(pts_parts[j + 2]),
                    ))

        images[image_id] = Image(
            image_id=image_id,
            qw=qw, qx=qx, qy=qy, qz=qz,
            tx=tx, ty=ty, tz=tz,
            camera_id=camera_id,
            name=name,
            points2d=points2d,
        )
        i += 1

    return images, comments


def parse_points3d(filepath: str) -> Tuple[Dict[int, Point3D], List[str]]:
    """Return {point3d_id: Point3D} and list of header comment lines."""
    points: Dict[int, Point3D] = {}
    comments: List[str] = []
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                comments.append(line.rstrip("\n"))
                continue
            parts = stripped.split()
            pid = int(parts[0])
            track: List[Tuple[int, int]] = []
            for j in range(8, len(parts), 2):
                track.append((int(parts[j]), int(parts[j + 1])))
            points[pid] = Point3D(
                point3d_id=pid,
                x=float(parts[1]), y=float(parts[2]), z=float(parts[3]),
                r=int(parts[4]), g=int(parts[5]), b=int(parts[6]),
                error=float(parts[7]),
                track=track,
            )
    return points, comments


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_cameras(filepath: str, cameras: Dict[int, Camera],
                  comments: List[str] | None = None) -> None:
    with open(filepath, "w") as f:
        if comments:
            for c in comments:
                f.write(c + "\n")
        for cam in sorted(cameras.values(), key=lambda c: c.camera_id):
            params_str = " ".join(f"{p}" for p in cam.params)
            f.write(f"{cam.camera_id} {cam.model} "
                    f"{cam.width} {cam.height} {params_str}\n")


def write_images(filepath: str, images: Dict[int, Image],
                 comments: List[str] | None = None) -> None:
    with open(filepath, "w") as f:
        if comments:
            for c in comments:
                f.write(c + "\n")
        for img in sorted(images.values(), key=lambda im: im.image_id):
            f.write(f"{img.image_id} "
                    f"{img.qw} {img.qx} {img.qy} {img.qz} "
                    f"{img.tx} {img.ty} {img.tz} "
                    f"{img.camera_id} {img.name}\n")
            pts_tokens: List[str] = []
            for p in img.points2d:
                pts_tokens.append(f"{p.x} {p.y} {p.point3d_id}")
            f.write(" ".join(pts_tokens) + "\n")


def write_points3d(filepath: str, points: Dict[int, Point3D],
                   comments: List[str] | None = None) -> None:
    with open(filepath, "w") as f:
        if comments:
            for c in comments:
                f.write(c + "\n")
        for pt in sorted(points.values(), key=lambda p: p.point3d_id):
            track_str = " ".join(f"{im} {idx}" for im, idx in pt.track)
            f.write(f"{pt.point3d_id} "
                    f"{pt.x} {pt.y} {pt.z} "
                    f"{pt.r} {pt.g} {pt.b} "
                    f"{pt.error} {track_str}\n")


# ---------------------------------------------------------------------------
# Convenience: load / save full COLMAP model
# ---------------------------------------------------------------------------

def load_colmap_model(model_dir: str):
    """Load cameras, images, points3D from *model_dir*.

    Returns (cameras, images, points3d) dicts plus their comments.
    """
    cameras, cam_comments = parse_cameras(
        os.path.join(model_dir, "cameras.txt"))
    images, img_comments = parse_images(
        os.path.join(model_dir, "images.txt"))
    points3d, pts_comments = parse_points3d(
        os.path.join(model_dir, "points3D.txt"))
    return (cameras, cam_comments,
            images, img_comments,
            points3d, pts_comments)


def save_colmap_model(output_dir: str,
                      cameras: Dict[int, Camera], cam_comments: List[str],
                      images: Dict[int, Image], img_comments: List[str],
                      points3d: Dict[int, Point3D], pts_comments: List[str],
                      ) -> None:
    os.makedirs(output_dir, exist_ok=True)
    write_cameras(os.path.join(output_dir, "cameras.txt"),
                  cameras, cam_comments)
    write_images(os.path.join(output_dir, "images.txt"),
                 images, img_comments)
    write_points3d(os.path.join(output_dir, "points3D.txt"),
                   points3d, pts_comments)
