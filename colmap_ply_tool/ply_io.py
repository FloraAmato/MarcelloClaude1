"""
Read and write PLY (Polygon File Format) point clouds.

Export includes a custom *point3d_id* property so that CloudCompare can
preserve the COLMAP point identity through filtering operations.  On
re-import the reader looks for that property first; if absent it falls
back to coordinate-based matching.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .colmap_io import Point3D


# ---------------------------------------------------------------------------
# Data class for a generic PLY vertex
# ---------------------------------------------------------------------------

@dataclass
class PlyVertex:
    x: float
    y: float
    z: float
    r: int = 0
    g: int = 0
    b: int = 0
    point3d_id: Optional[int] = None   # present only if exported by us


# ---------------------------------------------------------------------------
# Writer  (ASCII PLY — universally supported by CloudCompare)
# ---------------------------------------------------------------------------

def write_ply(filepath: str, points: Dict[int, Point3D]) -> None:
    """Export COLMAP points to an ASCII PLY file.

    Each vertex carries *point3d_id* as a scalar field so that
    CloudCompare can keep it after filtering / segmentation.
    """
    sorted_pts = sorted(points.values(), key=lambda p: p.point3d_id)

    with open(filepath, "w") as f:
        # --- header ---
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(sorted_pts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int point3d_id\n")
        f.write("end_header\n")

        # --- data ---
        for pt in sorted_pts:
            f.write(f"{pt.x} {pt.y} {pt.z} "
                    f"{pt.r} {pt.g} {pt.b} "
                    f"{pt.point3d_id}\n")


# ---------------------------------------------------------------------------
# Reader  (ASCII + little-endian binary PLY)
# ---------------------------------------------------------------------------

# Mapping from PLY type names to struct format characters and sizes.
_PLY_TYPE_MAP = {
    "char": ("b", 1), "int8": ("b", 1),
    "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2),
    "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4),
    "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4),
    "double": ("d", 8), "float64": ("d", 8),
}


def _parse_header(lines: List[str]):
    """Parse PLY header lines.

    Returns (format_str, vertex_count, properties) where *properties*
    is a list of (name, ply_type_string).
    """
    fmt = "ascii"
    vertex_count = 0
    properties: List[Tuple[str, str]] = []
    in_vertex_element = False

    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "format":
            fmt = tokens[1]           # "ascii", "binary_little_endian", …
        elif tokens[0] == "element":
            in_vertex_element = (tokens[1] == "vertex")
            if in_vertex_element:
                vertex_count = int(tokens[2])
        elif tokens[0] == "property" and in_vertex_element:
            # property <type> <name>
            if tokens[1] != "list":
                properties.append((tokens[2], tokens[1]))
        elif tokens[0] == "end_header":
            break

    return fmt, vertex_count, properties


def read_ply(filepath: str) -> List[PlyVertex]:
    """Read a PLY file and return a list of :class:`PlyVertex`.

    Supports ASCII and binary_little_endian formats.
    """
    # Read header (always ASCII text)
    header_lines: List[str] = []
    header_byte_length = 0
    with open(filepath, "rb") as f:
        while True:
            raw = f.readline()
            header_byte_length += len(raw)
            line = raw.decode("ascii", errors="replace").strip()
            header_lines.append(line)
            if line == "end_header":
                break

    fmt, vertex_count, properties = _parse_header(header_lines)

    # Build index maps for the properties we care about.
    prop_names = [name for name, _ in properties]
    idx_x = prop_names.index("x") if "x" in prop_names else None
    idx_y = prop_names.index("y") if "y" in prop_names else None
    idx_z = prop_names.index("z") if "z" in prop_names else None

    # Color: try "red"/"green"/"blue" first, then CloudCompare's short names.
    def _color_idx(long: str, short: str):
        if long in prop_names:
            return prop_names.index(long)
        if short in prop_names:
            return prop_names.index(short)
        return None

    idx_r = _color_idx("red", "r")
    idx_g = _color_idx("green", "g")
    idx_b = _color_idx("blue", "b")

    # point3d_id — CloudCompare may prefix scalar fields with "scalar_"
    idx_pid: Optional[int] = None
    for candidate in ("point3d_id", "scalar_point3d_id",
                       "scalar_Point3d_id", "scalar_Point3D_id"):
        if candidate in prop_names:
            idx_pid = prop_names.index(candidate)
            break

    vertices: List[PlyVertex] = []

    if fmt == "ascii":
        with open(filepath, "r") as f:
            # Skip header
            for _ in header_lines:
                f.readline()
            for _ in range(vertex_count):
                parts = f.readline().split()
                vals = [float(v) for v in parts]
                vertices.append(PlyVertex(
                    x=vals[idx_x] if idx_x is not None else 0.0,
                    y=vals[idx_y] if idx_y is not None else 0.0,
                    z=vals[idx_z] if idx_z is not None else 0.0,
                    r=int(vals[idx_r]) if idx_r is not None else 0,
                    g=int(vals[idx_g]) if idx_g is not None else 0,
                    b=int(vals[idx_b]) if idx_b is not None else 0,
                    point3d_id=(int(vals[idx_pid])
                                if idx_pid is not None else None),
                ))
    elif fmt == "binary_little_endian":
        struct_fmt = "<" + "".join(
            _PLY_TYPE_MAP[ptype][0] for _, ptype in properties)
        row_size = struct.calcsize(struct_fmt)

        with open(filepath, "rb") as f:
            f.seek(header_byte_length)
            for _ in range(vertex_count):
                raw = f.read(row_size)
                vals = struct.unpack(struct_fmt, raw)
                vertices.append(PlyVertex(
                    x=float(vals[idx_x]) if idx_x is not None else 0.0,
                    y=float(vals[idx_y]) if idx_y is not None else 0.0,
                    z=float(vals[idx_z]) if idx_z is not None else 0.0,
                    r=int(vals[idx_r]) if idx_r is not None else 0,
                    g=int(vals[idx_g]) if idx_g is not None else 0,
                    b=int(vals[idx_b]) if idx_b is not None else 0,
                    point3d_id=(int(vals[idx_pid])
                                if idx_pid is not None else None),
                ))
    else:
        raise ValueError(
            f"PLY format '{fmt}' not supported. "
            "Use ASCII or binary_little_endian.")

    return vertices
