#!/usr/bin/env python3
"""Render a dimensioned, marker-accurate picture of an AprilTag cube model.

Every coordinate comes from ``AprilTagCubeModel`` and the config being rendered
-- nothing is retyped -- so the figures are a direct visual audit of the geometry
the pipeline actually uses. The tags are the real DICT_APRILTAG_36h11 bitmaps,
pasted with the model's own corner convention (p0..p3 clockwise seen from
outside) and ``face_roll_deg``, so a printed cube can be checked sticker by
sticker.

The marker model only knows marker planes, so the surrounding solid is supplied
separately: by default the body + ``+Z`` protrusion implied by config.py, or an
explicit ``--solid`` JSON for a cube of a different shape (see
targets/gt_cube/cube_solid.json).

Outputs
  cube_model_geometry.png  3D views + dimensioned front/right/top elevations
  cube_model_net.png       unfolded net: what each face must look like from outside
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as Polygon2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration_pipeline.apriltag_cube import (  # noqa: E402
    AprilTagCubeModel,
    validate_cube_config,
)
from calibration_pipeline.config import (  # noqa: E402
    CUBE_BODY_BOTTOM_Z_M,
    CUBE_BODY_TOP_Z_M,
    TOP_MARKER_PLANE_Z_M,
    TOP_PROTRUSION_HEIGHT_M,
    CubeConfig,
    get_default_cube_config,
)
from calibration_pipeline.cube_config import (  # noqa: E402
    load_cube_config_from_json_file,
    load_cube_config_from_meta,
)
from calibration_pipeline.result_paths import ABLATION_RESULT_ROOT

BG = "#11151b"
PANEL = "#161b22"
INK = "#e6edf3"
MUTED = "#9aa4b0"
GRID = "#39424e"
SURFACE = "#39424e"
EDGE = "#7d8896"
DIM = "#ffb347"
TIER_COLORS = ("#39424e", "#4a5866", "#586a7a", "#647889", "#708698")
ID_COLORS = {
    0: "#f2a31a", 1: "#f05238", 2: "#33a8f0",
    3: "#2ec27e", 4: "#946bf0", 5: "#f259b3",
}
# Printed sticker margin (quiet zone) drawn around the black tag border, as a
# fraction of the marker side. Visual aid for contrast only; the model's marker
# size is the black border, so this margin must stay inside the face bounds.
QUIET_ZONE = 0.05
TAG_LIFT_MM = 0.4  # draw tags this far outside the surface so they are not z-fought


# --------------------------------------------------------------------------- #
# the solid around the markers
# --------------------------------------------------------------------------- #
@dataclass
class Box:
    """An axis-aligned block of the physical cube, in millimetres."""

    label: str
    x: Tuple[float, float]
    y: Tuple[float, float]
    z: Tuple[float, float]
    color: str = SURFACE

    @property
    def size(self) -> Tuple[float, float, float]:
        return (self.x[1] - self.x[0], self.y[1] - self.y[0], self.z[1] - self.z[0])

    def center(self) -> np.ndarray:
        return np.array([sum(self.x) / 2.0, sum(self.y) / 2.0, sum(self.z) / 2.0])

    def faces(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Six faces as (quad[4x3], outward normal)."""
        (x0, x1), (y0, y1), (z0, z1) = self.x, self.y, self.z
        return [
            (np.array([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]]), np.array([0, 0, 1.0])),
            (np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]]), np.array([0, 0, -1.0])),
            (np.array([[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]]), np.array([1.0, 0, 0])),
            (np.array([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]]), np.array([-1.0, 0, 0])),
            (np.array([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]]), np.array([0, 1.0, 0])),
            (np.array([[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]]), np.array([0, -1.0, 0])),
        ]

    def edges(self) -> List[np.ndarray]:
        corners = np.array([[cx, cy, cz] for cz in self.z for cy in self.y for cx in self.x])
        pairs = [(0, 1), (1, 3), (3, 2), (2, 0), (4, 5), (5, 7), (7, 6), (6, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]
        return [corners[list(pair)] for pair in pairs]


def default_solid(cfg: CubeConfig, model: AprilTagCubeModel) -> List[Box]:
    """Body + ``+Z`` protrusion implied by config.py and the rendered config.

    The footprint and the top-marker plane are read back from the config, so a
    frozen session config (e.g. the pre-2026-09 cube) is drawn as it really was
    instead of being mixed with the current module constants. The 57mm body and
    the origin offset never moved across that CAD revision, so they still come
    from config.py.
    """
    half = float(cfg.cube_side_m) * 500.0
    top_ids = [int(m) for m in cfg.marker_ids if cfg.id_to_face.get(int(m)) == "+Z"]
    if top_ids:
        top_z = float(np.mean([model.marker_pose_in_rig(m)[2, 3] for m in top_ids])) * 1000.0
    else:
        top_z = TOP_MARKER_PLANE_Z_M * 1000.0
    body_hi, body_lo = CUBE_BODY_TOP_Z_M * 1000.0, CUBE_BODY_BOTTOM_Z_M * 1000.0

    boxes = [Box(f"body {2 * half:.0f}x{2 * half:.0f}x{body_hi - body_lo:.0f}",
                 (-half, half), (-half, half), (body_lo, body_hi), TIER_COLORS[0])]
    if top_z - body_hi > 0.05:
        boxes.append(Box(f"+Z protrusion {top_z - body_hi:.0f}",
                         (-half, half), (-half, half), (body_hi, top_z), TIER_COLORS[1]))
    return boxes


def load_solid(path: str) -> Tuple[List[Box], str]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if str(data.get("units", "mm")).lower() != "mm":
        raise SystemExit(f"{path}: only 'mm' units are supported")
    boxes = []
    for index, entry in enumerate(data["boxes"]):
        boxes.append(Box(
            label=str(entry.get("label", f"box {index}")),
            x=(float(entry["x"][0]), float(entry["x"][1])),
            y=(float(entry["y"][0]), float(entry["y"][1])),
            z=(float(entry["z"][0]), float(entry["z"][1])),
            color=str(entry.get("color", TIER_COLORS[index % len(TIER_COLORS)])),
        ))
    return boxes, str(data.get("name", os.path.basename(path)))


def envelope(boxes: Sequence[Box]) -> Dict[str, float]:
    return {
        "x0": min(b.x[0] for b in boxes), "x1": max(b.x[1] for b in boxes),
        "y0": min(b.y[0] for b in boxes), "y1": max(b.y[1] for b in boxes),
        "z0": min(b.z[0] for b in boxes), "z1": max(b.z[1] for b in boxes),
    }


# --------------------------------------------------------------------------- #
# tags
# --------------------------------------------------------------------------- #
def tag_bitmap(dictionary_name: str, marker_id: int) -> np.ndarray:
    """Return the marker as an NxN boolean grid (True = white cell)."""
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    cells = int(dictionary.markerSize) + 2  # data bits + 1-cell black border
    scale = 8
    img = cv2.aruco.generateImageMarker(dictionary, int(marker_id), cells * scale, borderBits=1)
    return img[scale // 2::scale, scale // 2::scale][:cells, :cells] > 127


def tag_quads(model: AprilTagCubeModel, marker_id: int, dictionary_name: str,
              lift_mm: float = TAG_LIFT_MM) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
    """Cell quads of the printed tag in rig coordinates [mm], plus the sticker quad.

    The bitmap is mapped with the model's own corner order: p0 = the tag's own
    top-left, p1 = top-right, p2 = bottom-right, p3 = bottom-left (clockwise seen
    from outside the cube), so a convention error shows up as a rotated tag
    rather than being hidden.
    """
    bits = tag_bitmap(dictionary_name, marker_id)
    n = bits.shape[0]
    normal = model.marker_pose_in_rig(marker_id)[:3, 2]
    corners = model.marker_corners_in_rig(marker_id) * 1000.0 + lift_mm * normal
    p0, p1, _p2, p3 = corners

    def point(a: float, b: float) -> np.ndarray:  # a: left->right, b: top->bottom
        return p0 + a * (p1 - p0) + b * (p3 - p0)

    quads, colors = [], []
    for r in range(n):
        for c in range(n):
            a0, a1, b0, b1 = c / n, (c + 1) / n, r / n, (r + 1) / n
            quads.append(np.array([point(a0, b0), point(a1, b0), point(a1, b1), point(a0, b1)]))
            colors.append("#f5f5f0" if bits[r, c] else "#101010")

    m = QUIET_ZONE
    sticker = np.array([point(-m, -m), point(1 + m, -m), point(1 + m, 1 + m), point(-m, 1 + m)])
    return quads, colors, sticker


# --------------------------------------------------------------------------- #
# 3D panel
# --------------------------------------------------------------------------- #
def draw_3d(ax, cfg: CubeConfig, model: AprilTagCubeModel, boxes: Sequence[Box],
            elev: float, azim: float, title: str) -> None:
    cam = np.array([
        np.cos(np.deg2rad(elev)) * np.cos(np.deg2rad(azim)),
        np.cos(np.deg2rad(elev)) * np.sin(np.deg2rad(azim)),
        np.sin(np.deg2rad(elev)),
    ])
    quads: List[np.ndarray] = []
    facecolors: List[str] = []
    edgecolors: List[str] = []

    for box in boxes:
        for quad, normal in box.faces():
            if float(np.dot(normal, cam)) <= 0.02:
                continue  # back face
            quads.append(quad)
            facecolors.append(box.color)
            edgecolors.append(EDGE)

    for marker_id in cfg.marker_ids:
        normal = model.marker_pose_in_rig(marker_id)[:3, 2]
        if float(np.dot(normal, cam)) <= 0.02:
            continue
        cells, colors, sticker = tag_quads(model, marker_id, cfg.dictionary_name)
        quads.append(sticker)
        facecolors.append("#f5f5f0")
        edgecolors.append(ID_COLORS[int(marker_id)])
        quads.extend(cells)
        facecolors.extend(colors)
        edgecolors.extend(colors)

    ax.add_collection3d(Poly3DCollection(quads, facecolors=facecolors, edgecolors=edgecolors,
                                         linewidths=0.35, zsort="average"))
    for box in boxes:
        for seg in box.edges():
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=EDGE, linewidth=0.8, alpha=0.8)

    for marker_id in cfg.marker_ids:
        pose = model.marker_pose_in_rig(marker_id)
        center, normal = pose[:3, 3] * 1000.0, pose[:3, 2]
        if float(np.dot(normal, cam)) <= 0.02:
            continue
        ax.plot(*zip(center, center + 13.0 * normal), color=ID_COLORS[int(marker_id)], linewidth=1.6)
        ax.text(*(center + 16.0 * normal), f"ID {marker_id}", color=ID_COLORS[int(marker_id)],
                fontsize=10, fontweight="bold", ha="center")

    env = envelope(boxes)
    reach = max(abs(v) for v in env.values()) + 8.0
    for vec, color, label in ((np.array([1.0, 0, 0]), "#ff5f5f", "+X"),
                              (np.array([0, 1.0, 0]), "#5fe08a", "+Y"),
                              (np.array([0, 0, 1.0]), "#5fb4ff", "+Z")):
        ax.plot(*zip(np.zeros(3), vec * reach * 0.62), color=color, linewidth=1.8)
        ax.text(*(vec * reach * 0.70), label, color=color, fontsize=9)
    ax.scatter([0], [0], [0], color="#ffffff", s=28, depthshade=False)
    ax.text(2, 2, -reach * 0.09, "origin (0,0,0)", color=INK, fontsize=8)

    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_zlim(-reach, reach)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X [mm]", color=MUTED, fontsize=8)
    ax.set_ylabel("Y [mm]", color=MUTED, fontsize=8)
    ax.set_zlabel("Z [mm]", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.set_title(title, color=INK, fontsize=12, pad=4)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = (0.22, 0.26, 0.31, 0.5)


# --------------------------------------------------------------------------- #
# orthographic panels
# --------------------------------------------------------------------------- #
VIEWS = {
    # name: (forward = viewing direction, screen right, screen up)
    "front": (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
    "right": (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    "top": (np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
}


def _project(points: np.ndarray, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    return np.c_[np.asarray(points, float).reshape(-1, 3) @ right,
                 np.asarray(points, float).reshape(-1, 3) @ up]


def _box_rect(box: Box, right: np.ndarray, up: np.ndarray) -> np.ndarray:
    """The box's outline in a 2D view, as an axis-aligned rectangle."""
    flat = _project(np.array([[x, y, z] for x in box.x for y in box.y for z in box.z]), right, up)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _dim(ax, p0, p1, text: str, side: float = 0.0, axis: str = "x", fontsize: int = 8) -> None:
    """Dimension line between two 2D points, offset along x (vertical dim) or y."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    off = np.array([side, 0.0]) if axis == "x" else np.array([0.0, side])
    a, b = p0 + off, p1 + off
    ax.annotate("", xy=a, xytext=b,
                arrowprops=dict(arrowstyle="<->", color=DIM, linewidth=0.9, shrinkA=0, shrinkB=0))
    ax.plot([p0[0], a[0]], [p0[1], a[1]], color=DIM, linewidth=0.5, alpha=0.6)
    ax.plot([p1[0], b[0]], [p1[1], b[1]], color=DIM, linewidth=0.5, alpha=0.6)
    mid = (a + b) / 2.0
    ax.text(mid[0], mid[1], f" {text} ", color=DIM, fontsize=fontsize, rotation=90 if axis == "x" else 0,
            ha="center", va="center", bbox=dict(fc=PANEL, ec="none", pad=1.0))


def draw_ortho(ax, cfg: CubeConfig, model: AprilTagCubeModel, boxes: Sequence[Box],
               view: str, title: str) -> None:
    forward, right, up = VIEWS[view]
    ax.set_facecolor(PANEL)

    # solid silhouette, far boxes first
    for box in sorted(boxes, key=lambda b: float(b.center() @ forward), reverse=True):
        ax.add_patch(Polygon2D(_box_rect(box, right, up), closed=True,
                               facecolor=box.color, edgecolor=EDGE, linewidth=1.0))

    for marker_id in cfg.marker_ids:
        normal = model.marker_pose_in_rig(marker_id)[:3, 2]
        if float(np.dot(normal, -forward)) < 0.9:
            continue
        cells, colors, sticker = tag_quads(model, marker_id, cfg.dictionary_name, lift_mm=0.0)
        ax.add_patch(Polygon2D(_project(sticker, right, up), closed=True,
                               facecolor="#f5f5f0", edgecolor=ID_COLORS[int(marker_id)], linewidth=1.2))
        for quad, color in zip(cells, colors):
            ax.add_patch(Polygon2D(_project(quad, right, up), closed=True,
                                   facecolor=color, edgecolor=color, linewidth=0.2))
        c2d = _project(model.marker_pose_in_rig(marker_id)[:3, 3] * 1000.0, right, up)[0]
        ax.text(c2d[0], c2d[1], f"ID {marker_id}", color=ID_COLORS[int(marker_id)], fontsize=9,
                fontweight="bold", ha="center", va="center",
                bbox=dict(fc="#000000cc", ec="none", pad=1.2))

    env = envelope(boxes)
    corners = np.array([[env["x0"], env["y0"], env["z0"]], [env["x1"], env["y1"], env["z1"]]])
    flat = _project(corners, right, up)
    h0, h1 = float(min(flat[:, 0])), float(max(flat[:, 0]))
    v0, v1 = float(min(flat[:, 1])), float(max(flat[:, 1]))
    span_h, span_v = h1 - h0, v1 - v0

    if view in ("front", "right"):
        levels = {0.0: "origin  z = 0"}
        for box in boxes:
            for z in box.z:
                levels.setdefault(z, f"z = {z:+.1f}")
        for marker_id in cfg.marker_ids:
            normal = model.marker_pose_in_rig(marker_id)[:3, 2]
            if abs(float(normal @ np.array([0, 0, 1.0]))) > 0.9:
                z = float(model.marker_pose_in_rig(marker_id)[2, 3] * 1000.0)
                levels[z] = f"top-marker plane  z = {z:+.1f}"
        caption_x = h0 - 0.78 * span_h
        text_y = -1e9
        for z, label in sorted(levels.items()):
            ax.plot([caption_x, h1], [z, z], color=DIM, linewidth=0.5, linestyle=":", alpha=0.55)
            text_y = max(z + 0.015 * span_v, text_y + 0.062 * span_v)
            ax.text(caption_x, text_y, label, color=DIM, fontsize=7.5)

        # one height dimension per distinct tier, then the overall height
        tiers = sorted({(b.z[0], b.z[1]) for b in boxes})
        labels = {(b.z[0], b.z[1]): b.label for b in boxes}
        for index, (z0, z1) in enumerate(tiers):
            _dim(ax, (h1, z0), (h1, z1), f"{labels[(z0, z1)].split()[0]} {z1 - z0:.0f}",
                 side=0.09 * span_h, axis="x", fontsize=7.5)
        _dim(ax, (h1, v0), (h1, v1), f"overall {span_v:.0f}", side=0.26 * span_h, axis="x")
        _dim(ax, (h0, v0), (h1, v0), f"{span_h:.0f}", side=-0.09 * span_v, axis="y")
        # width of any tier narrower than the envelope, dimensioned at its own top
        drawn: set = set()
        for box in boxes:
            rect = _box_rect(box, right, up)
            lo, hi = rect[0], rect[2]
            width = hi[0] - lo[0]
            key = (round(lo[0], 1), round(hi[0], 1), round(hi[1], 1))
            if abs(width - span_h) < 0.5 or key in drawn:
                continue
            drawn.add(key)
            _dim(ax, (lo[0], hi[1]), (hi[0], hi[1]), f"{width:.0f}",
                 side=0.035 * span_v, axis="y", fontsize=7.5)
        for marker_id in cfg.marker_ids:
            normal = model.marker_pose_in_rig(marker_id)[:3, 2]
            if float(np.dot(normal, -forward)) < 0.9:
                continue
            z = model.marker_corners_in_rig(marker_id)[:, 2] * 1000.0
            _dim(ax, (h0, float(z.min())), (h0, float(z.max())),
                 f"tag {model.marker_size(marker_id) * 1000:.0f}",
                 side=-0.09 * span_h, axis="x", fontsize=7.5)
        ax.set_xlim(h0 - 0.82 * span_h, h1 + 0.40 * span_h)
        ax.set_ylim(v0 - 0.16 * span_v, v1 + 0.10 * span_v)
    else:
        _dim(ax, (h0, v0), (h1, v0), f"{span_h:.0f}", side=-0.10 * span_v, axis="y")
        _dim(ax, (h1, v0), (h1, v1), f"{span_v:.0f}", side=0.10 * span_h, axis="x")
        for marker_id in cfg.marker_ids:
            normal = model.marker_pose_in_rig(marker_id)[:3, 2]
            if float(np.dot(normal, -forward)) < 0.9:
                continue
            corners2d = _project(model.marker_corners_in_rig(marker_id) * 1000.0, right, up)
            _dim(ax, (h0, float(corners2d[:, 1].min())), (h0, float(corners2d[:, 1].max())),
                 f"{model.marker_size(marker_id) * 1000:.0f}", side=-0.10 * span_h, axis="x", fontsize=7.5)
            cv = _project(model.marker_pose_in_rig(marker_id)[:3, 3] * 1000.0, right, up)[0][1]
            ax.plot([h0, h1], [cv, cv], color=DIM, linewidth=0.5, linestyle=":", alpha=0.6)
            ax.text(h1 + 0.02 * span_h, cv, f"{cv:+.0f}", color=DIM, fontsize=7.5, va="center")
        ax.set_xlim(h0 - 0.30 * span_h, h1 + 0.26 * span_h)
        ax.set_ylim(v0 - 0.16 * span_v, v1 + 0.10 * span_v)

    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=11, pad=6)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.4)


# --------------------------------------------------------------------------- #
# spec table
# --------------------------------------------------------------------------- #
def spec_lines(cfg: CubeConfig, model: AprilTagCubeModel, boxes: Sequence[Box],
               source: str, solid_name: str) -> List[str]:
    ok, problems = validate_cube_config(cfg)
    env = envelope(boxes)
    lines = [
        f"config          : {source}",
        f"solid           : {solid_name}",
        f"dictionary      : {cfg.dictionary_name}",
        f"envelope        : {env['x1'] - env['x0']:.0f} (X) x {env['y1'] - env['y0']:.0f} (Y) "
        f"x {env['z1'] - env['z0']:.0f} (Z) mm",
        f"                  x [{env['x0']:+.1f}, {env['x1']:+.1f}]  "
        f"y [{env['y0']:+.1f}, {env['y1']:+.1f}]  z [{env['z0']:+.1f}, {env['z1']:+.1f}]",
        f"marker inset    : {float(getattr(cfg, 'marker_inset_m', 0.0)) * 1000:.2f} mm",
        "",
    ]
    for box in boxes:
        sx, sy, sz = box.size
        lines.append(f"  {box.label:<26} {sx:5.1f} x {sy:5.1f} x {sz:5.1f}  z [{box.z[0]:+.1f}, {box.z[1]:+.1f}]")
    lines += ["", f"{'id':>2} {'face':>4} {'size':>6} {'center (x,y,z) mm':>24} {'roll':>6}"]
    for marker_id in cfg.marker_ids:
        mid = int(marker_id)
        center = model.marker_pose_in_rig(mid)[:3, 3] * 1000.0
        lines.append(
            f"{mid:>2} {cfg.id_to_face[mid]:>4} {model.marker_size(mid) * 1000:>5.0f}mm "
            f"({center[0]:+7.1f},{center[1]:+7.1f},{center[2]:+7.1f}) {cfg.face_roll_deg.get(mid, 0.0):>5.0f}d"
        )
    lines += ["", f"validate_cube_config: {'PASS' if ok else 'FAIL'}"]
    lines += [f"  ! {p}" for p in problems[:5]]
    if problems:
        lines += [
            "  (validate_cube_config() checks the marker planes against the",
            "   config.py module constants, so any cube other than the one",
            "   config.py describes reports FAIL here by construction.)",
        ]
    return lines


def draw_spec(ax, lines: Sequence[str], fontsize: float = 8.6) -> None:
    ax.set_facecolor(PANEL)
    ax.axis("off")
    ax.set_title("model spec", color=INK, fontsize=11, pad=6)
    ax.text(0.02, 0.99, "\n".join(lines), color=INK, fontsize=fontsize, family="monospace",
            va="top", ha="left", transform=ax.transAxes, linespacing=1.5)


# --------------------------------------------------------------------------- #
# unfolded net
# --------------------------------------------------------------------------- #
NET_BASIS = {
    "+Z": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0])),
    "+X": (np.array([0, 1.0, 0]), np.array([0, 0, 1.0])),
    "+Y": (np.array([-1.0, 0, 0]), np.array([0, 0, 1.0])),
    "-X": (np.array([0, -1.0, 0]), np.array([0, 0, 1.0])),
    "-Y": (np.array([1.0, 0, 0]), np.array([0, 0, 1.0])),
}
NET_FORWARD = {"+Z": np.array([0, 0, -1.0]), "+X": np.array([-1.0, 0, 0]),
               "-X": np.array([1.0, 0, 0]), "+Y": np.array([0, -1.0, 0]),
               "-Y": np.array([0, 1.0, 0])}
NET_SLOTS = {"+Z": (1, 0), "-X": (0, 1), "-Y": (1, 1), "+X": (2, 1), "+Y": (3, 1)}


def draw_net(ax, cfg: CubeConfig, model: AprilTagCubeModel, boxes: Sequence[Box]) -> None:
    """Unfolded net: each face as seen from outside, so stickers can be checked."""
    ax.set_facecolor(PANEL)
    env = envelope(boxes)
    pitch = max(env["x1"] - env["x0"], env["y1"] - env["y0"], env["z1"] - env["z0"]) + 20.0
    pitch_row = pitch * 1.22  # extra room so a tall face panel clears the next row's title

    face_ids: Dict[str, List[int]] = {}
    for marker_id in cfg.marker_ids:
        face_ids.setdefault(cfg.id_to_face[int(marker_id)], []).append(int(marker_id))

    for face, (col, row) in NET_SLOTS.items():
        right, up = NET_BASIS[face]
        forward = NET_FORWARD[face]
        shift = np.array([col * pitch, -row * pitch_row])
        for box in sorted(boxes, key=lambda b: float(b.center() @ forward), reverse=True):
            ax.add_patch(Polygon2D(_box_rect(box, right, up) + shift, closed=True,
                                   facecolor=box.color, edgecolor=EDGE, linewidth=0.9))
        top = _project(np.array([[env["x0"], env["y0"], env["z0"]],
                                 [env["x1"], env["y1"], env["z1"]]]), right, up).max(axis=0) + shift
        ax.text(shift[0], top[1] + 0.05 * pitch, f"face {face}", color=INK, fontsize=10, ha="center")

        for marker_id in face_ids.get(face, []):
            cells, colors, sticker = tag_quads(model, marker_id, cfg.dictionary_name, lift_mm=0.0)
            ax.add_patch(Polygon2D(_project(sticker, right, up) + shift, closed=True,
                                   facecolor="#f5f5f0", edgecolor=ID_COLORS[marker_id], linewidth=1.2))
            for quad, color in zip(cells, colors):
                ax.add_patch(Polygon2D(_project(quad, right, up) + shift, closed=True,
                                       facecolor=color, edgecolor=color, linewidth=0.25))
            corners2d = _project(model.marker_corners_in_rig(marker_id) * 1000.0, right, up) + shift
            for index, pt in enumerate(corners2d):
                ax.scatter(*pt, color=ID_COLORS[marker_id], s=14, zorder=5)
                ax.text(pt[0], pt[1], f" p{index}", color=ID_COLORS[marker_id], fontsize=7, zorder=5)
            center2d = _project(model.marker_pose_in_rig(marker_id)[:3, 3] * 1000.0, right, up)[0] + shift
            ax.text(center2d[0], center2d[1],
                    f"ID {marker_id}\nroll {cfg.face_roll_deg.get(marker_id, 0.0):.0f}deg",
                    color=ID_COLORS[marker_id], fontsize=8, fontweight="bold", ha="center", va="center",
                    bbox=dict(fc="#000000cc", ec="none", pad=1.4))

    ax.set_aspect("equal")
    ax.set_xlim(-0.62 * pitch, 3.62 * pitch)
    ax.set_ylim(-1.66 * pitch_row, 0.70 * pitch)
    ax.axis("off")


def net_legend_lines(cfg: CubeConfig, model: AprilTagCubeModel) -> List[str]:
    """Where the detector's first corner (p0) lands on each face, seen from outside."""
    where: Dict[int, str] = {}
    for marker_id in cfg.marker_ids:
        mid = int(marker_id)
        right, up = NET_BASIS[cfg.id_to_face[mid]]
        offset = model.marker_corners_in_rig(mid)[0] - model.marker_pose_in_rig(mid)[:3, 3]
        where[mid] = f"{'top' if float(offset @ up) > 0 else 'bottom'}-" \
                     f"{'right' if float(offset @ right) > 0 else 'left'}"
    unique = sorted(set(where.values()))
    if len(unique) == 1:
        corner_note = [f"   After face_roll_deg, p0 lands at the {unique[0]} corner of EVERY face,",
                       "   so the six tags are mounted consistently."]
    else:
        corner_note = ["   " + ", ".join(f"ID {mid}({cfg.id_to_face[mid]}) {pos}" for mid, pos in where.items())]
    return [
        "How to check this drawing against the printed cube",
        "",
        "1. Stand the cube upright (+Z up) and look straight at one face. The panel above is",
        "   exactly what that face must look like, tag bitmap included; grey blocks behind the",
        "   tag are the rest of the solid seen from that direction.",
        "2. p0 is the corner the detector decodes first (the tag's own top-left).",
    ] + corner_note + [
        "3. A printed tag rotated 90 / 180 / 270 deg from the panel means face_roll_deg is wrong",
        "   for that ID -- re-run the face-roll self-calibration rather than editing by hand.",
    ]


# --------------------------------------------------------------------------- #
def resolve_cfg(args) -> Tuple[CubeConfig, str]:
    if args.cube_config:
        cfg, source = load_cube_config_from_json_file(args.cube_config)
        if cfg is None:
            raise SystemExit(f"could not read cube config: {args.cube_config}")
        return cfg, f"{source}: {args.cube_config}"
    if args.session_root:
        cfg, source = load_cube_config_from_meta(args.session_root)
        return cfg, f"{source}: {os.path.join(args.session_root, 'meta.json')}"
    return get_default_cube_config(), "config.py:CubeConfig (current CAD revision)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session-root", default=None,
                        help="render the cube_config frozen in this session's meta.json")
    parser.add_argument("--cube-config", default=None, help="render an explicit cube config JSON")
    parser.add_argument("--solid", default=None,
                        help="JSON describing the physical solid (mm boxes); "
                             "default is the body + protrusion implied by config.py")
    parser.add_argument("--title", default=None, help="override the figure title")
    parser.add_argument("--output-dir", default=f"{ABLATION_RESULT_ROOT}/cube_model")
    args = parser.parse_args()

    cfg, source = resolve_cfg(args)
    model = AprilTagCubeModel(cfg)
    if args.solid:
        boxes, solid_name = load_solid(args.solid)
    else:
        boxes, solid_name = default_solid(cfg, model), "derived from config.py (body + protrusion)"
    os.makedirs(args.output_dir, exist_ok=True)
    env = envelope(boxes)
    lines = spec_lines(cfg, model, boxes, source, solid_name)
    title = args.title or (f"AprilTag cube model  |  envelope {env['x1'] - env['x0']:.0f} x "
                           f"{env['y1'] - env['y0']:.0f} x {env['z1'] - env['z0']:.0f} mm  |  "
                           "tags drawn from the model's own corner order and face_roll_deg")

    fig = plt.figure(figsize=(19, 12), dpi=150, facecolor=BG)
    grid = fig.add_gridspec(2, 3, height_ratios=(1.1, 1.0), hspace=0.16, wspace=0.18)
    for index, (elev, azim, subtitle) in enumerate(
            ((22, -52, "3D  |  +X / -Y / +Z visible"), (22, 128, "3D  |  -X / +Y / +Z visible"))):
        draw_3d(fig.add_subplot(grid[0, index], projection="3d", facecolor=BG),
                cfg, model, boxes, elev, azim, subtitle)
    draw_spec(fig.add_subplot(grid[0, 2]), lines, fontsize=8.0)
    draw_ortho(fig.add_subplot(grid[1, 0]), cfg, model, boxes, "front", "FRONT  (from -Y)   X right, Z up")
    draw_ortho(fig.add_subplot(grid[1, 1]), cfg, model, boxes, "right", "RIGHT  (from +X)   Y right, Z up")
    draw_ortho(fig.add_subplot(grid[1, 2]), cfg, model, boxes, "top", "TOP  (from +Z)   X right, Y up")
    fig.suptitle(title, color=INK, fontsize=15, y=0.965)
    fig.text(0.5, 0.012,
             "Marker planes come from AprilTagCubeModel; the surrounding solid comes from the --solid "
             "description. No dimension is retyped in this script.",
             color=MUTED, fontsize=10, ha="center")
    geometry_path = os.path.join(args.output_dir, "cube_model_geometry.png")
    fig.savefig(geometry_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 11), dpi=150, facecolor=BG)
    net_grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 0.30), hspace=0.02)
    draw_net(fig.add_subplot(net_grid[0]), cfg, model, boxes)
    legend_ax = fig.add_subplot(net_grid[1])
    legend_ax.axis("off")
    legend_ax.text(0.03, 0.98, "\n".join(net_legend_lines(cfg, model)), color=INK, fontsize=10,
                   family="monospace", va="top", ha="left", transform=legend_ax.transAxes,
                   linespacing=1.6, bbox=dict(fc=PANEL, ec=GRID, pad=9.0))
    fig.suptitle("Unfolded net  |  every face as seen from outside the cube  |  "
                 "p0->p3 clockwise, roll = printed in-plane rotation", color=INK, fontsize=15, y=0.95)
    net_path = os.path.join(args.output_dir, "cube_model_net.png")
    fig.savefig(net_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))
    print(f"\nwrote {geometry_path}\nwrote {net_path}")


if __name__ == "__main__":
    main()
