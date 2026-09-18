#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_pointcloud_cubeframe.py -- session2 15세트 × 고정캠 3대 depth를
큐브 좌표계로 모아 하나의 큐브 포인트클라우드로 합친다.

각 세트 k: P_cube = T_base_cube(k)^-1 · T_base_Ci · P_cam.  T_base_cube(k)는 FK 앵커
(FK(place_k) @ T_flange_cube, A3와 같은 정의). depth PNG는 color 해상도로 정렬돼 있어 color K로 역투영.
세트별로 윗면(z=+64.5) 높이가 큐브 좌표에서 얼마나 흩어지는지로 캘리브레이션+depth 일관성을 본다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from calibration_pipeline.apriltag_cube import inv_T  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402
from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from paths import SESSION2_DIR  # noqa: E402
from session2_pick_and_place import compute_ordered_targets  # noqa: E402

FIXED = ("039422061216", "fixed2", "fixed3")
TOP_Z = 64.5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session2-capture-subdir", required=True)
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--fit-json", required=True)
    ap.add_argument("--methods-json", required=True)
    ap.add_argument("--row", default="A5")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sets", default=None, help="예: 0-14 (기본 전체)")
    args = ap.parse_args()
    plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False

    intr = Path(args.zeus_intrinsics_dir)
    K, D = load_intrinsics_by_label(intr, REPO_ROOT / "ur3_calibration/intrinsics", intr / "device_map.json")
    m = json.loads(Path(args.methods_json).read_text())
    T_bC = {int(k): np.asarray(v, float) for k, v in m["rows"][args.row]["all"]["transforms"]["T_base_Ci"].items()}
    T_fc = np.asarray(json.loads(Path(args.fit_json).read_text())["T_gripper_cube"], float)
    items = compute_ordered_targets(SESSION2_DIR)
    root = SESSION2_DIR / args.session2_capture_subdir
    sets = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if args.sets:
        a, b = args.sets.split("-"); sets = [s for s in sets if int(a) <= s <= int(b)]
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    clouds, stats = [], []
    for k in sets:
        T_bcube = pose6_to_T(items[k]["target"]) @ T_fc
        T_cube_base = inv_T(T_bcube)
        for lab in FIXED:
            cid = LOCAL_CAM_IDS[lab]
            p = root / f"{k:03d}" / f"cam_{lab}_depth.png"
            if not p.is_file():
                continue
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float64)
            h, w = d.shape; v, u = np.mgrid[0:h, 0:w]; ok = d > 0
            pix = np.stack([u[ok], v[ok]], 1).astype(np.float64); z = d[ok]
            n = cv2.undistortPoints(pix.reshape(-1, 1, 2), K[cid], D[cid]).reshape(-1, 2)
            P_cam = np.column_stack([n[:, 0] * z, n[:, 1] * z, z]) / 1000.0
            P_base = (T_bC[cid][:3, :3] @ P_cam.T).T + T_bC[cid][:3, 3]
            P = ((T_cube_base[:3, :3] @ P_base.T).T + T_cube_base[:3, 3]) * 1000.0
            box = (np.abs(P[:, 0]) < 60) & (np.abs(P[:, 1]) < 60) & (P[:, 2] > -35) & (P[:, 2] < 80)
            Q = P[box]
            clouds.append((k, lab, Q))
            top = Q[(np.abs(Q[:, 0]) < 14) & (np.abs(Q[:, 1]) > 14) & (np.abs(Q[:, 1]) < 42) & (np.abs(Q[:, 2] - TOP_Z) < 12)]
            side = Q[(np.abs(Q[:, 1]) < 25) & (np.abs(Q[:, 2]) < 25)]  # 본체 옆면(±x 면) 후보
            stats.append({"set": k, "cam": lab, "n": int(len(Q)), "top_n": int(len(top)),
                          "top_dz": float(np.median(top[:, 2]) - TOP_Z) if len(top) > 30 else None,
                          "top_std": float(top[:, 2].std()) if len(top) > 30 else None})

    # 세트별 윗면 높이 (3대 합산)
    print(f"{'set':>3} | " + " | ".join(f"{l[:6]:>6} dz/std" for l in FIXED) + " | 3대 합산 dz")
    per_set = {}
    for k in sets:
        row = [s for s in stats if s["set"] == k]
        cells = []; allz = []
        for lab in FIXED:
            s = next((r for r in row if r["cam"] == lab), None)
            if s and s["top_dz"] is not None:
                cells.append(f"{s['top_dz']:+5.1f}/{s['top_std']:4.1f}")
                allz.append(s["top_dz"])
            else:
                cells.append("     -   ")
        per_set[k] = float(np.mean(allz)) if allz else None
        print(f"{k:>3} | " + " | ".join(cells) + f" | {per_set[k]:+5.1f}" if allz else f"{k:>3} | " + " | ".join(cells) + " |   -")
    dz = np.array([v for v in per_set.values() if v is not None])
    print(f"\n세트별 윗면 높이(큐브 좌표, 3대 평균) 편차: 평균 {dz.mean():+.2f}mm, std {dz.std():.2f}mm, 최대|.| {np.abs(dz).max():.2f}mm (n={len(dz)})")
    cam_dz = {lab: np.array([s["top_dz"] for s in stats if s["cam"] == lab and s["top_dz"] is not None]) for lab in FIXED}
    for lab, a in cam_dz.items():
        print(f"  {lab}: 세트 간 std {a.std():.2f}mm, 평균 {a.mean():+.2f}mm (n={len(a)})")

    # 그림: 큐브 좌표계에서 45개 시점 합침 (세트별 색)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    cmap = plt.get_cmap("tab20")
    rng = np.random.default_rng(0)
    for k, lab, Q in clouds:
        idx = rng.choice(len(Q), min(len(Q), 6000), replace=False); R = Q[idx]; c = cmap(k % 20)
        axes[0].scatter(R[:, 0], R[:, 1], s=1, color=c, alpha=0.35)
        axes[1].scatter(R[:, 0], R[:, 2], s=1, color=c, alpha=0.35)
        axes[2].scatter(R[:, 1], R[:, 2], s=1, color=c, alpha=0.35)
    for ax, (a, b), t in zip(axes, ((0, 1), (0, 2), (1, 2)), ("위에서 (x-y)", "옆에서 (x-z)", "옆에서 (y-z)")):
        ax.set_title(t); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.set_xlabel("xyz"[a] + " mm"); ax.set_ylabel("xyz"[b] + " mm")
    axes[0].add_patch(plt.Rectangle((-29.5, -29.5), 59, 59, fill=False, ec="k", lw=1.2, ls="--"))
    axes[0].add_patch(plt.Rectangle((-15.5, -44), 31, 88, fill=False, ec="k", lw=0.8, ls=":"))
    for ax in axes[1:]:
        ax.axhline(TOP_Z, c="k", ls="--", lw=0.8); ax.axhline(29.5, c="gray", ls=":", lw=0.8); ax.axhline(-29.5, c="gray", ls=":", lw=0.8)
    axes[1].add_patch(plt.Rectangle((-29.5, -29.5), 59, 59, fill=False, ec="k", lw=1.2, ls="--"))
    axes[2].add_patch(plt.Rectangle((-29.5, -29.5), 59, 59, fill=False, ec="k", lw=1.2, ls="--"))
    fig.suptitle(f"[{args.row}] session2 {len(sets)}세트 × 고정캠 3대 depth를 큐브 좌표계로 모음 (색 = 세트, 점선 = 큐브 도면)", fontsize=13)
    fig.tight_layout(); fig.savefig(out / f"cubeframe_all_sets_{args.row}.png", dpi=100)
    json.dump({"row": args.row, "per_set_top_dz_mm": per_set, "stats": stats}, open(out / f"cubeframe_stats_{args.row}.json", "w"), indent=2)
    print("wrote", out / f"cubeframe_all_sets_{args.row}.png")


if __name__ == "__main__":
    main()
