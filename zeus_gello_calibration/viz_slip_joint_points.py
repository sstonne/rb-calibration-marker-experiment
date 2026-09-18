#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/viz_slip_joint_points.py -- 3대 joint 슬립 판정에 쓴 코너를 실제 사진 위에 찍어 보여준다.

세트별 그림 1장: 고정캠 3대 × (held 사진 큐브 부분 확대). 각 카메라에
  ● 초록  held 검출 코너
  ● 파랑  released 검출 코너, 초록→파랑 화살표 = 이동 (×배율)
  ● 주황  picked 검출 코너
  ✕ 흰색  joint 6-DoF 포즈(3대 공동 적합)를 이 카메라에 재투영한 위치 (held)
제목에 joint 판정 mm/deg와 카메라별 코너 평균 이동 px.
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
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from eval_joint_relative import FIXED_LABELS, LOCAL_CAM_IDS, load_pose, project  # noqa: E402
from fit_grasp_offset import load_intrinsics_by_label  # noqa: E402
from paths import SESSION2_DIR  # noqa: E402

GT_CUBE_CONFIG = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"
COL = {"held": "#22c55e", "released": "#3b82f6", "picked": "#f97316"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session2-capture-subdir", required=True)
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--methods-json", required=True)
    ap.add_argument("--slip-json", required=True, help="eval_slip_joint.py 결과 (제목용)")
    ap.add_argument("--row", default="A5")
    ap.add_argument("--sets", default=None, help="예: 2,4,10 (기본 전체)")
    ap.add_argument("--arrow-scale", type=float, default=20.0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False

    intr = Path(args.zeus_intrinsics_dir)
    K, D = load_intrinsics_by_label(intr, REPO_ROOT / "ur3_calibration/intrinsics", intr / "device_map.json")
    cfg, _ = load_cube_config_from_json_file(GT_CUBE_CONFIG)
    cube = AprilTagCubeTarget(cfg)
    T_bC = {int(k): np.asarray(v, float) for k, v in
            json.loads(Path(args.methods_json).read_text())["rows"][args.row]["all"]["transforms"]["T_base_Ci"].items()}
    slip = {r["set"]: r for r in json.loads(Path(args.slip_json).read_text())["per_set"]}
    root = SESSION2_DIR / args.session2_capture_subdir
    sets = [int(s) for s in args.sets.split(",")] if args.sets else sorted(int(p.name) for p in (root / "held").iterdir() if p.name.isdigit())
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    for k in sets:
        data = {}
        for tag in ("held", "released", "picked"):
            f = root / tag / f"{k:03d}"
            if f.is_dir():
                T, pc = load_pose(f, cube, K, D, T_bC)
                if T is not None:
                    data[tag] = (T, pc)
        if "held" not in data:
            print(f"set {k}: held 없음"); continue
        Th, pch = data["held"]
        fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
        for ax, lab in zip(axes, FIXED_LABELS):
            cid = LOCAL_CAM_IDS[lab]
            img = cv2.cvtColor(cv2.imread(str(root / "held" / f"{k:03d}" / f"cam_{lab}.png")), cv2.COLOR_BGR2RGB)
            if cid not in pch:
                ax.imshow(img); ax.set_title(f"{lab}: 검출 없음"); ax.axis("off"); continue
            oh, ph = pch[cid]
            proj = project(Th, T_bC[cid], oh, K[cid], D[cid])
            cx, cy = ph.mean(0); r = max(np.ptp(ph[:, 0]), np.ptp(ph[:, 1])) * 0.75 + 30
            ax.imshow(img); ax.set_xlim(cx - r, cx + r); ax.set_ylim(cy + r, cy - r)
            ax.scatter(ph[:, 0], ph[:, 1], s=28, c=COL["held"], edgecolors="k", linewidths=0.4, label="held 코너", zorder=4)
            ax.scatter(proj[:, 0], proj[:, 1], s=40, marker="x", c="white", linewidths=1.0, label="joint 포즈 재투영", zorder=5)
            info = []
            for tag in ("released", "picked"):
                if tag not in data or cid not in data[tag][1]:
                    continue
                ot, pt = data[tag][1][cid]
                # 같은 3D 코너끼리 짝 맞추기
                mh = {tuple(np.round(o, 6)): p for o, p in zip(oh, ph)}
                pairs = [(mh[tuple(np.round(o, 6))], p) for o, p in zip(ot, pt) if tuple(np.round(o, 6)) in mh]
                if not pairs:
                    continue
                A = np.array([a for a, _ in pairs]); B = np.array([b for _, b in pairs]); dv = B - A
                ax.scatter(B[:, 0], B[:, 1], s=14, c=COL[tag], edgecolors="k", linewidths=0.3, label=f"{tag} 코너", zorder=4)
                ax.quiver(A[:, 0], A[:, 1], dv[:, 0] * args.arrow_scale, dv[:, 1] * args.arrow_scale, angles="xy", scale_units="xy", scale=1,
                          color=COL[tag], width=0.004, alpha=0.9, zorder=3)
                info.append(f"{tag}: 평균 {np.linalg.norm(dv, axis=1).mean():.2f}px (×{args.arrow_scale:.0f} 화살표)")
            rms = np.sqrt(np.mean(np.sum((proj - ph) ** 2, axis=1)))
            ax.set_title(f"{lab}  joint 재투영 잔차 {rms:.2f}px\n" + " | ".join(info), fontsize=10)
            ax.axis("off")
        axes[0].legend(loc="lower left", fontsize=8, framealpha=0.85)
        s = slip.get(k, {})
        t = f"set {k}: 3대 joint 판정 — "
        for tag, nm in (("released", "놓은 뒤"), ("picked", "다시 잡은 뒤")):
            if tag in s:
                m = s[tag]["mm"]; t += f"{nm} |Δ|={s[tag]['norm_mm']:.2f}mm (x{m[0]:+.2f} y{m[1]:+.2f} z{m[2]:+.2f}) {s[tag]['deg']:.2f}°   "
        fig.suptitle(t, fontsize=13, y=0.99)
        fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out / f"slip_points_set{k:02d}.png", dpi=100); plt.close(fig)
        print("wrote", out / f"slip_points_set{k:02d}.png")


if __name__ == "__main__":
    main()
