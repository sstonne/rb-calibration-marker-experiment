#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_pointcloud_workspace_by_method.py -- 작업 셀 전체 포인트클라우드를
방법별(T_base_Ci) base 좌표로 합쳤을 때 카메라 3대가 서로 얼마나 맞는지 비교.

FK와 무관한 지표 두 가지 + FK 기준 지표 하나:
  1. 바닥 평면 일치: 카메라별로 바닥(보드 면) 평면을 맞추고, 작업 공간 중심에서 카메라 쌍 간 높이 차(mm)와 기울기 차(deg)
  2. 큐브 윗면 카메라 쌍 최근접점 거리(mm): 세트별 큐브 윗면 영역에서 A→B 최근접 거리 중앙값
  3. 큐브 윗면 z - FK 앵커 윗면 z (mm): 카메라별 편차(평균/세트 간 std)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from robot.backends.zeus_client import pose6_to_T  # noqa: E402
from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from paths import SESSION2_DIR  # noqa: E402
from session2_pick_and_place import compute_ordered_targets  # noqa: E402

FIXED = ("039422061216", "fixed2", "fixed3")
TOP_Z = 64.5
FLOOR_Z = -29.5  # 큐브 본체 바닥 = 보드 면 (큐브 원점 기준)
WS = dict(x=(-490, -110), y=(10, 740))  # 3대 공통 바닥 시야


def fit_plane(P):
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[2]
    if n[2] < 0: n = -n
    return c, n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session2-capture-subdir", required=True)
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--fit-json", required=True)
    ap.add_argument("--methods-json", required=True)
    ap.add_argument("--rows", default="A1,A2,A5,B2")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    intr = Path(args.zeus_intrinsics_dir)
    K, D = load_intrinsics_by_label(intr, REPO_ROOT / "ur3_calibration/intrinsics", intr / "device_map.json")
    m = json.loads(Path(args.methods_json).read_text())
    T_fc = np.asarray(json.loads(Path(args.fit_json).read_text())["T_gripper_cube"], float)
    items = compute_ordered_targets(SESSION2_DIR)
    root = SESSION2_DIR / args.session2_capture_subdir
    sets = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    s = args.stride

    # 카메라 좌표 점군은 방법과 무관 -> 한 번만 역투영
    cam_pts = {}
    for k in sets:
        for lab in FIXED:
            p = root / f"{k:03d}" / f"cam_{lab}_depth.png"
            if not p.is_file():
                continue
            cid = LOCAL_CAM_IDS[lab]
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED).astype(np.float64)[::s, ::s]
            h, w = d.shape; v, u = np.mgrid[0:h, 0:w]; ok = (d > 0) & (d < 1500)
            pix = np.stack([u[ok] * s, v[ok] * s], 1).astype(np.float64); z = d[ok]
            n = cv2.undistortPoints(pix.reshape(-1, 1, 2), K[cid], D[cid]).reshape(-1, 2)
            cam_pts[(k, lab)] = np.column_stack([n[:, 0] * z, n[:, 1] * z, z])  # mm
    anchors = {k: (pose6_to_T(items[k]["target"]) @ T_fc)[:3, 3] * 1000 for k in sets}
    ws_c = np.array([np.mean(WS["x"]), np.mean(WS["y"])])

    results = {}
    for row in args.rows.split(","):
        T_bC = {int(c): np.asarray(v, float) for c, v in m["rows"][row]["all"]["transforms"]["T_base_Ci"].items()}
        floor = {lab: [] for lab in FIXED}      # (세트, 평면 c, n)
        top_dz = {lab: [] for lab in FIXED}
        nn = {}
        for k in sets:
            a = anchors[k]; tz = a[2] + TOP_Z; base = {}
            for lab in FIXED:
                if (k, lab) not in cam_pts:
                    continue
                cid = LOCAL_CAM_IDS[lab]
                P = (T_bC[cid][:3, :3] @ cam_pts[(k, lab)].T).T + T_bC[cid][:3, 3] * 1000
                base[lab] = P
                # 바닥: 공통 시야 안, 큐브 주변 150mm 제외, 큐브 바닥 높이 ±25mm
                fz = a[2] + FLOOR_Z
                sel = (P[:, 0] > WS["x"][0]) & (P[:, 0] < WS["x"][1]) & (P[:, 1] > WS["y"][0]) & (P[:, 1] < WS["y"][1]) \
                    & (np.abs(P[:, 2] - fz) < 25) & (np.hypot(P[:, 0] - a[0], P[:, 1] - a[1]) > 150)
                if sel.sum() > 500:
                    c, nrm = fit_plane(P[sel])
                    # 1차 정합 후 outlier 제거 재적합
                    r = (P[sel] - c) @ nrm; c, nrm = fit_plane(P[sel][np.abs(r) < 8])
                    floor[lab].append((k, c, nrm))
                topsel = (np.abs(P[:, 0] - a[0]) < 40) & (np.abs(P[:, 1] - a[1]) < 40) & (np.abs(P[:, 2] - tz) < 12)
                if topsel.sum() > 30:
                    top_dz[lab].append(float(np.median(P[topsel, 2]) - tz))
                base[lab + "_top"] = P[topsel]
            for i in range(3):
                for j in range(i + 1, 3):
                    A, B = base.get(FIXED[i] + "_top"), base.get(FIXED[j] + "_top")
                    if A is None or B is None or len(A) < 30 or len(B) < 30:
                        continue
                    dnn, _ = cKDTree(B).query(A)
                    nn.setdefault(f"{FIXED[i][:6]}-{FIXED[j][:6]}", []).append(float(np.median(dnn)))
        # 바닥 평면: 세트별 카메라 쌍 높이/기울기 차
        pair_dz, pair_tilt = {}, {}
        for i in range(3):
            for j in range(i + 1, 3):
                fi = {k: (c, n) for k, c, n in floor[FIXED[i]]}; fj = {k: (c, n) for k, c, n in floor[FIXED[j]]}
                key = f"{FIXED[i][:6]}-{FIXED[j][:6]}"
                for k in fi.keys() & fj.keys():
                    (ci, ni), (cj, nj) = fi[k], fj[k]
                    zi = ci[2] - (ni[0] * (ws_c[0] - ci[0]) + ni[1] * (ws_c[1] - ci[1])) / ni[2]
                    zj = cj[2] - (nj[0] * (ws_c[0] - cj[0]) + nj[1] * (ws_c[1] - cj[1])) / nj[2]
                    pair_dz.setdefault(key, []).append(float(zi - zj))
                    pair_tilt.setdefault(key, []).append(float(np.degrees(np.arccos(np.clip(ni @ nj, -1, 1)))))
        results[row] = {
            "floor_pair_dz_mm": {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in pair_dz.items()},
            "floor_pair_tilt_deg": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in pair_tilt.items()},
            "top_nn_median_mm": {k: {"median": float(np.median(v)), "max": float(np.max(v)), "n": len(v)} for k, v in nn.items()},
            "top_dz_vs_fk_mm": {lab: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for lab, v in top_dz.items() if v},
        }

    lab_row = {"A1": "A1", "A2": "A2", "A5": "A3 (FtC FK fixed)", "B2": "B1 (cube only)"}
    print("\n[1] 바닥 평면 카메라 쌍 높이 차 (작업 공간 중심, mm, 세트 평균±std) / 기울기 차 (deg)")
    pairs = list(next(iter(results.values()))["floor_pair_dz_mm"])
    print(f"{'row':<20}" + "".join(f"{p:>26}" for p in pairs))
    for row, r in results.items():
        print(f"{lab_row.get(row,row):<20}" + "".join(f"{r['floor_pair_dz_mm'][p]['mean']:+6.2f}±{r['floor_pair_dz_mm'][p]['std']:.2f} / {r['floor_pair_tilt_deg'][p]['mean']:.2f}°".rjust(26) for p in pairs))
    print("\n[2] 큐브 윗면 카메라 쌍 최근접 거리 (mm, 세트 중앙값의 중앙값 / 최대)")
    for row, r in results.items():
        print(f"{lab_row.get(row,row):<20}" + "".join(f"{r['top_nn_median_mm'][p]['median']:.2f} / {r['top_nn_median_mm'][p]['max']:.2f}".rjust(26) for p in pairs if p in r["top_nn_median_mm"]))
    print("\n[3] 큐브 윗면 z - FK 앵커 (mm, 카메라별 평균±세트 간 std)")
    for row, r in results.items():
        print(f"{lab_row.get(row,row):<20}" + "".join(f"{lab[:6]} {v['mean']:+.2f}±{v['std']:.2f}".rjust(22) for lab, v in r["top_dz_vs_fk_mm"].items()))
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
