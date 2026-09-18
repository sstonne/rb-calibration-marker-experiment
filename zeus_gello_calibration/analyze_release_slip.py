#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/analyze_release_slip.py -- session2의 held/released 진단
촬영으로 (1) 릴리즈 슬립, (2) 테이블 높이에서의 카메라-FK 편향을 분리한다.

session2_pick_and_place.py 가 placement마다 같은 로봇 자세에서 그리퍼 열기 전
(held/NNN)과 연 후(released/NNN)를 찍어둔다. 고정캠들로 각 사진의 큐브 pose를
base 좌표로 추정하면:
  released − held  : 같은 카메라·같은 자세라 카메라/FK 편향이 상쇄 -> 순수 릴리즈 슬립
  held − FK anchor : 큐브가 아직 그리퍼에 물려 있으므로 FK@T_gripper_cube 가 그대로
                     성립해야 함 -> 남는 차이는 릴리즈와 무관한 카메라-FK 편향(테이블 높이)
  released − anchor: 위 둘의 합 = 지금까지 held-out에서 보던 "비전 vs FK" 불일치

사용법:
  python analyze_release_slip.py --fit fit_통합_no-fk_0914.json \
      --capture-subdir capture_placed_0914 --grasp pass1_grasp_offset_replayed_0914.json \
      --cube-config ../targets/gt_cube/cube_config.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T, rodrigues_to_Rt  # noqa: E402
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from calibration_pipeline.reprojection import pose_delta  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from gt_pick_test import load_fit  # noqa: E402
from paths import SESSION2_DIR  # noqa: E402
from session2_pick_and_place import compute_ordered_targets  # noqa: E402

FIXED_LABELS = ("039422061216", "fixed2", "fixed3")


def cube_in_base(folder: Path, cube_target, K_map, D_map, T_base_cam, thr_px):
    cands, per_cam = [], {}
    for label in FIXED_LABELS:
        p = folder / f"cam_{label}.png"
        if not p.is_file():
            continue
        cid = LOCAL_CAM_IDS[label]
        if cid not in T_base_cam:
            continue
        img = cv2.imread(str(p))
        ok, rv, tv, used, rep = cube_target.solve_pnp_cube(img, K_map[cid], D_map[cid],
                                                           reproj_thr_mean_px=thr_px, return_reproj=True)
        if not ok:
            continue
        T = T_base_cam[cid] @ rodrigues_to_Rt(rv, tv)
        cands.append(T)
        per_cam[label] = T
    if not cands:
        return None, per_cam
    return (cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]), per_cam


def delta_vec(T_ref, T):
    """T_ref 기준 T의 차이: base 프레임 평행이동(mm) 3 + 회전각(deg)."""
    d = inv_T(T_ref) @ T
    ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())))
    return (T[:3, 3] - T_ref[:3, 3]) * 1000.0, ang


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", required=True, help="fit JSON (T_base_cam, T_gripper_cube)")
    ap.add_argument("--grasp", default=None, help="T_gripper_cube 출처 JSON (기본: --fit 의 것)")
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR))
    ap.add_argument("--capture-subdir", default="capture_placed_0914")
    ap.add_argument("--cube-config", default=None)
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--reproj-thr-px", type=float, default=10.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    K_map, D_map = load_intrinsics_by_label(Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))
    T_gripper_cube, T_base_cam, _ = load_fit(Path(args.fit))
    if args.grasp:
        T_gripper_cube = np.asarray(json.loads(Path(args.grasp).read_text())["T_gripper_cube"], float)
    T_gripper_cube = np.asarray(T_gripper_cube, float)
    if args.cube_config:
        cfg, _ = load_cube_config_from_json_file(args.cube_config)
    else:
        cfg = get_default_cube_config()
    cube_target = AprilTagCubeTarget(cfg)

    s2 = Path(args.session2_dir)
    root = s2 / args.capture_subdir
    items = compute_ordered_targets(s2)
    rows = []
    print(f"{'set':>3} | {'released-held (slip) dx dy dz mm, rot':>42} | {'held-anchor dx dy dz mm, rot':>36} | {'released-anchor':>22}")
    for idx in sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()):
        held_dir, rel_dir = root / "held" / f"{idx:03d}", root / "released" / f"{idx:03d}"
        if not held_dir.is_dir() or not rel_dir.is_dir() or idx >= len(items):
            continue
        T_h, pc_h = cube_in_base(held_dir, cube_target, K_map, D_map, T_base_cam, args.reproj_thr_px)
        T_r, pc_r = cube_in_base(rel_dir, cube_target, K_map, D_map, T_base_cam, args.reproj_thr_px)
        if T_h is None or T_r is None:
            print(f"{idx:>3} | 검출 실패 (held cams={sorted(pc_h)}, released cams={sorted(pc_r)})")
            continue
        # anchor: 실제 place 자세(held 촬영 순간의 robot.json)에 T_gripper_cube
        rj = json.loads((held_dir / "robot.json").read_text())
        T_anchor = pose6_to_T(rj["pose"]) @ T_gripper_cube
        slip_t, slip_r = delta_vec(T_h, T_r)
        ha_t, ha_r = delta_vec(T_anchor, T_h)
        ra_t, ra_r = delta_vec(T_anchor, T_r)
        # 카메라별 슬립: 같은 카메라의 held/released 두 장을 그 카메라 기준으로만 빼서
        # 외부 파라미터 오차가 완전히 상쇄된 값 (기존 3대 평균 pose 차이와 별도)
        per_cam_slip = {}
        for label in FIXED_LABELS:
            if label in pc_h and label in pc_r:
                st, sr = delta_vec(pc_h[label], pc_r[label])
                per_cam_slip[label] = {"mm": st.tolist(), "deg": sr}
        rows.append({"set": idx, "slip_mm": slip_t.tolist(), "slip_deg": slip_r, "per_camera_slip": per_cam_slip,
                     "held_minus_anchor_mm": ha_t.tolist(), "held_minus_anchor_deg": ha_r,
                     "released_minus_anchor_mm": ra_t.tolist(), "released_minus_anchor_deg": ra_r,
                     "held_cams": sorted(pc_h), "released_cams": sorted(pc_r)})
        print(f"{idx:>3} | {slip_t[0]:6.2f} {slip_t[1]:6.2f} {slip_t[2]:6.2f}  {slip_r:5.2f}°  (cams {len(pc_h)}/{len(pc_r)}) "
              f"| {ha_t[0]:6.2f} {ha_t[1]:6.2f} {ha_t[2]:6.2f}  {ha_r:5.2f}° | {np.linalg.norm(ra_t):6.2f} mm {ra_r:5.2f}°")
    if not rows:
        print("결과 없음"); return

    def stats(key):
        A = np.array([r[key] for r in rows])
        med = np.median(A, axis=0); mad = np.median(np.abs(A - med), axis=0) * 1.4826
        return med, mad, np.linalg.norm(A, axis=1)
    print("\n=== 요약 (base 프레임, mm) ===")
    for key, name in (("slip_mm", "릴리즈 슬립 (released − held)"), ("held_minus_anchor_mm", "카메라-FK 편향 (held − anchor)"),
                      ("released_minus_anchor_mm", "합계 (released − anchor)")):
        med, mad, norms = stats(key)
        print(f"  {name:>32}: 중앙값 {np.round(med, 2)}  MAD {np.round(mad, 2)}  |·| 평균 {norms.mean():.2f} mm (n={len(rows)})")
    print("\n=== 카메라별 릴리즈 슬립 (released − held, 같은 카메라끼리; base 프레임 mm) ===")
    print(f"{'set':>3} | " + " | ".join(f"{lab[:12]:>28}" for lab in FIXED_LABELS) + " | 카메라 간 편차(std)")
    per_cam_all = {lab: [] for lab in FIXED_LABELS}
    spread = []
    for r in rows:
        cells = []
        vs = []
        for lab in FIXED_LABELS:
            v = r["per_camera_slip"].get(lab)
            if v is None:
                cells.append(f"{'-':>28}")
                continue
            a = np.array(v["mm"]); per_cam_all[lab].append(a); vs.append(a)
            cells.append(f"{a[0]:6.2f} {a[1]:6.2f} {a[2]:6.2f} |{np.linalg.norm(a):5.2f}| {v['deg']:4.2f}°")
        if len(vs) >= 2:
            sd = np.array(vs).std(axis=0); spread.append(sd)
            sp = f"{sd[0]:.2f} {sd[1]:.2f} {sd[2]:.2f}"
        else:
            sp = "-"
        print(f"{r['set']:>3} | " + " | ".join(cells) + f" | {sp}")
    for lab in FIXED_LABELS:
        A = np.array(per_cam_all[lab])
        if len(A) == 0:
            continue
        med = np.median(A, axis=0); mad = np.median(np.abs(A - med), axis=0) * 1.4826
        print(f"  {lab:>13}: 중앙값 {np.round(med, 2)}  MAD {np.round(mad, 2)}  |·| 평균 {np.linalg.norm(A, axis=1).mean():.2f} mm (n={len(A)})")
    if spread:
        print(f"  카메라 간 편차(std) 평균: {np.round(np.mean(spread, axis=0), 2)} mm  -> 슬립 측정 자체의 노이즈 바닥")
    out = Path(args.out) if args.out else REPO_ROOT / "zeus_gello_calibration" / "results" / "slip" / f"release_slip_{args.capture_subdir}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
