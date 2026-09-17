#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_joint_relative.py -- 고정캠 3대 joint 삼각측량으로
session1/session2(held) 절대·상대 정확도(mm, deg, px)를 낸다.

joint 포즈: 최종 캘리브레이션의 T_base_Ci로 고정캠 3대 코너를 한 번에 재투영
최적화한 T_base_cube(t). 카메라 한 대짜리 PnP의 깊이 노이즈가 3대에서 상쇄되므로
이 값이 mm 정확도의 기준 지표다 (eval_heldout_and_consistency.py의 joint 지표와 같은 방식).

  절대: joint(t) vs FK(t) @ T_flange_cube
  상대: joint(i) @ [FK 상대변환 i->j] 로 예측한 j 포즈 vs joint(j)  (모든 쌍, 양방향)
  px : 그 포즈를 3대에 재투영해 검출 코너와 비교한 RMSE

사용법:
  python eval_joint_relative.py --session1-capture-subdir capture_replayed_0917_1920x1080 \
      --session2-capture-subdir capture_placed_0917_1920x1080 \
      --zeus-intrinsics-dir ../intrinsics_1920x1080_rgbd720 \
      --fit-json fit_통합_0917_1920x1080.json \
      --methods-json ../ABLATION_TEST_result_0917/.../ABLATION_TEST_table1_methods.json --row A5
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T, rodrigues_to_Rt  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402

from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from paths import SESSION1_DIR, SESSION2_DIR  # noqa: E402

FIXED_LABELS = ("039422061216", "fixed2", "fixed3")
GT_CUBE_CONFIG = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"


def T_from_x(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_matrix(Rotation.from_rotvec(x[:3]).as_matrix()).as_matrix()
    T[:3, 3] = x[3:]
    return T


def project(T_base_cube, T_base_cam, obj, K, D):
    T = inv_T(T_base_cam) @ T_base_cube
    p, _ = cv2.projectPoints(obj, cv2.Rodrigues(T[:3, :3])[0], T[:3, 3], K, D)
    return p.reshape(-1, 2)


def load_pose(folder, cube, K, D, T_base_Ci):
    per_cam, cands = {}, []
    for lab in FIXED_LABELS:
        p = folder / f"cam_{lab}.png"
        if not p.is_file():
            continue
        cid = LOCAL_CAM_IDS[lab]
        if cid not in T_base_Ci:
            continue
        img = cv2.imread(str(p))
        corners_list, ids = cube.detect(img)
        if ids is None:
            continue
        obj, pix = [], []
        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if not cube.model.has_marker(mid):
                continue
            obj.append(cube.model.marker_corners_in_rig(mid))
            pix.append(cube.model.reorder_image_corners(mid, np.asarray(c).reshape(4, 2)))
        if not obj:
            continue
        ok, rv, tv, *_ = cube.solve_pnp_cube(img, K[cid], D[cid], reproj_thr_mean_px=10.0, return_reproj=True)
        if not ok:
            continue
        per_cam[cid] = (np.concatenate(obj), np.concatenate(pix))
        cands.append(T_base_Ci[cid] @ rodrigues_to_Rt(rv, tv))
    if len(per_cam) < 2:
        return None, per_cam
    T0 = cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]
    x0 = np.r_[Rotation.from_matrix(T0[:3, :3]).as_rotvec(), T0[:3, 3]]

    def resid(x):
        T = T_from_x(x)
        return np.concatenate([(project(T, T_base_Ci[c], o, K[c], D[c]) - px).ravel()
                               for c, (o, px) in per_cam.items()])

    sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    return T_from_x(sol.x), per_cam


def reproj_err(T_base_cube, per_cam, K, D, T_base_Ci):
    return np.concatenate([np.linalg.norm(project(T_base_cube, T_base_Ci[c], o, K[c], D[c]) - px, axis=1)
                           for c, (o, px) in per_cam.items()])


def delta(Ta, Tb):
    d = inv_T(Ta) @ Tb
    return (float(np.linalg.norm(d[:3, 3]) * 1000.0),
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec()))))


def rmse(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def evaluate(name, dir_path, cube, K, D, T_base_Ci, T_fc):
    idxs = sorted(int(p.name) for p in dir_path.iterdir() if p.is_dir() and p.name.isdigit())
    J, PC, FK = {}, {}, {}
    for i in idxs:
        f = dir_path / f"{i:03d}"
        T, pc = load_pose(f, cube, K, D, T_base_Ci)
        if T is None:
            continue
        J[i], PC[i] = T, pc
        FK[i] = pose6_to_T(json.loads((f / "robot.json").read_text())["pose"]) @ T_fc
    ks = sorted(J)
    abs_mm = np.array([delta(FK[i], J[i])[0] for i in ks])
    abs_deg = np.array([delta(FK[i], J[i])[1] for i in ks])
    abs_px = np.concatenate([reproj_err(FK[i], PC[i], K, D, T_base_Ci) for i in ks])
    fit_px = np.concatenate([reproj_err(J[i], PC[i], K, D, T_base_Ci) for i in ks])
    rel_mm, rel_deg, rel_px = [], [], []
    for a, b in combinations(ks, 2):
        for s, d in ((a, b), (b, a)):
            T_pred = J[s] @ (inv_T(FK[s]) @ FK[d])
            m, g = delta(T_pred, J[d])
            rel_mm.append(m); rel_deg.append(g)
            rel_px.append(reproj_err(T_pred, PC[d], K, D, T_base_Ci))
    rel_mm, rel_deg = np.array(rel_mm), np.array(rel_deg)
    rel_px = np.concatenate(rel_px) if rel_px else np.array([np.nan])
    out = {
        "n_poses": len(ks), "n_pairs": len(rel_mm) // 2,
        "joint_fit_px_rmse": rmse(fit_px),
        "absolute": {"mm_mean": float(abs_mm.mean()), "mm_median": float(np.median(abs_mm)),
                     "mm_p95": float(np.percentile(abs_mm, 95)), "deg_mean": float(abs_deg.mean()),
                     "px_rmse": rmse(abs_px)},
        "relative": {"mm_mean": float(rel_mm.mean()), "mm_median": float(np.median(rel_mm)),
                     "mm_p95": float(np.percentile(rel_mm, 95)), "deg_mean": float(rel_deg.mean()),
                     "deg_median": float(np.median(rel_deg)), "px_rmse": rmse(rel_px)},
    }
    a, r = out["absolute"], out["relative"]
    print(f"\n=== {name}: joint 포즈 {len(ks)}개, 상대 쌍 {out['n_pairs']}개 ===")
    print(f"  [joint 자체 적합 잔차]            px RMSE {out['joint_fit_px_rmse']:.2f}")
    print(f"  [절대: joint vs FK@T_flange_cube]  mm 평균 {a['mm_mean']:.2f} 중앙값 {a['mm_median']:.2f} P95 {a['mm_p95']:.2f} | "
          f"deg {a['deg_mean']:.2f} | px RMSE {a['px_rmse']:.2f}")
    print(f"  [상대: FK 예측 vs joint, 모든 쌍]  mm 평균 {r['mm_mean']:.2f} 중앙값 {r['mm_median']:.2f} P95 {r['mm_p95']:.2f} | "
          f"deg {r['deg_mean']:.2f} | px RMSE {r['px_rmse']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session1-dir", default=str(SESSION1_DIR))
    ap.add_argument("--session1-capture-subdir", required=True)
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR))
    ap.add_argument("--session2-capture-subdir", required=True)
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=None, help="기본: <zeus-intrinsics-dir>/device_map.json")
    ap.add_argument("--cube-config", default=str(GT_CUBE_CONFIG))
    ap.add_argument("--fit-json", required=True, help="T_gripper_cube 출처 (fit_grasp_offset.py 출력)")
    ap.add_argument("--methods-json", required=True, help="table1_zeus.py 출력 ABLATION_TEST_table1_methods.json")
    ap.add_argument("--row", default="A5", help="T_base_Ci를 가져올 row (기본 A5)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    zeus_intr = Path(args.zeus_intrinsics_dir)
    device_map = Path(args.device_map) if args.device_map else zeus_intr / "device_map.json"
    K, D = load_intrinsics_by_label(zeus_intr, Path(args.ur3_intrinsics_dir), device_map)
    cfg, _ = load_cube_config_from_json_file(args.cube_config)
    cube = AprilTagCubeTarget(cfg)
    T_fc = np.asarray(json.loads(Path(args.fit_json).read_text())["T_gripper_cube"], float)
    methods = json.loads(Path(args.methods_json).read_text())
    T_base_Ci = {int(k): np.asarray(v, float)
                 for k, v in methods["rows"][args.row]["all"]["transforms"]["T_base_Ci"].items()}
    print(f"row {args.row} 최종 T_base_Ci + 고정캠 {len(T_base_Ci)}대 joint 삼각측량")

    result = {"row": args.row, "fit_json": str(args.fit_json), "methods_json": str(args.methods_json)}
    result["session1"] = evaluate("session1 (handheld)", Path(args.session1_dir) / args.session1_capture_subdir,
                                  cube, K, D, T_base_Ci, T_fc)
    held = Path(args.session2_dir) / args.session2_capture_subdir / "held"
    if held.is_dir():
        result["session2_held"] = evaluate("session2 held (placement)", held, cube, K, D, T_base_Ci, T_fc)
    else:
        print(f"\n[skip] session2 held 폴더 없음: {held}")
    out = Path(args.out) if args.out else Path(args.methods_json).with_name("joint_relative_eval.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
