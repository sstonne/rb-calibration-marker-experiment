#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_slip_joint.py -- session2 전(held)·중(released)·후(picked) 슬립을
고정캠 3대 joint 강체 적합으로 잰다 (mm/deg).

카메라 한 대 PnP는 깊이축이 약해 mm가 튄다(저해상도에서 1.3mm대). 3대 코너를 한꺼번에
묶어 6-DoF 큐브 포즈를 풀면 깊이가 서로 보완돼 실제 이동만 남는다.
  joint 포즈: 최종 캘리브레이션(row A5) T_base_Ci로 3대 코너 재투영 최적화
  슬립     : held 기준 released / picked 포즈 차이 (base 프레임 mm, deg)
  일관성   : joint 적합 잔차 px (3대가 한 강체를 보고 있는지)

사용법:
  python eval_slip_joint.py --session2-capture-subdir capture_placed_0917_1920x1080 \
      --zeus-intrinsics-dir ../intrinsics_1920x1080_rgbd720 \
      --methods-json ../ABLATION_TEST_result_0917/zeus_0917_1920x1080_newintr/table1_zeus/ABLATION_TEST_table1_methods.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from eval_joint_relative import load_pose, reproj_err  # noqa: E402
from fit_grasp_offset import load_intrinsics_by_label  # noqa: E402
from paths import SESSION2_DIR  # noqa: E402

GT_CUBE_CONFIG = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"


def delta_xyz(Ta, Tb):
    d = inv_T(Ta) @ Tb
    return (Tb[:3, 3] - Ta[:3, 3]) * 1000.0, float(np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR))
    ap.add_argument("--session2-capture-subdir", required=True)
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--cube-config", default=str(GT_CUBE_CONFIG))
    ap.add_argument("--methods-json", required=True, help="table1_zeus 결과 (T_base_Ci 출처)")
    ap.add_argument("--row", default="A5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    intr = Path(args.zeus_intrinsics_dir)
    K, D = load_intrinsics_by_label(intr, Path(args.ur3_intrinsics_dir),
                                    Path(args.device_map) if args.device_map else intr / "device_map.json")
    cfg, _ = load_cube_config_from_json_file(args.cube_config)
    cube = AprilTagCubeTarget(cfg)
    methods = json.loads(Path(args.methods_json).read_text())
    T_base_Ci = {int(k): np.asarray(v, float)
                 for k, v in methods["rows"][args.row]["all"]["transforms"]["T_base_Ci"].items()}
    root = Path(args.session2_dir) / args.session2_capture_subdir
    idxs = sorted(int(p.name) for p in (root / "held").iterdir() if p.name.isdigit())

    rows = []
    print(f"{'set':>3} | {'열기(held→released) dx dy dz mm | deg':>40} | {'재파지(held→picked) dx dy dz mm | deg':>40} | 적합잔차 px (held/rel/pk)")
    for k in idxs:
        res = {"set": k}
        poses, fit_px = {}, {}
        for tag in ("held", "released", "picked"):
            f = root / tag / f"{k:03d}"
            if not f.is_dir():
                continue
            T, pc = load_pose(f, cube, K, D, T_base_Ci)
            if T is None:
                continue
            poses[tag] = T
            fit_px[tag] = float(np.sqrt(np.mean(reproj_err(T, pc, K, D, T_base_Ci) ** 2)))
        if "held" not in poses:
            print(f"{k:>3} | (held 검출 실패)"); continue
        cells = []
        for tag in ("released", "picked"):
            if tag in poses:
                t, r = delta_xyz(poses["held"], poses[tag])
                res[tag] = {"mm": t.tolist(), "norm_mm": float(np.linalg.norm(t)), "deg": r}
                cells.append(f"{t[0]:6.2f} {t[1]:6.2f} {t[2]:6.2f} |{np.linalg.norm(t):5.2f}| {r:5.2f}")
            else:
                cells.append(f"{'-':>40}")
        res["fit_px"] = fit_px
        rows.append(res)
        print(f"{k:>3} | {cells[0]:>40} | {cells[1]:>40} | " + "/".join(f"{fit_px.get(t, float('nan')):.2f}" for t in ("held", "released", "picked")))

    print("\n=== 요약 (3대 joint, base 프레임) ===")
    summary = {}
    for tag, name in (("released", "열기 (전→중)"), ("picked", "재파지 (전→후)")):
        n = np.array([r[tag]["norm_mm"] for r in rows if tag in r]); g = np.array([r[tag]["deg"] for r in rows if tag in r])
        v = np.array([r[tag]["mm"] for r in rows if tag in r])
        if not len(n):
            continue
        summary[tag] = {"n": int(len(n)), "mm_mean": float(n.mean()), "mm_median": float(np.median(n)), "mm_p95": float(np.percentile(n, 95)),
                        "mm_max": float(n.max()), "deg_mean": float(g.mean()), "deg_median": float(np.median(g)),
                        "xyz_median": np.median(v, axis=0).tolist()}
        print(f"  {name}: |이동| 평균 {n.mean():.2f} 중앙값 {np.median(n):.2f} P95 {np.percentile(n,95):.2f} 최대 {n.max():.2f} mm | "
              f"회전 평균 {g.mean():.2f} 중앙값 {np.median(g):.2f} deg | xyz 중앙값 {np.round(np.median(v,axis=0),2)} (n={len(n)})")
    fp = np.array([r["fit_px"]["held"] for r in rows if "held" in r["fit_px"]])
    print(f"  joint 적합 잔차(held) 평균 {fp.mean():.2f} px -> 3대가 같은 강체를 보고 있음")
    out = Path(args.out) if args.out else Path(args.methods_json).parent.parent / "slip_joint.json"
    out.write_text(json.dumps({"row": args.row, "per_set": rows, "summary": summary}, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
