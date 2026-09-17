#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_data_ratio_a3.py -- 세션1/2/3 촬영 수를 같은 비율(30/50/70%)로
줄였을 때 A3(FtC FK fixed, unified, board+cube)의 held-out 정확도 변화.

비율마다 무작위 부분집합을 N번 뽑아 평균±std를 낸다. 각 부분집합에 대해
  1) session1 부분집합으로 T_flange_cube 재적합 (fit_grasp_offset.py)
  2) session2/3 부분집합으로 table1_zeus A5(=보고서 A3) LOPO
  3) held-out px(table1) + held-out joint mm/deg(고정캠 3대 공동 삼각측량 vs FK@T_flange_cube)
부분집합은 원본 폴더로의 심볼릭 링크로 만들며(원본 index 이름 유지), 끝나면 지운다.

사용법:
  python eval_data_ratio_a3.py --tag 0917_1920x1080 --zeus-intrinsics-dir ../intrinsics_1920x1080_rgbd720
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402

from eval_joint_relative import load_pose, delta  # noqa: E402
from fit_grasp_offset import load_intrinsics_by_label  # noqa: E402
from paths import SESSION1_DIR, SESSION2_DIR, SESSION3_DIR  # noqa: E402
from session2_pick_and_place import compute_ordered_targets  # noqa: E402

GT_CUBE_CONFIG = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"
ROW = "A5"  # table1_zeus 내부 키 (보고서의 A3)


def numeric_dirs(root: Path):
    return sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())


def make_subset(session_dir: Path, src_sub: str, dst_sub: str, keep):
    src, dst = session_dir / src_sub, session_dir / dst_sub
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir()
    for idx in keep:
        (dst / f"{idx:03d}").symlink_to(src / f"{idx:03d}")
    return dst


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run([str(c) for c in cmd], cwd=str(HERE), stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}\n{Path(log).read_text()[-2000:]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--cube-config", default=str(GT_CUBE_CONFIG))
    ap.add_argument("--ratios", default="0.3,0.5,0.7")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.tag
    intr = Path(args.zeus_intrinsics_dir).resolve()
    device_map = Path(args.device_map).resolve() if args.device_map else intr / "device_map.json"
    work = Path(args.work_dir).resolve() if args.work_dir else REPO_ROOT / "ABLATION_TEST_result_0917" / f"data_ratio_{tag}"
    work.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    s1_sub, s2_sub, s3_sub = f"capture_replayed_{tag}", f"capture_placed_{tag}", f"capture_replayed_{tag}"
    s1_all = numeric_dirs(SESSION1_DIR / s1_sub); s2_all = numeric_dirs(SESSION2_DIR / s2_sub); s3_all = numeric_dirs(SESSION3_DIR / s3_sub)
    print(f"전체: session1 {len(s1_all)} / session2 {len(s2_all)} / session3 {len(s3_all)}")

    K, D = load_intrinsics_by_label(intr, REPO_ROOT / "ur3_calibration" / "intrinsics", device_map)
    cfg, _ = load_cube_config_from_json_file(args.cube_config)
    cube = AprilTagCubeTarget(cfg)
    items = compute_ordered_targets(SESSION2_DIR)
    rng = random.Random(args.seed)

    results = {}
    for ratio in [float(x) for x in args.ratios.split(",")]:
        n1, n2, n3 = (max(5, round(ratio * len(s1_all))), max(3, round(ratio * len(s2_all))), max(5, round(ratio * len(s3_all))))
        rows = []
        for rep in range(args.repeats):
            k1, k2, k3 = sorted(rng.sample(s1_all, n1)), sorted(rng.sample(s2_all, n2)), sorted(rng.sample(s3_all, n3))
            name = f"_ratio{int(ratio*100)}_{rep}"
            d1 = make_subset(SESSION1_DIR, s1_sub, s1_sub + name, k1)
            d2 = make_subset(SESSION2_DIR, s2_sub, s2_sub + name, k2)
            d3 = make_subset(SESSION3_DIR, s3_sub, s3_sub + name, k3)
            wd = work / f"r{int(ratio*100)}_{rep}"; wd.mkdir(parents=True, exist_ok=True)
            fit_json = wd / "fit.json"
            try:
                run([py, "fit_grasp_offset.py", "--capture-subdir", d1.name, "--zeus-intrinsics-dir", intr,
                     "--device-map", device_map, "--cube-config", args.cube_config, "--out", fit_json], wd / "log_fit.txt")
                run([py, "table1_zeus.py", "--session1-capture-subdir", d1.name, "--session2-capture-subdir", d2.name,
                     "--session3-capture-subdir", d3.name, "--zeus-intrinsics-dir", intr, "--device-map", device_map,
                     "--fit-json", fit_json, "--cube-config", args.cube_config, "--s3-gripper-only", "--include-session2-board",
                     "--rows", ROW, "--report-dir", wd, "--out", wd / "t1.json"], wd / "log_table1.txt")
                m = json.loads((wd / "ABLATION_TEST_table1_methods.json").read_text())
                r = m["rows"][ROW]
                T_fc = np.asarray(json.loads(fit_json.read_text())["T_gripper_cube"], float)
                jm, jd = [], []
                for f in r["folds"]:
                    Tc = {int(k): np.asarray(v, float) for k, v in f["transforms"]["T_base_Ci"].items()}
                    for s in f["heldout_placement_ids"]:
                        s = int(s)
                        T, pc = load_pose(SESSION2_DIR / s2_sub / f"{s:03d}", cube, K, D, Tc)
                        if T is None:
                            continue
                        a, b = delta(pose6_to_T(items[s]["target"]) @ T_fc, T); jm.append(a); jd.append(b)
                row = {"ratio": ratio, "rep": rep, "n": [n1, n2, n3], "sets": k2,
                       "heldout_px": r["summary"]["heldout_test_cube_rmse_px"], "train_px": r["summary"]["train_cube_rmse_px"],
                       "joint_mm": float(np.mean(jm)), "joint_deg": float(np.mean(jd)),
                       "converged": f"{r['summary']['n_converged']}/{r['summary']['n_folds']}",
                       "T_fc_mm": (T_fc[:3, 3] * 1000).round(2).tolist()}
                rows.append(row)
                print(f"  ratio {ratio:.1f} rep {rep}: n={row['n']} held-out {row['heldout_px']:.2f}px  joint {row['joint_mm']:.2f}mm / {row['joint_deg']:.2f}deg  {row['converged']}")
            except Exception as exc:
                print(f"  ratio {ratio:.1f} rep {rep}: 실패 ({str(exc)[:200]})")
            finally:
                for d in (d1, d2, d3):
                    shutil.rmtree(d, ignore_errors=True)
        results[str(ratio)] = rows

    print(f"\n{'비율':>5} | {'s1/s2/s3 수':>11} | {'held-out px':>16} | {'joint mm':>14} | {'joint deg':>14} | n")
    summary = {}
    for k, rows in results.items():
        if not rows:
            continue
        px = np.array([r["heldout_px"] for r in rows]); mm = np.array([r["joint_mm"] for r in rows]); dg = np.array([r["joint_deg"] for r in rows])
        summary[k] = {"n_sessions": rows[0]["n"], "heldout_px_mean": px.mean(), "heldout_px_std": px.std(),
                      "joint_mm_mean": mm.mean(), "joint_mm_std": mm.std(), "joint_deg_mean": dg.mean(), "joint_deg_std": dg.std(), "repeats": len(rows)}
        print(f"{float(k)*100:4.0f}% | {'/'.join(map(str, rows[0]['n'])):>11} | {px.mean():6.2f} ± {px.std():4.2f}     | {mm.mean():5.2f} ± {mm.std():4.2f}   | {dg.mean():5.2f} ± {dg.std():4.2f}   | {len(rows)}")
    out = Path(args.out) if args.out else work / "data_ratio_a3.json"
    out.write_text(json.dumps({"tag": tag, "row": ROW, "results": results, "summary": summary}, indent=2, ensure_ascii=False, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
