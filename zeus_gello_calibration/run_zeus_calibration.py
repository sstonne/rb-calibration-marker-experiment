#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/run_zeus_calibration.py -- Zeus 캘리브레이션 표준 실행기.

어떤 촬영분(해상도/날짜)이든 아래 4단계를 같은 옵션으로 돌린다. 옵션은 여기서만 고정한다.

  1. fit_grasp_offset.py        session1 -> T_gripper_cube (P1 VISION fit)
  2. table1_zeus.py             A0~B3, leave-one-placement-out (--s3-gripper-only, --include-session2-board)
  3. eval_heldout_and_consistency.py   joint held-out mm/deg/px (정확도 기준 지표)
  4. eval_joint_relative.py     session1/session2 절대·상대 mm/deg/px (row A5 T_base_Ci)

고정 규칙:
  - 큐브 기하: targets/gt_cube/cube_config.json (GT 큐브, 자체보정된 값)
  - session3 보드는 그리퍼캠만 사용 (--s3-gripper-only)
  - session2 보드 관측 포함 (--include-session2-board): 고정캠 base 앵커
  - 큐브 관측 정책 legacy (그리퍼캠 단면 관측 허용)

사용법:
  python run_zeus_calibration.py --tag 0917_1920x1080 --zeus-intrinsics-dir ../intrinsics_1920x1080_rgbd720
  python run_zeus_calibration.py --tag 0914 --zeus-intrinsics-dir ../intrinsics
    (--tag T => session1/3: capture_replayed_T, session2: capture_placed_T; 개별 지정도 가능)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
GT_CUBE_CONFIG = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"


def run(step, cmd, log_path):
    print(f"\n[{step}] {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run([str(c) for c in cmd], cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT)
    print(f"[{step}] exit={proc.returncode} ({time.time()-t0:.0f}s) log: {log_path}")
    if proc.returncode != 0:
        print(Path(log_path).read_text()[-3000:])
        sys.exit(f"[{step}] failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="촬영분 태그 (예: 0917_1920x1080). capture 폴더명과 출력 이름에 쓰임")
    ap.add_argument("--zeus-intrinsics-dir", required=True)
    ap.add_argument("--device-map", default=None, help="기본: <zeus-intrinsics-dir>/device_map.json")
    ap.add_argument("--session1-capture-subdir", default=None)
    ap.add_argument("--session2-capture-subdir", default=None)
    ap.add_argument("--session3-capture-subdir", default=None)
    ap.add_argument("--cube-config", default=str(GT_CUBE_CONFIG))
    ap.add_argument("--results-root", default=None,
                    help="기본: ABLATION_TEST_result_<MMDD>/zeus_<tag>")
    ap.add_argument("--skip-fit", action="store_true", help="1단계 생략 (기존 fit json 재사용)")
    ap.add_argument("--skip-table1", action="store_true")
    ap.add_argument("--skip-heldout", action="store_true")
    ap.add_argument("--skip-relative", action="store_true")
    args = ap.parse_args()

    tag = args.tag
    s1 = args.session1_capture_subdir or f"capture_replayed_{tag}"
    s2 = args.session2_capture_subdir or f"capture_placed_{tag}"
    s3 = args.session3_capture_subdir or f"capture_replayed_{tag}"
    intr = Path(args.zeus_intrinsics_dir).resolve()
    device_map = Path(args.device_map).resolve() if args.device_map else intr / "device_map.json"
    results = Path(args.results_root).resolve() if args.results_root else \
        REPO_ROOT / f"ABLATION_TEST_result_{time.strftime('%m%d')}" / f"zeus_{tag}"
    results.mkdir(parents=True, exist_ok=True)
    fit_json = HERE / f"fit_통합_{tag}.json"
    table1_dir = results / "table1_zeus"
    methods_json = table1_dir / "ABLATION_TEST_table1_methods.json"
    heldout_json = results / f"heldout_and_consistency_{tag}.json"
    relative_json = results / f"joint_relative_eval_{tag}.json"
    py = sys.executable

    print(f"tag={tag}\n  session1={s1}\n  session2={s2}\n  session3={s3}\n  intrinsics={intr}\n  results={results}")

    if not args.skip_fit:
        run("1/4 fit_grasp_offset", [py, "fit_grasp_offset.py",
             "--capture-subdir", s1, "--zeus-intrinsics-dir", intr, "--device-map", device_map,
             "--cube-config", args.cube_config, "--out", fit_json], results / "log_1_fit.txt")
    if not args.skip_table1:
        table1_dir.mkdir(parents=True, exist_ok=True)
        run("2/4 table1_zeus", [py, "table1_zeus.py",
             "--session1-capture-subdir", s1, "--session2-capture-subdir", s2, "--session3-capture-subdir", s3,
             "--zeus-intrinsics-dir", intr, "--device-map", device_map, "--fit-json", fit_json,
             "--cube-config", args.cube_config, "--s3-gripper-only", "--include-session2-board",
             "--report-dir", table1_dir, "--out", table1_dir / "ABLATION_TEST_table1_zeus.json"],
            results / "log_2_table1.txt")
    if not args.skip_heldout:
        run("3/4 eval_heldout_and_consistency", [py, "eval_heldout_and_consistency.py",
             "--session1-capture-subdir", s1, "--session2-capture-subdir", s2, "--session3-capture-subdir", s3,
             "--zeus-intrinsics-dir", intr, "--device-map", device_map, "--fit-json", fit_json,
             "--cube-config", args.cube_config, "--s3-gripper-only", "--out", heldout_json],
            results / "log_3_heldout.txt")
    if not args.skip_relative:
        run("4/4 eval_joint_relative", [py, "eval_joint_relative.py",
             "--session1-capture-subdir", s1, "--session2-capture-subdir", s2,
             "--zeus-intrinsics-dir", intr, "--device-map", device_map, "--fit-json", fit_json,
             "--cube-config", args.cube_config, "--methods-json", methods_json, "--row", "A5",
             "--out", relative_json], results / "log_4_relative.txt")

    # 요약
    lines = [f"# Zeus 캘리브레이션 요약 ({tag})", ""]
    if methods_json.is_file():
        m = json.loads(methods_json.read_text())
        lines += ["## table1_zeus held-out (px)", "", "| row | held-out Cube RMSE px |", "|---|---:|"]
        for row, r in m["rows"].items():
            v = r["summary"].get("heldout_test_cube_rmse_px") if isinstance(r.get("summary"), dict) else None
            lines.append(f"| {row} | {v:.3f} |" if isinstance(v, (int, float)) else f"| {row} | pending |")
        lines.append("")
    if heldout_json.is_file():
        h = json.loads(heldout_json.read_text())["results"]
        lines += ["## joint held-out (3대 공동 삼각측량 vs FK 정답) -- 정확도 기준 지표", "",
                  "| 방식 | held-out px | joint mm | joint deg | 카메라 합의 mm |", "|---|---:|---:|---:|---:|"]
        for k, r in h.items():
            hc = r["heldout_cross"]
            lines.append(f"| {k} | {r['heldout_cube_rmse_px']:.3f} | {hc['heldout_joint_translation_mean_mm']:.2f} | "
                         f"{hc['heldout_joint_rotation_mean_deg']:.2f} | {hc['cam_common_translation_mm']:.2f} |")
        lines.append("")
    if relative_json.is_file():
        rj = json.loads(relative_json.read_text())
        lines += ["## session1/2 joint 절대·상대 (row A5)", "",
                  "| 세션 | 절대 mm (평균/중앙/P95) | 절대 deg | 절대 px | 상대 mm (평균/중앙/P95) | 상대 deg | 상대 px |",
                  "|---|---|---:|---:|---|---:|---:|"]
        for k in ("session1", "session2_held"):
            if k in rj:
                a, r = rj[k]["absolute"], rj[k]["relative"]
                lines.append(f"| {k} | {a['mm_mean']:.2f}/{a['mm_median']:.2f}/{a['mm_p95']:.2f} | {a['deg_mean']:.2f} | {a['px_rmse']:.2f} | "
                             f"{r['mm_mean']:.2f}/{r['mm_median']:.2f}/{r['mm_p95']:.2f} | {r['deg_mean']:.2f} | {r['px_rmse']:.2f} |")
    summary = results / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {summary}")


if __name__ == "__main__":
    main()
