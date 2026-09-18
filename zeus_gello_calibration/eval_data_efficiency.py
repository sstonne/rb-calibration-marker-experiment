#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/eval_data_efficiency.py -- session2 placement 개수를
30%/50%/70%(그리고 참고로 100%)만 써서 캘리브레이션했을 때, 통합_no-fk /
통합_raw-fk / 독립_no-fk 각각이 나머지(held-out) placement를 얼마나 잘
설명하는지 비교한다 -- "데이터가 적을 때 통합의 이점이 더 커지는가/작아지는가"
를 보는 실험.

session1(42, grasp+FK)과 session3(15, 그리퍼 보드)는 항상 전부 포함한다 --
줄이는 건 session2의 15개 placement 중 학습에 쓰는 개수뿐이다.

session2가 15개뿐이라 특정 조합을 한 번만 뽑으면 운에 따라 결과가 크게
흔들린다. 그래서 각 비율마다 서로 다른 랜덤 부분집합으로 N번(기본 8번)
반복해서 평균±표준편차를 낸다.

정답(ground truth)은 eval_heldout_and_consistency.py와 같은 방식 --
"그 세트에서 로봇이 실제로 명령받아 간 FK 위치 @ session1의 T_gripper_cube"
(비전 무관, 완전 독립).

사용법:
  python eval_data_efficiency.py
  python eval_data_efficiency.py --ratios 0.3,0.5,0.7,1.0 --repeats 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import inv_T  # noqa: E402
from calibration_pipeline.reprojection import pose_delta, project_points  # noqa: E402
from calibration_pipeline.table1 import estimate_board_handeye_initial  # noqa: E402

from eval_heldout_and_consistency import camera_cube_estimate, fit_frozen  # noqa: E402
from fit_calibration_methods import (  # noqa: E402
    GRIPPER_LOCAL_ID, SESSION1_DIR_DEFAULT, SESSION3_DIR_DEFAULT, fk_anchor_cubes, load_all_data,
    rmse_px,
)
from session2_pick_and_place import SESSION2_DIR_DEFAULT  # noqa: E402


def evaluate_on_sets(cams, gtc, data, robot_T_all, K_map, D_map, test_set_ids):
    """이미 학습된(cams, gtc)로, test_set_ids의 큐브들을 FK 기반(비전 무관)
    정답과 비교 -- px RMSE + 실제 mm/deg 오차 둘 다 반환."""
    grasp_init = data["grasp_init"]
    obs_all_s2 = data["obs_s2_fixed"] + data["obs_s2_gripper"]
    px_errs = []
    mm_list, deg_list = [], []
    for s in test_set_ids:
        if s not in data["items_by_index"]:
            continue
        T_gt = fk_anchor_cubes({s: data["items_by_index"][s]}, grasp_init)[s]
        set_obs = [o for o in obs_all_s2 if o.set_idx is not None and int(o.set_idx) == s]
        set_errs = []
        for o in set_obs:
            c = int(o.cam)
            if c == GRIPPER_LOCAL_ID:
                if int(o.event) not in robot_T_all:
                    continue
                T_base_cam = robot_T_all[int(o.event)] @ gtc
            else:
                if c not in cams:
                    continue
                T_base_cam = cams[c]
            pred = project_points(inv_T(T_base_cam) @ T_gt, o.object_points, K_map[c], D_map[c])
            set_errs.extend(np.linalg.norm(pred - o.image_points, axis=1).tolist())
        if not set_errs:
            continue
        px_errs.extend(set_errs)

        cands = [camera_cube_estimate(o, cams, gtc, robot_T_all, K_map, D_map, GRIPPER_LOCAL_ID)
                 for o in set_obs]
        cands = [c for c in cands if c is not None]
        if cands:
            T_vision = cands[0] if len(cands) == 1 else cp.robust_se3_average(cands, None)[0]
            d_mm, d_deg = pose_delta(T_vision, T_gt)
            mm_list.append(d_mm)
            deg_list.append(d_deg)
    return {
        "px_rmse": rmse_px(px_errs) if px_errs else float("nan"),
        "mm_mean": float(np.mean(mm_list)) if mm_list else float("nan"),
        "deg_mean": float(np.mean(deg_list)) if deg_list else float("nan"),
        "n_test_sets": len(test_set_ids),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session1-dir", default=str(SESSION1_DIR_DEFAULT))
    ap.add_argument("--session1-capture-subdir", default="capture_replayed")
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR_DEFAULT))
    ap.add_argument("--session2-capture-subdir", default="capture_placed")
    ap.add_argument("--session3-dir", default=str(SESSION3_DIR_DEFAULT))
    ap.add_argument("--session3-capture-subdir", default="capture_replayed")
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--fit-json", default=str(REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "pass1_grasp_offset_replayed.json"))
    ap.add_argument("--fixed-min-corners", type=int, default=8)
    ap.add_argument("--cube-observation-policy", default="legacy", choices=("legacy", "core_multiface"))
    ap.add_argument("--ratios", default="0.3,0.5,0.7")
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "results" / "heldout" / "data_efficiency.json"))
    args = ap.parse_args()

    data = load_all_data(args)
    K_map, D_map = data["K_map"], data["D_map"]
    all_set_ids = sorted(data["items_by_index"])
    n_total = len(all_set_ids)
    robot_T_all = {**data["robot_T_s1"], **data["robot_T_s2_gripper"], **data["robot_T_s3"]}

    gtc_init, board_init, _eih_diag = estimate_board_handeye_initial(
        data["obs_s3"], data["robot_T_s3"], K_map, D_map, GRIPPER_LOCAL_ID)

    ratios = [float(r) for r in args.ratios.split(",")]
    rng = random.Random(args.seed)

    conditions = [
        ("통합", "no_fk", "통합_no-fk"),
        ("통합", "fixed_fk", "통합_raw-fk"),
        ("독립_true", "no_fk", "독립_no-fk"),
    ]

    results = {}
    for ratio in ratios:
        n_train = max(1, round(ratio * n_total))
        n_test = n_total - n_train
        print(f"\n=== ratio={ratio:.0%} (train {n_train}/{n_total}개, test {n_test}개, {args.repeats}회 반복) ===")
        per_condition = {label: {"px": [], "mm": [], "deg": []} for _, _, label in conditions}

        for rep in range(args.repeats):
            shuffled = all_set_ids[:]
            rng.shuffle(shuffled)
            train_ids = set(shuffled[:n_train])
            test_ids = [s for s in all_set_ids if s not in train_ids]
            if not test_ids:
                continue

            data_fold = dict(data)
            data_fold["obs_s2_fixed"] = [o for o in data["obs_s2_fixed"]
                                         if o.set_idx is None or int(o.set_idx) in train_ids]
            data_fold["obs_s2_gripper"] = [o for o in data["obs_s2_gripper"]
                                           if o.set_idx is None or int(o.set_idx) in train_ids]
            data_fold["items_by_index"] = {k: v for k, v in data["items_by_index"].items() if k in train_ids}

            for method, fk_mode, label in conditions:
                try:
                    cams, gtc = fit_frozen(method, data_fold, fk_mode, gtc_init, board_init)
                except Exception as exc:
                    print(f"    [WARN] {label} rep={rep}: fit 실패 ({exc}), 건너뜀")
                    continue
                metrics = evaluate_on_sets(cams, gtc, data, robot_T_all, K_map, D_map, test_ids)
                if not np.isnan(metrics["px_rmse"]):
                    per_condition[label]["px"].append(metrics["px_rmse"])
                if not np.isnan(metrics["mm_mean"]):
                    per_condition[label]["mm"].append(metrics["mm_mean"])
                    per_condition[label]["deg"].append(metrics["deg_mean"])

        results[f"{ratio:.0%}"] = {}
        for _, _, label in conditions:
            v = per_condition[label]
            results[f"{ratio:.0%}"][label] = {
                "px_mean": float(np.mean(v["px"])) if v["px"] else float("nan"),
                "px_std": float(np.std(v["px"])) if v["px"] else float("nan"),
                "mm_mean": float(np.mean(v["mm"])) if v["mm"] else float("nan"),
                "mm_std": float(np.std(v["mm"])) if v["mm"] else float("nan"),
                "deg_mean": float(np.mean(v["deg"])) if v["deg"] else float("nan"),
                "n_valid_repeats": len(v["px"]),
            }
            r = results[f"{ratio:.0%}"][label]
            print(f"  {label:>14}  px={r['px_mean']:.3f}±{r['px_std']:.3f}  "
                  f"mm={r['mm_mean']:.3f}±{r['mm_std']:.3f}  deg={r['deg_mean']:.3f}  "
                  f"(n={r['n_valid_repeats']})")

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
