#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/gt_compare_fits.py -- 큐브 사진 한 장으로 4가지 fit의
추론값(큐브 중심 base-frame 위치)을 비교한다. 로봇은 움직이지 않는다(현재
자세 그대로 읽기만 함).

GT 큐브를 바닥에 두고 한 번만 촬영(고정캠 3대 + 그리퍼캠) + 그 순간의 로봇
pose(FK)를 같이 읽음 -> 그 이미지들에 대해 fit_통합_no-fk.json /
fit_독립_no-fk.json / fit_통합_raw-fk.json / fit_독립_raw-fk.json 각각의
카메라 extrinsics(고정캠 T_base_Ci + 그리퍼캠 T_gripper_cam)로 큐브 pose를
계산 -> 네 결과의 x,y,z,rz와 서로 간 차이(mm)를 출력한다. 검출(PnP)은
fit마다 다시 안 하고 한 번만 하고, 그 결과에 각 fit의 extrinsics만 다르게
적용한다(검출 자체는 fit과 무관하니까).

그리퍼캠은 지금 로봇이 있는 자세에서 실제로 GT 큐브를 볼 수 있어야
잡힌다(session2 파킹 자세 근처처럼) -- 안 보이면 그 fit의 그리퍼캠 칸은
"검출 실패"로 빠지고 고정캠들로만 합쳐진다.

사용법:
  python gt_compare_fits.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, rodrigues_to_Rt  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from robot.backends.zeus_client import T_to_pose6, ZeusClient, pose6_to_T  # noqa: E402

from capture_session import (  # noqa: E402
    load_camera_labels, connect_cameras, stop_cameras, LiveView, grab_frames, write_capture,
    read_robot_state, DEVICE_MAP_DEFAULT, ROBOT_IP_DEFAULT, ROBOT_PORT_DEFAULT,
)
from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from gt_pick_test import GT_CUBE_CONFIG_PATH, FIXED_LABELS, load_fit  # noqa: E402
from session2_pick_and_place import ROBOT_TIMEOUT_DEFAULT  # noqa: E402
from zeus_gello_calibration.paths import (  # noqa: E402
    ZEUS_DATA_ROOT,
    require_zeus_data_path,
)
from capture_pipeline.paths import resolve_dated_dir

def detect_cube_pose_all(frames, K_map, D_map, T_base_cam, T_gripper_cam, T_base_gripper_now,
                         cube_target, reproj_thr_mean_px):
    """고정캠 3대 + 그리퍼캠(있으면)에서 GT 큐브를 검출해 합친다."""
    candidates = []
    report = {}
    for label in FIXED_LABELS:
        if label not in frames or frames[label][0] is None:
            report[label] = "프레임 없음"
            continue
        local_id = LOCAL_CAM_IDS[label]
        if local_id not in T_base_cam:
            report[label] = "이 fit에 extrinsics 없음 (건너뜀)"
            continue
        K, D = K_map[local_id], D_map[local_id]
        ok, rvec, tvec, used, reproj = cube_target.solve_pnp_cube(
            frames[label][0], K, D, reproj_thr_mean_px=reproj_thr_mean_px, return_reproj=True)
        if not ok:
            report[label] = f"검출 실패 (markers={used})"
            continue
        T_cam_cube = rodrigues_to_Rt(rvec, tvec)
        candidates.append(T_base_cam[local_id] @ T_cam_cube)
        report[label] = (f"검출됨: markers={used}, reproj_err_mean={reproj['err_mean']:.2f}px, "
                         f"n_points={reproj['n_points']}")

    if T_gripper_cam is None:
        report["gripper"] = "이 fit에 T_gripper_cam 없음"
    elif "gripper" not in frames or frames["gripper"][0] is None:
        report["gripper"] = "프레임 없음"
    else:
        local_id = LOCAL_CAM_IDS["gripper"]
        K, D = K_map[local_id], D_map[local_id]
        ok, rvec, tvec, used, reproj = cube_target.solve_pnp_cube(
            frames["gripper"][0], K, D, reproj_thr_mean_px=reproj_thr_mean_px, return_reproj=True)
        if not ok:
            report["gripper"] = f"검출 실패 (markers={used})"
        else:
            T_cam_cube = rodrigues_to_Rt(rvec, tvec)
            candidates.append(T_base_gripper_now @ T_gripper_cam @ T_cam_cube)
            report["gripper"] = (f"검출됨: markers={used}, reproj_err_mean={reproj['err_mean']:.2f}px, "
                                 f"n_points={reproj['n_points']}")

    if not candidates:
        return None, report
    if len(candidates) == 1:
        return candidates[0], report
    T_base_cube, diag = cp.robust_se3_average(candidates, None)
    report["_dispersion"] = diag
    return T_base_cube, report


# 학습 목적함수(px 재투영 vs mm pose 오차)별로 3가지 방식 x 2 = 6개.
# px는 fit_calibration_methods.py가, mm은 fit_calibration_methods_mm.py가 각각
# fit_<조건>[_mm].json으로 저장해둔다. "독립_raw-fk"는 큐브 위치가 상수라 통합과
# 수학적으로 완전히 같은 답을 내서(block-separable) 따로 안 만든다 -- 예전엔
# 여기 들어있었는데 지금 "독립"의 정의(핸드오프 없는 완전 독립)와 안 맞는 옛날
# 파일이라 제거했다.
DEFAULT_FITS = [
    ("통합_no-fk_px", REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_통합_no-fk.json"),
    ("통합_no-fk_mm", REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_통합_no-fk_mm.json"),
    ("통합_raw-fk_px", REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_통합_raw-fk.json"),
    ("통합_raw-fk_mm", REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_통합_raw-fk_mm.json"),
    ("독립_no-fk_px", REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_독립_no-fk.json"),
    ("독립_no-fk_mm", REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "fit_독립_no-fk_mm.json"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device-map", default=str(DEVICE_MAP_DEFAULT))
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--gt-cube-config", default=str(GT_CUBE_CONFIG_PATH))
    ap.add_argument("--reproj-thr-px", type=float, default=10.0)
    ap.add_argument("--no-cam-reset", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument(
        "--out-root",
        default=str(resolve_dated_dir("gt_compare_captures", ZEUS_DATA_ROOT)),
        help=f"GT capture output (must stay inside {ZEUS_DATA_ROOT}; "
             "이름에 촬영 날짜 _MMDD 가 붙는다)",
    )
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    ap.add_argument("--robot-port", type=int, default=ROBOT_PORT_DEFAULT)
    ap.add_argument("--robot-timeout", type=float, default=ROBOT_TIMEOUT_DEFAULT)
    args = ap.parse_args()
    try:
        out_root = require_zeus_data_path(args.out_root, label="--out-root")
    except ValueError as exc:
        ap.error(str(exc))

    cube_cfg, src = load_cube_config_from_json_file(args.gt_cube_config)
    if cube_cfg is None:
        print(f"[ERROR] GT 큐브 config를 못 읽었습니다: {args.gt_cube_config}")
        return
    cube_target = AprilTagCubeTarget(cube_cfg)
    K_map, D_map = load_intrinsics_by_label(
        Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))

    labels = load_camera_labels(Path(args.device_map))
    cams, used_labels = connect_cameras(labels, no_reset=args.no_cam_reset)
    view = None
    if not args.no_preview:
        view = LiveView(cams, used_labels, window_name="gt_compare_fits (q/ESC=닫기)")
        view.start()

    rb = ZeusClient(args.robot_ip, args.robot_port, timeout=args.robot_timeout)
    rb.connect()
    try:
        input("\nGT 큐브를 바닥에 놓고 Enter를 누르면 한 장 촬영합니다 (로봇은 안 움직임) > ")
        if view is not None:
            view.show()
        frames = grab_frames(cams, used_labels)
        robot_state = read_robot_state(rb, {"note": "gt_compare_fits: 로봇 이동 없음, 현재 자세 그대로 읽음"})
        T_base_gripper_now = pose6_to_T(robot_state["pose"])
        out_dir = out_root / time.strftime("%Y%m%d_%H%M%S")
        write_capture(frames, out_dir, robot_state)
    finally:
        rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)

    print(f"촬영 시점 로봇 pose: {[round(v, 2) for v in robot_state['pose']]}")

    results = []
    for label, fit_path in DEFAULT_FITS:
        if not fit_path.is_file():
            print(f"[WARN] {fit_path} 없음, 건너뜀")
            continue
        _T_gripper_cube, T_base_cam, T_gripper_cam = load_fit(fit_path)
        T_base_cube, report = detect_cube_pose_all(
            frames, K_map, D_map, T_base_cam, T_gripper_cam, T_base_gripper_now,
            cube_target, args.reproj_thr_px)
        if T_base_cube is None:
            print(f"[{label}] 검출 실패: {report}")
            continue
        pose6 = T_to_pose6(T_base_cube)
        results.append((label, pose6, report))

    print(f"\n{'method':>14} {'x_mm':>10} {'y_mm':>10} {'z_mm':>10} {'rz_deg':>10}")
    for label, pose6, _ in results:
        print(f"{label:>14} {pose6[0]:>10.2f} {pose6[1]:>10.2f} {pose6[2]:>10.2f} {pose6[3]:>10.2f}")

    if len(results) >= 2:
        print("\n방식 간 위치 차이 (mm, xyz 유클리드 거리):")
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                li, pi, _ = results[i]
                lj, pj, _ = results[j]
                d = float(np.linalg.norm(np.asarray(pi[:3]) - np.asarray(pj[:3])))
                print(f"  {li} vs {lj}: {d:.3f} mm")

    for label, _pose6, report in results:
        print(f"\n[{label}] 카메라별 검출 상세:")
        for cam_label, msg in report.items():
            print(f"  [{cam_label}] {msg}")


if __name__ == "__main__":
    main()
