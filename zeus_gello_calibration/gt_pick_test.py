#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/gt_pick_test.py -- 외부 GT 큐브 vision 기반 pick 정확도 테스트

구형 calibration collector는 "미리 저장해둔 좌표"로 pick/place했지만, 이건 그
반대인 별도 External GT 도구다. targets/gt_cube/의 검증용 큐브를 바닥 아무 곳에나 두고, 이
스크립트가 고정 카메라들로 실시간으로 그 큐브를 검출 -> 우리가 fit한
T_gripper_cube/카메라 extrinsics(fit_grasp_offset.py 결과)로 base 좌표계
pick 목표를 계산 -> 그 자리로 movel해서 집는다. 그 다음 (물리적으로 눈금이
보이는 자리에서) 실제로 얼마나 정확히 집었는지 눈으로 읽어서 그 오차를
기록하는 게 "외부 GT 실험"의 핵심이다 -- 지금까지의 reprojection RMSE/
cross-camera dispersion 같은 self-consistency 지표와 달리, 카메라/FK 계산
체인과 완전히 무관한 물리적 눈금으로 검증하는 것 (같은 체인으로 만든 값을
GT로 쓰면 순환 논리가 된다는 논의 참고).

GT 큐브는 메인 큐브와 마커 6장(AprilTag ID/크기)은 동일하지만 몸체 형상이
다르다(targets/gt_cube/README.md). 마커 ID가 메인 큐브와 겹치므로, 메인
큐브가 같이 시야에 들어오면 안 된다.

pick 자세 계산은 검출된 큐브 pose에 T_gripper_cube(session1 fit, 실제 grasp
offset) 전체를 곱해서 만든다: T_base_flange_target = T_base_cube @
inv(T_gripper_cube). 예전엔 x,y,rz만 큐브 원점에서 가져오고 z/ry/rx는
GRASP_REF_POSE 고정값으로 대체했는데, 이건 "로봇이 정확히 큐브 중심을
잡는다"는 근사였다 -- 실제로는 T_gripper_cube의 x,y가 0이 아니라(0.5~1mm
정도 벗어난 지점을 실제로 잡음) 이 근사가 틀렸다는 게 session1 fit으로
확인됐다. 그래서 지금은 그 근사 없이 실제 offset을 통째로 반영한다.
(메인 큐브와 GT 큐브의 "바닥에서 object-frame origin까지 높이"가 우연히
같다는 사실(둘 다 29.5mm)은 여전히 유효하지만, 이젠 pick 목표 계산에
그 사실을 직접 이용하지 않는다 -- T_gripper_cube 자체가 이미 그 관계를
암묵적으로 반영하고 있기 때문.)

그리퍼 카메라("gripper" 라벨)는 쓰지 않는다 -- fit_grasp_offset.py에서도
그리퍼캠은 쥔 큐브를 못 봐서 fit에서 제외됐고(0/16 PnP 성공), 이 스크립트가
검출하려는 "바닥에 놓인" 큐브 역시 그리퍼캠 시야에 안 들어온다.

사용법:
  python gt_pick_test.py                       # dry-run: 검출만 하고 이동 안 함
  python gt_pick_test.py --execute              # 검출 -> 확인 -> pick 실행 (1회)
  python gt_pick_test.py --execute --trials 5   # 5번 반복 (매번 큐브 새로 놓고 Enter)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T, rodrigues_to_Rt  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from robot.backends.zeus_client import ZeusClient, ZeusError, T_to_pose6  # noqa: E402

from capture_session import (  # noqa: E402
    load_camera_labels, connect_cameras, stop_cameras, LiveView, grab_frames,
    ROBOT_IP_DEFAULT, ROBOT_PORT_DEFAULT, DEVICE_MAP_DEFAULT,
)
from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from session2_pick_and_place import (  # noqa: E402
    APPROACH_MM_DEFAULT, MOVE_LIN_SPEED,
    DESCEND_LIN_SPEED, GRIP_TIMEOUT_S, SETTLE_S, ROBOT_TIMEOUT_DEFAULT, approach_of,
)
from zeus_gello_calibration.paths import (  # noqa: E402
    ZEUS_DATA_ROOT,
    require_zeus_data_path,
)

GT_CUBE_CONFIG_PATH = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"
FIT_JSON_DEFAULT = REPO_ROOT / "zeus_gello_calibration" / "results" / "fits" / "pass1_grasp_offset_replayed.json"
FIXED_LABELS = ("039422061216", "fixed2", "fixed3")  # gripper 캠 제외 (바닥 큐브를 못 봄)


def load_fit(fit_json: Path):
    d = json.loads(fit_json.read_text())
    T_gripper_cube = d["T_gripper_cube"]  # 지금은 안 쓰지만 fit 출처 기록용으로 같이 로드
    T_base_cam = {}
    for key, T in d["T_base_cam"].items():
        local_id = int(key.split("_", 1)[0])
        T_base_cam[local_id] = np.asarray(T, dtype=np.float64)
    T_gripper_cam = np.asarray(d["T_gripper_cam"], dtype=np.float64) if "T_gripper_cam" in d else None
    return T_gripper_cube, T_base_cam, T_gripper_cam


def detect_cube_pose(frames, K_map, D_map, T_base_cam, cube_target, reproj_thr_mean_px):
    """각 고정 카메라에서 GT 큐브를 검출해 T_base_cube 후보를 모으고, 여러 대가
    잡으면 robust 평균으로 합친다. 검출 실패한 카메라는 조용히 건너뛴다."""
    candidates = []
    report = {}
    for label in FIXED_LABELS:
        if label not in frames:
            report[label] = "이 트라이얼 프레임 없음"
            continue
        color, _depth = frames[label]
        if color is None:
            report[label] = "프레임 없음"
            continue
        local_id = LOCAL_CAM_IDS[label]
        if local_id not in T_base_cam:
            report[label] = "이 fit에 extrinsics 없음 (건너뜀)"
            continue
        K, D = K_map[local_id], D_map[local_id]
        ok, rvec, tvec, used, reproj = cube_target.solve_pnp_cube(
            color, K, D, reproj_thr_mean_px=reproj_thr_mean_px, return_reproj=True)
        if not ok:
            report[label] = f"검출 실패 (markers={used})"
            continue
        T_cam_cube = rodrigues_to_Rt(rvec, tvec)
        T_base_cube_c = T_base_cam[local_id] @ T_cam_cube
        candidates.append(T_base_cube_c)
        report[label] = (f"검출됨: markers={used}, reproj_err_mean={reproj['err_mean']:.2f}px, "
                         f"n_points={reproj['n_points']}")
    if not candidates:
        return None, report
    if len(candidates) == 1:
        return candidates[0], report
    T_base_cube, diag = cp.robust_se3_average(candidates, None)
    report["_dispersion"] = diag
    return T_base_cube, report


def build_pick_target(T_base_cube, T_gripper_cube):
    """실제 grasp offset(T_gripper_cube, session1 fit)을 통째로 반영한다.

    예전엔 검출된 큐브의 x,y,rz만 쓰고 z/ry/rx는 GRASP_REF_POSE 고정값으로
    대체했는데, 이건 "로봇이 정확히 큐브 중심을 잡는다"는 근사였다. 그런데
    session1 fit으로 실제 grasp offset을 알아낸 지금은(x,y로도 0.5~1mm 정도
    벗어난 지점을 실제로 잡고 있었음, z도 178.75mm 자체가 큐브 치수가 아니라
    "이 높이면 잡힌다"는 실측 검증값일 뿐), 이 근사를 걷어내고 T_base_cube @
    inv(T_gripper_cube)를 그대로 써서 x,y,z,rz,ry,rx 전부 실측 offset대로
    계산한다. 외부 GT 검증처럼 mm 단위 정확도를 따질 때는 이 차이가 중요하다."""
    T_target = np.asarray(T_base_cube, dtype=np.float64) @ inv_T(np.asarray(T_gripper_cube, dtype=np.float64))
    pose6 = T_to_pose6(T_target)
    return [float(v) for v in pose6]


def run_pick(rb: ZeusClient, target, approach_mm, move_speed, descend_speed):
    approach = approach_of(target, approach_mm)
    print(f"  -> approach 이동: {[round(v, 1) for v in approach]}")
    rb.movel(approach, lin_speed=move_speed)
    time.sleep(SETTLE_S)
    print(f"  -> 수직 하강: {[round(v, 1) for v in target]}")
    rb.movel(target, lin_speed=descend_speed)
    time.sleep(SETTLE_S)
    print("  -> 그리퍼 닫기 (pick)")
    rb.grip("close", timeout_s=GRIP_TIMEOUT_S)
    time.sleep(SETTLE_S)
    print(f"  -> 수직 상승: {[round(v, 1) for v in approach]}")
    rb.movel(approach, lin_speed=descend_speed)
    time.sleep(SETTLE_S)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    ap.add_argument("--robot-port", type=int, default=ROBOT_PORT_DEFAULT)
    ap.add_argument("--robot-timeout", type=float, default=ROBOT_TIMEOUT_DEFAULT)
    ap.add_argument("--device-map", default=str(DEVICE_MAP_DEFAULT))
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--fit-json", default=str(FIT_JSON_DEFAULT),
                    help="T_gripper_cube/카메라 extrinsics 출처 (fit_grasp_offset.py 결과)")
    ap.add_argument("--gt-cube-config", default=str(GT_CUBE_CONFIG_PATH))
    ap.add_argument("--reproj-thr-px", type=float, default=10.0)
    ap.add_argument("--approach-mm", type=float, default=APPROACH_MM_DEFAULT)
    ap.add_argument("--move-speed", type=float, default=MOVE_LIN_SPEED)
    ap.add_argument("--descend-speed", type=float, default=DESCEND_LIN_SPEED)
    ap.add_argument("--trials", type=int, default=1, help="반복 횟수 -- 매 트라이얼마다 큐브를 새로 놓고 진행")
    ap.add_argument(
        "--log",
        default=str(ZEUS_DATA_ROOT / "external_gt" / "gt_pick_test_log.json"),
        help=f"External GT trial log (must stay inside {ZEUS_DATA_ROOT})",
    )
    ap.add_argument("--execute", action="store_true", help="실제로 검출+이동 (없으면 검출만 하고 이동 안 함)")
    ap.add_argument("--no-cam-reset", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()
    try:
        log_path = require_zeus_data_path(args.log, label="--log")
    except ValueError as exc:
        ap.error(str(exc))

    cube_cfg, src = load_cube_config_from_json_file(args.gt_cube_config)
    if cube_cfg is None:
        print(f"[ERROR] GT 큐브 config를 못 읽었습니다: {args.gt_cube_config}")
        return
    print(f"GT 큐브 config 로드: {args.gt_cube_config} ({src})")
    cube_target = AprilTagCubeTarget(cube_cfg)

    K_map, D_map = load_intrinsics_by_label(
        Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))
    T_gripper_cube, T_base_cam, T_gripper_cam = load_fit(Path(args.fit_json))
    print(f"고정 카메라 extrinsics 로드: {sorted(T_base_cam)} (from {args.fit_json})")
    if T_gripper_cube is None:
        print("[ERROR] 이 fit-json엔 T_gripper_cube가 없습니다. pick 목표를 계산할 수 없습니다.")
        return
    t_mm = np.asarray(T_gripper_cube, dtype=np.float64)[:3, 3] * 1000
    print(f"T_gripper_cube translation_mm={np.round(t_mm, 2).tolist()} (실제 grasp offset 그대로 사용) "
          f"approach={args.approach_mm}mm\n")

    labels = load_camera_labels(Path(args.device_map))
    cams, used_labels = connect_cameras(labels, no_reset=args.no_cam_reset)
    view = None
    if not args.no_preview:
        view = LiveView(cams, used_labels, window_name="gt_pick_test (q/ESC=닫기)")
        view.start()

    rb = None
    if args.execute:
        rb = ZeusClient(args.robot_ip, args.robot_port, timeout=args.robot_timeout)
        rb.connect()
        print("*** 실제 로봇이 자동으로 움직이고 그리퍼를 조작합니다. GELLO 텔레옵은 완전히 "
              "종료된 상태여야 합니다. 비상정지에 손이 닿는지 확인하세요. ***\n")

    log_entries = []
    try:
        for trial in range(1, args.trials + 1):
            cmd = input(f"\n=== 트라이얼 {trial}/{args.trials} === "
                        "GT 큐브를 바닥에 놓고 Enter (q=중단) > ").strip().lower()
            if cmd == "q":
                break

            if view is not None:
                view.show()
            frames = grab_frames(cams, used_labels)
            T_base_cube, report = detect_cube_pose(
                frames, K_map, D_map, T_base_cam, cube_target, args.reproj_thr_px)
            for label, msg in report.items():
                print(f"  [{label}] {msg}")

            if T_base_cube is None:
                print("  [ERROR] 어떤 카메라에서도 GT 큐브를 검출하지 못했습니다. 이 트라이얼 건너뜁니다.")
                continue

            target = build_pick_target(T_base_cube, T_gripper_cube)
            print(f"  검출된 pick 목표: {[round(v, 2) for v in target]}")

            entry = {
                "trial": trial,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "T_base_cube": T_base_cube.tolist(),
                "pick_target_pose6": target,
                "per_camera": {k: v for k, v in report.items() if k != "_dispersion"},
            }
            if "_dispersion" in report:
                entry["cross_camera_dispersion"] = report["_dispersion"]

            if not args.execute:
                print("  (dry-run) --execute 없이는 이동하지 않습니다.")
                log_entries.append(entry)
                continue

            cmd = input("  이 목표로 pick 진행할까요? Enter=진행 / s=건너뛰기 / q=중단 > ").strip().lower()
            if cmd == "q":
                log_entries.append(entry)
                break
            if cmd == "s":
                log_entries.append(entry)
                continue

            try:
                run_pick(rb, target, args.approach_mm, args.move_speed, args.descend_speed)
            except ZeusError as e:
                print(f"  [ZeusError] {e}")
                entry["error"] = str(e)
                log_entries.append(entry)
                continue

            measured = input("  실측 오차(눈금 읽은 값, mm)를 입력하세요 (모르면 그냥 Enter) > ").strip()
            if measured:
                entry["measured_error_mm"] = measured
            log_entries.append(entry)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if rb is not None:
            rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if log_path.exists():
        try:
            existing = json.loads(log_path.read_text())
        except Exception:
            existing = []
    existing.extend(log_entries)
    log_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    print(f"\n완료 -- {len(log_entries)}개 트라이얼 기록 -> {log_path}")


if __name__ == "__main__":
    main()
