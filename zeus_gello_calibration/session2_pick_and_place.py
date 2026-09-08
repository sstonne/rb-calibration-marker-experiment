#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/session2_pick_and_place.py -- Zeus로 session2 pick-and-place
+ 촬영 (ur3_calibration/session2_pick_and_place.py 와 같은 개념)

START_JOINTS/START_POSE로 지정한 자세(그리퍼에 큐브를 쥔 채)에서 시작해서,
session2에 저장된 15곳에 순서대로 "놓고 -> (다음 자리로) 다시 집어서 놓고
-> ..."를 반복하며 매번 놓은 직후 고정 촬영 위치로 가서 카메라 4대 + 로봇
상태를 저장한다.

*** Zeus는 rz,ry,rx가 진짜 오일러각(pose6_to_T의 extrinsic ZYX)이라서, UR3 때
rotation vector 표현 때문에 필요했던 "짐벌락 회피/moveJ 안전장치" 같은 게
필요 없다 -- z,ry,rx를 고정값으로 그대로 대입하고 rz만 갈아끼우면 그 자체로
"기울기 고정, yaw만 회전"이 정확히 성립한다 (rz,ry,rx는 world 축 기준으로
순서대로 곱해지는 extrinsic 표현이라 rz만 바꾸는 게 그대로 world Z축 회전이다).
그래도 이동 순서는 인접 자세 간 rz 차이가 작아지도록 정렬해서 불필요하게
큰 회전을 연달아 하지 않게 한다.

세션2 목표 자세 = [x_i, y_i, Z_FIXED, rz_i, RY_FIXED, RX_FIXED]
  x_i, y_i, rz_i : session2 각 캡처의 저장된 pose에서 그대로 가져옴
  Z_FIXED/RY_FIXED/RX_FIXED : GRASP_REF_POSE(아래, 실측값)에서 고정

카메라 촬영 위치(그리퍼가 비었을 때 파킹하는 자세)는 아직 별도로 안 정해서
**임시로 session3의 첫 번째 저장된 자세를 그대로 쓴다** (session3 캡처 폴더의
joints를 읽어서 movej) -- 나중에 제대로 된 파킹 자세가 생기면 --cam-pose-joints
로 바꿔 끼우면 된다.

*** 실제 로봇에 물건을 집었다 놓았다 하는 자동 이동 스크립트다. ***
GELLO 텔레옵은 반드시 완전히 종료(Ctrl+C)한 상태여야 한다 (motion ownership
을 이 스크립트가 가져야 movel/movej/grip을 보낼 수 있음 -- server/zeus_gello.py
참고). 처음 실행은 로봇 옆에서 비상정지에 손이 닿는 상태로, --execute 만 주고
(스텝별 Enter 확인) 진행할 것.

사용법:
  python session2_pick_and_place.py                     # dry-run: 계획만 출력
  python session2_pick_and_place.py --execute            # 스텝별 Enter로 확인하며 실행
  python session2_pick_and_place.py --execute --no-step  # (검증 후) 연속 실행
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot.backends.zeus_client import ZeusClient, ZeusError  # noqa: E402

from capture_session import (  # noqa: E402
    load_camera_labels, connect_cameras, stop_cameras, LiveView,
    grab_frames, write_capture, read_robot_state,
    ROBOT_IP_DEFAULT, ROBOT_PORT_DEFAULT, DEVICE_MAP_DEFAULT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT_DEFAULT = Path(__file__).resolve().parent / "data"

SESSION2_DIR_DEFAULT = DATA_ROOT_DEFAULT / "session2_floor_board_dual_cam"
SESSION3_DIR_DEFAULT = DATA_ROOT_DEFAULT / "session3_wrist_motion_gripper_cam"

# 실측: "Pos, -292.02, 400.03, 178.75, -90.00, -0.00, 180.00" -- 여기서
# z/ry/rx만 고정값으로 쓴다 (x,y는 session2 각 자세에서 그대로 가져오므로 안 씀).
GRASP_REF_POSE = [-292.02, 400.03, 178.75, -90.00, -0.00, 180.00]
Z_FIXED = GRASP_REF_POSE[2]
RY_FIXED = GRASP_REF_POSE[4]
RX_FIXED = GRASP_REF_POSE[5]

# 0단계 시작 자세: session1 끝난 자세 대신 지정받은 고정 자세로 이동한다.
# joints는 실측 지정값, pose는 실제로 이 joints로 movej했을 때 get_state()가
# 돌려준 값(align_rotation이 여기서부터 회전 정렬을 시작하는 기준점으로 씀).
START_JOINTS = [49.47, -20.79, -115.64, 0.01, -43.57, -130.53]
START_POSE = [-261.367, 355.019, 259.716, -90.007, -0.005, 179.996]

# 촬영 파킹 자세(그리퍼 카메라로 찍으러 가는 위치): session3 첫 캡처 대신
# 지정받은 고정 joints로 이동한다. pose는 실제로 이 joints로 movej한 뒤
# get_state()로 읽은 실측값.
CAM_POSE_JOINTS = [36.48, 14.59, -95.20, 3.08, -80.23, -113.78]
CAM_POSE_POSE = [-80.785, 270.964, 603.407, -120.547, 4.803, 161.178]

APPROACH_MM_DEFAULT = 50.0   # 5cm -- pick/place 직후 수직 유지 거리
MOVE_LIN_SPEED = 30.0        # mm/s? -- zeus_client movel의 lin_speed, 저속으로 시작
DESCEND_LIN_SPEED = 15.0     # 수직 하강/상승은 더 느리게
JNT_SPEED_PARK = 10.0        # 파킹 자세로 갈 때 (movej, joints 직접)
GRIP_TIMEOUT_S = 3.0
SETTLE_S = 0.3
# movel/movej는 서버가 이동 완료까지 응답을 안 보내는 blocking 방식이라, 파킹
# 위치가 멀어지면(z=603mm 등) 이동거리가 길어져 ZeusClient 기본 타임아웃(10초)을
# 넘길 수 있다 -- 실제로 로봇은 정상 도착했는데 클라이언트만 먼저 타임아웃난
# 사례가 있어서 여유있게 잡음.
ROBOT_TIMEOUT_DEFAULT = 30.0


def approach_of(pose6, offset_mm):
    p = list(pose6)
    p[2] += offset_mm  # Zeus pose z는 이미 mm 단위
    return p


def load_latest_state(session_dir: Path, subdir_candidates=("capture_replayed", "capture"),
                       index: int = -1):
    """세션 폴더에서 지정한 인덱스(기본 -1=마지막)의 저장된 joints/pose를 읽는다.
    capture_replayed가 있으면 그걸 우선한다(더 정확한 정지 상태 재현)."""
    for sub in subdir_candidates:
        d = session_dir / sub
        if not d.is_dir():
            continue
        dirs = sorted((p for p in d.iterdir() if p.is_dir()), key=lambda p: int(p.name))
        if not dirs:
            continue
        chosen = dirs[index]
        data = json.loads((chosen / "robot.json").read_text())
        return data["joints"], data["pose"], chosen
    raise FileNotFoundError(f"{session_dir} 아래에 저장된 캡처가 없습니다.")


def align_rotation(pose_now, target_pose):
    """target_pose의 회전(rz,ry,rx)만 가져오고 위치(x,y,z)는 pose_now 그대로 -- 즉
    "제자리에서 회전만 먼저 맞추는" 중간 목표. 위치+회전을 한 movel에 같이
    넣으면 궤적이 커서 Unreachable이 나기 쉬운데, 회전 따로/이동 따로 나누면
    각 movel이 훨씬 단순해진다."""
    return [pose_now[0], pose_now[1], pose_now[2], target_pose[3], target_pose[4], target_pose[5]]


def compute_ordered_targets(session2_dir: Path) -> list:
    """session2 캡처들의 pose에서 x,y,rz만 뽑아 목표 자세를 만들고,
    인접 자세 간 rz 차이가 작아지도록 정렬한다."""
    capture_root = session2_dir / "capture"
    dirs = sorted((p for p in capture_root.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    items = []
    for d in dirs:
        data = json.loads((d / "robot.json").read_text())
        x, y, _z, rz, _ry, _rx = data["pose"]
        target = [x, y, Z_FIXED, rz, RY_FIXED, RX_FIXED]
        items.append({"orig_index": int(d.name), "rz": rz, "target": target})
    items.sort(key=lambda it: it["rz"])
    return items


def mv(pose6, speed, desc):
    return {"kind": "movel", "pose": pose6, "speed": speed, "desc": desc}


def mj(joints, speed, desc):
    return {"kind": "movej", "joints": joints, "speed": speed, "desc": desc}


def grip(state, desc):
    return {"kind": "grip", "state": state, "desc": desc}


def cap(desc):
    return {"kind": "capture", "desc": desc}


def build_plan(items, start_pose, start_joints, cam_pose, cam_pose_joints, approach_mm, return_home,
               move_speed=MOVE_LIN_SPEED, descend_speed=DESCEND_LIN_SPEED, jnt_speed=JNT_SPEED_PARK):
    """이동 순서: 위치+회전을 한 movel에 같이 넣지 않는다 -- 큰 회전 변화가
    있는 구간마다 먼저 "제자리에서 회전만 정렬"(align_rotation)한 뒤에
    위치를 옮긴다. 그래야 각 movel이 더 단순해져서 Unreachable이 덜 난다."""
    steps = []

    def place_only_block(label, current_pose, dest_pose):
        steps.append(mv(align_rotation(current_pose, dest_pose), move_speed,
                        f"[{label}] place 방향 정렬 (제자리 회전)"))
        steps.append(mv(approach_of(dest_pose, approach_mm), move_speed, f"[{label}] place approach 이동"))
        steps.append(mv(dest_pose, descend_speed, f"[{label}] place 수직 하강"))
        steps.append(grip("open", f"[{label}] 그리퍼 열기 (place)"))
        steps.append(mv(approach_of(dest_pose, approach_mm), descend_speed, f"[{label}] place 수직 상승"))
        steps.append(mj(cam_pose_joints, jnt_speed, f"[{label}] 고정 촬영 위치로 이동"))
        steps.append(cap(f"[{label}] 촬영 자리"))

    def pick_place_block(label, source_pose, dest_pose):
        # 직전 스텝이 항상 촬영 파킹(cam_pose)이므로 거기서부터 방향 정렬.
        steps.append(mv(align_rotation(cam_pose, source_pose), move_speed,
                        f"[{label}] pick 방향 정렬 (제자리 회전)"))
        steps.append(mv(approach_of(source_pose, approach_mm), move_speed, f"[{label}] pick approach 이동"))
        steps.append(mv(source_pose, descend_speed, f"[{label}] pick 수직 하강"))
        steps.append(grip("close", f"[{label}] 그리퍼 닫기 (pick)"))
        steps.append(mv(approach_of(source_pose, approach_mm), descend_speed, f"[{label}] pick 수직 상승"))
        place_only_block(label, approach_of(source_pose, approach_mm), dest_pose)

    steps.append(mj(start_joints, jnt_speed, "0단계: 지정된 시작 자세로 이동 (큐브 쥔 상태)"))

    current = None
    for order, item in enumerate(items):
        label = f"{order + 1}/{len(items)} (원본 #{item['orig_index']})"
        if current is None:
            place_only_block(label, start_pose, item["target"])
        else:
            pick_place_block(label, current, item["target"])
        current = item["target"]

    if return_home and current is not None:
        pick_place_block("원위치 복귀", current, GRASP_REF_POSE)

    return steps


def print_plan(steps):
    for i, step in enumerate(steps):
        if step["kind"] == "movel":
            print(f"[{i + 1:3d}] movel  {step['desc']:<40s} "
                  f"{[round(v, 1) for v in step['pose']]}")
        elif step["kind"] == "movej":
            print(f"[{i + 1:3d}] movej  {step['desc']:<40s} "
                  f"{[round(v, 1) for v in step['joints']]}")
        elif step["kind"] == "grip":
            print(f"[{i + 1:3d}] grip   {step['desc']:<40s} state={step['state']}")
        else:
            print(f"[{i + 1:3d}] ----   {step['desc']}")


def execute_plan(steps, rb: ZeusClient, cams, labels, out_root: Path, view, no_step: bool,
                  skip_steps: int = 0):
    # 건너뛰는 스텝 중 몇 개가 "촬영" 스텝이었는지 세어서, 재개했을 때 캡처
    # 번호가 처음부터 다시 매겨지며 기존 파일을 덮어쓰지 않게 한다.
    capture_counter = sum(1 for s in steps[:skip_steps] if s["kind"] == "capture")

    for i, step in enumerate(steps):
        if i < skip_steps:
            continue
        desc = step["desc"]
        if not no_step:
            if view is not None:
                view.show()
            cmd = input(f"\n[{i + 1}/{len(steps)}] {desc}\nEnter=진행 / q=중단 > ").strip().lower()
            if cmd == "q":
                print("중단했습니다.")
                return
        else:
            print(f"[{i + 1}/{len(steps)}] {desc}")

        try:
            if step["kind"] == "movel":
                rb.movel(step["pose"], lin_speed=step["speed"])
            elif step["kind"] == "movej":
                rb.movej(step["joints"], jnt_speed=step["speed"])
            elif step["kind"] == "grip":
                rb.grip(step["state"], timeout_s=GRIP_TIMEOUT_S)
            else:  # capture
                time.sleep(SETTLE_S)
                if view is not None:
                    view.show()
                robot_state = read_robot_state(rb, {"capture_index": capture_counter, "step_index": i})
                frames = grab_frames(cams, labels)
                write_capture(frames, out_root / f"{capture_counter:03d}", robot_state)
                capture_counter += 1
        except ZeusError as exc:
            print(f"\n  [ERROR] 스텝 {i + 1} 실패, 중단합니다: {exc}")
            print(f"  이어서 하려면: --skip-steps {i} (이 스텝부터 다시 시도) "
                  f"또는 --skip-steps {i + 1} (이 스텝 건너뛰고 다음부터)")
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    ap.add_argument("--robot-port", type=int, default=ROBOT_PORT_DEFAULT)
    ap.add_argument("--robot-timeout", type=float, default=ROBOT_TIMEOUT_DEFAULT,
                    help="소켓 응답 대기 시간(초) -- movel/movej는 완료될 때까지 서버가 응답을 "
                         "안 보내는 blocking 방식이라, 이동거리가 길면 기본값(10초)보다 오래 걸릴 수 있음")
    ap.add_argument("--device-map", default=str(DEVICE_MAP_DEFAULT))
    ap.add_argument("--session2-dir", default=str(SESSION2_DIR_DEFAULT))
    ap.add_argument("--session3-dir", default=str(SESSION3_DIR_DEFAULT),
                    help="촬영 파킹 자세를 임시로 여기서(첫 캡처) 가져옴")
    ap.add_argument("--out-root", default=str(SESSION2_DIR_DEFAULT / "capture_placed"))
    ap.add_argument("--approach-mm", type=float, default=APPROACH_MM_DEFAULT)
    ap.add_argument("--move-speed", type=float, default=MOVE_LIN_SPEED, help="수평 이동/회전정렬 movel 속도")
    ap.add_argument("--descend-speed", type=float, default=DESCEND_LIN_SPEED, help="수직 하강/상승 movel 속도")
    ap.add_argument("--jnt-speed", type=float, default=JNT_SPEED_PARK, help="movej(0단계, 촬영 파킹) 속도")
    ap.add_argument("--return-home", action="store_true", help="마지막에 큐브를 GRASP_REF_POSE 위치로 복귀")
    ap.add_argument("--execute", action="store_true", help="실제로 이동/그리퍼/촬영 (없으면 dry-run)")
    ap.add_argument("--no-step", action="store_true", help="스텝마다 Enter로 확인하지 않고 연속 실행")
    ap.add_argument("--skip-steps", type=int, default=0,
                    help="이미 실행된 스텝 수 -- 실패/중단 후 이어서 재실행할 때 그만큼 건너뜀")
    ap.add_argument("--no-cam-reset", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    start_joints, start_pose, start_src = START_JOINTS, START_POSE, "고정 지정값 (START_JOINTS/START_POSE)"

    cam_pose_joints = CAM_POSE_JOINTS
    if CAM_POSE_POSE is not None:
        cam_pose, cam_src = CAM_POSE_POSE, "고정 지정값 (CAM_POSE_JOINTS/CAM_POSE_POSE)"
    else:
        if args.execute:
            print("[ERROR] CAM_POSE_POSE가 아직 미확정입니다 (로봇 연결해서 CAM_POSE_JOINTS로 movej 후 "
                  "get_state()로 채워야 함). 정확한 값 없이 --execute 하면 회전-정렬 이동이 잘못된 "
                  "위치를 기준으로 계산돼서 위험합니다. 로봇이 연결되면 값을 채운 뒤 다시 실행하세요.")
            return
        # dry-run 미리보기 전용 근사치 (실제 실행 전에는 반드시 CAM_POSE_POSE를 채워야 함)
        _joints_unused, cam_pose, _src = load_latest_state(
            Path(args.session3_dir), subdir_candidates=("capture",), index=0)
        cam_src = ("[미확정 -- session3 첫 캡처 pose로 근사, dry-run 미리보기 전용] "
                   "CAM_POSE_JOINTS는 반영됨, CAM_POSE_POSE는 아직 실측 필요")
    items = compute_ordered_targets(Path(args.session2_dir))

    print(f"고정값: z={Z_FIXED}mm  ry={RY_FIXED}deg  rx={RX_FIXED}deg  approach={args.approach_mm}mm")
    print(f"시작 자세 <- {start_src}")
    print(f"촬영 파킹 자세 <- {cam_src}\n")

    steps = build_plan(items, start_pose, start_joints, cam_pose, cam_pose_joints,
                        args.approach_mm, args.return_home,
                        move_speed=args.move_speed, descend_speed=args.descend_speed,
                        jnt_speed=args.jnt_speed)
    print_plan(steps)

    if not args.execute:
        print(f"\n(dry-run) 총 {len(steps)}스텝 계획. 실제로 움직이지 않았습니다. --execute 를 주면 실행합니다.")
        return

    labels = load_camera_labels(Path(args.device_map))
    cams, used_labels = connect_cameras(labels, no_reset=args.no_cam_reset)
    view = None
    if not args.no_preview:
        view = LiveView(cams, used_labels, window_name="session2_pick_and_place (q/ESC=닫기)")
        view.start()

    rb = ZeusClient(args.robot_ip, args.robot_port, timeout=args.robot_timeout)
    rb.connect()
    print("\n*** 실제 로봇이 자동으로 움직이고 그리퍼를 조작합니다. "
          "GELLO 텔레옵은 완전히 종료된 상태여야 합니다. 비상정지에 손이 닿는지 확인하세요. ***")
    if args.skip_steps:
        print(f"처음 {args.skip_steps}스텝은 이미 실행된 것으로 보고 건너뜁니다.")
    if input("계속하려면 'go' 입력: ").strip().lower() != "go":
        print("취소했습니다.")
        rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)
        return

    out_root = Path(args.out_root)
    try:
        execute_plan(steps, rb, cams, used_labels, out_root, view, args.no_step, args.skip_steps)
    finally:
        rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)

    print(f"\n완료 -- {out_root} 확인해보세요.")


if __name__ == "__main__":
    main()
