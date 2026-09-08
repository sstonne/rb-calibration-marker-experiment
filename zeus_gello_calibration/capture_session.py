#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/capture_session.py -- GELLO로 Zeus를 움직이며 실시간 촬영+로봇 pose 저장

UR3 때는 "포즈만 먼저 따고(capture_poses.py) 나중에 따로 촬영(capture_run.py)"
2단계였는데, 이번엔 요청대로 GELLO로 옮기는 그 순간 로봇 pose/joint와 카메라
4대(고정 3 + 그리퍼 1) 사진을 **한 번에** 저장한다.

전제: 로봇 움직임은 이 스크립트가 하지 않는다. 사람이 별도 터미널에서
GELLO 텔레옵(jello/gello_zeus_real_teleop.py)을 이미 돌려서 로봇을 조종하고
있고, 이 스크립트는 같은 로봇 서버(server/zeus_gello.py, port 12350)에
**읽기 전용**으로 별도 접속해서 get_state()만 폴링한다 -- zeus_gello.py는
get_state/ping을 움직임 소유권과 무관하게 아무 연결에서나 받아주도록 만든
서버라 이게 가능하다 (motion 명령은 이 스크립트에서 전혀 보내지 않음).

세션 구분은 UR3 쪽과 동일한 의미:
  1: handheld_fixed_cam       -- 그리퍼로 큐브를 들고 움직이며 고정캠 촬영
  2: floor_board_dual_cam     -- 큐브를 바닥 여러 곳에 두고 고정캠+그리퍼캠
  3: wrist_motion_gripper_cam -- 손목만 움직이며 그리퍼캠으로 바닥 보드 촬영

사용법:
  # 먼저 다른 터미널에서 GELLO 텔레옵을 켜서 로봇을 조종할 수 있는 상태로 둔다.
  python capture_session.py --session 1
  python capture_session.py --session 2 --num-poses 15
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_pipeline.camera import RealSenseCamera  # noqa: E402
from robot.backends.zeus_client import ZeusClient  # noqa: E402

ROBOT_IP_DEFAULT = "192.168.0.23"
ROBOT_PORT_DEFAULT = 12350
CAM_WIDTH, CAM_HEIGHT, CAM_FPS = 1280, 720, 15
DEVICE_MAP_DEFAULT = Path(__file__).resolve().parents[1] / "intrinsics" / "device_map.json"

SESSIONS = {
    1: {
        "name": "handheld_fixed_cam",
        "description": "그리퍼로 큐브를 들고 움직이며 고정 카메라들이 촬영 (eye-to-hand)",
    },
    2: {
        "name": "floor_board_dual_cam",
        "description": "큐브를 바닥에 두고(여러 위치) 고정 카메라 + 그리퍼 카메라 촬영",
    },
    3: {
        "name": "wrist_motion_gripper_cam",
        "description": "손목만 움직이며 바닥 마커보드를 그리퍼 카메라로 촬영 (eye-in-hand)",
    },
}


def load_camera_labels(device_map_path: Path) -> dict:
    """serial -> camN -> label(fixed1/2/3, gripper). Zeus는 device_map.json에
    이미 gripper_cam_idx가 정확히 박혀있어서(UR3와 달리 손볼 필요 없음),
    그걸 그대로 쓴다."""
    dm = json.loads(device_map_path.read_text())
    serial_to_idx = {s: int(i) for s, i in dm["serial_to_idx"].items()}
    gripper_idx = int(dm["gripper_cam_idx"])
    fixed_serials = sorted(
        (s for s, i in serial_to_idx.items() if i != gripper_idx),
        key=lambda s: serial_to_idx[s],
    )
    labels = {}
    for n, serial in enumerate(fixed_serials, start=1):
        labels[serial] = f"fixed{n}"
    for s, i in serial_to_idx.items():
        if i == gripper_idx:
            labels[s] = "gripper"
    return labels


def connect_cameras(labels: dict, no_reset: bool = False):
    if not no_reset:
        RealSenseCamera.reset_all_devices()
    devices = RealSenseCamera.list_devices()
    if not devices:
        raise RuntimeError("연결된 RealSense 카메라가 없습니다.")

    known_serials = set(labels.keys())
    detected_serials = set(devices.keys())
    unknown = detected_serials - known_serials
    missing = known_serials - detected_serials
    if unknown:
        print(f"[WARN] intrinsics/device_map.json에 없는(=intrinsics 없는) 카메라가 연결되어 있습니다: "
              f"{sorted(unknown)} -- 이 카메라로 찍은 데이터는 intrinsics를 못 찾을 수 있습니다.")
    if missing:
        print(f"[WARN] device_map.json에 등록된 카메라가 지금 안 보입니다(연결 확인 필요): {sorted(missing)}")

    print(f"카메라 {len(devices)}대 발견:")
    cams = {}
    used_labels = {}
    for serial, name in devices.items():
        label = labels.get(serial, serial)
        used_labels[serial] = label
        print(f"  {serial}  {name}  -> {label}")
        cam = RealSenseCamera(serial, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS,
                               use_color=True, use_depth=True, align_depth_to_color=True,
                               lock_color_exposure=False)
        cam.start()
        cams[serial] = cam
    return cams, used_labels


def stop_cameras(cams):
    for cam in cams.values():
        cam.stop()


class LiveView:
    """4대 카메라를 2x2로 붙인 미리보기. OpenCV(Qt5 백엔드)는 창을 메인 스레드가
    아닌 곳에서 다루면 검은 화면만 뜨는 문제가 있어(ur3_calibration/capture_run.py
    에서 실측 확인함), 백그라운드 스레드 대신 메인 루프에서 매 반복마다
    show()를 직접 호출하는 방식으로 만들었다."""

    TILE_W, TILE_H = 640, 360

    def __init__(self, cams: dict, labels: dict, window_name: str = "zeus_gello_calibration (q/ESC=닫기)"):
        self.cams = cams
        self.labels = labels
        self.window_name = window_name
        self.order = sorted(self.cams.keys(), key=lambda s: self.labels.get(s, s))
        self._closed = False

    def start(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.show()

    def show(self, wait_ms: int = 30, status: str = ""):
        """한 프레임 갱신하고 눌린 키를 돌려준다 (없으면 -1).

        wait_ms만큼 cv2.waitKey로 대기하면서 그 사이 계속 이 함수를 반복
        호출해야 실시간으로 보인다 -- input()으로 터미널에서 블로킹하며
        기다리면 그동안 화면이 멈춘 것처럼 보인다(예전 버그)."""
        if self._closed:
            return -1
        tiles = []
        for serial in self.order:
            color, _depth, _ts = self.cams[serial].get_latest()
            label = self.labels.get(serial, serial)
            if color is None:
                tile = np.zeros((self.TILE_H, self.TILE_W, 3), dtype=np.uint8)
            else:
                tile = cv2.resize(color, (self.TILE_W, self.TILE_H))
            cv2.putText(tile, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            tiles.append(tile)
        while len(tiles) < 4:
            tiles.append(np.zeros((self.TILE_H, self.TILE_W, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(tiles[0:2]), np.hstack(tiles[2:4])])
        if status:
            cv2.putText(grid, status, (10, grid.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow(self.window_name, grid)
        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord("q"), 27):
            self.stop()
            return key
        return key if key != 255 else -1

    def stop(self):
        if self._closed:
            return
        self._closed = True
        cv2.destroyWindow(self.window_name)


# PNG 압축 레벨(0~9, 낮을수록 빠르고 파일은 큼) -- 기본값(보통 1~3)도 나쁘진
# 않지만, 8장(컬러4+depth4)을 SPACE 누른 직후 즉시 써야 해서 최대한 빠르게.
PNG_FAST = [cv2.IMWRITE_PNG_COMPRESSION, 1]


def grab_frames(cams: dict, labels: dict) -> dict:
    """카메라 버퍼에서 최신 프레임을 메모리로만 즉시 복사해온다 (디스크 I/O 없음,
    거의 즉시 끝남) -- SPACE를 누른 그 순간의 프레임을 확정해두기 위함."""
    frames = {}
    for serial, cam in cams.items():
        color, depth, _ts = cam.get_latest()
        label = labels.get(serial, serial)
        frames[label] = (color, depth)
    return frames


def write_capture(frames: dict, out_dir: Path, robot_state: dict) -> None:
    """실제 디스크 쓰기 -- 느릴 수 있는 부분이라 백그라운드 스레드에서 돌려서
    메인 루프(미리보기 갱신)가 멈추지 않게 한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for label, (color, depth) in frames.items():
        if color is None:
            print(f"  [WARN] {label}: 프레임 없음, 건너뜀")
            continue
        cv2.imwrite(str(out_dir / f"cam_{label}.png"), color, PNG_FAST)
        if depth is not None:
            cv2.imwrite(str(out_dir / f"cam_{label}_depth.png"), depth, PNG_FAST)
        saved.append(label)
    (out_dir / "robot.json").write_text(json.dumps(robot_state, indent=2, ensure_ascii=False) + "\n")
    print(f"  저장됨 -> {out_dir}  (카메라 {len(saved)}대: {', '.join(saved)})")


def read_robot_state(rb: ZeusClient, extra=None) -> dict:
    state = rb.get_state()
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pose_convention": "pose=[x,y,z,rz,ry,rx] mm/deg (i611); joints deg",
        "pose": state["pose"],
        "joints": state["joints"],
        "gripper": state["gripper"],
    }
    if extra:
        payload.update(extra)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", type=int, required=True, choices=sorted(SESSIONS))
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    ap.add_argument("--robot-port", type=int, default=ROBOT_PORT_DEFAULT)
    ap.add_argument("--device-map", default=str(DEVICE_MAP_DEFAULT))
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--num-poses", type=int, default=15, help="목표 촬영 개수 (기본 15)")
    ap.add_argument("--reset", action="store_true", help="기존 세션 폴더 있어도 처음부터(0번)")
    ap.add_argument("--no-cam-reset", action="store_true", help="카메라 시작 전 하드웨어 리셋 생략")
    ap.add_argument("--no-preview", action="store_true", help="4대 미리보기 창을 띄우지 않음")
    args = ap.parse_args()

    info = SESSIONS[args.session]
    session_dir = Path(args.out_root) / f"session{args.session}_{info['name']}"
    capture_root = session_dir / "capture"

    start_index = 0
    if capture_root.exists() and not args.reset:
        existing = sorted(p.name for p in capture_root.iterdir() if p.is_dir())
        if existing:
            start_index = int(existing[-1]) + 1
            print(f"기존 {start_index}개 발견, 이어서 {start_index}번부터 저장합니다 (--reset 으로 초기화 가능).")

    print(f"=== 세션 {args.session}: {info['name']} ===")
    print(info["description"])
    print(f"목표 {args.num_poses}개, 로봇: {args.robot_ip}:{args.robot_port}")
    print("다른 터미널에서 GELLO 텔레옵이 이미 돌고 있어야 합니다 (이 스크립트는 로봇을 움직이지 않음).\n")

    labels = load_camera_labels(Path(args.device_map))
    cams, used_labels = connect_cameras(labels, no_reset=args.no_cam_reset)
    view = None
    if not args.no_preview:
        view = LiveView(cams, used_labels)
        view.start()

    rb = ZeusClient(args.robot_ip, args.robot_port)
    rb.connect()
    print("로봇 서버 연결됨 (읽기 전용, get_state만 사용).")
    if view is not None:
        print("GELLO로 원하는 자세로 옮긴 뒤, 미리보기 창에서: SPACE=촬영 / q 또는 ESC=조기 종료\n")
    else:
        print("GELLO로 원하는 자세로 옮긴 뒤: Enter=촬영 / q=조기 종료 (미리보기 꺼짐)\n")

    count = start_index
    try:
        while count < start_index + args.num_poses:
            status = f"[{count - start_index}/{args.num_poses}] SPACE=촬영  q/ESC=종료"
            if view is not None:
                # wait_ms만큼씩 계속 새로 그려야 실시간으로 보인다 (한 번만 그리고
                # 터미널 input()으로 블로킹하면 그 사이 화면이 멈춘 것처럼 보임).
                key = view.show(wait_ms=30, status=status)
                if view._closed:
                    print("미리보기 창에서 종료했습니다.")
                    break
                if key != 32:  # SPACE 아니면 계속 루프 돌며 화면만 갱신
                    continue
            else:
                cmd = input(f"[{count - start_index}/{args.num_poses}] > ").strip().lower()
                if cmd == "q":
                    break

            robot_state = read_robot_state(rb, {"capture_index": count})
            frames = grab_frames(cams, used_labels)
            out_dir = capture_root / f"{count:03d}"
            write_capture(frames, out_dir, robot_state)
            print(f"  pose_mm_deg={[round(v, 1) for v in robot_state['pose']]}")
            count += 1
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)

    print(f"\n완료 -- {capture_root} 에 {count - start_index}개 저장됨 (총 {count}개)")
    if count - start_index < args.num_poses:
        print(f"(목표 {args.num_poses}개 중 {count - start_index}개만 저장됨 -- 이어서 하려면 같은 명령 다시 실행)")


if __name__ == "__main__":
    main()
