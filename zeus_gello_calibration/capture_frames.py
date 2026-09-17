#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/capture_frames.py -- 카메라 4대 RGB + depth 를 N장 찍어 저장.
(외부 GT 촬영용. 로봇은 움직이지 않고, 연결되어 있으면 촬영 순간 pose만 같이 기록.)

저장: zeus_gello_calibration/data/gt_frames_<MMDD>/<timestamp>/<NNN>/cam_<label>.png, cam_<label>_depth.png, robot.json
(write_capture 가 data/ 안만 허용하므로 --out-root 도 data/ 안이어야 함)

사용법:
  python capture_frames.py --shots 5                    # Enter 누를 때마다 1장, 5장
  python capture_frames.py --shots 5 --no-robot         # 로봇 서버 없이 사진만
  python capture_frames.py --shots 5 --no-cam-reset     # USB 리셋 생략 (리셋이 카메라를 떨어뜨릴 때)
  python capture_frames.py --shots 5 --width 1920 --height 1080 --depth-width 1280 --depth-height 720 \\
      --device-map ../intrinsics_1920x1080_rgbd720/device_map.json   # 해상도 지정 (폴더명에 자동으로 붙음)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from robot.backends.zeus_client import ZeusClient  # noqa: E402

from capture_session import (  # noqa: E402
    DEVICE_MAP_DEFAULT, ROBOT_IP_DEFAULT, ROBOT_PORT_DEFAULT, LiveView,
    connect_cameras, grab_frames, load_camera_labels, read_robot_state, stop_cameras, write_capture,
    validate_intrinsics_stream, CAM_WIDTH, CAM_HEIGHT, CAM_FPS,
)

from paths import ZEUS_DATA_ROOT  # noqa: E402

# write_capture 가 zeus_gello_calibration/data/ 안만 허용 -- 촬영 데이터 규칙(<이름>_<MMDD>)대로
OUT_ROOT_DEFAULT = ZEUS_DATA_ROOT / f"gt_frames_{time.strftime('%m%d')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--out-root", default=str(OUT_ROOT_DEFAULT))
    ap.add_argument("--device-map", default=str(DEVICE_MAP_DEFAULT))
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    ap.add_argument("--robot-port", type=int, default=ROBOT_PORT_DEFAULT)
    ap.add_argument("--robot", action="store_true", help="로봇 서버에 읽기 전용 접속해서 촬영 순간 pose를 같이 기록 (기본: 접속 안 함)")
    ap.add_argument("--no-cam-reset", action="store_true")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--width", type=int, default=CAM_WIDTH, help="RealSense color/depth width")
    ap.add_argument("--height", type=int, default=CAM_HEIGHT, help="RealSense color/depth height")
    ap.add_argument("--depth-width", type=int, default=None, help="RealSense depth width")
    ap.add_argument("--depth-height", type=int, default=None, help="RealSense depth height")
    ap.add_argument("--fps", type=int, default=CAM_FPS, help="RealSense stream FPS")
    args = ap.parse_args()

    if (args.depth_width is None) != (args.depth_height is None):
        ap.error("--depth-width와 --depth-height는 함께 지정해야 합니다.")
    if (
        args.width <= 0 or args.height <= 0 or args.fps <= 0
        or (args.depth_width is not None and args.depth_width <= 0)
        or (args.depth_height is not None and args.depth_height <= 0)
    ):
        ap.error("--width, --height, --depth-width, --depth-height, --fps 값은 모두 양수여야 합니다.")
    depth_width = int(args.depth_width) if args.depth_width is not None else int(args.width)
    depth_height = int(args.depth_height) if args.depth_height is not None else int(args.height)

    # 해상도별로 여러 번 찍을 때 폴더가 안 겹치도록 자동으로 붙인다.
    res_suffix = f"_{args.width}x{args.height}"
    if not args.out_root.endswith(res_suffix):
        args.out_root = args.out_root + res_suffix

    out_dir = Path(args.out_root) / time.strftime("%Y%m%d_%H%M%S")
    labels = load_camera_labels(Path(args.device_map))
    validate_intrinsics_stream(
        Path(args.device_map), args.width, args.height, args.fps,
        depth_width=depth_width, depth_height=depth_height,
    )
    cams, used_labels = connect_cameras(
        labels, no_reset=args.no_cam_reset,
        width=args.width, height=args.height, fps=args.fps,
        depth_width=depth_width, depth_height=depth_height,
    )
    view = None
    if not args.no_preview:
        view = LiveView(cams, used_labels, window_name="capture_frames (q/ESC=닫기)")
        view.start()

    rb = None
    if args.robot:
        try:
            rb = ZeusClient(args.robot_ip, args.robot_port)
            rb.connect()
            print("로봇 서버 연결됨 (읽기 전용, pose 기록용).")
        except Exception as exc:
            print(f"[WARN] 로봇 서버 연결 실패, pose 없이 진행: {exc}")
            rb = None

    if view is not None:
        print(f"저장 폴더: {out_dir}\n{args.shots}장 촬영. **미리보기 창을 클릭한 뒤** SPACE 또는 Enter=촬영 / q 또는 ESC=종료\n")
    else:
        print(f"저장 폴더: {out_dir}\n{args.shots}장 촬영. 터미널에서 Enter=촬영 / q=종료\n")
    n = 0
    try:
        while n < args.shots:
            if view is not None:
                # 미리보기 갱신하면서 키 대기 -- 키 입력은 OpenCV 창에 포커스가 있을 때만 잡힘
                key = view.show(wait_ms=30, status=f"[{n}/{args.shots}] SPACE/Enter=촬영  q/ESC=종료 (창 클릭 후)")
                if view._closed:
                    break
                if key not in (32, 13, 10):
                    continue
            else:
                cmd = input(f"[{n}/{args.shots}] > ").strip().lower()
                if cmd == "q":
                    break
            if rb is not None:
                robot_state = read_robot_state(rb, {"capture_index": n, "note": "capture_frames: 로봇 이동 없음"})
            else:
                robot_state = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "pose": None, "joints": None,
                               "capture_index": n, "note": "capture_frames: 로봇 미접속"}
            frames = grab_frames(cams, used_labels)
            write_capture(frames, out_dir / f"{n:03d}", robot_state)
            if robot_state.get("pose"):
                print(f"  pose_mm_deg={[round(v, 2) for v in robot_state['pose']]}")
            n += 1
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if rb is not None:
            rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)
    print(f"\n완료 -- {out_dir} 에 {n}장 저장됨")


if __name__ == "__main__":
    main()
