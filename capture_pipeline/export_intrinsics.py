"""
연결된 모든 RealSense 카메라의 factory intrinsics를 저장한다.
 - Saves per-camera npz (K, D, depth_scale, etc.)
 - Saves device_map.json with serial -> cam_idx mapping
 * 그리퍼 카메라는 여기서 지정하지 않는다. 기존 device_map.json의 gripper 값은 그대로 유지한다.

명령어:
python3 01_export_intrinsics.py \
--out_dir ./intrinsics \
--color_w 1920 \
--color_h 1080 \
--depth_w 1280 \
--depth_h 720 \
--fps 15

결과물:
  intrinsics/
    device_map.json
    cam0.npz, cam1.npz, cam2.npz, cam3.npz, cam4.npz
    depth_scales.json
    intrinsics_by_serial/
"""

import os
import json
import time
import argparse
import numpy as np
# ****** K,D = intrinsics -> camera matrix, distortion coeffs
def _intr_to_KD(intr):
    fx, fy = float(intr.fx), float(intr.fy)
    cx, cy = float(intr.ppx), float(intr.ppy)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64).reshape(-1, 1)
    return K, D

def main():
    parser = argparse.ArgumentParser(description="Dump intrinsics for all RealSense cameras")
    parser.add_argument("--out_dir", type=str, default="intrinsics")
    # 프로젝트 표준은 color/depth 모두 1280x720@15. 더 높은 color 해상도를 쓰려면
    # depth는 D4xx RGB-D에서 지원되는 해상도(예: 1280x720)에 남겨둔 채 align한다.
    parser.add_argument("--color_w", type=int, default=1280)
    parser.add_argument("--color_h", type=int, default=720)
    parser.add_argument("--depth_w", type=int, default=None)
    parser.add_argument("--depth_h", type=int, default=None)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    if (args.depth_w is None) != (args.depth_h is None):
        parser.error("--depth_w and --depth_h must be supplied together")
    depth_w = int(args.depth_w) if args.depth_w is not None else int(args.color_w)
    depth_h = int(args.depth_h) if args.depth_h is not None else int(args.color_h)
    if min(int(args.color_w), int(args.color_h), depth_w, depth_h, int(args.fps)) <= 0:
        parser.error("--color_w, --color_h, --depth_w, --depth_h, --fps must be positive")

    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as error:
        raise SystemExit(
            "[ERROR] pyrealsense2가 없습니다. RealSense Python 환경에서 "
            "01번을 실행하세요.") from error

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    by_serial_dir = os.path.join(out_dir, "intrinsics_by_serial")
    os.makedirs(by_serial_dir, exist_ok=True)

    map_path = os.path.join(out_dir, "device_map.json")
    scales_path = os.path.join(out_dir, "depth_scales.json")

    # ****** 장치 검색 및 매핑
    ctx = rs.context()
    devs = ctx.query_devices()
    if len(devs) == 0:
        print("[ERROR] No RealSense devices found.")
        return

    detected = []
    for d in devs:
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name) if d.supports(rs.camera_info.name) else "Unknown"
        detected.append({"serial": serial, "name": name})

    detected_serials = [x["serial"] for x in detected]
    print(f"[INFO] Found {len(detected)} RealSense devices:")
    for x in detected:
        print(f"  serial={x['serial']}  name={x['name']}")

    # ****** device map 생성/업데이트 (serial -> cam_idx)
    existing_map = None
    if os.path.exists(map_path):
        with open(map_path, "r") as f:
            existing_map = json.load(f)

    if existing_map is not None:
        serial_to_idx = dict(existing_map.get("serial_to_idx", {}))
        next_idx = 0 if len(serial_to_idx) == 0 else (max(serial_to_idx.values()) + 1)
        for s in detected_serials:
            if s not in serial_to_idx:
                serial_to_idx[s] = next_idx
                next_idx += 1
        print("[INFO] Updated existing device_map.json")
    else:
        detected_serials_sorted = sorted(detected_serials)
        serial_to_idx = {s: i for i, s in enumerate(detected_serials_sorted)}
        print("[INFO] Created new device_map.json (sorted by serial)")

    # ****** 그리퍼카메라 인덱스: 지정하지 않고 기존 맵의 값만 유지
    gripper_serial = None
    gripper_cam_idx = None
    if existing_map is not None:
        gripper_cam_idx = existing_map.get("gripper_cam_idx")
        gripper_serial = existing_map.get("gripper_serial")
    if gripper_cam_idx is not None:
        gripper_cam_idx = int(gripper_cam_idx)

    map_obj = {
        "created_at_epoch": existing_map.get("created_at_epoch", time.time()) if existing_map else time.time(),
        "updated_at_epoch": time.time(),
        "serial_to_idx": serial_to_idx,
        "gripper_cam_idx": gripper_cam_idx,
        "gripper_serial": gripper_serial,
        "detected_now": detected,
    }
    with open(map_path, "w") as f:
        json.dump(map_obj, f, indent=2)
    print(f"[SAVE] {map_path}")

    # ****** 각 장치의 intrinsics 읽어서 저장
    # 이번에 연결되지 않은 카메라의 기존 depth scale은 유지한다 (카메라를 나눠 연결하는 경우).
    previous_scales = {}
    if os.path.exists(scales_path):
        with open(scales_path, "r") as f:
            previous_scales = json.load(f).get("serial_to_depth_scale_m_per_unit", {})
    depth_scales = {"updated_at_epoch": time.time(),
                    "serial_to_depth_scale_m_per_unit": dict(previous_scales)}
    idx_serial_pairs = sorted([(serial_to_idx[s], s) for s in detected_serials], key=lambda x: x[0])

    print(
        "\n[INFO] Stream profile: "
        f"color {int(args.color_w)}x{int(args.color_h)}@{int(args.fps)}, "
        f"depth {depth_w}x{depth_h}@{int(args.fps)}"
    )
    print("[INFO] Camera index assignment:")
    for idx, s in idx_serial_pairs:
        tag = " (GRIPPER)" if idx == gripper_cam_idx else " (FIXED)"
        print(f"  cam{idx}: serial={s}{tag}")

    for cam_idx, serial in idx_serial_pairs:
        dev = None
        for d in devs:
            if d.get_info(rs.camera_info.serial_number) == serial:
                dev = d
                break
        if dev is None:
            continue

        # Depth scale
        try:
            ds = float(dev.first_depth_sensor().get_depth_scale())
            depth_scales["serial_to_depth_scale_m_per_unit"][serial] = ds
        except Exception as e:
            print(f"[WARN] depth_scale read failed for {serial}: {e}")
            ds = None

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, args.color_w, args.color_h, rs.format.bgr8, args.fps)
        config.enable_stream(rs.stream.depth, depth_w, depth_h, rs.format.z16, args.fps)

        try:
            profile = pipeline.start(config)

            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            Kc, Dc = _intr_to_KD(color_stream.get_intrinsics())

            depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
            Kd, Dd = _intr_to_KD(depth_stream.get_intrinsics())

            try:
                extr = depth_stream.get_extrinsics_to(color_stream)
                R_dc = np.array(extr.rotation, dtype=np.float64).reshape(3, 3)
                t_dc = np.array(extr.translation, dtype=np.float64).reshape(3, 1)
            except Exception:
                R_dc = np.eye(3, dtype=np.float64)
                t_dc = np.zeros((3, 1), dtype=np.float64)

            is_gripper = (cam_idx == gripper_cam_idx)

            npz_path = os.path.join(out_dir, f"cam{cam_idx}.npz")
            np.savez(npz_path,
                     serial=serial,
                     is_gripper=is_gripper,
                     color_K=Kc, color_D=Dc,
                     depth_K=Kd, depth_D=Dd,
                     depth_scale_m_per_unit=(ds if ds is not None else np.nan),
                     color_w=args.color_w, color_h=args.color_h,
                     depth_w=depth_w, depth_h=depth_h, fps=args.fps,
                     R_depth_to_color=R_dc, t_depth_to_color=t_dc)
            print(f"[SAVE] {npz_path}")

            serial_npz = os.path.join(by_serial_dir, f"serial_{serial}.npz")
            np.savez(serial_npz, serial=serial, cam_idx=cam_idx,
                     color_K=Kc, color_D=Dc, depth_K=Kd, depth_D=Dd,
                     depth_scale_m_per_unit=(ds if ds is not None else np.nan),
                     color_w=args.color_w, color_h=args.color_h,
                     depth_w=depth_w, depth_h=depth_h, fps=args.fps)
            print(f"[SAVE] {serial_npz}")

            tag = "GRIPPER" if is_gripper else "FIXED"
            print(f"[INFO] cam{cam_idx} ({tag}) serial={serial}")
            print(f"       color K:\n{Kc}")
            print(f"       depth_scale = {ds}")
        finally:
            try:
                pipeline.stop()
            except Exception:
                pass

    with open(scales_path, "w") as f:
        json.dump(depth_scales, f, indent=2)
    print(f"\n[SAVE] {scales_path}")
    print("[DONE] 01_export_intrinsics.py complete.")


if __name__ == "__main__":
    main()
