#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/gt_eval_offline.py -- gt_compare_fits.py가 저장해둔
외부 GT 촬영(gt_compare_captures/<timestamp>/)을 다시 읽어서, 임의의 fit
JSON 목록에 대해 로봇/카메라 없이 오프라인으로 외부 GT 오차를 계산한다.

새 캘리브레이션 변형(corrected-FK 등)을 만들 때마다 실물 실험을 다시 하지
않고 같은 사진으로 비교하기 위한 것. 오차 정의는 METHODS_ANALYSIS.md "결과 2"와
같다: x,y는 offset 없이, z는 `검출 z − (GT flange z − 그 fit의 T_gripper_cube z)`,
rz는 offset 없이 직접 비교.

사용법:
  # 어느 폴더가 어느 트라이얼인지 모를 때: fit 하나로 검출 xy만 출력
  python gt_eval_offline.py --list --fits fit_통합_no-fk.json

  # 트라이얼 = 폴더:GT flange pose6 (x,y,z,rz,ry,rx)
  python gt_eval_offline.py \
    --trial 20260909_163046:-293.015,399.985,178.71,-90.008,-0.012,179.999 \
    --trial 20260909_163921:-343.00,450.00,178.71,-66.01,-0.01,180.00 \
    --trial 20260909_164503:-193.03,339.97,178.71,-126.00,-0.01,-180.00 \
    --fits fit_통합_no-fk.json fit_통합_raw-fk.json fit_독립_no-fk.json fit_cfk_*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from robot.backends.zeus_client import T_to_pose6, pose6_to_T  # noqa: E402

from fit_grasp_offset import load_intrinsics_by_label  # noqa: E402
from gt_compare_fits import detect_cube_pose_all  # noqa: E402
from gt_pick_test import GT_CUBE_CONFIG_PATH, load_fit  # noqa: E402

CAPTURE_ROOT = REPO_ROOT / "zeus_gello_calibration" / "gt_compare_captures"
LABELS = ("039422061216", "fixed2", "fixed3", "gripper")


def load_capture(folder: Path, assumed_flange_pose=None):
    frames = {}
    for label in LABELS:
        p = folder / f"cam_{label}.png"
        frames[label] = (cv2.imread(str(p)) if p.is_file() else None, None)
    robot = json.loads((folder / "robot.json").read_text())
    pose = robot.get("pose")
    if pose is None:
        if assumed_flange_pose is not None:
            # 로봇 미접속 촬영이지만 촬영 자세를 아는 경우(--assume-flange-dz):
            # GT flange pose 에서 z 만 dz 올린 자세로 그리퍼캠을 base 로 옮긴다.
            return frames, pose6_to_T(list(assumed_flange_pose))
        # 로봇 미접속 촬영(capture_frames.py --robot 없이): 그리퍼캠은 base 좌표로
        # 못 옮기므로 제외하고 고정캠만 쓴다.
        frames["gripper"] = (None, None)
        return frames, np.eye(4)
    return frames, pose6_to_T(pose)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fits", nargs="+", required=True, help="fit JSON 경로/glob (zeus_gello_calibration 기준)")
    ap.add_argument("--capture-root", default=str(CAPTURE_ROOT),
                    help="트라이얼 폴더들의 부모 (기본 gt_compare_captures/; capture_frames.py 결과는 data/gt_frames_<MMDD>/<timestamp>)")
    ap.add_argument("--trial", action="append", default=[], help="<capture_folder>:<x,y,z,rz,ry,rx GT flange pose>")
    ap.add_argument("--list", action="store_true", help="폴더별 검출 xy만 출력 (트라이얼 식별용)")
    ap.add_argument("--per-camera", action="store_true", help="카메라별(단일 PnP) 오차를 따로 출력")
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--gt-cube-config", default=str(GT_CUBE_CONFIG_PATH))
    ap.add_argument("--reproj-thr-px", type=float, default=10.0)
    ap.add_argument("--assume-flange-dz", type=float, default=None,
                    help="robot.json 에 pose 가 없을 때, 촬영 자세 = GT flange pose 의 z 에 이 값(mm)을 더한 것으로 가정 (그리퍼캠 사용)")
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "results" / "gt_eval" / "gt_eval_offline.json"))
    args = ap.parse_args()

    base = REPO_ROOT / "zeus_gello_calibration"
    fit_paths = []
    for pat in args.fits:
        matches = sorted(glob.glob(str(base / pat))) or sorted(glob.glob(pat))
        fit_paths.extend(matches or [str(base / pat)])
    fits = {}
    for p in fit_paths:
        p = Path(p)
        label = p.stem.replace("fit_", "")
        fits[label] = load_fit(p)

    K_map, D_map = load_intrinsics_by_label(Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))
    cube_cfg, _ = load_cube_config_from_json_file(args.gt_cube_config)
    cube_target = AprilTagCubeTarget(cube_cfg)

    if args.list:
        label0, (Tgc, T_base_cam, T_gripper_cam) = next(iter(fits.items()))
        print(f"[{label0}] 폴더별 검출 큐브 pose (x, y, z, rz):")
        for folder in sorted(Path(args.capture_root).iterdir()):
            if not folder.is_dir():
                continue
            frames, T_bg = load_capture(folder)
            T, report = detect_cube_pose_all(frames, K_map, D_map, T_base_cam, T_gripper_cam, T_bg, cube_target, args.reproj_thr_px)
            if T is None:
                print(f"  {folder.name}: 검출 실패")
                continue
            p6 = T_to_pose6(T)
            n_ok = sum(1 for v in report.values() if isinstance(v, str) and v.startswith("검출됨"))
            print(f"  {folder.name}: x={p6[0]:8.2f} y={p6[1]:8.2f} z={p6[2]:7.2f} rz={p6[3]:8.2f}  (cams={n_ok})")
        return

    trials = []
    for t in args.trial:
        folder, pose = t.split(":")
        trials.append((folder, [float(v) for v in pose.split(",")]))
    if not trials:
        ap.error("--trial 최소 1개 필요 (또는 --list)")

    def assumed(gt):
        if args.assume_flange_dz is None:
            return None
        return [gt[0], gt[1], gt[2] + args.assume_flange_dz, gt[3], gt[4], gt[5]]

    if args.per_camera:
        from calibration_pipeline.apriltag_cube import rodrigues_to_Rt
        from fit_grasp_offset import LOCAL_CAM_IDS
        per_cam = {}
        print(f"{'fit':>14} | {'camera':>13} | {'trial':>16} | {'dx':>6} {'dy':>6} {'dz':>6} {'drz':>6} | {'xyz':>6}")
        for label, (T_gripper_cube, T_base_cam, T_gripper_cam) in fits.items():
            tgc_z = float(np.asarray(T_gripper_cube)[2, 3] * 1000.0)
            for cam_label in LABELS:
                rows = []
                for folder, gt in trials:
                    frames, T_bg = load_capture(Path(args.capture_root) / folder, assumed(gt))
                    img = frames.get(cam_label, (None, None))[0]
                    if img is None:
                        continue
                    lid = LOCAL_CAM_IDS[cam_label]
                    ok, rvec, tvec, used, _ = cube_target.solve_pnp_cube(
                        img, K_map[lid], D_map[lid], reproj_thr_mean_px=args.reproj_thr_px, return_reproj=True)
                    if not ok:
                        print(f"{label:>14} | {cam_label:>13} | {folder:>16} | 검출 실패 (markers={used})")
                        continue
                    T_cam_cube = rodrigues_to_Rt(rvec, tvec)
                    if cam_label == "gripper":
                        if T_gripper_cam is None:
                            continue
                        T = T_bg @ T_gripper_cam @ T_cam_cube
                    else:
                        if lid not in T_base_cam:
                            continue
                        T = T_base_cam[lid] @ T_cam_cube
                    e = T_to_pose6(T)
                    dx, dy = e[0] - gt[0], e[1] - gt[1]
                    dz = e[2] - (gt[2] - tgc_z)
                    drz = ((e[3] - gt[3]) + 180.0) % 360.0 - 180.0
                    xyz = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                    rows.append({"trial": folder, "dx": dx, "dy": dy, "dz": dz, "drz": drz, "xyz": xyz})
                    print(f"{label:>14} | {cam_label:>13} | {folder:>16} | {dx:>6.2f} {dy:>6.2f} {dz:>6.2f} {drz:>6.2f} | {xyz:>6.2f}")
                if rows:
                    per_cam[(label, cam_label)] = rows
        print(f"\n{'fit':>14} | {'camera':>13} | {'|dx|':>6} {'|dy|':>6} {'|dz|':>6} {'xyz TRE':>8} {'|drz|':>6} | n")
        for (label, cam_label), rows in per_cam.items():
            a = {k: float(np.mean([abs(r[k]) for r in rows])) for k in ("dx", "dy", "dz", "drz")}
            xyz = float(np.mean([r["xyz"] for r in rows]))
            print(f"{label:>14} | {cam_label:>13} | {a['dx']:>6.2f} {a['dy']:>6.2f} {a['dz']:>6.2f} {xyz:>8.2f} {a['drz']:>6.2f} | {len(rows)}")
        Path(args.out).with_name(Path(args.out).stem + "_per_camera.json").write_text(
            json.dumps({f"{l}/{c}": r for (l, c), r in per_cam.items()}, indent=2))
        return

    results = {}
    print(f"{'fit':>22} | {'trial':>16} | {'dx':>6} {'dy':>6} {'dz':>6} {'drz':>6} | {'xyz':>6}")
    for label, (T_gripper_cube, T_base_cam, T_gripper_cam) in fits.items():
        tgc_z = float(np.asarray(T_gripper_cube)[2, 3] * 1000.0)
        rows = []
        for folder, gt in trials:
            frames, T_bg = load_capture(Path(args.capture_root) / folder, assumed(gt))
            T, report = detect_cube_pose_all(frames, K_map, D_map, T_base_cam, T_gripper_cam, T_bg, cube_target, args.reproj_thr_px)
            if T is None:
                print(f"{label:>22} | {folder:>16} | 검출 실패")
                continue
            e = T_to_pose6(T)
            dx, dy = e[0] - gt[0], e[1] - gt[1]
            dz = e[2] - (gt[2] - tgc_z)
            drz = ((e[3] - gt[3]) + 180.0) % 360.0 - 180.0
            xyz = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            rows.append({"trial": folder, "dx": dx, "dy": dy, "dz": dz, "drz": drz, "xyz": xyz,
                         "n_cams": sum(1 for v in report.values() if isinstance(v, str) and v.startswith("검출됨"))})
            print(f"{label:>22} | {folder:>16} | {dx:>6.2f} {dy:>6.2f} {dz:>6.2f} {drz:>6.2f} | {xyz:>6.2f}")
        if rows:
            agg = {k: float(np.mean([abs(r[k]) for r in rows])) for k in ("dx", "dy", "dz", "drz")}
            agg["xyz"] = float(np.mean([r["xyz"] for r in rows]))
            results[label] = {"trials": rows, "mean_abs": agg}

    print(f"\n{'fit':>22} | {'|dx|':>6} {'|dy|':>6} {'|dz|':>6} {'xyz TRE':>8} {'|drz|':>6}")
    for label, r in sorted(results.items(), key=lambda kv: kv[1]["mean_abs"]["dz"]):
        a = r["mean_abs"]
        print(f"{label:>22} | {a['dx']:>6.2f} {a['dy']:>6.2f} {a['dz']:>6.2f} {a['xyz']:>8.2f} {a['drz']:>6.2f}")
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
