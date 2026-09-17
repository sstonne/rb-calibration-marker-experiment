#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/fit_grasp_offset.py -- fit T_gripper_cube for Zeus.

Mirrors ur3_calibration/fit_grasp_offset.py's method (same solver-calling
pattern: cv2.calibrateRobotWorldHandEye per camera for an initializer, then
calibration_pipeline.reprojection.solve_corner_reprojection for the joint
bundle adjustment). What is genuinely different here, and why:

1. No meta.json for this dataset. Zeus's session1 capture
   (zeus_gello_calibration/data/session1_handheld_fixed_cam_0909/capture/<idx:03d>/)
   was written by the retired legacy collector as one folder per pose with
   cam_<label>.png files + robot.json -- there is no 05_calibrate.py-style
   meta.json describing captures/cams/set_index. calibration_pipeline's own
   cube detector (cube_detection.detect_corner_observations, reached via
   observations.load_cube_pixel_observations) is meta.json-shaped, so this
   script builds a small in-memory synthetic meta dict (never written to
   disk, never touching the real data/ folder) that satisfies exactly the
   fields that loader reads: captures[].event_id/cube_gripped/grasp_id/
   cams[cam_idx].saved/.rgb_path. This is the intended "adapt only the
   data-loading part" move, not a reimplementation of the detector.

2. Camera identity/intrinsics needs two sources. Session1 has 4 camera
   labels: "039422061216", "fixed2", "fixed3", "gripper". Three of those
   ("fixed2"/"fixed3"/"gripper") are Zeus's own RealSense units, whose serials
   and gripper-camera role are recorded in Zeus's own intrinsics/device_map.json
   (mirrors the legacy collector's camera-label derivation:
   sort non-gripper serials by serial_to_idx to get fixed1/2/3, remaining
   serial is "gripper"). The fourth label, "039422061216", is NOT one of
   Zeus's 4 registered serials -- already root-caused (see task instructions
   this script was written against): it is the same physical RealSense unit
   used on the UR3 rig, temporarily swapped into this session, whose
   intrinsics live only at ur3_calibration/intrinsics/cam0.npz (that file's
   own "serial" field is checked against "039422061216" at runtime below, so
   a future re-run fails loudly if that assumption ever stops holding).

3. No board / eye-in-hand modeling at all. Unlike the UR3 script (which also
   had session2 board views to anchor the gripper camera's eye-in-hand
   extrinsic via estimate_board_handeye_initial), this dataset has ONLY cube
   observations from one session. There is therefore no data to distinguish
   "camera rigidly follows the gripper" from "camera has some fixed-but-
   unknown pose in the base frame" for any of the 4 cameras, including the
   one physically mounted near the gripper (labeled "gripper" here). This
   script follows the task's instructed model: treat all up to 4 cameras
   uniformly as UNKNOWN-BUT-CONSTANT extrinsics T_base_Ci (eye-to-hand form),
   solved jointly with the one constant T_gripper_cube via
       T_base_gripper[i] @ T_gripper_cube ~= T_base_Ci @ T_cam_cube[i]     (*)
   exactly the AX=ZB form cv2.calibrateRobotWorldHandEye solves (A=T_base_
   gripper, B=T_cam_cube, X=T_gripper_cube, Z=T_base_Ci), then refined by
   calibration_pipeline.reprojection.solve_corner_reprojection with variable
   keys T_base_Ci (all 4) + T_gripper_cube_by_grasp (one grasp, id 0). No
   T_gripper_cam / T_base_board variables exist in this solve; passing a
   gripper_cam_idx sentinel that matches none of the 4 local camera ids (-1)
   keeps every observation routed through the "fixed camera" residual path
   in reprojection.py's CornerReprojectionProblem, which is what (*) requires
   even for the "gripper"-labeled camera.

4. Zeus pose convention. T_base_gripper for each capture comes from
   robot.backends.zeus_client.pose6_to_T on that capture's robot.json["pose"]
   ([x,y,z,rz,ry,rx] mm/deg, extrinsic ZYX) -- NOT
   capture_pipeline/robot.py::euler_deg_to_matrix, which encodes a different
   (UR3/rotvec-family) convention and would silently give wrong results here.

Output: JSON with the fitted T_gripper_cube (matrix + mm/rotvec-deg), each
fitted T_base_Ci, training reprojection RMSE, and the cross-camera initial
dispersion sanity check (mirrors the UR3 report's <0.1mm/0.14deg number).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline import se3 as cp  # noqa: E402
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T  # noqa: E402
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.observations import load_cube_pixel_observations  # noqa: E402
from calibration_pipeline.path_evaluation import solve_observed_pose  # noqa: E402
from calibration_pipeline.reprojection import (  # noqa: E402
    PoseState, SolverOptions, solve_corner_reprojection, variable_keys,
)
from calibration_pipeline.runtime import load_intrinsics_with_depth_scale  # noqa: E402
from robot.backends.zeus_client import pose6_to_T  # noqa: E402

# Fixed local camera-id scheme used only inside this script (the pipeline
# just needs *some* small stable int per camera). Order/ids are arbitrary;
# what matters is that K_map/D_map/observations/robot_T all agree on them.
LOCAL_CAM_IDS = {"039422061216": 0, "fixed2": 1, "fixed3": 2, "gripper": 3}
BORROWED_UR3_LABEL = "039422061216"
NOT_GRIPPER_SENTINEL = -1  # no local cam id equals this -> all 4 cams treated
                           # as fixed eye-to-hand extrinsics (see docstring point 3)


def resolve_zeus_camera_serials(device_map_path: Path) -> dict:
    """label -> serial for Zeus's own 3 cameras (fixed2/fixed3/gripper).

    Reproduces the retired collector's label mapping (same sort-by-idx rule),
    so the labels recorded in this session's folders decode to the
    same serials/cameras that were physically connected when it was captured.
    """
    dm = json.loads(device_map_path.read_text())
    serial_to_idx = {s: int(i) for s, i in dm["serial_to_idx"].items()}
    gripper_idx = int(dm["gripper_cam_idx"])
    fixed_serials = sorted(
        (s for s, i in serial_to_idx.items() if i != gripper_idx),
        key=lambda s: serial_to_idx[s],
    )
    label_to_serial = {}
    for n, serial in enumerate(fixed_serials, start=1):
        label_to_serial[f"fixed{n}"] = serial
    for s, i in serial_to_idx.items():
        if i == gripper_idx:
            label_to_serial["gripper"] = s
    return label_to_serial, serial_to_idx


def find_intrinsics_override(zeus_intrinsics_dir: Path, serial: str):
    """`<zeus_intrinsics_dir>/overrides/<serial>*.npz` 가 있으면 그 파일(이름순 마지막 = 최신)을
    돌려준다. 카메라 하나만 다시 찍은 내부 파라미터를 모든 스크립트에 한 번에 적용하기 위한
    규약. 환경변수 ZEUS_INTRINSICS_NO_OVERRIDE=1 이면 무시."""
    if os.environ.get("ZEUS_INTRINSICS_NO_OVERRIDE"):
        return None
    cands = sorted((zeus_intrinsics_dir / "overrides").glob(f"{serial}*.npz"))
    return cands[-1] if cands else None


def load_intrinsics_npz_generic(path: Path):
    """K/D 또는 color_K/color_D 키를 가진 npz 에서 (K, D) 를 읽는다."""
    z = np.load(path, allow_pickle=True)
    kk = "K" if "K" in z else "color_K"
    dk = "D" if "D" in z else "color_D"
    return np.asarray(z[kk], dtype=np.float64).reshape(3, 3), np.asarray(z[dk], dtype=np.float64).reshape(-1)


def load_intrinsics_by_label(zeus_intrinsics_dir: Path, ur3_intrinsics_dir: Path,
                             device_map_path: Path) -> tuple:
    """Return (K_map, D_map) keyed by LOCAL_CAM_IDS, from the two directories.
    `<zeus_intrinsics_dir>/overrides/<serial>*.npz` 가 있으면 그 카메라는 그 값을 우선 쓴다.

    BORROWED_UR3_LABEL(039422061216)은 원래 Zeus 쪽에 그 유닛의 자체 calibration이
    없어서 UR3 rig의 charuco 결과를 빌려 쓰던 카메라다. 지금 device_map에 그 serial의
    자체 항목(serial_to_idx)이 있으면 -- 즉 이 zeus_intrinsics_dir가 그 카메라도 직접
    찍었으면 -- 해상도가 안 맞을 수 있는 UR3(항상 1280x720) 값 대신 그 자체 값을 쓴다.
    UR3 borrow는 device_map에 그 serial이 아예 없는(과거) 경우의 fallback으로만 남는다.
    """
    label_to_serial, serial_to_idx = resolve_zeus_camera_serials(device_map_path)
    K_map, D_map = {}, {}
    for label, local_id in LOCAL_CAM_IDS.items():
        serial_for_label = BORROWED_UR3_LABEL if label == BORROWED_UR3_LABEL else label_to_serial[label]
        override = find_intrinsics_override(Path(zeus_intrinsics_dir), serial_for_label)
        has_own_zeus_entry = serial_for_label in serial_to_idx
        if override is not None:
            K, D = load_intrinsics_npz_generic(override)
            print(f"[intrinsics] {label} (serial {serial_for_label}): override {override.name} 사용 (fx={K[0,0]:.1f})")
        elif label == BORROWED_UR3_LABEL and not has_own_zeus_entry:
            K, D, _ = load_intrinsics_with_depth_scale(str(ur3_intrinsics_dir), 0)
            npz = np.load(ur3_intrinsics_dir / "cam0.npz", allow_pickle=True)
            actual_serial = str(npz["serial"]) if "serial" in npz else None
            if actual_serial != BORROWED_UR3_LABEL:
                raise RuntimeError(
                    f"expected {ur3_intrinsics_dir/'cam0.npz'} serial "
                    f"{BORROWED_UR3_LABEL!r}, got {actual_serial!r} -- the "
                    "borrowed-camera assumption this script relies on no "
                    "longer holds")
            print(f"[intrinsics] {label} (serial {serial_for_label}): device_map에 자체 항목이 없어 "
                  f"UR3 rig 값을 빌려 씀 (fx={K[0,0]:.1f}, 해상도가 zeus_intrinsics_dir와 다를 수 있음)")
        else:
            zeus_idx = serial_to_idx[serial_for_label]
            K, D, _ = load_intrinsics_with_depth_scale(str(zeus_intrinsics_dir), zeus_idx)
        K_map[local_id] = K
        D_map[local_id] = D
    return K_map, D_map


def build_synthetic_meta(session_root: Path, capture_indices, capture_subdir: str = "capture") -> dict:
    """In-memory meta dict satisfying load_cube_pixel_observations's schema.

    Every capture here is a session1 "cube gripped by the robot" pose, so
    every capture gets cube_gripped=True / grasp_id=0. cams[local_id] is only
    present when that capture actually saved that camera's image (all 16
    captures have all 4 in this dataset, but this stays robust if that ever
    changes).
    """
    label_by_id = {v: k for k, v in LOCAL_CAM_IDS.items()}
    captures = []
    for idx in capture_indices:
        folder = f"{idx:03d}"
        cams = {}
        for local_id, label in label_by_id.items():
            rel = f"{capture_subdir}/{folder}/cam_{label}.png"
            if (session_root / rel).is_file():
                cams[str(local_id)] = {"saved": True, "rgb_path": rel}
        captures.append({
            "event_id": int(idx),
            "cube_gripped": True,
            "grasp_id": 0,
            "cams": cams,
        })
    return {"captures": captures}


def load_robot_T(session_root: Path, capture_indices, capture_subdir: str = "capture") -> dict:
    """event_id -> T_base_gripper (metres), via Zeus's own pose6_to_T."""
    robot_T = {}
    for idx in capture_indices:
        robot_json = session_root / capture_subdir / f"{idx:03d}" / "robot.json"
        state = json.loads(robot_json.read_text())
        robot_T[int(idx)] = pose6_to_T(state["pose"])
    return robot_T


def estimate_grasp_offset_one_camera(cube_obs_c, robot_T, K_map, D_map, cam_idx):
    """Eye-to-hand init for one camera c: solve A[i]@X = Z@B[i] (AX=ZB form).

    A[i] = T_base_gripper[i] (robot FK, known), B[i] = T_cam_cube[i] (PnP),
    X = T_gripper_cube, Z = T_base_C_c -- identical call pattern to
    ur3_calibration/fit_grasp_offset.py::estimate_grasp_offset_one_camera.
    Returns (T_gripper_cube, T_base_C_c, diag) or raises RuntimeError.
    """
    import cv2

    a_R, a_t, b_R, b_t = [], [], [], []
    for obs in cube_obs_c:
        event = int(obs.event)
        if event not in robot_T:
            continue
        T_cam_cube = solve_observed_pose(obs, K_map, D_map)
        if T_cam_cube is None:
            continue
        T_base_gripper = np.asarray(robot_T[event], dtype=np.float64)
        a_R.append(T_base_gripper[:3, :3])
        a_t.append(T_base_gripper[:3, 3].reshape(3, 1))
        b_R.append(T_cam_cube[:3, :3])
        b_t.append(T_cam_cube[:3, 3].reshape(3, 1))
    if len(a_R) < 5:
        raise RuntimeError(f"cam{cam_idx}: only {len(a_R)} usable gripped-cube poses (<5)")
    best = None
    for name, method in (("SHAH", cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH),
                         ("LI", cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)):
        try:
            R_x, t_x, R_z, t_z = cv2.calibrateRobotWorldHandEye(
                a_R, a_t, b_R, b_t, method=method)
        except Exception:
            continue
        T_gripper_cube = np.eye(4)
        T_gripper_cube[:3, :3] = np.asarray(R_x).reshape(3, 3)
        T_gripper_cube[:3, 3] = np.asarray(t_x).reshape(3)
        T_base_cam = np.eye(4)
        T_base_cam[:3, :3] = np.asarray(R_z).reshape(3, 3)
        T_base_cam[:3, 3] = np.asarray(t_z).reshape(3)
        errs_mm, errs_deg = [], []
        for T_bg_R, T_bg_t, T_cc_R, T_cc_t in zip(a_R, a_t, b_R, b_t):
            T_bg = np.eye(4); T_bg[:3, :3] = T_bg_R; T_bg[:3, 3] = T_bg_t.reshape(3)
            T_cc = np.eye(4); T_cc[:3, :3] = T_cc_R; T_cc[:3, 3] = T_cc_t.reshape(3)
            lhs = T_bg @ T_gripper_cube
            rhs = T_base_cam @ T_cc
            err = inv_T(lhs) @ rhs
            errs_mm.append(float(np.linalg.norm(err[:3, 3]) * 1000.0))
            errs_deg.append(float(np.degrees(np.linalg.norm(
                Rotation.from_matrix(err[:3, :3]).as_rotvec()))))
        score = float(np.median(errs_mm)) + 5.0 * float(np.median(errs_deg))
        if best is None or score < best[0]:
            best = (score, name, T_gripper_cube, T_base_cam,
                    {"n_poses": len(a_R), "method": name,
                     "residual_median_mm": float(np.median(errs_mm)),
                     "residual_median_deg": float(np.median(errs_deg)),
                     "residual_max_mm": float(np.max(errs_mm)),
                     "residual_max_deg": float(np.max(errs_deg))})
    if best is None:
        raise RuntimeError(f"cam{cam_idx}: both robot-world-hand-eye methods failed")
    _, name, T_gripper_cube, T_base_cam, diag = best
    return T_gripper_cube, T_base_cam, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-root", default=str(
        REPO_ROOT / "zeus_gello_calibration" / "data" / "session1_handheld_fixed_cam_0909"))
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--cube-observation-policy", default="legacy", choices=("legacy", "core_multiface"))
    ap.add_argument("--fixed-min-corners", type=int, default=8)
    ap.add_argument("--capture-subdir", default="capture",
                    help="session_root 아래 실제 캡처 폴더 이름 (예: capture_replayed)")
    ap.add_argument("--out", default=str(REPO_ROOT / "zeus_gello_calibration" / "pass1_grasp_offset.json"))
    ap.add_argument("--cube-config", default=None,
                    help="큐브 마커 config JSON (기본: config.py 메인 큐브). 0914 촬영처럼 GT 큐브로 찍었으면 targets/gt_cube/cube_config.json")
    args = ap.parse_args()

    session_root = Path(args.session_root)
    capture_dirs = sorted(p for p in (session_root / args.capture_subdir).iterdir() if p.is_dir())
    capture_indices = [int(p.name) for p in capture_dirs]
    print(f"session root: {session_root}  ({len(capture_indices)} captures: {capture_indices})")

    K_map, D_map = load_intrinsics_by_label(
        Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))
    for label, local_id in LOCAL_CAM_IDS.items():
        print(f"  local cam {local_id} ({label}): fx={K_map[local_id][0,0]:.1f} fy={K_map[local_id][1,1]:.1f}")

    meta = build_synthetic_meta(session_root, capture_indices, args.capture_subdir)
    robot_T = load_robot_T(session_root, capture_indices, args.capture_subdir)

    if args.cube_config:
        from calibration_pipeline.cube_config import load_cube_config_from_json_file
        cube_cfg, cube_src = load_cube_config_from_json_file(args.cube_config)
        if cube_cfg is None:
            raise SystemExit(f"cube config를 못 읽었습니다: {args.cube_config}")
        print(f"cube config: {args.cube_config} ({cube_src})")
    else:
        cube_cfg = get_default_cube_config()
    cube = AprilTagCubeTarget(cube_cfg)
    all_cam_ids = sorted(LOCAL_CAM_IDS.values())
    observations, diag = load_cube_pixel_observations(
        str(session_root), meta, cube, K_map, D_map, all_cam_ids,
        gripper_cam_idx=NOT_GRIPPER_SENTINEL, exclude_gripped=False,
        fixed_min_corners=args.fixed_min_corners, image_scale=1.0,
        observation_policy=args.cube_observation_policy,
    )
    print(f"loaded {len(observations)} cube observations "
          f"(policy={args.cube_observation_policy})")

    # Report raw PnP-accepted counts per camera (before the min-corner /
    # policy selection above) -- this is the number the task report asks for.
    label_by_id = {v: k for k, v in LOCAL_CAM_IDS.items()}
    pnp_accepted_by_cam = {c: 0 for c in all_cam_ids}
    for record in diag.get("observation_quality_by_event_camera", []):
        if record.get("pnp_accepted"):
            pnp_accepted_by_cam[int(record["camera_id"])] += 1
    print("PnP-accepted cube detections per camera (out of "
          f"{len(capture_indices)} captures):")
    for c in all_cam_ids:
        print(f"  cam{c} ({label_by_id[c]}): {pnp_accepted_by_cam[c]} pnp-accepted, "
              f"{sum(1 for o in observations if int(o.cam) == c)} selected for calibration")

    # Per-camera initial T_gripper_cube + T_base_C_c (AX=ZB form; see
    # estimate_grasp_offset_one_camera docstring).
    grasp_estimates = []
    cam_init = {}
    per_camera_diag = {}
    for c in all_cam_ids:
        cube_obs_c = [o for o in observations if int(o.cam) == c and o.grasp_idx is not None]
        if len(cube_obs_c) < 5:
            print(f"  cam{c} ({label_by_id[c]}): only {len(cube_obs_c)} usable cube observations, "
                  "skipping as an initializer")
            continue
        try:
            gripper_cube_c, cam_c_base, diag_c = estimate_grasp_offset_one_camera(
                cube_obs_c, robot_T, K_map, D_map, c)
        except RuntimeError as exc:
            print(f"  cam{c} ({label_by_id[c]}): hand-eye init failed ({exc}), skipping as an initializer")
            continue
        grasp_estimates.append(gripper_cube_c)
        cam_init[c] = cam_c_base
        per_camera_diag[c] = diag_c
        t_mm = gripper_cube_c[:3, 3] * 1000.0
        print(f"  cam{c} ({label_by_id[c]}): initial T_gripper_cube translation_mm="
              f"{t_mm.round(1).tolist()} (n_poses={diag_c['n_poses']}, method={diag_c['method']}, "
              f"residual_median_mm={diag_c['residual_median_mm']:.2f}, "
              f"residual_median_deg={diag_c['residual_median_deg']:.2f})")

    if not grasp_estimates:
        raise RuntimeError(
            "no camera could initialize T_gripper_cube from session1's gripped-cube views "
            "-- too few valid PnP solves to proceed (see per-camera counts above)")

    grasp_init, grasp_init_diag = cp.robust_se3_average(grasp_estimates, None)
    print(f"initial T_gripper_cube (robust avg of {len(grasp_estimates)} cams): "
          f"translation_mm={(grasp_init[:3,3]*1000).round(1).tolist()} dispersion={grasp_init_diag}")

    missing_cams = sorted(set(all_cam_ids) - set(cam_init))
    if missing_cams:
        print(f"cameras with no independent initializer (excluded from the joint solve): "
              f"{[label_by_id[c] for c in missing_cams]}")

    used_cam_ids = sorted(cam_init)
    state = PoseState(
        cams={c: cam_init[c] for c in used_cam_ids},
        gtc=np.eye(4),  # unused: NOT_GRIPPER_SENTINEL means no observation reads this
        board=None,
        cubes={},
        grasps={0: grasp_init},
    )
    used_observations = [o for o in observations if int(o.cam) in cam_init]
    keys = variable_keys(["T_base_Ci", "T_gripper_cube_by_grasp"], state)
    print(f"solving with {len(keys)} free SE(3) variables over {len(used_observations)} observations "
          f"({len(observations) - len(used_observations)} dropped: camera(s) with no initializer)")
    final_state, solve_diag = solve_corner_reprojection(
        observations=used_observations,
        variable_keys_=keys,
        reference_state=state,
        robot_T=robot_T,
        K_map=K_map,
        D_map=D_map,
        gripper_cam_idx=NOT_GRIPPER_SENTINEL,
        options=SolverOptions(),
    )
    print(f"solve success={solve_diag['success']} "
          f"train_reprojection_rmse_px={solve_diag['train_reprojection_rmse_px']:.4f} "
          f"(initial was {solve_diag['initial_reprojection_rmse_px']:.4f})")

    T_gripper_cube = final_state.grasps[0]
    t_mm = (T_gripper_cube[:3, 3] * 1000.0).tolist()
    rot_deg = Rotation.from_matrix(T_gripper_cube[:3, :3]).as_rotvec(degrees=True)
    print(f"FITTED T_gripper_cube: translation_mm={np.round(t_mm, 2).tolist()} "
          f"|t|_mm={np.linalg.norm(t_mm):.2f} rotvec_deg={np.round(rot_deg, 2).tolist()} "
          f"|r|_deg={np.linalg.norm(rot_deg):.2f}")

    result = {
        "T_gripper_cube": T_gripper_cube.tolist(),
        "T_gripper_cube_translation_mm": t_mm,
        "T_gripper_cube_rotvec_deg": rot_deg.tolist(),
        "T_base_cam": {
            f"{c}_{label_by_id[c]}": final_state.cams[c].tolist() for c in used_cam_ids
        },
        "local_cam_ids": LOCAL_CAM_IDS,
        "cameras_excluded_from_joint_solve": [label_by_id[c] for c in missing_cams],
        "pnp_accepted_per_camera": {label_by_id[c]: pnp_accepted_by_cam[c] for c in all_cam_ids},
        "selected_observations_per_camera": {
            label_by_id[c]: sum(1 for o in observations if int(o.cam) == c) for c in all_cam_ids
        },
        "n_captures": len(capture_indices),
        "solve_diagnostics": {
            "success": solve_diag["success"],
            "message": solve_diag["message"],
            "nfev": solve_diag["nfev"],
            "train_reprojection_rmse_px": solve_diag["train_reprojection_rmse_px"],
            "initial_reprojection_rmse_px": solve_diag["initial_reprojection_rmse_px"],
            "n_parameters": solve_diag["n_parameters"],
            "n_residuals": solve_diag["n_residuals"],
            "jacobian_rank_deficient": solve_diag["jacobian"]["rank_deficient"],
        },
        "per_camera_ax_zb_init_diagnostics": {
            label_by_id[c]: per_camera_diag[c] for c in per_camera_diag
        },
        "grasp_init_dispersion_across_cams": grasp_init_diag,
        "n_cams_used_for_grasp_init": len(grasp_estimates),
        "cube_observation_policy": args.cube_observation_policy,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
