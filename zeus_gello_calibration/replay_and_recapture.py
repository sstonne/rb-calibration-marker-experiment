#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/replay_and_recapture.py -- 저장된 세션의 joint 자세로
로봇을 직접(자동) 다시 움직여서 pose/joint - 이미지 동기화 오차를 없애고 재촬영

GELLO 텔레옵으로 촬영할 때는 SPACE를 누르는 순간에도 사람 손이 계속 움직이고
있을 수 있어서, "그 순간 읽은 pose"와 "그 순간 찍힌 이미지"가 미세하게 어긋날
(시간 지연으로 인한 위치 offset) 위험이 있다. 이 스크립트는 그 위험을 없애기
위해, 이미 저장된 capture/<idx>/robot.json 의 joint 값으로 **로봇을 완전히
멈춰 세운 뒤** 촬영한다(움직이는 도중에 찍는 게 아니라 도착 후 정지 상태에서
촬영 -> pose와 이미지가 확실히 같은 순간을 가리킴).

*** 이 스크립트는 실제로 로봇을 자동으로 움직인다(movej). ***
지금까지 이 프로젝트의 Zeus 쪽 스크립트들은 전부 GELLO 텔레옵(사람이 직접
조종)이거나 get_state()만 읽는 읽기 전용이었는데, 이건 처음으로 자동 이동
명령을 보낸다. 원래 GELLO로 자유롭게 움직이며 찍은 16개 자세를 순서 그대로
재생하는 거라 연속된 자세끼리는 원래도 사람이 부드럽게 이어서 움직인
경로이긴 하지만, **그래도 처음 실행은 반드시 로봇 옆에서 비상정지에 손이
닿는 상태로, 저속(--jnt-speed 낮게)으로 스텝별 확인(--no-step 없이)하며
진행할 것.** 단일 세션 재촬영 모드 외에 --all-phases를 사용하면 저장된 P1/P2/P3
pose를 모두 실행하고 하나의 새 calibration session에 저장한다. P2는 저장 pose의
x/y/yaw를 쓰고 실측 GRASP_REF_POSE의 z/roll/pitch로 안전한 평면 placement를 만든 뒤,
release 후 고정 CAM_POSE_JOINTS에서 한 번 촬영한다.

--regrasp-joints J1 J2 J3 J4 J5 J6 를 주면, 저장된 자세로 이동하기 전에
사용자 입력을 받아가며 큐브를 새로 쥐는 절차를 먼저 수행한다:
  1) Enter 입력 대기
  2) 그리퍼 열기 -> 지정한 joints로 movej
  3) "큐브를 놓아주세요" 안내 후 Enter 입력 대기
  4) 그리퍼 닫기
  이후 평소처럼 저장된 자세들로 이동+촬영을 진행한다.

기본은 기존 capture/<idx>/ 폴더를 덮어쓰지 않고 별도 폴더
(capture_replayed/<idx>/)에 저장한다. 해당 폴더가 이미 차 있으면 실행을 거부하므로
--output-subdir로 새 이름을 지정해야 한다. --overwrite-source를 주면 원래
capture/<idx>/를 덮어쓰지만, 원본 보존을 위해 사용하지 않는 것을 권장한다.

사용법:
  python replay_and_recapture.py --all-phases                    # P1/P2/P3 dry-run
  python replay_and_recapture.py --all-phases --execute          # 모든 동작 스텝별 확인
  python replay_and_recapture.py --all-phases --execute --no-step  # 검증 후 연속 자동 촬영
  python replay_and_recapture.py --session 1                     # dry-run: 계획만 출력
  python replay_and_recapture.py --session 1 --execute --motion-only  # 이동 경로만 저속 검증
  python replay_and_recapture.py --session 1 --execute \\
      --output-subdir capture_replayed_step_check                 # 스텝별 확인하며 촬영
  python replay_and_recapture.py --session 1 --execute --no-step \\
      --output-subdir capture_replayed_auto                       # 검증 후 연속 촬영
  python replay_and_recapture.py --session 1 --execute --regrasp-joints  # 재파지부터 (기본 자세)
  python replay_and_recapture.py --session 1 --execute \\
      --regrasp-joints 24.48 -33.28 -111.05 -179.99 35.68 24.48    # 재파지 자세 직접 지정
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot.backends.zeus_client import ZeusClient  # noqa: E402
from capture_pipeline.robot import euler_deg_to_matrix  # noqa: E402
from capture_pipeline.session import allocate_next_capture_session  # noqa: E402
from capture_pipeline.waypoint_safety import PROTOCOL_SAVED_POSE_REPLAY  # noqa: E402
from calibration_pipeline.board_config import (  # noqa: E402
    charuco_config_to_dict, describe_charuco_config, list_charuco_boards,
    resolve_charuco_config,
)
from calibration_pipeline.config import get_default_cube_config  # noqa: E402
from calibration_pipeline.cube_config import cube_config_to_dict  # noqa: E402

from zeus_gello_calibration.capture_session import (  # noqa: E402
    SESSIONS, load_camera_labels, connect_cameras, stop_cameras, LiveView,
    grab_frames, write_capture, read_robot_state,
    ROBOT_IP_DEFAULT, ROBOT_PORT_DEFAULT, DEVICE_MAP_DEFAULT,
    CAM_WIDTH, CAM_HEIGHT, CAM_FPS,
    validate_intrinsics_stream,
)
from zeus_gello_calibration.paths import (  # noqa: E402
    ZEUS_DATA_ROOT,
    require_zeus_data_path,
)
from zeus_gello_calibration.session2_pick_and_place import (  # noqa: E402
    APPROACH_MM_DEFAULT as P2_APPROACH_MM_DEFAULT,
    CAM_POSE_JOINTS as P2_CAMERA_JOINTS,
    CAM_POSE_POSE as P2_CAMERA_POSE,
    DESCEND_LIN_SPEED as P2_DESCEND_SPEED_DEFAULT,
    GRASP_REF_POSE as P2_GRASP_REF_POSE,
    MOVE_LIN_SPEED as P2_MOVE_SPEED_DEFAULT,
    START_JOINTS as P2_START_JOINTS,
    START_POSE as P2_START_POSE,
)
from capture_pipeline.paths import resolve_dated_dir  # noqa: E402

JNT_SPEED_DEFAULT = 10.0   # GELLO 텔레옵과 동일한 "실기 테스트로 정한" 기본값
OVERLAP_DEFAULT = 0.0      # 블렌딩 없이 매번 완전히 멈춰야 정확한 정지 후 촬영이 됨
SETTLE_S = 0.3             # movej 리턴 직후 잔진동/카메라 버퍼 안정화 대기
ROBOT_TIMEOUT_DEFAULT = 30.0  # movej는 완료까지 응답 없는 blocking 방식이라 기본 10초로는 부족할 수 있음
COMBINED_REPLAY_PROTOCOL = PROTOCOL_SAVED_POSE_REPLAY
GRIP_TIMEOUT_S = 3.0
GRIPPER_OPEN_STATE = ["0", "1", "0", "0"]
GRIPPER_CLOSED_STATE = ["0", "0", "0", "1"]

# session1 재파지 기본 자세 (실측 지정값) -- --regrasp-joints를 값 없이 주면 이걸 씀.
REGRASP_JOINTS_DEFAULT = {
    1: [24.48, -33.28, -111.05, -179.99, 35.68, 24.48],
}


def run_regrasp_sequence(rb, joints, jnt_speed, overlap):
    """큐브를 새로 쥐는 절차: Enter -> 그리퍼 열기+이동 -> "놓아주세요" -> Enter -> 그리퍼 닫기."""
    input("\n[재파지] Enter를 누르면 그리퍼를 열고 지정 자세로 이동합니다 > ")
    rb.grip("open")
    time.sleep(SETTLE_S)
    rb.movej(joints, jnt_speed=jnt_speed, overlap=overlap)
    time.sleep(SETTLE_S)
    print(f"[재파지] 이동 완료: {[round(v, 2) for v in joints]}")
    input("[재파지] 큐브를 그리퍼 위치에 놓아주세요. 다 놓으셨으면 Enter > ")
    rb.grip("close")
    time.sleep(SETTLE_S)
    print("[재파지] 그리퍼 닫음 -- 저장된 자세로 이동+촬영을 시작합니다.\n")


def _validated_six(value, label: str) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != 6
        or any(
            not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in value
        )
    ):
        raise ValueError(f"{label}: 유한한 숫자 6개여야 합니다.")
    return [float(item) for item in value]


def load_saved_states(capture_root: Path) -> list:
    items = []
    if not capture_root.is_dir():
        return items
    for d in sorted(
        (p for p in capture_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    ):
        robot_json = d / "robot.json"
        if not robot_json.exists():
            continue
        data = json.loads(robot_json.read_text())
        joints = _validated_six(data.get("joints"), f"{robot_json}: joints")
        pose = _validated_six(data.get("pose"), f"{robot_json}: pose")
        items.append({
            "index": int(d.name),
            "joints": joints,
            "pose": pose,
            "gripper": data.get("gripper"),
            "src_dir": d,
        })
    return items


def load_saved_joints(capture_root: Path) -> list:
    """Backward-compatible joint-only view used by the single-session mode."""
    return [
        {"index": item["index"], "joints": item["joints"], "src_dir": item["src_dir"]}
        for item in load_saved_states(capture_root)
    ]


def resolve_combined_sources(data_root: Path) -> dict:
    sources = {}
    for session_number in (1, 2, 3):
        info = SESSIONS[session_number]
        session_dir = resolve_dated_dir(
            f"session{session_number}_{info['name']}", data_root, create=False
        )
        states = load_saved_states(session_dir / "capture")
        if not states:
            raise ValueError(
                f"{session_dir / 'capture'}에 유효한 저장 pose가 없습니다."
            )
        sources[session_number] = {
            "session_dir": session_dir,
            "states": states,
        }
    return sources


def build_combined_event_plan(sources: dict) -> list[dict]:
    events = []
    for state in sources[1]["states"]:
        events.append({
            "planned_event_id": f"P1_{state['index']:03d}",
            "phase": "P1_MOVING_RIG",
            "target_state": "gripped",
            "source_index": state["index"],
            "source_dir": str(state["src_dir"]),
            "capture_joints": state["joints"],
            "capture_pose": state["pose"],
            "placement_id": None,
            "view_index": state["index"],
        })
    p2_states = sorted(sources[2]["states"], key=lambda state: state["pose"][3])
    for placement_index, state in enumerate(p2_states):
        placement_id = f"PLACEMENT_{placement_index:03d}"
        source_pose = state["pose"]
        placement_pose = [
            source_pose[0], source_pose[1], P2_GRASP_REF_POSE[2],
            source_pose[3], P2_GRASP_REF_POSE[4], P2_GRASP_REF_POSE[5],
        ]
        events.append({
            "planned_event_id": f"P2_{placement_id}_VIEW_0",
            "phase": "P2_PICK_PLACE",
            "target_state": "released",
            "source_index": state["index"],
            "source_dir": str(state["src_dir"]),
            "source_pose": source_pose,
            "placement_pose": placement_pose,
            "placement_joints": state["joints"],
            "placement_index": placement_index,
            "capture_joints": list(P2_CAMERA_JOINTS),
            "capture_pose": list(P2_CAMERA_POSE),
            "placement_id": placement_id,
            "view_index": 0,
        })
    last_placement = events[-1]["placement_id"]
    for state in sources[3]["states"]:
        events.append({
            "planned_event_id": f"P3_{state['index']:03d}",
            "phase": "P3_STATIONARY_RIG",
            "target_state": "stationary",
            "source_index": state["index"],
            "source_dir": str(state["src_dir"]),
            "capture_joints": state["joints"],
            "capture_pose": state["pose"],
            "placement_id": last_placement,
            "view_index": state["index"],
        })
    for capture_index, event in enumerate(events):
        event["capture_index"] = capture_index
    return events


def _atomic_json_write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _phase_counts(events: list[dict]) -> dict[str, int]:
    counts = {}
    for event in events:
        phase = event["phase"]
        counts[phase] = counts.get(phase, 0) + 1
    return counts


def _camera_mapping(device_map_path: Path, labels: dict) -> tuple[dict, int]:
    payload = json.loads(device_map_path.read_text())
    serial_to_idx = {str(k): int(v) for k, v in payload["serial_to_idx"].items()}
    detected = {str(serial) for serial in labels}
    configured = set(serial_to_idx)
    if detected != configured:
        raise ValueError(
            "연결 카메라 serial이 device_map.json과 다릅니다: "
            f"missing={sorted(configured - detected)}, unknown={sorted(detected - configured)}"
        )
    label_to_idx = {
        label: serial_to_idx[str(serial)]
        for serial, label in labels.items()
    }
    return label_to_idx, int(payload["gripper_cam_idx"])


def _initial_combined_meta(capture_root: Path, events: list[dict],
                           label_to_idx: dict, gripper_cam_idx: int,
                           source_dirs: dict, camera_stream: dict,
                           board: str | None = None) -> dict:
    cube_config = cube_config_to_dict(get_default_cube_config())
    # 촬영 중 검출은 하지 않지만 후단(04/05)이 이 값으로 이미지를 해석하므로
    # 실제로 찍은 보드를 기록해야 한다.
    board_cfg, board_source = resolve_charuco_config(board)
    board_config = charuco_config_to_dict(board_cfg)
    return {
        "root_folder": str(capture_root.resolve()),
        "capture_protocol": COMBINED_REPLAY_PROTOCOL,
        "gripper_cam_idx": int(gripper_cam_idx),
        "cam_indices": sorted(int(index) for index in label_to_idx.values()),
        "n_gripper_cams": 1,
        "n_fixed_cams": max(0, len(label_to_idx) - 1),
        "cube_config_source": "code_default:get_default_cube_config",
        "cube_config": cube_config,
        "charuco_board_config_source": board_source,
        "charuco_board_config": board_config,
        "capture_config": {
            "schema_version": "saved_pose_replay_capture_config_v1",
            "capture_protocol": COMBINED_REPLAY_PROTOCOL,
            "camera_stream": dict(camera_stream),
            "event_count_policy": "all_saved_source_poses",
            "expected_event_count": len(events),
            "expected_phase_counts": _phase_counts(events),
            "p2_views_per_placement": 1,
            "p2_camera_joints_deg": list(P2_CAMERA_JOINTS),
            "p2_camera_pose_6dof_mm_deg": list(P2_CAMERA_POSE),
            "p2_placement_pose_policy": "saved_x_y_yaw_plus_grasp_ref_z_roll_pitch",
            "p2_grasp_ref_pose_6dof_mm_deg": list(P2_GRASP_REF_POSE),
            "p2_start_joints_deg": list(P2_START_JOINTS),
            "p2_start_pose_6dof_mm_deg": list(P2_START_POSE),
            "source_sessions": {
                str(number): str(info["session_dir"])
                for number, info in source_dirs.items()
            },
        },
        "planned_events": events,
        "captures": [],
    }


def _write_combined_progress(capture_root: Path, meta: dict,
                             events: list[dict], status: str) -> None:
    completed_ids = [
        str(capture["planned_event_id"])
        for capture in meta["captures"]
        if capture.get("selected_for_analysis") is True
    ]
    expected_ids = [str(event["planned_event_id"]) for event in events]
    completed_set = set(completed_ids)
    manifest = {
        "schema_version": "saved_pose_replay_manifest_v1",
        "capture_protocol": COMBINED_REPLAY_PROTOCOL,
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_event_count": len(expected_ids),
        "selected_event_count": len(completed_ids),
        "expected_phase_counts": _phase_counts(events),
        "completed_planned_event_ids": completed_ids,
        "missing_planned_event_ids": [
            event_id for event_id in expected_ids if event_id not in completed_set
        ],
    }
    _atomic_json_write(capture_root / "meta.json", meta)
    _atomic_json_write(capture_root / "capture_protocol_manifest.json", manifest)


def _capture_combined_event(rb, cams, labels, label_to_idx, gripper_cam_idx,
                            view, capture_root: Path, meta: dict,
                            events: list[dict], event: dict,
                            release_state=None) -> bool:
    if view is not None:
        view.show()
    robot_state = read_robot_state(rb, {
        "capture_index": event["capture_index"],
        "planned_event_id": event["planned_event_id"],
        "phase": event["phase"],
        "replayed_from": event["source_dir"],
    })
    frames = grab_frames(cams, labels)
    event_dir = capture_root / "events" / f"{event['capture_index']:03d}"
    write_capture(frames, event_dir, robot_state)
    relative_dir = event_dir.relative_to(capture_root)
    camera_records = {}
    all_saved = True
    for label, (color, depth) in frames.items():
        camera_index = label_to_idx[label]
        saved = color is not None
        all_saved = all_saved and saved
        camera_records[str(camera_index)] = {
            "saved": saved,
            "transport_success": saved,
            "is_gripper": camera_index == gripper_cam_idx,
            "rgb_path": (
                str(relative_dir / f"cam_{label}.png") if saved else None
            ),
            "depth_path": (
                str(relative_dir / f"cam_{label}_depth.png")
                if saved and depth is not None else None
            ),
        }
    pose = [float(value) for value in robot_state["pose"]]
    joints = [float(value) for value in robot_state["joints"]]
    placement_index = (
        event["placement_index"] if event["phase"] == "P2_PICK_PLACE" else None
    )
    set_index = placement_index
    if event["phase"] == "P1_MOVING_RIG":
        set_index = None
    elif event["phase"] == "P3_STATIONARY_RIG":
        set_index = len([
            planned for planned in events if planned["phase"] == "P2_PICK_PLACE"
        ])
    p2_count = len([
        planned for planned in events if planned["phase"] == "P2_PICK_PLACE"
    ])
    grasp_id = (
        0 if event["phase"] == "P1_MOVING_RIG"
        else int(event["placement_index"]) + 1
        if event["phase"] == "P2_PICK_PLACE"
        else p2_count
    )
    capture_block = (
        "B_eyetohand" if event["target_state"] == "gripped"
        else "A_placement"
    )
    capture = {
        "event_id": int(event["capture_index"]),
        "capture_index": int(event["capture_index"]),
        "protocol_version": COMBINED_REPLAY_PROTOCOL,
        "planned_event_id": event["planned_event_id"],
        "attempt_index": 0,
        "phase": event["phase"],
        "target_state": event["target_state"],
        "placement_id": event["placement_id"],
        "view_index": int(event["view_index"]),
        "source_pose_dir": event["source_dir"],
        "robot_pose_6dof": pose,
        "capture_gripper_pose_6dof": pose,
        "robot_pose_matrix_4x4": euler_deg_to_matrix(*pose).tolist(),
        "capture_gripper_pose_matrix_4x4": euler_deg_to_matrix(*pose).tolist(),
        "capture_robot_joints_6dof": joints,
        "robot_state": {
            "flange_pose_6dof_mm_deg": pose,
            "joints_deg": joints,
            "gripper_io": robot_state.get("gripper"),
        },
        "set_index": set_index,
        "grasp_id": grasp_id,
        "cube_gripped": event["target_state"] == "gripped",
        "capture_block": capture_block,
        "capture_gate": {
            "status": "NOT_EVALUATED",
            "reason": "saved_pose_replay_captures_all_stopped_robot_events",
            "capture_block": capture_block,
        },
        "analysis_group_id": (
            f"P2:{event['placement_id']}"
            if event["phase"] == "P2_PICK_PLACE"
            else "P3:STATIONARY_RIG"
            if event["phase"] == "P3_STATIONARY_RIG"
            else f"P1:{event['planned_event_id']}"
        ),
        "split_unit_id": (
            event["placement_id"] if event["phase"] == "P2_PICK_PLACE" else None
        ),
        "release_state": release_state,
        "cams": camera_records,
        "transport_integrity": {
            "pass": all_saved,
            "status": "PASS" if all_saved else "FAIL",
        },
        "attempt_status": "captured" if all_saved else "transport_failed",
        "selected_for_analysis": all_saved,
    }
    meta["captures"].append(capture)
    _write_combined_progress(capture_root, meta, events, "in_progress")
    print(
        f"[SAVE] {event['planned_event_id']} -> {event_dir} "
        f"({len(camera_records)} cameras)"
    )
    return all_saved


def _step_allowed(description: str, no_step: bool) -> bool:
    if no_step:
        print(f"\n[AUTO] {description}")
        return True
    command = input(f"\n{description}\nEnter=진행 / q=중단 > ").strip().lower()
    return command != "q"


def _approach_pose(pose: list[float], approach_mm: float) -> list[float]:
    result = list(pose)
    result[2] += float(approach_mm)
    return result


def _align_rotation_pose(current: list[float], target: list[float]) -> list[float]:
    return [
        float(current[0]), float(current[1]), float(current[2]),
        float(target[3]), float(target[4]), float(target[5]),
    ]


def _move_to_p2_contact(rb, target_pose: list[float], move_speed: float,
                        descend_speed: float, approach_mm: float,
                        description: str) -> None:
    current_pose = [float(value) for value in rb.get_state()["pose"]]
    aligned = _align_rotation_pose(current_pose, target_pose)
    approach = _approach_pose(target_pose, approach_mm)
    print(f"  {description}: 현재 위치에서 target 회전 정렬")
    rb.movel(aligned, lin_speed=move_speed, overlap=0.0)
    print(f"  {description}: target +Z {approach_mm:.1f} mm 접근")
    rb.movel(approach, lin_speed=move_speed, overlap=0.0)
    print(f"  {description}: 저장된 flange pose로 수직 하강")
    rb.movel(target_pose, lin_speed=descend_speed, overlap=0.0)


def _retreat_from_p2_contact(rb, target_pose: list[float],
                             descend_speed: float, approach_mm: float) -> None:
    rb.movel(
        _approach_pose(target_pose, approach_mm),
        lin_speed=descend_speed,
        overlap=0.0,
    )


def _require_gripper_state(rb, expected: list[str], label: str) -> dict:
    state = rb.get_state()
    actual = list(state.get("gripper") or [])
    if actual != expected:
        raise RuntimeError(
            f"{label}: gripper state expected {expected}, received {actual}"
        )
    return state


def _update_allocated_session_manifest(manifest_path: Path, status: str,
                                       expected_count: int,
                                       captured_count: int) -> None:
    payload = json.loads(manifest_path.read_text())
    payload.update({
        "status": status,
        "capture_protocol": COMBINED_REPLAY_PROTOCOL,
        "expected_event_count": int(expected_count),
        "captured_event_count": int(captured_count),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    _atomic_json_write(manifest_path, payload)


def run_combined_replay(rb, cams, labels, view, data_root: Path,
                        device_map_path: Path, session_label: str,
                        sources: dict, events: list[dict], args) -> tuple[Path | None, bool]:
    p1_events = [event for event in events if event["phase"] == "P1_MOVING_RIG"]
    p2_events = [event for event in events if event["phase"] == "P2_PICK_PLACE"]
    p3_events = [event for event in events if event["phase"] == "P3_STATIONARY_RIG"]

    live_state = rb.get_state()
    if list(live_state.get("gripper") or []) != GRIPPER_CLOSED_STATE:
        raise RuntimeError(
            "P1 시작 전에 gripper CLOSED 상태가 확인되지 않았습니다. "
            "큐브를 파지하거나 --regrasp-joints를 사용하세요."
        )

    allocated = None
    capture_root = None
    meta = None
    label_to_idx = {}
    gripper_cam_idx = -1
    if not args.motion_only:
        label_to_idx, gripper_cam_idx = _camera_mapping(device_map_path, labels)
        allocated = allocate_next_capture_session(str(data_root), label=session_label)
        capture_root = Path(allocated.capture_root)
        meta = _initial_combined_meta(
            capture_root, events, label_to_idx, gripper_cam_idx, sources,
            {
                "color_w": int(args.width),
                "color_h": int(args.height),
                "depth_w": int(
                    args.depth_width if args.depth_width is not None else args.width
                ),
                "depth_h": int(
                    args.depth_height if args.depth_height is not None else args.height
                ),
                "fps": int(args.fps),
            },
            board=getattr(args, "board", None),
        )
        _write_combined_progress(capture_root, meta, events, "in_progress")
        _update_allocated_session_manifest(
            Path(allocated.manifest_path), "capturing", len(events), 0
        )
        print(f"\n[SESSION] {allocated.session_root}")
        print(f"[SESSION] 촬영 데이터 -> {capture_root}")

    def capture(event: dict, release_state=None) -> bool:
        if args.motion_only:
            return True
        time.sleep(SETTLE_S)
        return _capture_combined_event(
            rb, cams, labels, label_to_idx, gripper_cam_idx, view,
            capture_root, meta, events, event, release_state=release_state,
        )

    completed = True
    previous_placement_pose = None
    try:
        for event in p1_events:
            if not _step_allowed(
                f"[{event['planned_event_id']}] 저장 joint로 이동 후 촬영",
                args.no_step,
            ):
                completed = False
                break
            rb.movej(
                event["capture_joints"],
                jnt_speed=args.jnt_speed,
                overlap=args.overlap,
            )
            if not capture(event):
                completed = False
                break
        if not completed:
            return capture_root, False

        if not _step_allowed(
            "[P1 -> P2] cube 파지 상태로 검증된 P2 시작 joint에 이동",
            args.no_step,
        ):
            return capture_root, False
        rb.movej(
            P2_START_JOINTS,
            jnt_speed=args.jnt_speed,
            overlap=args.overlap,
        )

        for placement_number, event in enumerate(p2_events):
            if not _step_allowed(
                f"[{event['planned_event_id']}] 저장 pose에 place하고 CAM_POSE에서 촬영",
                args.no_step,
            ):
                completed = False
                break

            if previous_placement_pose is not None:
                _move_to_p2_contact(
                    rb, previous_placement_pose,
                    args.p2_move_speed, args.p2_descend_speed,
                    args.p2_approach_mm,
                    f"P2 #{placement_number:03d} 이전 cube pick",
                )
                rb.grip("close", timeout_s=GRIP_TIMEOUT_S)
                time.sleep(SETTLE_S)
                _require_gripper_state(
                    rb, GRIPPER_CLOSED_STATE,
                    f"P2 #{placement_number:03d} pick",
                )
                _retreat_from_p2_contact(
                    rb, previous_placement_pose,
                    args.p2_descend_speed, args.p2_approach_mm,
                )

            target_pose = event["placement_pose"]
            _move_to_p2_contact(
                rb, target_pose,
                args.p2_move_speed, args.p2_descend_speed,
                args.p2_approach_mm,
                f"P2 #{placement_number:03d} cube place",
            )
            pre_release = read_robot_state(rb, {"state_role": "pre_release"})
            rb.grip("open", timeout_s=GRIP_TIMEOUT_S)
            time.sleep(SETTLE_S)
            _require_gripper_state(
                rb, GRIPPER_OPEN_STATE,
                f"P2 #{placement_number:03d} release",
            )
            post_release = read_robot_state(rb, {"state_role": "post_release"})
            _retreat_from_p2_contact(
                rb, target_pose,
                args.p2_descend_speed, args.p2_approach_mm,
            )
            rb.movej(
                P2_CAMERA_JOINTS,
                jnt_speed=args.jnt_speed,
                overlap=args.overlap,
            )
            capture_ok = capture(event, release_state={
                "schema_version": "saved_pose_release_state_v1",
                "placement_id": event["placement_id"],
                "source_pose_dir": event["source_dir"],
                "place_pose_6dof_mm_deg": list(target_pose),
                "pre_release": pre_release,
                "post_release": post_release,
            })
            previous_placement_pose = target_pose
            if not capture_ok:
                completed = False
                break
        if not completed:
            return capture_root, False

        for event in p3_events:
            if not _step_allowed(
                f"[{event['planned_event_id']}] stationary cube 유지, 저장 joint로 이동 후 촬영",
                args.no_step,
            ):
                completed = False
                break
            rb.movej(
                event["capture_joints"],
                jnt_speed=args.jnt_speed,
                overlap=args.overlap,
            )
            if not capture(event):
                completed = False
                break
    finally:
        if capture_root is not None and meta is not None and allocated is not None:
            status = "complete" if completed and len(meta["captures"]) == len(events) else "incomplete"
            _write_combined_progress(capture_root, meta, events, status)
            _update_allocated_session_manifest(
                Path(allocated.manifest_path),
                status,
                len(events),
                len(meta["captures"]),
            )
    return capture_root, completed


def require_subdir_name(value: str) -> str:
    """Keep replay output inside the selected session directory."""
    path = Path(value)
    if value in {"", ".", ".."} or path.name != value or path.is_absolute():
        raise argparse.ArgumentTypeError("단일 폴더 이름만 허용합니다 (예: capture_replayed_auto).")
    return value


def directory_has_files(path: Path) -> bool:
    return path.is_dir() and any(path.iterdir())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--session", type=int, choices=sorted(SESSIONS),
                      help="기존 단일 phase만 재촬영")
    mode.add_argument("--all-phases", action="store_true",
                      help="저장된 session1/2/3 pose를 P1/P2/P3로 연속 실행해 새 session 하나에 저장")
    ap.add_argument("--robot-ip", default=ROBOT_IP_DEFAULT)
    ap.add_argument("--robot-port", type=int, default=ROBOT_PORT_DEFAULT)
    ap.add_argument("--robot-timeout", type=float, default=ROBOT_TIMEOUT_DEFAULT)
    ap.add_argument("--device-map", default=str(DEVICE_MAP_DEFAULT))
    ap.add_argument(
        "--data-root", "--out-root",
        dest="data_root",
        default=str(ZEUS_DATA_ROOT),
        help=f"Zeus capture data root (must stay inside {ZEUS_DATA_ROOT})",
    )
    ap.add_argument(
        "--session-dir",
        default=None,
        help="재생할 session 폴더를 직접 지정. 생략하면 --data-root에서 --session에 맞는 최신 폴더를 찾음",
    )
    ap.add_argument(
        "--output-subdir",
        type=require_subdir_name,
        default="capture_replayed",
        help="재촬영 저장 폴더 이름 (기본: capture_replayed). 기존 데이터가 있으면 실행 거부",
    )
    ap.add_argument(
        "--session-label",
        default="zeus_saved_pose_replay",
        help="--all-phases 결과 session 폴더의 설명 label",
    )
    ap.add_argument("--jnt-speed", type=float, default=JNT_SPEED_DEFAULT)
    ap.add_argument("--overlap", type=float, default=OVERLAP_DEFAULT)
    ap.add_argument("--width", type=int, default=CAM_WIDTH, help="RealSense color/depth width")
    ap.add_argument("--height", type=int, default=CAM_HEIGHT, help="RealSense color/depth height")
    ap.add_argument("--depth-width", type=int, default=None, help="RealSense depth width")
    ap.add_argument("--depth-height", type=int, default=None, help="RealSense depth height")
    ap.add_argument("--fps", type=int, default=CAM_FPS, help="RealSense stream FPS")
    ap.add_argument("--board", default=None,
                    help="장면에 둔 ChArUco 보드 정의. targets/charuco_boards/ 의 이름"
                         f" ({', '.join(list_charuco_boards()) or '없음'}) 또는 JSON 경로."
                         " 생략하면 config.py 기본 보드. meta.json 에 그대로 기록된다")
    ap.add_argument("--p2-approach-mm", type=float, default=P2_APPROACH_MM_DEFAULT)
    ap.add_argument("--p2-move-speed", type=float, default=P2_MOVE_SPEED_DEFAULT)
    ap.add_argument("--p2-descend-speed", type=float, default=P2_DESCEND_SPEED_DEFAULT)
    ap.add_argument("--overwrite-source", "--overwrite", dest="overwrite_source", action="store_true",
                    help="원래 capture/<idx>/ 폴더를 덮어씀. 원본 보존을 위해 사용 비권장")
    ap.add_argument("--execute", action="store_true", help="실제로 이동/촬영 (없으면 dry-run)")
    ap.add_argument("--no-step", action="store_true", help="스텝마다 Enter로 확인하지 않고 연속 실행")
    camera_reset = ap.add_mutually_exclusive_group()
    camera_reset.add_argument(
        "--no-cam-reset",
        action="store_true",
        help="카메라 시작 전 hardware reset 생략",
    )
    camera_reset.add_argument(
        "--cam-reset",
        action="store_true",
        help="카메라를 순차 hardware reset. --all-phases는 기본적으로 reset을 생략함",
    )
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--motion-only", action="store_true",
                    help="카메라/저장 없이 robot 동작만 수행. --all-phases에서는 P2 gripper 동작도 포함")
    ap.add_argument("--regrasp-joints", type=float, nargs="*", default=None,
                    metavar="J",
                    help="저장된 자세로 이동하기 전에, 사용자 확인을 받아가며 이 joints에서 "
                         "큐브를 새로 쥐는 절차(그리퍼 열기->이동->대기->그리퍼 닫기)를 먼저 수행. "
                         "값 6개를 직접 주거나(J1..J6), 값 없이 --regrasp-joints만 주면 "
                         "REGRASP_JOINTS_DEFAULT[세션번호]를 씀")
    args = ap.parse_args()
    try:
        board_cfg, board_source = resolve_charuco_config(args.board)
    except (FileNotFoundError, ValueError) as error:
        ap.error(str(error))
    print(f"[BOARD] {board_source}: {describe_charuco_config(board_cfg)}")

    if args.session == 2 and args.execute:
        ap.error(
            "session2는 place/pick/gripper 동작이 필요한 데이터입니다. 단순 movej 재생으로는 "
            "물리 상태를 재현할 수 없으므로 session2_pick_and_place.py를 사용하세요."
        )

    if args.regrasp_joints is not None:
        if not args.all_phases and args.session != 1:
            ap.error("--regrasp-joints는 cube를 계속 파지하는 session1에서만 사용할 수 있습니다.")
        if len(args.regrasp_joints) == 0:
            default_session = 1 if args.all_phases else args.session
            if default_session not in REGRASP_JOINTS_DEFAULT:
                print(f"[ERROR] --regrasp-joints를 값 없이 줬는데 세션 {default_session}용 기본값이 없습니다. "
                      "직접 6개 값을 주세요 (J1..J6).")
                return
            args.regrasp_joints = REGRASP_JOINTS_DEFAULT[default_session]
        elif len(args.regrasp_joints) != 6:
            print(f"[ERROR] --regrasp-joints는 6개 값(J1..J6)이거나 값 없이 줘야 합니다 "
                  f"({len(args.regrasp_joints)}개 받음).")
            return

    try:
        data_root = require_zeus_data_path(args.data_root, label="--data-root")
    except ValueError as exc:
        ap.error(str(exc))
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
    args.depth_width = depth_width
    args.depth_height = depth_height

    if args.all_phases:
        if args.session_dir is not None:
            ap.error("--session-dir는 단일 --session 모드에서만 사용할 수 있습니다.")
        if args.overwrite_source:
            ap.error("--all-phases는 새 session만 할당하므로 --overwrite-source를 사용할 수 없습니다.")
        if args.p2_approach_mm <= 0 or args.p2_move_speed <= 0 or args.p2_descend_speed <= 0:
            ap.error("P2 approach와 speed 값은 모두 양수여야 합니다.")
        try:
            sources = resolve_combined_sources(data_root)
            events = build_combined_event_plan(sources)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            ap.error(f"P1/P2/P3 원본 pose를 읽을 수 없습니다: {exc}")

        phase_counts = _phase_counts(events)
        print("=== 저장 pose 기반 P1/P2/P3 단일 session 자동 촬영 ===")
        for number in (1, 2, 3):
            source = sources[number]
            print(
                f"P{number}: {source['session_dir'] / 'capture'} "
                f"({len(source['states'])} poses)"
            )
        print(f"총 planned events: {len(events)} ({phase_counts})")
        print(
            "P2: 저장 x/y/yaw + GRASP_REF z/roll/pitch에 cube place -> "
            f"CAM_POSE_JOINTS에서 1회 촬영 ({len(sources[2]['states'])} placements)"
        )

        if not args.execute:
            print("\n(dry-run) robot과 카메라는 연결하지 않았습니다.")
            return

        cams, used_labels, view = [], {}, None
        if args.motion_only:
            print(
                "\n--motion-only: 카메라와 파일 저장은 생략하지만 "
                "P2 pick/place 및 gripper 동작은 실제 수행합니다."
            )
        else:
            labels = load_camera_labels(Path(args.device_map))
            validate_intrinsics_stream(
                Path(args.device_map), args.width, args.height, args.fps,
                depth_width=depth_width, depth_height=depth_height,
            )
            skip_camera_reset = args.no_cam_reset or not args.cam_reset
            if skip_camera_reset:
                print("[INFO] P1/P2/P3 통합 촬영: 카메라 전체 hardware reset을 생략합니다.")
            cams, used_labels = connect_cameras(
                labels,
                no_reset=skip_camera_reset,
                width=args.width,
                height=args.height,
                fps=args.fps,
                depth_width=depth_width,
                depth_height=depth_height,
            )
            if not args.no_preview:
                view = LiveView(
                    cams, used_labels,
                    window_name="P1/P2/P3 combined replay (q/ESC=preview close)",
                )
                view.start()

        rb = ZeusClient(args.robot_ip, args.robot_port, timeout=args.robot_timeout)
        rb.connect()
        try:
            if args.regrasp_joints is not None:
                run_regrasp_sequence(
                    rb, args.regrasp_joints, args.jnt_speed, args.overlap
                )
            print(
                "\n*** P1 이동, P2 pick/place, P3 이동을 실제로 수행합니다. "
                "비상정지에 손이 닿는 상태인지 확인하세요. ***"
            )
            prompt = (
                f"{len(events)}-event 연속 자동 실행을 시작하려면 'go' 입력: "
                if args.no_step
                else "P1/P2/P3 스텝별 실행을 시작하려면 'go' 입력: "
            )
            if input(prompt).strip().lower() != "go":
                print("취소했습니다.")
                return
            output_root, completed = run_combined_replay(
                rb, cams, used_labels, view, data_root,
                Path(args.device_map), args.session_label,
                sources, events, args,
            )
            if args.motion_only:
                print(
                    "\n완료: P1/P2/P3 robot 동작만 검증했습니다. 저장 데이터는 없습니다."
                    if completed
                    else "\n중단: robot 동작 검증을 끝까지 완료하지 않았습니다."
                )
            elif completed:
                print(f"\n완료: P1/P2/P3가 하나의 session에 저장됐습니다: {output_root}")
            else:
                print(f"\n중단: 부분 session이 incomplete로 보존됐습니다: {output_root}")
        except (KeyboardInterrupt, EOFError):
            print("\n사용자 중단: robot stop을 요청합니다.")
            try:
                rb.stop()
            except Exception:
                pass
        finally:
            rb.close()
            if view is not None:
                view.stop()
            if cams:
                stop_cameras(cams)
        return

    info = SESSIONS[args.session]
    try:
        session_dir = (
            require_zeus_data_path(args.session_dir, label="--session-dir")
            if args.session_dir
            else resolve_dated_dir(
                f"session{args.session}_{info['name']}", data_root, create=False
            )
        )
    except ValueError as exc:
        ap.error(str(exc))
    capture_root = session_dir / "capture"
    out_root = capture_root if args.overwrite_source else session_dir / args.output_subdir

    try:
        items = load_saved_joints(capture_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        ap.error(f"원본 robot.json을 읽을 수 없습니다: {exc}")
    if not items:
        ap.error(f"{capture_root}에 유효한 robot.json 캡처가 없습니다.")
    if (
        args.execute
        and not args.motion_only
        and not args.overwrite_source
        and directory_has_files(out_root)
    ):
        ap.error(
            f"{out_root}에 기존 결과가 있습니다. 원본을 보존하려면 "
            "--output-subdir에 새 이름을 지정하세요."
        )

    print(f"=== 세션 {args.session}: {info['name']} 재생+재촬영 ===")
    print(f"원본: {capture_root}  ({len(items)}개)")
    print(f"저장 위치: {out_root}{' (원본 덮어씀)' if args.overwrite_source else ' (원본은 그대로 둠)'}")
    print(f"jnt_speed={args.jnt_speed}  overlap={args.overlap}\n")
    if args.regrasp_joints is not None:
        print(f"--regrasp-joints 지정됨: 시작 전에 {[round(v, 2) for v in args.regrasp_joints]}에서 "
              "재파지 절차(그리퍼 열기->이동->대기->닫기)를 먼저 수행합니다.\n")
    elif args.session == 1:
        print("큐브가 세션1 촬영 때와 같은 방식으로 그리퍼에 그대로 물려있어야 합니다 "
              "(--regrasp-joints 없이는 이 스크립트가 그리퍼를 건드리지 않습니다).\n")
    else:
        print("session3의 stationary target은 원본 촬영 때와 같은 위치에 고정하고, "
              "로봇과 gripper camera만 이동해야 합니다.\n")

    for it in items:
        print(f"  #{it['index']:03d}  joints(deg)={[round(v, 1) for v in it['joints']]}")

    if not args.execute:
        print(f"\n(dry-run) 총 {len(items)}개 이동+촬영 계획. --execute 를 주면 실행합니다.")
        return

    cams, used_labels, view = [], [], None
    if args.motion_only:
        print("--motion-only: 카메라를 연결하지 않고 movej 이동만 수행합니다 (촬영/저장 없음).\n")
    else:
        labels = load_camera_labels(Path(args.device_map))
        validate_intrinsics_stream(
            Path(args.device_map), args.width, args.height, args.fps,
            depth_width=depth_width, depth_height=depth_height,
        )
        cams, used_labels = connect_cameras(
            labels,
            no_reset=args.no_cam_reset and not args.cam_reset,
            width=args.width,
            height=args.height,
            fps=args.fps,
            depth_width=depth_width,
            depth_height=depth_height,
        )
        if not args.no_preview:
            view = LiveView(cams, used_labels, window_name="replay_and_recapture (q/ESC=닫기)")
            view.start()

    rb = ZeusClient(args.robot_ip, args.robot_port, timeout=args.robot_timeout)
    rb.connect()
    print("\n*** 실제 로봇이 자동으로 움직입니다. 비상정지에 손이 닿는 상태인지 확인하세요. ***")

    if args.regrasp_joints is not None:
        run_regrasp_sequence(rb, args.regrasp_joints, args.jnt_speed, args.overlap)

    prompt = (
        "전체 연속 자동 실행을 시작하려면 'go' 입력: "
        if args.no_step
        else "스텝별 실행을 시작하려면 'go' 입력: "
    )
    if input(prompt).strip().lower() != "go":
        print("취소했습니다.")
        rb.close()
        if view is not None:
            view.stop()
        stop_cameras(cams)
        return

    try:
        for it in items:
            desc = f"[{it['index']:03d}/{len(items)}] movej -> {[round(v, 1) for v in it['joints']]}"
            if not args.no_step:
                cmd = input(f"\n{desc}\nEnter=이동+촬영 / q=중단 > ").strip().lower()
                if cmd == "q":
                    print("중단했습니다.")
                    break
            else:
                print(desc)

            rb.movej(it["joints"], jnt_speed=args.jnt_speed, overlap=args.overlap)
            time.sleep(SETTLE_S)  # 잔진동 + 카메라 버퍼가 새 프레임으로 채워질 시간

            if args.motion_only:
                continue

            if view is not None:
                view.show()

            robot_state = read_robot_state(rb, {"capture_index": it["index"], "replayed_from": str(it["src_dir"])})
            frames = grab_frames(cams, used_labels)
            out_dir = out_root / f"{it['index']:03d}"
            write_capture(frames, out_dir, robot_state)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        rb.close()
        if view is not None:
            view.stop()
        if cams:
            stop_cameras(cams)

    if args.motion_only:
        print("\n완료 -- movej 이동만 수행했습니다 (촬영/저장 없음).")
    else:
        print(f"\n완료 -- {out_root} 확인해보세요.")


if __name__ == "__main__":
    main()
