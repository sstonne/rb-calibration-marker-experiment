#!/usr/bin/env python3
"""
촬영 완료 후 저장된 영상을 다시 검출하여 캘리브레이션 관측 품질을 평가한다.

Cube와 ChArUco 관측을 standard/strict 기준으로 선별하고, 선택된 2D·3D corner,
제외 사유, 재촬영 후보 및 검토용 오버레이를 생성한다. 원본 촬영 데이터는 수정하지 않는다.

[실행 명령어]
python3 04_filter_observations.py \
  --session-root data/session02_NOUSE_session04_0814/calib_train \
  --intrinsics-dir intrinsics
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from calibration_pipeline.apriltag_cube import (
    CUBE_CORNER_REFINEMENT_MODES,
    AprilTagCubeTarget,
)
from calibration_pipeline.charuco import CharucoTarget
from calibration_pipeline.config import get_default_cube_config
from calibration_pipeline.board_config import (
    charuco_config_to_dict,
    charuco_topology,
    load_charuco_config_from_meta,
)
from calibration_pipeline.cube_detection import detect_corner_observations
from calibration_pipeline.cube_config import cube_config_to_dict
from calibration_pipeline.observations import POST_CAPTURE_MANIFEST_SCHEMA
from calibration_pipeline.runtime import (
    get_capture_set_index,
    load_intrinsics_with_depth_scale,
    resolve_cube_config_for_run,
)
from capture_pipeline.waypoint_safety import (
    PROTOCOL_COMPOSITE_RIG_45,
    PROTOCOL_SAVED_POSE_REPLAY,
)


PHASE_AWARE_CAPTURE_PROTOCOLS = {
    PROTOCOL_COMPOSITE_RIG_45,
    PROTOCOL_SAVED_POSE_REPLAY,
}


OUTPUT_NAMES = {
    "manifest": "Step2b_observation_manifest.json",
    "selected": "Step2b_selected_observations.csv",
    "quarantine": "Step2b_quarantine_observations.csv",
    "rejected": "Step2b_rejected_observations.csv",
    "retake": "Step2b_retake_candidates.csv",
    "overlay": "Step2b_review_overlay.jpg",
    "readme": "CAPTURE_FILTER.md",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _default_output_dir(session_root: Path) -> Path:
    if session_root.name == "calib_train":
        return session_root.parent / "calib_out" / "capture_filter"
    return session_root / "capture_filter"


def _capture_index(meta: dict) -> Dict[int, dict]:
    return {
        int(capture["event_id"]): capture
        for capture in meta.get("captures", [])
        if int(capture.get("event_id", -1)) >= 0
    }


def _saved_camera_rows(meta: dict) -> Iterable[Tuple[dict, int, dict]]:
    for capture in meta.get("captures", []):
        if int(capture.get("event_id", -1)) < 0:
            continue
        if (capture.get("protocol_version") in PHASE_AWARE_CAPTURE_PROTOCOLS
                and capture.get("selected_for_analysis") is not True):
            continue
        for camera_text, camera_info in capture.get("cams", {}).items():
            if camera_info.get("saved"):
                yield capture, int(camera_text), camera_info


def _image_provenance(session_root: Path, meta: dict) -> dict:
    output = {}
    for _, _, camera_info in _saved_camera_rows(meta):
        relative = str(camera_info.get("rgb_path", ""))
        if not relative or relative in output:
            continue
        path = (session_root / relative).resolve()
        output[relative] = {
            "path": str(path),
            "sha256": _sha256(path) if path.is_file() else None,
        }
    return dict(sorted(output.items()))


def _core_support(record: Mapping) -> bool:
    return bool(
        record.get("pnp_accepted")
        and int(record.get("observed_face_count") or 0) >= 2
        and int(record.get("noncoplanar_face_count") or 0) >= 2
        and record.get("is_planar") is False
        and int(record.get("positive_depth_candidate_count") or 0) >= 1
        and record.get("quality_tier") == "nonplanar_multiface"
    )


def _cube_policy_decision(record: Mapping, *, max_rmse_px: float,
                          min_inlier_fraction: float) -> Tuple[bool, str]:
    status = str(record.get("status", "unknown"))
    if not record.get("pnp_accepted"):
        return False, status
    if not _core_support(record):
        tier = str(record.get("quality_tier", "unknown"))
        return False, f"noncore_{tier}"
    rmse = float(record.get("pnp_rmse_px") or math.inf)
    if rmse > float(max_rmse_px):
        return False, f"pnp_rmse_above_{float(max_rmse_px):g}px"
    inlier = float(record.get("pnp_inlier_fraction") or 0.0)
    if inlier < float(min_inlier_fraction):
        return False, f"inlier_fraction_below_{float(min_inlier_fraction):g}"
    return True, "selected"


def _disposition(selected: bool, reason: str, recovered: bool = False) -> str:
    if selected:
        return "recovered" if recovered else "selected"
    if reason.startswith("noncore_") or reason.startswith("pnp_rmse_above_") \
            or reason.startswith("inlier_fraction_below_") \
            or reason.startswith("charuco_corners_below_"):
        return "quarantine"
    return "rejected"


def _cube_records(session_root: Path, meta: dict, cube, K_map, D_map,
                  camera_ids: Sequence[int], gripper: int,
                  image_scale: float, policies: Mapping[str, dict],
                  exclude_gripped: bool = True):
    observations, diagnostics = detect_corner_observations(
        root=str(session_root),
        meta=meta,
        cube=cube,
        K_map=K_map,
        D_map=D_map,
        all_cam_ids=list(camera_ids),
        gripper_cam_idx=int(gripper),
        max_err_fixed=3.0,
        max_err_gripper=5.0,
        min_aspect_fixed=0.0,
        min_aspect_gripper=0.35,
        exclude_gripped=exclude_gripped,
        image_scale=float(image_scale),
    )
    observation_map = {
        (int(observation.event), int(observation.cam)): observation
        for observation in observations
    }
    captures = _capture_index(meta)
    records = []
    for quality in diagnostics.get("observation_quality_by_event_camera", []):
        event = int(quality["event_id"])
        camera = int(quality["camera_id"])
        capture = captures[event]
        camera_info = capture["cams"][str(camera)]
        observation = observation_map.get((event, camera))
        recovered = bool(quality.get("recovered_core_observation"))
        selected_by_policy, reason_by_policy, disposition_by_policy = {}, {}, {}
        for name, policy in policies.items():
            selected, reason = _cube_policy_decision(
                quality,
                max_rmse_px=float(policy["cube_max_pnp_rmse_px"]),
                min_inlier_fraction=float(
                    policy["cube_min_inlier_fraction"]),
            )
            selected_by_policy[name] = bool(selected)
            reason_by_policy[name] = reason
            disposition_by_policy[name] = _disposition(
                selected, reason, recovered=recovered)
        record = {
            "observation_id": f"cube:E{event:04d}:cam{camera}",
            "target": "cube",
            "event_id": event,
            "camera_id": camera,
            "set_idx": quality.get("set_idx"),
            # grasp 모델(T_gripper_cube[grasp_idx])용 -- cube_gripped 이벤트면
            # 프로토콜 버전과 무관하게 기록한다 (legacy 변환 세션의 session1 쥔
            # 큐브도 05의 --include_gripped_cube 로 쓰이려면 필요).
            "grasp_idx": (
                int(capture.get("grasp_id"))
                if (capture.get("cube_gripped")
                    and capture.get("grasp_id") is not None)
                else None
            ),
            "capture_block": capture.get("capture_gate", {}).get(
                "capture_block"),
            "cube_gripped": bool(capture.get("cube_gripped")),
            "protocol_version": capture.get("protocol_version"),
            "phase": capture.get("phase"),
            "placement_id": capture.get("placement_id"),
            "view_index": capture.get("view_index"),
            "analysis_group_id": capture.get("analysis_group_id"),
            "split_unit_id": capture.get("split_unit_id"),
            "image_path": str(camera_info.get("rgb_path", "")),
            "corner_count": int(
                len(observation.image_points) if observation is not None else 0),
            "object_points": (
                np.asarray(observation.object_points, dtype=np.float64).tolist()
                if observation is not None else []),
            "image_points": (
                np.asarray(observation.image_points, dtype=np.float64).tolist()
                if observation is not None else []),
            "selected_by_policy": selected_by_policy,
            "reason_by_policy": reason_by_policy,
            "disposition_by_policy": disposition_by_policy,
            **dict(quality),
        }
        records.append(_jsonable(record))
    return records, diagnostics


def _board_records(session_root: Path, meta: dict, board_cfg, image_scale: float,
                   policies: Mapping[str, dict]) -> List[dict]:
    detector = CharucoTarget(board_cfg)
    output = []
    for capture, camera, camera_info in _saved_camera_rows(meta):
        event = int(capture["event_id"])
        set_index = get_capture_set_index(capture)
        relative = str(camera_info.get("rgb_path", ""))
        path = session_root / relative
        image = cv2.imread(str(path)) if relative else None
        status = "pending"
        object_points = np.empty((0, 3), dtype=np.float64)
        image_points = np.empty((0, 2), dtype=np.float64)
        charuco_ids: List[int] = []
        if image is None:
            status = "unreadable_image" if relative else "missing_rgb_path"
        else:
            if float(image_scale) != 1.0:
                interpolation = (
                    cv2.INTER_AREA if float(image_scale) < 1.0
                    else cv2.INTER_CUBIC)
                image = cv2.resize(
                    image, None, fx=float(image_scale), fy=float(image_scale),
                    interpolation=interpolation)
            try:
                corners, ids, count, _, _ = detector.detect(image)
            except Exception:
                corners, ids, count = None, None, 0
                status = "detection_error"
            if status == "pending" and (corners is None or ids is None or count < 4):
                status = "no_charuco_or_below_4_corners"
            elif status == "pending":
                try:
                    matched_object, matched_image = detector.board.matchImagePoints(
                        corners, ids)
                except Exception:
                    matched_object, matched_image = None, None
                if matched_object is None or matched_image is None \
                        or len(matched_object) < 4:
                    status = "charuco_match_failed"
                else:
                    object_points = np.asarray(
                        matched_object, dtype=np.float64).reshape(-1, 3)
                    image_points = (
                        np.asarray(matched_image, dtype=np.float64).reshape(-1, 2)
                        / float(image_scale)
                    )
                    charuco_ids = [int(value) for value in np.asarray(ids).reshape(-1)]
                    status = "accepted"

        selected_by_policy, reason_by_policy, disposition_by_policy = {}, {}, {}
        for name, policy in policies.items():
            minimum = int(policy["board_min_charuco_corners"])
            selected = bool(status == "accepted" and len(image_points) >= minimum)
            reason = (
                "selected" if selected else
                f"charuco_corners_below_{minimum}"
                if status == "accepted" else status
            )
            selected_by_policy[name] = selected
            reason_by_policy[name] = reason
            disposition_by_policy[name] = _disposition(selected, reason)
        output.append(_jsonable({
            "observation_id": f"board:E{event:04d}:cam{camera}",
            "target": "board",
            "event_id": event,
            "camera_id": camera,
            "set_idx": None if set_index is None else int(set_index),
            "grasp_idx": (
                int(capture.get("grasp_id"))
                if (capture.get("protocol_version") in PHASE_AWARE_CAPTURE_PROTOCOLS
                    and capture.get("cube_gripped")
                    and capture.get("grasp_id") is not None)
                else None
            ),
            "capture_block": capture.get("capture_gate", {}).get(
                "capture_block"),
            "cube_gripped": bool(capture.get("cube_gripped")),
            "protocol_version": capture.get("protocol_version"),
            "phase": capture.get("phase"),
            "placement_id": capture.get("placement_id"),
            "view_index": capture.get("view_index"),
            "analysis_group_id": capture.get("analysis_group_id"),
            "split_unit_id": capture.get("split_unit_id"),
            "image_path": relative,
            "corner_count": int(len(image_points)),
            "charuco_ids": charuco_ids,
            "object_points": object_points.tolist(),
            "image_points": image_points.tolist(),
            "status": status,
            "selected_by_policy": selected_by_policy,
            "reason_by_policy": reason_by_policy,
            "disposition_by_policy": disposition_by_policy,
        }))
    return output


def _event_summaries(meta: dict, records: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[int(record["event_id"])].append(record)
    output = []
    for capture in meta.get("captures", []):
        event = int(capture.get("event_id", -1))
        if event < 0:
            continue
        event_records = grouped[event]
        summary = {
            "event_id": event,
            "set_idx": get_capture_set_index(capture),
            "capture_block": capture.get("capture_gate", {}).get(
                "capture_block"),
            "cube_gripped": bool(capture.get("cube_gripped")),
            "protocol_version": capture.get("protocol_version"),
            "phase": capture.get("phase"),
            "placement_id": capture.get("placement_id"),
            "view_index": capture.get("view_index"),
            "analysis_group_id": capture.get("analysis_group_id"),
            "split_unit_id": capture.get("split_unit_id"),
        }
        for policy in ("standard", "strict"):
            cube = [record for record in event_records
                    if record["target"] == "cube"
                    and record["selected_by_policy"][policy]]
            board = [record for record in event_records
                     if record["target"] == "board"
                     and record["selected_by_policy"][policy]]
            if cube and board:
                status = "selected_cube_and_board"
            elif cube:
                status = "selected_cube_only"
            elif board:
                status = "board_only"
            else:
                status = "rejected"
            summary[f"{policy}_status"] = status
            summary[f"{policy}_cube_camera_ids"] = sorted(
                int(record["camera_id"]) for record in cube)
            summary[f"{policy}_board_camera_ids"] = sorted(
                int(record["camera_id"]) for record in board)
        output.append(_jsonable(summary))
    return output


def _summary(records: Sequence[dict], events: Sequence[dict]) -> dict:
    output = {
        "observations_total": int(len(records)),
        "events_total": int(len(events)),
    }
    for policy in ("standard", "strict"):
        for target in ("cube", "board"):
            subset = [record for record in records if record["target"] == target]
            output[f"{policy}_{target}_selected"] = int(sum(
                bool(record["selected_by_policy"][policy]) for record in subset))
        output[f"{policy}_event_status_counts"] = dict(sorted(Counter(
            str(event[f"{policy}_status"]) for event in events).items()))
    cube = [record for record in records if record["target"] == "cube"]
    output["cube_recovered_standard"] = int(sum(
        record["disposition_by_policy"]["standard"] == "recovered"
        for record in cube))
    output["cube_standard_disposition_counts"] = dict(sorted(Counter(
        record["disposition_by_policy"]["standard"] for record in cube).items()))
    output["cube_standard_reason_counts"] = dict(sorted(Counter(
        record["reason_by_policy"]["standard"] for record in cube
        if not record["selected_by_policy"]["standard"]).items()))
    for target in ("cube", "board"):
        subset = [record for record in records if record["target"] == target]
        output[f"{target}_strict_additional_reason_counts"] = dict(sorted(Counter(
            record["reason_by_policy"]["strict"] for record in subset
            if record["selected_by_policy"]["standard"]
            and not record["selected_by_policy"]["strict"]).items()))
    board = [record for record in records if record["target"] == "board"]
    output["board_standard_reason_counts"] = dict(sorted(Counter(
        record["reason_by_policy"]["standard"] for record in board
        if not record["selected_by_policy"]["standard"]).items()))
    return output


def _retake_rows(events: Sequence[dict]) -> List[dict]:
    rows = []
    for event in events:
        # Gripped-cube events are excluded from the current calibration cube
        # contract, so their board-only status is intentional, not a retake.
        if event.get("cube_gripped"):
            continue
        if event["standard_cube_camera_ids"]:
            continue
        has_board = bool(event["standard_board_camera_ids"])
        rows.append({
            "event_id": int(event["event_id"]),
            "set_idx": event.get("set_idx"),
            "capture_block": event.get("capture_block"),
            "priority": "medium" if has_board else "high",
            "reason": (
                "missing_standard_core_cube; board observation remains usable"
                if has_board else
                "neither standard core cube nor board observation is usable"
            ),
            "board_camera_ids": ",".join(
                f"cam{int(value)}"
                for value in event["standard_board_camera_ids"]),
        })
    return rows


def _csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _write_csv(path: Path, rows: Sequence[dict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(fields), extrasaction="ignore",
            lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _flat_observation_row(record: Mapping) -> dict:
    return {
        "observation_id": record["observation_id"],
        "target": record["target"],
        "event_id": record["event_id"],
        "camera_id": record["camera_id"],
        "set_idx": record.get("set_idx"),
        "capture_block": record.get("capture_block"),
        "image_path": record.get("image_path"),
        "corner_count": record.get("corner_count", 0),
        "marker_ids": record.get("marker_ids", record.get("charuco_ids", [])),
        "observed_faces": record.get("observed_faces", []),
        "quality_tier": record.get("quality_tier"),
        "pnp_rmse_px": record.get("pnp_rmse_px"),
        "pnp_inlier_fraction": record.get("pnp_inlier_fraction"),
        "detection_method": record.get("detection_method"),
        "standard_selected": record["selected_by_policy"]["standard"],
        "standard_disposition": record["disposition_by_policy"]["standard"],
        "standard_reason": record["reason_by_policy"]["standard"],
        "strict_selected": record["selected_by_policy"]["strict"],
        "strict_disposition": record["disposition_by_policy"]["strict"],
        "strict_reason": record["reason_by_policy"]["strict"],
    }


OBSERVATION_CSV_FIELDS = (
    "observation_id", "target", "event_id", "camera_id", "set_idx",
    "capture_block", "image_path", "corner_count", "marker_ids",
    "observed_faces", "quality_tier", "pnp_rmse_px",
    "pnp_inlier_fraction", "detection_method", "standard_selected",
    "standard_disposition", "standard_reason", "strict_selected",
    "strict_disposition", "strict_reason",
)


def _draw_review_overlay(path: Path, session_root: Path,
                         records: Sequence[dict], meta: dict,
                         max_panels: int = 48) -> int:
    review = [
        record for record in records
        if record["target"] == "cube"
        and record["disposition_by_policy"]["standard"] != "selected"
    ][:int(max_panels)]
    if not review:
        return 0
    capture_map = _capture_index(meta)
    panel_w, image_h, header_h = 360, 203, 58
    panels = []
    colors = {
        "recovered": (35, 205, 70),
        "quarantine": (20, 175, 245),
        "rejected": (40, 55, 225),
    }
    for record in review:
        image = cv2.imread(str(session_root / record["image_path"]))
        if image is None:
            image = np.full((image_h, panel_w, 3), 40, dtype=np.uint8)
        scale_x = panel_w / float(image.shape[1])
        scale_y = image_h / float(image.shape[0])
        resized = cv2.resize(image, (panel_w, image_h), interpolation=cv2.INTER_AREA)
        disposition = record["disposition_by_policy"]["standard"]
        color = colors[disposition]
        points = np.asarray(record.get("image_points", []), dtype=np.float64).reshape(-1, 2)
        for start in range(0, len(points), 4):
            polygon = points[start:start + 4]
            if len(polygon) != 4:
                continue
            polygon = np.column_stack([
                polygon[:, 0] * scale_x, polygon[:, 1] * scale_y,
            ]).round().astype(np.int32)
            cv2.polylines(resized, [polygon], True, color, 2, cv2.LINE_AA)
        if not len(points):
            camera_info = capture_map[int(record["event_id"])]["cams"][
                str(int(record["camera_id"]))]
            for marker in camera_info.get("markers", []) or []:
                polygon = np.asarray(
                    marker.get("corners_2d", []), dtype=np.float64).reshape(-1, 2)
                if len(polygon) != 4:
                    continue
                polygon = np.column_stack([
                    polygon[:, 0] * scale_x, polygon[:, 1] * scale_y,
                ]).round().astype(np.int32)
                cv2.polylines(
                    resized, [polygon], True, (220, 80, 220), 1, cv2.LINE_AA)
        header = np.full((header_h, panel_w, 3), 25, dtype=np.uint8)
        title = (
            f"E{int(record['event_id']):02d}/cam{int(record['camera_id'])}  "
            f"{disposition.upper()}"
        )
        cv2.putText(header, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.54, color, 2, cv2.LINE_AA)
        reason = str(record["reason_by_policy"]["standard"])
        cv2.putText(header, reason[:49], (8, 47), cv2.FONT_HERSHEY_SIMPLEX,
                    0.40, (225, 225, 225), 1, cv2.LINE_AA)
        panels.append(np.vstack([header, resized]))
    columns = min(4, len(panels))
    rows = int(math.ceil(len(panels) / columns))
    blank = np.full_like(panels[0], 25)
    panels.extend([blank] * (rows * columns - len(panels)))
    contact = np.vstack([
        np.hstack(panels[row * columns:(row + 1) * columns])
        for row in range(rows)
    ])
    if not cv2.imwrite(str(path), contact, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise RuntimeError(f"failed to write overlay: {path}")
    return len(review)


def _md_list(values: Sequence[int]) -> str:
    return ", ".join(f"cam{int(value)}" for value in values) or "—"


def _write_readme(path: Path, manifest: dict, retakes: Sequence[dict],
                  overlay_count: int) -> None:
    summary = manifest["summary"]
    policies = manifest["policies"]
    events = manifest["events"]
    records = manifest["observations"]
    cube_review = [
        record for record in records
        if record["target"] == "cube"
        and not record["selected_by_policy"]["standard"]
    ]
    recovered = [
        record for record in records
        if record["target"] == "cube"
        and record["disposition_by_policy"]["standard"] == "recovered"
    ]
    board_review = [
        record for record in records
        if record["target"] == "board"
        and not record["selected_by_policy"]["standard"]
    ]
    strict_additional = [
        record for record in records
        if record["selected_by_policy"]["standard"]
        and not record["selected_by_policy"]["strict"]
    ]
    lines = [
        "# 04 Post-capture Observation Filter",
        "",
        f"- Session: `{manifest['source']['session_root']}`",
        f"- 생성 시각(UTC): `{manifest['generated_utc']}`",
        "- 원본 RGB/meta/intrinsics: **수정하지 않음**",
        "- Calibration 입력: 재검출 결과의 native-pixel 2D corner를 manifest에 고정",
        "- Cube corner refinement: "
        f"`{manifest['scope'].get('cube_corner_refinement_mode', 'apriltag')}`",
        "",
        "## 결과 요약",
        "",
        "| 항목 | Standard | Strict |",
        "|---|---:|---:|",
        f"| Cube 선택 관측 | {summary['standard_cube_selected']} | {summary['strict_cube_selected']} |",
        f"| Board 선택 관측 | {summary['standard_board_selected']} | {summary['strict_board_selected']} |",
        f"| Cube 재검출 복구 | {summary['cube_recovered_standard']} | {sum(r['disposition_by_policy']['strict'] == 'recovered' for r in recovered)} |",
        "",
        "Cube standard disposition: " + ", ".join(
            f"`{key}` {value}" for key, value in
            summary["cube_standard_disposition_counts"].items()),
        "",
        "## 정책",
        "",
        "- `standard`: cube는 서로 다른 방향의 non-coplanar face 2개 이상, "
        f"positive-depth PnP, RMSE ≤ {policies['standard']['cube_max_pnp_rmse_px']} px. "
        f"Board는 ChArUco corner ≥ {policies['standard']['board_min_charuco_corners']}.",
        "- `strict`: 같은 기하 조건에 "
        f"RMSE ≤ {policies['strict']['cube_max_pnp_rmse_px']} px, "
        f"inlier fraction ≥ {policies['strict']['cube_min_inlier_fraction']}, "
        f"Board corner ≥ {policies['strict']['board_min_charuco_corners']}.",
        "- `recovered`: 기본 검출은 core가 아니었지만 offline 재검출로 standard를 통과.",
        "- `quarantine`: corner/PnP는 있으나 face 또는 임계값 부족. 자동 calibration에서 제외.",
        "- `rejected`: marker 미검출, 영상 오류, PnP 실패/초과 등으로 frozen corner가 없음.",
        "",
        "## 시각 검토",
        "",
        f"Standard의 recovered/quarantine/rejected cube 관측 {overlay_count}개를 한 장에 모았습니다.",
        "초록은 recovered, 주황은 quarantine, 빨강은 rejected입니다. Rejected에 보라색 선이 있으면 촬영 당시 meta에 저장된 구형 검출 corner입니다.",
        "",
        f"![04 review overlay]({OUTPUT_NAMES['overlay']})",
        "",
        "## Standard 제외 cube 관측",
        "",
        "| Event/camera | 결과 | Marker IDs | Faces | RMSE | 이유 |",
        "|---|---|---|---|---:|---|",
    ]
    for record in sorted(cube_review, key=lambda item: (
            int(item["event_id"]), int(item["camera_id"]))):
        rmse = record.get("pnp_rmse_px")
        lines.append(
            f"| E{int(record['event_id']):02d}/cam{int(record['camera_id'])} "
            f"| {record['disposition_by_policy']['standard']} "
            f"| {', '.join(map(str, record.get('marker_ids', []))) or '—'} "
            f"| {', '.join(record.get('observed_faces', [])) or '—'} "
            f"| {float(rmse):.3f} px " if rmse is not None else
            f"| E{int(record['event_id']):02d}/cam{int(record['camera_id'])} "
            f"| {record['disposition_by_policy']['standard']} "
            f"| {', '.join(map(str, record.get('marker_ids', []))) or '—'} "
            f"| {', '.join(record.get('observed_faces', [])) or '—'} | — "
        )
        lines[-1] += f"| `{record['reason_by_policy']['standard']}` |"
    lines.extend([
        "",
        "## Standard 제외 board 관측",
        "",
        "| Event/camera | Corner 수 | 상태 | 이유 |",
        "|---|---:|---|---|",
    ])
    for record in sorted(board_review, key=lambda item: (
            int(item["event_id"]), int(item["camera_id"]))):
        lines.append(
            f"| E{int(record['event_id']):02d}/cam{int(record['camera_id'])} "
            f"| {int(record['corner_count'])} | {record['status']} "
            f"| `{record['reason_by_policy']['standard']}` |")
    if not board_review:
        lines.append("| — | — | — | 제외 관측 없음 |")
    lines.extend([
        "",
        "## Strict에서 추가 제외되는 관측",
        "",
        "Standard는 통과했지만 strict RMSE/inlier/board-corner 기준에서 추가 제외되는 관측입니다.",
        "",
        "| Target | Event/camera | Corner 수 | RMSE | Inlier | 이유 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for record in sorted(strict_additional, key=lambda item: (
            str(item["target"]), int(item["event_id"]), int(item["camera_id"]))):
        rmse = record.get("pnp_rmse_px")
        inlier = record.get("pnp_inlier_fraction")
        lines.append(
            f"| {record['target']} "
            f"| E{int(record['event_id']):02d}/cam{int(record['camera_id'])} "
            f"| {int(record['corner_count'])} "
            f"| {float(rmse):.3f} px " if rmse is not None else
            f"| {record['target']} "
            f"| E{int(record['event_id']):02d}/cam{int(record['camera_id'])} "
            f"| {int(record['corner_count'])} | — "
        )
        lines[-1] += (
            f"| {float(inlier):.3f} " if inlier is not None else "| — ")
        lines[-1] += f"| `{record['reason_by_policy']['strict']}` |"
    if not strict_additional:
        lines.append("| — | — | — | — | — | 추가 제외 없음 |")
    lines.extend([
        "",
        "## 재촬영 후보",
        "",
        "현재 calibration 계약에서 cube를 사용하지 않는 gripped-cube event는 재촬영 후보에서 제외했습니다.",
        "",
        "| Event | Set | 우선순위 | 남아 있는 board cameras | 이유 |",
        "|---:|---:|---|---|---|",
    ])
    for row in retakes:
        lines.append(
            f"| {int(row['event_id']):02d} | {row.get('set_idx', '—')} "
            f"| {row['priority']} | {row['board_camera_ids'] or '—'} | {row['reason']} |")
    if not retakes:
        lines.append("| — | — | — | — | 재촬영 후보 없음 |")
    lines.extend([
        "",
        "## Event별 선택 결과",
        "",
        "| Event | Set | Block | Standard | Cube cams | Board cams | Strict |",
        "|---:|---:|---|---|---|---|---|",
    ])
    for event in events:
        lines.append(
            f"| {int(event['event_id']):02d} | {event.get('set_idx', '—')} "
            f"| {event.get('capture_block') or '—'} | {event['standard_status']} "
            f"| {_md_list(event['standard_cube_camera_ids'])} "
            f"| {_md_list(event['standard_board_camera_ids'])} "
            f"| {event['strict_status']} |")
    lines.extend([
        "",
        "## Calibration에서 frozen manifest 사용",
        "",
        "```bash",
        "python3 05_calibrate.py \\",
        f"  --root_folder {manifest['source']['session_root']} \\",
        f"  --intrinsics_dir {manifest['source']['intrinsics_dir']} \\",
        f"  --observation-manifest {path.parent / OUTPUT_NAMES['manifest']} \\",
        "  --observation-filter-policy standard",
        "```",
        "",
        "`strict` 비교 시 마지막 값만 `strict`로 바꾸면 됩니다. Manifest를 사용할 때 05 calibration은 detector를 다시 실행하지 않으며, meta/intrinsics/선택 RGB의 SHA-256이 달라지면 중단합니다.",
        "",
        "## 산출물",
        "",
        f"- `{OUTPUT_NAMES['manifest']}`: frozen 2D/3D corner, 정책, source SHA-256",
        f"- `{OUTPUT_NAMES['selected']}`: standard 선택 관측",
        f"- `{OUTPUT_NAMES['quarantine']}`: 복구되지 않은 저품질/planar 관측",
        f"- `{OUTPUT_NAMES['rejected']}`: 검출/PnP 실패 관측",
        f"- `{OUTPUT_NAMES['retake']}`: event 단위 재촬영 후보",
        f"- `{OUTPUT_NAMES['overlay']}`: 육안 검토 contact sheet",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_filter(args) -> dict:
    session_root = Path(args.session_root).resolve()
    intrinsics_dir = Path(args.intrinsics_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve() if args.output_dir
        else _default_output_dir(session_root).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = session_root / "meta.json"
    with meta_path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    capture_protocol = meta.get("capture_config", {}).get("capture_protocol")
    is_final_protocol = capture_protocol in PHASE_AWARE_CAPTURE_PROTOCOLS
    if is_final_protocol:
        selected_captures = [
            capture for capture in meta.get("captures", [])
            if capture.get("protocol_version") == capture_protocol
            and capture.get("selected_for_analysis") is True
        ]
        expected_count = int(
            meta.get("capture_config", {}).get(
                "expected_event_count",
                45 if capture_protocol == PROTOCOL_COMPOSITE_RIG_45 else 0,
            )
        )
        planned_ids = [
            str(capture.get("planned_event_id") or "")
            for capture in selected_captures
        ]
        if (
            expected_count <= 0
            or len(selected_captures) != expected_count
            or len(set(planned_ids)) != expected_count
            or "" in planned_ids
        ):
            raise RuntimeError(
                "phase-aware protocol filter requires exactly one selected capture "
                f"for each of {expected_count} planned events; "
                f"found {len(selected_captures)}"
            )
        meta = dict(meta)
        meta["captures"] = selected_captures
    camera_ids = sorted({
        int(camera) for capture in meta.get("captures", [])
        for camera in capture.get("cams", {})
    })
    gripper = int(meta["gripper_cam_idx"])
    K_map, D_map, intrinsics = {}, {}, {}
    for camera in camera_ids:
        K_map[camera], D_map[camera], _ = load_intrinsics_with_depth_scale(
            str(intrinsics_dir), camera)
        intrinsic_path = (intrinsics_dir / f"cam{camera}.npz").resolve()
        intrinsics[str(camera)] = {
            "path": str(intrinsic_path),
            "sha256": _sha256(intrinsic_path),
        }
    cube_cfg, cube_cfg_source = resolve_cube_config_for_run(
        str(session_root), cube_config_json=getattr(args, "cube_config", None),
        default_cfg=get_default_cube_config())
    board_cfg, board_cfg_source = load_charuco_config_from_meta(
        str(session_root), require_frozen=True)
    cube = AprilTagCubeTarget(
        cube_cfg,
        corner_refinement_mode=str(args.cube_corner_refinement_mode),
    )
    policies = {
        "standard": {
            "selection_stage": "post_capture_before_split_and_calibration_fit",
            "frozen_native_pixel_corners": True,
            "cube_geometry": "nonplanar_multiface_with_at_least_2_faces",
            "cube_max_pnp_rmse_px": float(args.standard_cube_rmse_px),
            "cube_min_inlier_fraction": float(args.standard_min_inlier_fraction),
            "board_min_charuco_corners": int(args.standard_board_min_corners),
        },
        "strict": {
            "selection_stage": "post_capture_before_split_and_calibration_fit",
            "frozen_native_pixel_corners": True,
            "cube_geometry": "nonplanar_multiface_with_at_least_2_faces",
            "cube_max_pnp_rmse_px": float(args.strict_cube_rmse_px),
            "cube_min_inlier_fraction": float(args.strict_min_inlier_fraction),
            "board_min_charuco_corners": int(args.strict_board_min_corners),
        },
    }
    cube_records, cube_diagnostics = _cube_records(
        session_root, meta, cube, K_map, D_map, camera_ids, gripper,
        float(args.image_scale), policies,
        exclude_gripped=(
            False if is_final_protocol
            else not bool(getattr(args, "include_gripped_cube", False))
        ))
    board_records = _board_records(
        session_root, meta, board_cfg, float(args.image_scale), policies)
    records = sorted(cube_records + board_records, key=lambda record: (
        int(record["event_id"]), int(record["camera_id"]), record["target"]))
    events = _event_summaries(meta, records)
    source_images = _image_provenance(session_root, meta)
    cube_config = cube_config_to_dict(cube_cfg)
    board_config = charuco_config_to_dict(board_cfg)
    manifest = {
        "schema": POST_CAPTURE_MANIFEST_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "opencv_version": cv2.__version__,
        },
        "source": {
            "session_root": str(session_root),
            "intrinsics_dir": str(intrinsics_dir),
            "meta_json": {"path": str(meta_path), "sha256": _sha256(meta_path)},
            "intrinsics": intrinsics,
            "images": source_images,
            "cube_config_source": cube_cfg_source,
            "cube_config": cube_config,
            "cube_config_sha256": _canonical_sha256(cube_config),
            "charuco_board_config_source": board_cfg_source,
            "charuco_board_config": board_config,
            "charuco_board_config_sha256": _canonical_sha256(board_config),
            "charuco_board_topology": charuco_topology(board_cfg),
        },
        "policies": policies,
        "scope": {
            "cube": "saved non-gripped RGB frames; matches current calibration contract",
            "board": "every saved RGB frame",
            "raw_capture_modified": False,
            "image_scale": float(args.image_scale),
            "board_corner_refinement_mode": "CORNER_REFINE_NONE",
            "cube_corner_refinement_mode": str(
                args.cube_corner_refinement_mode),
        },
        "summary": {},
        "events": events,
        "observations": records,
        "cube_redetection_diagnostics": cube_diagnostics,
    }
    manifest["summary"] = _summary(records, events)
    manifest_path = output_dir / OUTPUT_NAMES["manifest"]
    manifest_path.write_text(
        json.dumps(_jsonable(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")

    flat = [_flat_observation_row(record) for record in records]
    standard_selected = [row for row in flat if row["standard_selected"]]
    quarantine = [row for row in flat
                  if row["standard_disposition"] == "quarantine"]
    rejected = [row for row in flat
                if row["standard_disposition"] == "rejected"]
    _write_csv(output_dir / OUTPUT_NAMES["selected"], standard_selected,
               OBSERVATION_CSV_FIELDS)
    _write_csv(output_dir / OUTPUT_NAMES["quarantine"], quarantine,
               OBSERVATION_CSV_FIELDS)
    _write_csv(output_dir / OUTPUT_NAMES["rejected"], rejected,
               OBSERVATION_CSV_FIELDS)
    retakes = _retake_rows(events)
    _write_csv(
        output_dir / OUTPUT_NAMES["retake"], retakes,
        ("event_id", "set_idx", "capture_block", "priority", "reason",
         "board_camera_ids"),
    )
    overlay_count = _draw_review_overlay(
        output_dir / OUTPUT_NAMES["overlay"], session_root, records, meta,
        max_panels=int(args.max_overlay_panels))
    _write_readme(
        output_dir / OUTPUT_NAMES["readme"], manifest, retakes, overlay_count)
    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "summary": manifest["summary"],
        "retake_count": int(len(retakes)),
        "overlay_count": int(overlay_count),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-root", default="data/session02_NOUSE_session04_0814/calib_train")
    parser.add_argument("--intrinsics-dir", default="intrinsics")
    parser.add_argument("--output-dir")
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument(
        "--cube-config", default=None,
        help="큐브 기하 JSON 경로로 프로젝트 기본(도면 nominal) 대신 명시적으로 override. "
             "resolve_cube_config_for_run()이 지원하는 그 경로 그대로 전달됨 "
             "(예: targets/gt_cube/cube_config.json).",
    )
    parser.add_argument(
        "--cube-corner-refinement-mode",
        choices=CUBE_CORNER_REFINEMENT_MODES,
        default="apriltag",
        help=("Cube corner measurement model. 'apriltag' preserves the "
              "canonical detector; 'line_intersection' refits the four black "
              "border edges and intersects them without changing geometry."),
    )
    parser.add_argument("--standard-cube-rmse-px", type=float, default=3.0)
    parser.add_argument("--standard-min-inlier-fraction", type=float, default=0.0)
    parser.add_argument("--standard-board-min-corners", type=int, default=4)
    parser.add_argument("--strict-cube-rmse-px", type=float, default=2.0)
    parser.add_argument("--strict-min-inlier-fraction", type=float, default=0.9)
    parser.add_argument("--strict-board-min-corners", type=int, default=12)
    parser.add_argument("--max-overlay-panels", type=int, default=48)
    parser.add_argument(
        "--include-gripped-cube", action="store_true",
        help=("Also keep cube_gripped=True observations instead of dropping "
              "them (needed when fixed-camera cube views only exist while "
              "the cube is gripped, e.g. UR3 session1). Off by default to "
              "preserve prior runs' behavior."))
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if not np.isfinite(args.image_scale) or args.image_scale <= 0.0:
        raise ValueError("--image-scale must be finite and positive")
    if args.strict_cube_rmse_px > args.standard_cube_rmse_px:
        raise ValueError("strict cube RMSE threshold must not exceed standard")
    if args.strict_min_inlier_fraction < args.standard_min_inlier_fraction:
        raise ValueError("strict inlier threshold must not be below standard")
    if args.strict_board_min_corners < args.standard_board_min_corners:
        raise ValueError("strict board corner threshold must not be below standard")
    result = run_filter(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
