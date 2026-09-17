#!/usr/bin/env python3
"""Run every executable Table 1 ablation on one shared baseline and backend.

The runner implements the contract in ``calibration_pipeline.schema``:

* all rows share robot FK as the hand-eye kinematic backbone;
* ``seq`` is eih-only fitting followed by an e2h-only camera fit, with an
  explicit freeze boundary and no alternating pass;
* ``U`` fits both paths in one objective;
* A3 uses FK hard fixed after a preregistered mechanical frame conversion;
* A4/A5/B1/B2 reuse one train-only corrected-FK cube artifact;
* every row starts from one train-only shared reference state for overlapping
  transforms, before its declared marker/FK treatment is applied;
* A0/A1/A2/A3/A4/A5/B1/B2/B3 are emitted by this runner only;
* A3/A5 use the same canonical corner solver directly, with cube poses frozen;
* A4/B1/B2 use one shared covariance-whitened robust corrected-FK soft factor;
* the primary metric is event-grouped, set-stratified held-out corner
  reprojection with every fitted transform frozen;
* a noise-free A1=A2 sequential-vs-unified test must pass before real data are fitted.

VISION rows use neither a cube-pose FK anchor nor Ridge, SE(3), or another
post-correction. Camera intrinsics and distortion are constants in every stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration_pipeline import se3 as cp
from calibration_pipeline.schema import (
    CORRECTED_FK_FACTOR_CONTRACT,
    ALIGNED_FK_FIXED_ROWS,
    DEFAULT_SPLIT_SEED,
    EVALUATION_COMPARISON_CONTRACT,
    FK_ALIGNMENT_SHARED_ROWS,
    FK_FIXED_CONTRACT,
    FK_FIXED_ROWS,
    INTER_CAMERA_COUPLING_CONTRACT,
    MAIN_ABLATION_CONDITIONS,
    NOISE_FREE_SANITY_TOLERANCES,
    OBJECTIVE_TERM_CONTRACT,
    OPTIMIZATION_STRUCTURE_CONTRACT,
    PRIMARY_METRIC,
    PRIMARY_SPLIT,
    POSE_SOURCE_FK_FIXED,
    POSE_SOURCE_ALIGNED_FK_FIXED,
    RAW_FK_FIXED_ROWS,
    RELATIVE_POSE_REPORTING_CONTRACT,
    RAW_FK_CUBE_CENTER_TO_OBJECT,
    SEQUENTIAL_STAGE_SPECS,
    TASK_POSE_PROXY_LABEL,
    UNIFIED_FREE_VARIABLES,
    VISION_ALIGNED_FK_FIXED_CONTRACT,
    AblationCondition,
    validate_fk_alignment_artifact,
    validate_main_runner_contract,
)
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, inv_T
from calibration_pipeline.runtime import (
    DEFAULT_SESSION_ROOT,
    apply_session_defaults,
    filter_meta_by_set_indices,
    get_capture_set_index,
    load_intrinsics_with_depth_scale,
    resolve_cube_config_for_run,
)
from calibration_pipeline.config import get_default_cube_config
from calibration_pipeline.observations import (
    load_cube_board_pixel_observations,
    load_pixel_observations_from_manifest,
)
from calibration_pipeline.pose_convention import apply_pose_convention_manifest
from calibration_pipeline.fk_alignment import estimate_board_free_fk_cube_artifact
from calibration_pipeline.fk_factor import (
    FKFactorSpec,
    FK_MODE_FACTOR,
    FK_MODE_NONE,
    HUBER_F_SCALE,
    SIGMA_FK_DEG,
    SIGMA_FK_MM,
    diagonal_covariance,
    factorized_objective_cost,
    solve_factorized_fk,
    validate_covariance,
)
from capture_pipeline.waypoint_safety import (
    PHASE_P1,
    PHASE_P2,
    PHASE_P3,
    P3_STATIONARY_SET_INDEX,
    PROTOCOL_COMPOSITE_RIG_45,
    PROTOCOL_SAVED_POSE_REPLAY,
)
from calibration_pipeline.path_evaluation import (
    build_frozen_path_evaluation_mask,
    evaluate_paths_with_frozen_mask,
    not_applicable_path_metrics,
    solve_observed_pose,
    validate_frozen_path_evaluation_mask,
)
from calibration_pipeline.reprojection import (
    FramePruneRefitOptions,
    PixelObs,
    PoseState,
    RESIDUAL_WEIGHTING_MODES,
    SE3Scaling,
    SolverOptions,
    observation_population,
    pose_delta,
    project_points,
    select_frame_prune_subset,
    set_state_transform,
    solve_corner_reprojection,
    state_transform,
    variable_keys,
    visual_objective_cost,
)
from calibration_pipeline.evaluation import (
    REPROJECTION_METRIC_CONTRACT,
    SET_EQUAL_WEIGHT_REPROJECTION_CONTRACT,
    canonical_json_sha256 as _canonical_json_sha256,
    jsonable as _jsonable,
    pixel_reprojection_metrics,
    observations_sha256,
    serialize_state,
    state_sha256,
)

SHARED_BASELINE_SCHEMA = "table1_shared_train_only_baseline_v2"
RUNNABLE_ROWS = ("A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3")
BASELINE_ROWS = ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3")
TRAIN_ONLY_CUBE_EVALUATION_SCHEMA = "train_only_cube_evaluation_pose_v1"
PENDING_ROWS = {
    "A6": {
        "status": "not_run",
        "reason": "independent train-only 6-DoF FK correction labels are unavailable",
        "baseline": "same shared train-only reference state as A4",
    }
}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_population_manifest(observations: Sequence[PixelObs]) -> dict:
    return {
        "sha256": observations_sha256(observations),
        "observations": int(len(observations)),
        "corners": int(sum(
            len(np.asarray(observation.image_points).reshape(-1, 2))
            for observation in observations)),
        "events": sorted({int(observation.event) for observation in observations}),
    }


def _source_data_provenance(
        args, camera_ids: Sequence[int], eligible_observations,
        train_observations, heldout_observations) -> dict:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    meta_path = os.path.abspath(os.path.join(args.root_folder, "meta.json"))
    intrinsics = {}
    for camera_id in sorted(int(value) for value in camera_ids):
        path = os.path.abspath(os.path.join(
            args.intrinsics_dir, f"cam{camera_id}.npz"))
        intrinsics[str(camera_id)] = {
            "path": path,
            "sha256": _file_sha256(path),
        }
    source_names = (
        "apriltag_cube.py", "charuco.py", "config.py", "cross_target.py",
        "cube_detection.py", "evaluation.py", "fk_alignment.py",
        "fk_factor.py", "marker_system.py", "observations.py",
        "path_evaluation.py", "pose_convention.py", "reprojection.py",
        "runtime.py", "schema.py", "se3.py", "table1.py",
    )
    implementation = {}
    for name in source_names:
        path = os.path.join(project_root, "calibration_pipeline", name)
        implementation[os.path.relpath(path, project_root)] = _file_sha256(path)
    provenance = {
        "meta_json": {"path": meta_path, "sha256": _file_sha256(meta_path)},
        "intrinsics": intrinsics,
        "implementation_sha256": implementation,
        "observation_populations": {
            "eligible": _observation_population_manifest(eligible_observations),
            "train": _observation_population_manifest(train_observations),
            "heldout": _observation_population_manifest(heldout_observations),
        },
    }
    observation_manifest = getattr(args, "observation_manifest", None)
    if observation_manifest:
        manifest_path = os.path.abspath(observation_manifest)
        provenance["observation_manifest"] = {
            "path": manifest_path,
            "sha256": _file_sha256(manifest_path),
            "policy": str(getattr(
                args, "observation_filter_policy", "standard")),
        }
    pose_manifest_path = os.path.abspath(os.path.join(
        args.root_folder, "pose_convention_manifest.json"))
    if os.path.isfile(pose_manifest_path):
        provenance["pose_convention_manifest"] = {
            "path": pose_manifest_path,
            "sha256": _file_sha256(pose_manifest_path),
        }
    return provenance


def validate_source_data_provenance(provenance: Mapping) -> None:
    """Fail closed when a stored artifact no longer matches files on disk."""
    entries = [provenance.get("meta_json", {})]
    entries.extend(provenance.get("intrinsics", {}).values())
    if provenance.get("pose_convention_manifest"):
        entries.append(provenance["pose_convention_manifest"])
    if provenance.get("observation_manifest"):
        entries.append(provenance["observation_manifest"])
    for entry in entries:
        path = str(entry.get("path", ""))
        expected = str(entry.get("sha256", ""))
        if not path or not expected or not os.path.isfile(path):
            raise ValueError(f"source provenance file is unavailable: {path!r}")
        if _file_sha256(path) != expected:
            raise ValueError(f"stored result is stale: source file changed: {path}")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for relative, expected in provenance.get("implementation_sha256", {}).items():
        path = os.path.join(project_root, str(relative))
        if not os.path.isfile(path) or _file_sha256(path) != str(expected):
            raise ValueError(
                f"stored result is stale: calibration implementation changed: {relative}")
    populations = provenance.get("observation_populations", {})
    if set(populations) != {"eligible", "train", "heldout"}:
        raise ValueError("source provenance lacks calibrated observation populations")
    for name, manifest in populations.items():
        if not str(manifest.get("sha256", "")):
            raise ValueError(f"source provenance lacks {name} observation SHA-256")


def filter_observations(observations: Sequence[PixelObs], condition: AblationCondition,
                        role: Optional[str], gripper: int,
                        registered_cams: Optional[Iterable[int]] = None) -> List[PixelObs]:
    markers = {"board"} if condition.target_set == "board" else (
        {"cube"} if condition.target_set == "cube" else {"board", "cube"})
    cams = None if registered_cams is None else {int(ci) for ci in registered_cams}
    out = []
    for obs in observations:
        if obs.marker not in markers:
            continue
        obs_role = "eih" if int(obs.cam) == int(gripper) else "e2h"
        if role is not None and obs_role != role:
            continue
        if obs_role == "e2h" and cams is not None and int(obs.cam) not in cams:
            continue
        out.append(obs)
    return out




def estimate_board_handeye_initial(eih_board: Sequence[PixelObs], robot_T, K_map, D_map,
                                   gripper: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    poses = []
    for obs in eih_board:
        if int(obs.cam) != int(gripper) or int(obs.event) not in robot_T:
            continue
        T_C_B = solve_observed_pose(obs, K_map, D_map)
        if T_C_B is not None:
            poses.append((int(obs.event), T_C_B))
    if len(poses) < 5:
        raise RuntimeError(f"train-only board hand-eye initialization needs >=5 poses, got {len(poses)}")
    methods = {
        "TSAI": cv2.CALIB_HAND_EYE_TSAI,
        "PARK": cv2.CALIB_HAND_EYE_PARK,
        "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
        "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    candidates = []
    for name, method in methods.items():
        try:
            R, t = cv2.calibrateHandEye(
                [robot_T[e][:3, :3] for e, _ in poses],
                [robot_T[e][:3, 3].reshape(3, 1) for e, _ in poses],
                [T[:3, :3] for _, T in poses],
                [T[:3, 3].reshape(3, 1) for _, T in poses],
                method=method,
            )
            gtc = np.eye(4)
            gtc[:3, :3] = np.asarray(R).reshape(3, 3)
            gtc[:3, 3] = np.asarray(t).reshape(3)
            boards = [robot_T[e] @ gtc @ T for e, T in poses]
            board = cp.robust_se3_average(boards, None)[0]
            deltas = [pose_delta(board, T) for T in boards]
            score = float(np.mean([d[0] for d in deltas])) + 5.0 * float(np.mean([d[1] for d in deltas]))
            if np.isfinite(score):
                candidates.append((score, name, gtc, board))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError("all train-only board hand-eye initializers failed")
    score, name, gtc, board = min(candidates, key=lambda x: x[0])
    return gtc, board, {"method": name, "score_mm_equivalent": score, "n_poses": len(poses)}


def average_visual_target(eih_obs: Sequence[PixelObs], marker: str, gtc: np.ndarray,
                          robot_T, K_map, D_map, gripper: int) -> Dict[int, np.ndarray]:
    by_set: Dict[int, List[np.ndarray]] = defaultdict(list)
    for obs in eih_obs:
        if obs.marker != marker or int(obs.cam) != int(gripper) or obs.set_idx is None:
            continue
        T_C_O = solve_observed_pose(obs, K_map, D_map)
        if T_C_O is None or int(obs.event) not in robot_T:
            continue
        by_set[int(obs.set_idx)].append(robot_T[int(obs.event)] @ gtc @ T_C_O)
    return {s: cp.robust_se3_average(values, None)[0] for s, values in by_set.items() if values}


def average_board_with_gtc(eih_board: Sequence[PixelObs], gtc: np.ndarray,
                           robot_T, K_map, D_map, gripper: int) -> np.ndarray:
    values = []
    for obs in eih_board:
        if obs.marker != "board" or int(obs.cam) != int(gripper):
            continue
        T_C_B = solve_observed_pose(obs, K_map, D_map)
        if T_C_B is not None and int(obs.event) in robot_T:
            values.append(robot_T[int(obs.event)] @ gtc @ T_C_B)
    if not values:
        raise RuntimeError("board initialization unavailable with the supplied gTc")
    return cp.robust_se3_average(values, None)[0]


def estimate_fixed_camera_initials(fixed_obs: Sequence[PixelObs], board: Optional[np.ndarray],
                                   cubes: Mapping[int, np.ndarray], K_map, D_map,
                                   gripper: int) -> Tuple[Dict[int, np.ndarray], dict]:
    values: Dict[int, List[np.ndarray]] = defaultdict(list)
    source: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for obs in fixed_obs:
        if int(obs.cam) == int(gripper):
            continue
        if obs.marker == "board":
            target = board
        elif obs.set_idx is None:
            continue   # 쥔 큐브(grasp 모델) 관측: 초기 카메라 추정엔 안 쓴다 (grasp offset 미지)
        else:
            target = cubes.get(int(obs.set_idx))
        if target is None:
            continue
        T_C_O = solve_observed_pose(obs, K_map, D_map)
        if T_C_O is None:
            continue
        values[int(obs.cam)].append(np.asarray(target, float) @ inv_T(T_C_O))
        source[int(obs.cam)][obs.marker] += 1
    cams = {ci: cp.robust_se3_average(Ts, None)[0] for ci, Ts in values.items() if Ts}
    return cams, {str(ci): dict(source[ci]) for ci in sorted(cams)}


def build_event_split(observations: Sequence[PixelObs], gripper: int, fraction: float,
                      seed: int, min_train_eih_cube_events: int) -> dict:
    events_by_set: Dict[int, set] = defaultdict(set)
    eih_cube_by_set: Dict[int, set] = defaultdict(set)
    for obs in observations:
        if obs.set_idx is None:
            continue
        s = int(obs.set_idx)
        events_by_set[s].add(int(obs.event))
        if obs.marker == "cube" and int(obs.cam) == int(gripper):
            eih_cube_by_set[s].add(int(obs.event))
    rng = np.random.default_rng(int(seed))
    train_events, test_events, eligible_sets = set(), set(), []
    per_set = {}
    dropped = {}
    for s in sorted(events_by_set):
        events = sorted(events_by_set[s])
        eih = set(eih_cube_by_set.get(s, set()))
        if len(eih) < int(min_train_eih_cube_events) + 1:
            dropped[str(s)] = {
                "reason": "insufficient_eih_cube_events_for_train_and_test",
                "n_events": len(events), "n_eih_cube_events": len(eih),
            }
            continue
        # Prefer holding out non-eih events for test before touching scarce
        # eih (gripper-camera cube) events. Randomize within each group so
        # tie-breaks among multiple candidates of the same kind stay seeded,
        # but never spend an eih event on the test split while a non-eih
        # event is available to serve that role instead -- eih coverage is
        # the bottleneck resource for per-set training, so wasting one on a
        # held-out slot it didn't need to fill only starves training for no
        # accuracy benefit (the held-out score does not care which kind of
        # event it consumed).
        non_eih = [e for e in events if e not in eih]
        eih_only = [e for e in events if e in eih]
        rng.shuffle(non_eih)
        rng.shuffle(eih_only)
        order = non_eih + eih_only
        wanted = max(1, int(round(float(fraction) * len(events))))
        chosen = []
        remaining_eih = set(eih)
        for event in order:
            if len(chosen) >= wanted:
                break
            next_remaining = remaining_eih - {event}
            if len(next_remaining) < int(min_train_eih_cube_events):
                continue
            chosen.append(event)
            remaining_eih = next_remaining
        if not chosen:
            dropped[str(s)] = {"reason": "unable_to_allocate_test_event"}
            continue
        eligible_sets.append(s)
        test_events.update(chosen)
        train_for_set = set(events) - set(chosen)
        train_events.update(train_for_set)
        per_set[str(s)] = {
            "train_events": sorted(train_for_set),
            "test_events": sorted(chosen),
            "train_eih_cube_events": sorted(remaining_eih),
        }
    if not eligible_sets:
        raise RuntimeError("no cube set supports the event-stratified split")
    return {
        "strategy": PRIMARY_SPLIT,
        "seed": int(seed),
        "test_fraction_requested": float(fraction),
        "min_train_eih_cube_events": int(min_train_eih_cube_events),
        "eligible_sets": eligible_sets,
        "train_events": sorted(train_events),
        "test_events": sorted(test_events),
        "per_set": per_set,
        "dropped_sets": dropped,
    }


COMPOSITE_PLACEMENT_SPLIT = "composite_rig_p2_placement_grouped_v1"
PHASE_AWARE_CAPTURE_PROTOCOLS = {
    PROTOCOL_COMPOSITE_RIG_45,
    PROTOCOL_SAVED_POSE_REPLAY,
}


def capture_protocol_from_meta(meta: Mapping) -> Optional[str]:
    config = meta.get("capture_config")
    if isinstance(config, Mapping) and config.get("capture_protocol"):
        return str(config["capture_protocol"])
    protocols = {
        str(capture.get("protocol_version"))
        for capture in meta.get("captures", [])
        if capture.get("protocol_version")
    }
    if len(protocols) == 1:
        return protocols.pop()
    return None


def selected_composite_protocol_captures(meta: Mapping) -> List[dict]:
    """Return one transport-valid attempt for every phase-aware planned event."""
    protocol = capture_protocol_from_meta(meta)
    if protocol not in PHASE_AWARE_CAPTURE_PROTOCOLS:
        raise RuntimeError(f"unsupported phase-aware capture protocol: {protocol!r}")
    config = meta.get("capture_config", {})
    expected_count = int(config.get(
        "expected_event_count",
        45 if protocol == PROTOCOL_COMPOSITE_RIG_45 else 0,
    ))
    captures = [
        capture for capture in meta.get("captures", [])
        if capture.get("protocol_version") == protocol
        and capture.get("selected_for_analysis") is True
    ]
    planned_ids = [str(capture.get("planned_event_id", "")) for capture in captures]
    if (
        expected_count <= 0
        or len(captures) != expected_count
        or len(set(planned_ids)) != expected_count
        or "" in planned_ids
    ):
        raise RuntimeError(
            "phase-aware analysis requires exactly one selected transport-valid "
            f"attempt for each of {expected_count} planned events"
        )
    expected_phase_counts = config.get("expected_phase_counts") or {
        PHASE_P1: 15, PHASE_P2: 20, PHASE_P3: 10,
    }
    expected_phase_counts = {
        str(phase): int(count) for phase, count in expected_phase_counts.items()
    }
    actual_phase_counts = {
        phase: sum(capture.get("phase") == phase for capture in captures)
        for phase in expected_phase_counts
    }
    if actual_phase_counts != expected_phase_counts:
        raise RuntimeError(
            "selected phase counts differ from capture_config: expected {}, got {}".format(
                expected_phase_counts, actual_phase_counts)
        )
    p3_sets = {get_capture_set_index(capture) for capture in captures
               if capture.get("phase") == PHASE_P3}
    p2_sets = {get_capture_set_index(capture) for capture in captures
               if capture.get("phase") == PHASE_P2}
    expected_p3_set = (
        P3_STATIONARY_SET_INDEX
        if protocol == PROTOCOL_COMPOSITE_RIG_45
        else len(p2_sets)
    )
    if p3_sets != {expected_p3_set} or expected_p3_set in p2_sets:
        raise RuntimeError(
            "P3 must use independent set_index {}; got {}".format(
                expected_p3_set, sorted(
                    value for value in p3_sets if value is not None))
        )
    return sorted(captures, key=lambda capture: int(capture["event_id"]))


def build_composite_placement_split(meta: Mapping, fraction: float,
                                    seed: int) -> dict:
    """Hold out whole P2 placements; P1/P3 remain calibration-only.

    A placement is the physical target pose. All views belonging to it must
    remain on the same side of the train/held-out boundary.
    """
    fraction = float(fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("test fraction must be strictly between 0 and 1")
    captures = selected_composite_protocol_captures(meta)
    p2_by_placement: Dict[str, List[dict]] = defaultdict(list)
    always_train = []
    for capture in captures:
        phase = capture.get("phase")
        event = int(capture["event_id"])
        if phase == PHASE_P2:
            placement_id = str(capture.get("placement_id") or "")
            if not placement_id:
                raise RuntimeError("P2 capture lacks placement_id")
            p2_by_placement[placement_id].append(capture)
        else:
            always_train.append(event)
    config = meta.get("capture_config", {})
    views_per_placement = int(config.get("p2_views_per_placement", 2))
    expected_p2_events = int((config.get("expected_phase_counts") or {}).get(
        PHASE_P2, 20
    ))
    if views_per_placement <= 0 or expected_p2_events % views_per_placement:
        raise RuntimeError("invalid P2 event/view count in capture_config")
    expected_placements = expected_p2_events // views_per_placement
    if len(p2_by_placement) != expected_placements:
        raise RuntimeError(
            "P2 split requires {} placement groups, got {}".format(
                expected_placements, len(p2_by_placement))
        )
    set_by_placement = {}
    for placement_id, group in sorted(p2_by_placement.items()):
        views = {int(capture.get("view_index", -1)) for capture in group}
        sets = {get_capture_set_index(capture) for capture in group}
        expected_views = set(range(views_per_placement))
        if (len(group) != views_per_placement or views != expected_views
                or len(sets) != 1 or None in sets):
            raise RuntimeError(
                "{} must contain views {} under one set_index".format(
                    placement_id, sorted(expected_views))
            )
        set_by_placement[placement_id] = int(next(iter(sets)))
    if len(set(set_by_placement.values())) != expected_placements:
        raise RuntimeError("each P2 placement must have a distinct set_index")

    placements = sorted(p2_by_placement)
    rng = np.random.default_rng(int(seed))
    shuffled = list(placements)
    rng.shuffle(shuffled)
    heldout_count = max(1, min(len(placements) - 1,
                               int(round(fraction * len(placements)))))
    heldout_placements = set(shuffled[:heldout_count])
    train_placements = set(placements) - heldout_placements
    train_events = set(always_train)
    test_events = set()
    per_set = {}
    for placement_id in placements:
        events = sorted(int(capture["event_id"])
                        for capture in p2_by_placement[placement_id])
        role = "heldout" if placement_id in heldout_placements else "train"
        destination = test_events if role == "heldout" else train_events
        destination.update(events)
        per_set[str(set_by_placement[placement_id])] = {
            "placement_id": placement_id,
            "split_role": role,
            "events": events,
        }
    if train_events & test_events:
        raise RuntimeError("placement-grouped split produced event leakage")
    return {
        "strategy": COMPOSITE_PLACEMENT_SPLIT,
        "seed": int(seed),
        "test_fraction_requested": fraction,
        "split_unit": "P2 placement_id",
        "p1_p3_role": "always_train_calibration_only",
        "eligible_sets": sorted(set_by_placement.values()),
        "train_events": sorted(train_events),
        "test_events": sorted(test_events),
        "train_placement_ids": sorted(train_placements),
        "heldout_placement_ids": sorted(heldout_placements),
        "always_train_events": sorted(always_train),
        "per_set": per_set,
        "dropped_sets": {},
    }


def build_protocol_event_split(observations: Sequence[PixelObs], meta: Mapping,
                               gripper: int, fraction: float, seed: int,
                               min_train_eih_cube_events: int) -> dict:
    if capture_protocol_from_meta(meta) in PHASE_AWARE_CAPTURE_PROTOCOLS:
        return build_composite_placement_split(meta, fraction, seed)
    return build_event_split(
        observations, gripper, fraction, seed, min_train_eih_cube_events)


def support_report(observations: Sequence[PixelObs], gripper: int) -> dict:
    counts: Dict[Tuple[int, str, str], Dict[str, int]] = defaultdict(
        lambda: {"observations": 0, "corners": 0, "events": set()})
    for obs in observations:
        if obs.set_idx is None:
            continue
        role = "eih" if int(obs.cam) == int(gripper) else "e2h"
        key = (int(obs.set_idx), obs.marker, role)
        counts[key]["observations"] += 1
        counts[key]["corners"] += len(np.asarray(obs.image_points).reshape(-1, 2))
        counts[key]["events"].add(int(obs.event))
    return {
        f"set{s}:{marker}:{role}": {
            "observations": v["observations"], "corners": v["corners"],
            "events": len(v["events"]),
        }
        for (s, marker, role), v in sorted(counts.items())
    }


def build_shared_reference_state(
        train_obs: Sequence[PixelObs], gripper: int, robot_T, K_map, D_map,
        board_gtc: np.ndarray, board_initial: np.ndarray,
        visual_cubes: Mapping[int, np.ndarray]) -> Tuple[PoseState, dict]:
    """Build one train-only initializer before any ablation treatment is applied.

    The shared prefit may use both marker families, but it is never scored and
    never sees held-out events.  Every row receives byte-identical values for
    transforms it shares with another row.  A row then changes only its
    declared target presence, FK pose source, and optimization freeze mask.
    """
    all_marker_condition = next(
        condition for condition in MAIN_ABLATION_CONDITIONS
        if condition.row == "A1")
    relevant = filter_observations(
        train_obs, all_marker_condition, None, gripper)
    fixed = filter_observations(
        relevant, all_marker_condition, "e2h", gripper)
    cams, cam_sources = estimate_fixed_camera_initials(
        fixed,
        np.asarray(board_initial, dtype=np.float64),
        visual_cubes,
        K_map,
        D_map,
        gripper,
    )
    if not cams:
        raise RuntimeError("shared train-only baseline cannot initialize a fixed camera")
    state = PoseState(
        cams=cams,
        gtc=np.asarray(board_gtc, dtype=np.float64),
        board=np.asarray(board_initial, dtype=np.float64),
        cubes={int(s): np.asarray(T, dtype=np.float64)
               for s, T in visual_cubes.items()},
    )
    return state, {
        "artifact_schema": SHARED_BASELINE_SCHEMA,
        "scope": "initialization_only",
        "heldout_information_used": False,
        "marker_information_used": ["board", "cube"],
        "fixed_camera_sources": cam_sources,
        "registered_camera_ids": sorted(cams),
        "shared_reference_state_sha256": state_sha256(state),
    }


def make_initial_state(
        condition: AblationCondition, shared_reference_state: PoseState,
        fixed_cubes: Mapping[int, np.ndarray],
        aligned_fixed_cubes: Optional[Mapping[int, np.ndarray]] = None,
        ) -> Tuple[PoseState, dict]:
    """Specialize one shared reference state only by the declared treatment."""
    state = shared_reference_state.clone()
    if "board" not in condition.target_set:
        state.board = None
    if "cube" not in condition.target_set:
        state.cubes = {}
        cube_source = "absent"
    elif condition.fk_to_cube == POSE_SOURCE_FK_FIXED:
        state.cubes = {
            int(s): np.asarray(T, dtype=np.float64).copy()
            for s, T in fixed_cubes.items()
        }
        cube_source = "raw_FK_pose_with_preregistered_mechanical_frame_map"
    elif condition.fk_to_cube == POSE_SOURCE_ALIGNED_FK_FIXED:
        if aligned_fixed_cubes is None:
            raise ValueError(
                f"{condition.row}: corrected-FK hard-fixed poses are required")
        state.cubes = {
            int(s): np.asarray(T, dtype=np.float64).copy()
            for s, T in aligned_fixed_cubes.items()
        }
        cube_source = "train_only_board_free_vision_aligned_FK_pose"
    else:
        cube_source = "shared_train_only_visual_initialization"
    return state, {
        "baseline_schema": SHARED_BASELINE_SCHEMA,
        "shared_reference_state_sha256": state_sha256(shared_reference_state),
        "row_reference_state_sha256": state_sha256(state),
        "T_base_Ci_and_T_gripper_cam_source": "shared_train_only_reference_state",
        "T_base_cube_by_set_source": cube_source,
        "registered_camera_ids": sorted(state.cams),
        "only_declared_treatment_applied": True,
    }


def canonical_solver_options(args, max_nfev: Optional[int] = None,
                             tol: Optional[float] = None) -> SolverOptions:
    tolerance = float(args.tol if tol is None else tol)
    return SolverOptions(
        method="trf",
        loss=str(getattr(args, "loss", "soft_l1")),
        f_scale_px=float(getattr(args, "f_scale_px", 2.0)),
        max_nfev=int(args.max_nfev if max_nfev is None else max_nfev),
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        scaling=SE3Scaling(
            rotation_scale_rad=float(getattr(args, "rotation_scale_rad", 1.0)),
            translation_scale_m=float(getattr(args, "translation_scale_m", 1.0)),
        ),
        x_scale_mode=str(getattr(args, "x_scale_mode", "jac")),
        residual_weighting=str(getattr(
            args, "residual_weighting", "per_corner")),
    )


def canonical_frame_prune_options(args) -> FramePruneRefitOptions:
    options = FramePruneRefitOptions(
        enabled=bool(getattr(args, "frame_prune_refit", True)),
        mad_multiplier=float(getattr(args, "frame_prune_mad_multiplier", 3.0)),
        minimum_rmse_px=float(getattr(args, "frame_prune_min_rmse_px", 4.0)),
        maximum_fraction=float(getattr(args, "frame_prune_max_fraction", 0.30)),
        minimum_observations_per_variable=int(getattr(
            args, "frame_prune_min_observations_per_variable", 2)),
        minimum_relative_improvement=float(getattr(
            args, "frame_prune_min_relative_improvement", 1e-6)),
    )
    options.validate()
    return options


def _solve_summary(diagnostics: Optional[Mapping]) -> Optional[dict]:
    if diagnostics is None:
        return None
    return {
        "success": bool(diagnostics.get("success", False)),
        "status": diagnostics.get("status"),
        "message": diagnostics.get("message"),
        "nfev": int(diagnostics.get("nfev", 0)),
        "elapsed_s": float(diagnostics.get("elapsed_s", 0.0)),
        "train_reprojection_rmse_px": diagnostics.get(
            "train_reprojection_rmse_px"),
    }


def run_frame_prune_refit(
        observations: Sequence[PixelObs],
        variable_keys_: Sequence[Tuple[str, int]],
        first_state: PoseState,
        first_diagnostics: Mapping,
        robot_T,
        K_map,
        D_map,
        gripper_cam_idx: int,
        solver_options: SolverOptions,
        prune_options: FramePruneRefitOptions,
        refit_solver: Callable[[Sequence[PixelObs]], Tuple[PoseState, dict]],
        full_objective_cost: Callable[[PoseState], float],
        ) -> Tuple[PoseState, dict]:
    """Apply one training-only frame prune/refit pass with safe rollback."""
    kept, selection = select_frame_prune_subset(
        observations=observations,
        variable_keys_=variable_keys_,
        fitted_state=first_state,
        robot_T=robot_T,
        K_map=K_map,
        D_map=D_map,
        gripper_cam_idx=gripper_cam_idx,
        solver_options=solver_options,
        prune_options=prune_options,
    )
    first_cost = float(full_objective_cost(first_state))
    candidate_state = None
    candidate_diagnostics = None
    candidate_cost = None
    refit_error = None
    if selection["attempted"]:
        try:
            candidate_state, candidate_diagnostics = refit_solver(kept)
            candidate_cost = float(full_objective_cost(candidate_state))
        except Exception as exc:  # rollback is the contract for a failed refit
            refit_error = f"{type(exc).__name__}: {exc}"

    relative_improvement = None
    accepted = False
    rollback_reason = None
    if not selection["attempted"]:
        rollback_reason = "refit_not_attempted"
    elif refit_error is not None:
        rollback_reason = "refit_exception"
    elif not bool(candidate_diagnostics.get("success", False)):
        rollback_reason = "refit_unsuccessful"
    elif not np.isfinite(candidate_cost):
        rollback_reason = "non_finite_candidate_full_objective"
    else:
        relative_improvement = float(
            (first_cost - candidate_cost) / max(abs(first_cost), 1e-15))
        accepted = bool(
            relative_improvement > prune_options.minimum_relative_improvement)
        if not accepted:
            rollback_reason = "insufficient_full_objective_improvement"

    chosen_state = candidate_state if accepted else first_state
    chosen_diagnostics = dict(
        candidate_diagnostics if accepted else first_diagnostics)
    selected_cost = candidate_cost if accepted else first_cost
    chosen_diagnostics["input_visual_residual_population"] = (
        observation_population(observations, gripper_cam_idx))
    chosen_diagnostics["effective_visual_residual_population"] = (
        observation_population(kept if accepted else observations,
                               gripper_cam_idx))
    chosen_diagnostics["frame_prune_refit"] = {
        "selection": selection,
        "first_fit": _solve_summary(first_diagnostics),
        "refit": _solve_summary(candidate_diagnostics),
        "refit_error": refit_error,
        "first_full_training_robust_cost": first_cost,
        "candidate_full_training_robust_cost": candidate_cost,
        "selected_full_training_robust_cost": selected_cost,
        "relative_full_objective_improvement": relative_improvement,
        "required_relative_improvement": float(
            prune_options.minimum_relative_improvement),
        "accepted": accepted,
        "rolled_back": bool(selection["attempted"] and not accepted),
        "rollback_reason": rollback_reason,
        "total_nfev": int(first_diagnostics.get("nfev", 0)) + int(
            0 if candidate_diagnostics is None
            else candidate_diagnostics.get("nfev", 0)),
        "heldout_observations_used": False,
        "heldout_observations_pruned": False,
    }
    return chosen_state, chosen_diagnostics


def solve_stage(observations: Sequence[PixelObs], free_families: Sequence[str],
                reference_state: PoseState, robot_T, K_map, D_map, gripper: int,
                seed: int, args, max_nfev: Optional[int] = None,
                tol: Optional[float] = None) -> Tuple[PoseState, dict]:
    keys = variable_keys(free_families, reference_state)
    solver_options = canonical_solver_options(
        args, max_nfev=max_nfev, tol=tol)
    prune_options = canonical_frame_prune_options(args)
    first_state, first_diagnostics = solve_corner_reprojection(
        observations=observations,
        variable_keys_=keys,
        reference_state=reference_state,
        robot_T=robot_T,
        K_map=K_map,
        D_map=D_map,
        gripper_cam_idx=gripper,
        options=solver_options,
        seed=seed,
        init_translation_mm=float(args.init_translation_mm),
        init_rotation_deg=float(args.init_rotation_deg),
    )
    return run_frame_prune_refit(
        observations=observations,
        variable_keys_=keys,
        first_state=first_state,
        first_diagnostics=first_diagnostics,
        robot_T=robot_T,
        K_map=K_map,
        D_map=D_map,
        gripper_cam_idx=gripper,
        solver_options=solver_options,
        prune_options=prune_options,
        refit_solver=lambda kept: solve_corner_reprojection(
            observations=kept,
            variable_keys_=keys,
            reference_state=first_state,
            robot_T=robot_T,
            K_map=K_map,
            D_map=D_map,
            gripper_cam_idx=gripper,
            options=solver_options,
            seed=0,
            init_translation_mm=0.0,
            init_rotation_deg=0.0,
        ),
        full_objective_cost=lambda state: visual_objective_cost(
            observations, keys, state, robot_T, K_map, D_map, gripper,
            solver_options),
    )


def cube_evaluation_observations(
        observations: Sequence[PixelObs], gripper: int,
        registered_cams: Iterable[int]) -> List[PixelObs]:
    """Select the frozen cube evaluation population independent of row target_set."""
    cams = {int(camera_id) for camera_id in registered_cams}
    selected = []
    for obs in observations:
        if obs.marker != "cube":
            continue
        camera_id = int(obs.cam)
        if camera_id != int(gripper) and camera_id not in cams:
            continue
        if obs.set_idx is None:
            if obs.grasp_idx is not None:
                continue   # 쥔 큐브(grasp 모델) 관측은 placement 큐브 평가 대상이 아님
            raise RuntimeError(
                f"cube evaluation observation lacks set index at event {obs.event}")
        selected.append(obs)
    return selected


def fit_train_only_cube_evaluation(
        final_state: PoseState,
        train_obs: Sequence[PixelObs],
        test_obs: Sequence[PixelObs],
        initial_cube_poses: Mapping[int, np.ndarray],
        gripper: int,
        robot_T,
        K_map,
        D_map,
        seed: int,
        args) -> dict:
    """Fit only cube target poses for the cube metric after calibration is frozen.

    This lets board-only rows (A0/B3) report cube RMSE without letting cube
    observations update camera extrinsics or hand-eye during calibration.
    """
    train_cube = cube_evaluation_observations(
        train_obs, gripper, final_state.cams)
    heldout_cube = cube_evaluation_observations(
        test_obs, gripper, final_state.cams)
    if not train_cube or not heldout_cube:
        raise RuntimeError("cube evaluation requires train and held-out cube observations")
    required_sets = sorted({
        int(obs.set_idx) for obs in list(train_cube) + list(heldout_cube)
    })
    missing = [
        set_index for set_index in required_sets
        if int(set_index) not in initial_cube_poses
    ]
    if missing:
        raise RuntimeError(
            f"cube evaluation pose source missing sets {missing}")
    reference = final_state.clone()
    reference.cubes = {
        int(set_index): np.asarray(initial_cube_poses[int(set_index)],
                                   dtype=np.float64).copy()
        for set_index in required_sets
    }
    keys = variable_keys(("T_base_cube_by_set",), reference)
    options = canonical_solver_options(args)
    fitted_state, diagnostics = solve_corner_reprojection(
        observations=train_cube,
        variable_keys_=keys,
        reference_state=reference,
        robot_T=robot_T,
        K_map=K_map,
        D_map=D_map,
        gripper_cam_idx=gripper,
        options=options,
        seed=int(seed),
        init_translation_mm=0.0,
        init_rotation_deg=0.0,
    )
    return {
        "artifact_schema": TRAIN_ONLY_CUBE_EVALUATION_SCHEMA,
        "pose_source": (
            "train_cube_nuisance_pose_fit_with_frozen_camera_and_hand_eye"),
        "camera_and_hand_eye_frozen": True,
        "free_variables": ["T_base_cube_by_set"],
        "train_cube_observations_used_for_pose_fit": True,
        "heldout_cube_observations_used_for_pose_fit": False,
        "external_ground_truth_used": False,
        "model_dependent_observation_rejection": False,
        "initial_cube_pose_source": "shared_train_only_visual_initialization",
        "train_population": observation_population(train_cube, gripper),
        "heldout_population": observation_population(heldout_cube, gripper),
        "fit_diagnostics": _solve_summary(diagnostics),
        "evaluation_state_sha256": state_sha256(fitted_state),
        "T_base_cube_by_set": {
            str(set_index): fitted_state.cubes[int(set_index)].tolist()
            for set_index in sorted(fitted_state.cubes)
        },
        "train": pixel_reprojection_metrics(
            train_cube, fitted_state, robot_T, K_map, D_map, gripper),
        "heldout": pixel_reprojection_metrics(
            heldout_cube, fitted_state, robot_T, K_map, D_map, gripper),
    }


def run_condition_once(condition: AblationCondition, initial_state: PoseState,
                       train_obs: Sequence[PixelObs], test_obs: Sequence[PixelObs],
                       gripper: int, robot_T, K_map, D_map, seed: int, args,
                       path_evaluation_mask: Mapping,
                       cube_evaluation_pose_source: Mapping[int, np.ndarray]) -> dict:
    relevant_train = filter_observations(
        train_obs, condition, None, gripper, initial_state.cams)
    relevant_test = filter_observations(
        test_obs, condition, None, gripper, initial_state.cams)
    if condition.unified == "seq":
        spec = SEQUENTIAL_STAGE_SPECS[condition.row]
        eih = filter_observations(relevant_train, condition, "eih", gripper,
                                  initial_state.cams)
        state1, d1 = solve_stage(
            eih, spec.stage1_free, initial_state,
            robot_T, K_map, D_map, gripper, seed, args)
        e2h = filter_observations(relevant_train, condition, "e2h", gripper,
                                  state1.cams)
        final_state, d2 = solve_stage(
            e2h, spec.stage2_free, state1,
            robot_T, K_map, D_map, gripper, seed, args)
        stages = {"stage1_eih": d1, "stage2_e2h": d2}
        converged = bool(d1["success"] and d2["success"])
    elif condition.row in FK_FIXED_ROWS:
        # Hard-FK is represented solely by the freeze mask: cube transforms are
        # present in the state but absent from A3's declared free variables.
        final_state, diag = solve_stage(
            relevant_train, UNIFIED_FREE_VARIABLES[condition.row], initial_state,
            robot_T, K_map, D_map, gripper, seed, args)
        diag["entry_point"] = "calibration_pipeline.reprojection.solve_corner_reprojection"
        diag["fk_mode"] = condition.fk_to_cube
        diag["shared_reference_state_sha256"] = state_sha256(initial_state)
        stages = {"joint_eih_e2h": diag}
        converged = bool(diag.get("success", False))
    else:
        final_state, diag = solve_stage(
            relevant_train, UNIFIED_FREE_VARIABLES[condition.row], initial_state,
            robot_T, K_map, D_map, gripper, seed, args)
        stages = {"joint_eih_e2h": diag}
        converged = bool(diag["success"])
    train_metrics = pixel_reprojection_metrics(
        relevant_train, final_state, robot_T, K_map, D_map, gripper)
    test_metrics = pixel_reprojection_metrics(
        relevant_test, final_state, robot_T, K_map, D_map, gripper)
    cube_evaluation = fit_train_only_cube_evaluation(
        final_state, train_obs, test_obs, cube_evaluation_pose_source,
        gripper, robot_T, K_map, D_map, seed, args)
    if condition.target_set in {"cube", "cube+board"}:
        # Use the full held-out cube pool and the same pre-fit mask for every
        # cube-bearing row.  The row-specific target filter must not alter the
        # path-consistency evaluation population.
        path = evaluate_paths_with_frozen_mask(
            list(train_obs) + list(test_obs),
            final_state.cams, final_state.gtc, robot_T,
            gripper, K_map, D_map, path_evaluation_mask)
    else:
        path = not_applicable_path_metrics(
            path_evaluation_mask, "cube target absent in this condition")
    path.pop("predicted_by_set", None)
    return {
        "seed": int(seed),
        "converged": converged,
        "stages": stages,
        "train_reprojection": train_metrics,
        "heldout_reprojection": test_metrics,
        "cube_evaluation_reprojection": cube_evaluation,
        "heldout_path_metrics": path,
        "transforms": serialize_state(final_state),
    }


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fk_covariances(path: str, set_ids: Sequence[int]) -> Tuple[dict, dict]:
    """Load preregistered physical FK covariance without blind-GT leakage."""
    with open(path) as handle:
        payload = json.load(handle)
    if payload.get("artifact_schema") != "fk_factor_covariance_v1":
        raise ValueError("unknown FK covariance artifact schema")
    if payload.get("twist_order") != [
            "rx_rad", "ry_rad", "rz_rad", "tx_m", "ty_m", "tz_m"]:
        raise ValueError("FK covariance twist order/units do not match the contract")
    if payload.get("blind_external_gt_used") is not False:
        raise ValueError("FK covariance artifact must exclude blind external GT")
    if payload.get("preregistered_before_blind_test") is not True:
        raise ValueError("FK covariance must be preregistered before blind evaluation")
    source = str(payload.get("measurement_source", "")).strip()
    if not source or source.startswith("REPLACE_"):
        raise ValueError("FK covariance measurement_source is a placeholder")
    estimator = str(payload.get("covariance_estimator", "")).strip()
    if not estimator or estimator.startswith("REPLACE_"):
        raise ValueError("FK covariance covariance_estimator is a placeholder")
    try:
        n_repeats = int(payload.get("n_repeats", 0))
    except (TypeError, ValueError):
        n_repeats = 0
    # A sample covariance in d=6 dimensions has rank at most N-1.  Requiring
    # N>=7 is therefore the mathematical minimum for a full-rank 6x6 estimate.
    if n_repeats < 7:
        raise ValueError(
            "FK covariance artifact needs at least seven physical repeats "
            "for a potentially full-rank 6D sample covariance")
    shared = payload.get("shared_covariance_6x6")
    per_set = payload.get("per_set_covariance_6x6", {})
    covariances = {}
    for set_index in set_ids:
        raw = per_set.get(str(int(set_index)), shared)
        if raw is None:
            raise ValueError(f"FK covariance missing for set {set_index}")
        covariances[int(set_index)] = validate_covariance(
            np.asarray(raw, dtype=np.float64))
    return covariances, {
        "path": os.path.abspath(path),
        "sha256": _sha256_file(path),
        "measurement_source": source,
        "covariance_estimator": estimator,
        "n_repeats": n_repeats,
        "minimum_repeats_for_full_rank_6d_covariance": 7,
        "confirmatory_ready": True,
        "status": "confirmatory_measured_covariance",
    }


def preflight_fk_covariances(set_ids: Sequence[int]) -> Tuple[dict, dict]:
    """Return the frozen Simulation prior, explicitly marked non-confirmatory."""
    covariance = diagonal_covariance(SIGMA_FK_MM, SIGMA_FK_DEG)
    return (
        {int(set_index): covariance.copy() for set_index in set_ids},
        {
            "path": None,
            "measurement_source": "Simulation frozen isotropic prior",
            "translation_std_mm": SIGMA_FK_MM,
            "rotation_std_deg": SIGMA_FK_DEG,
            "confirmatory_ready": False,
            "status": "preflight_simulation_prior",
            "warning": (
                "Simulation-matched covariance is a preflight prior, not measured "
                "robot covariance"),
        },
    )


def _solve_with_fk_factor(observations: Sequence[PixelObs], free_families,
                          reference_state: PoseState, data, covariances,
                          seed: int, args, use_factor: bool = True,
                          explicit_variable_keys=None):
    keys = (
        variable_keys(free_families, reference_state)
        if explicit_variable_keys is None else list(explicit_variable_keys))
    solver_options = canonical_solver_options(args)
    prune_options = canonical_frame_prune_options(args)
    fk_spec = FKFactorSpec(
        mode=FK_MODE_FACTOR if use_factor else FK_MODE_NONE,
        loss="huber",
        robust_scale=HUBER_F_SCALE,
    )
    first_state, first_diagnostics = solve_factorized_fk(
        observations=observations,
        variable_keys_=keys,
        reference_state=reference_state,
        robot_T=data.robot_T,
        K_map=data.K_map,
        D_map=data.D_map,
        gripper_cam_idx=data.gripper,
        options=solver_options,
        fk_targets=data.aligned_fk_cubes,
        fk_covariances=covariances,
        fk_spec=fk_spec,
        seed=seed,
        init_translation_mm=float(args.init_translation_mm),
        init_rotation_deg=float(args.init_rotation_deg),
    )
    return run_frame_prune_refit(
        observations=observations,
        variable_keys_=keys,
        first_state=first_state,
        first_diagnostics=first_diagnostics,
        robot_T=data.robot_T,
        K_map=data.K_map,
        D_map=data.D_map,
        gripper_cam_idx=data.gripper,
        solver_options=solver_options,
        prune_options=prune_options,
        refit_solver=lambda kept: solve_factorized_fk(
            observations=kept,
            variable_keys_=keys,
            reference_state=first_state,
            robot_T=data.robot_T,
            K_map=data.K_map,
            D_map=data.D_map,
            gripper_cam_idx=data.gripper,
            options=solver_options,
            fk_targets=data.aligned_fk_cubes,
            fk_covariances=covariances,
            fk_spec=fk_spec,
            seed=0,
            init_translation_mm=0.0,
            init_rotation_deg=0.0,
        ),
        full_objective_cost=lambda state: factorized_objective_cost(
            observations=observations,
            variable_keys_=keys,
            state=state,
            robot_T=data.robot_T,
            K_map=data.K_map,
            D_map=data.D_map,
            gripper_cam_idx=data.gripper,
            options=solver_options,
            fk_targets=data.aligned_fk_cubes,
            fk_covariances=covariances,
            fk_spec=fk_spec,
        ),
    )


def run_factor_condition_once(condition: AblationCondition, initial_state: PoseState,
                              data, covariances, seed: int, args) -> dict:
    """Run A4/B1/B2 with one visual backend and one FK-factor definition."""
    relevant_train = filter_observations(
        data.train_obs, condition, None, data.gripper, initial_state.cams)
    relevant_test = filter_observations(
        data.test_obs, condition, None, data.gripper, initial_state.cams)
    if condition.row in {"A4", "B2"}:
        final_state, diagnostics = _solve_with_fk_factor(
            relevant_train, UNIFIED_FREE_VARIABLES[condition.row],
            initial_state, data, covariances, seed, args)
        stages = {"joint_eih_e2h": diagnostics}
        converged = bool(diagnostics.get("success", False))
    elif condition.row == "B1":
        eih = filter_observations(
            relevant_train, condition, "eih", data.gripper, initial_state.cams)
        stage1, d1 = _solve_with_fk_factor(
            eih,
            ("T_gripper_cam", "T_base_board", "T_base_cube_by_set"),
            initial_state, data, covariances, seed, args)
        final_state = stage1.clone()
        per_camera = {}
        fixed = filter_observations(
            relevant_train, condition, "e2h", data.gripper, initial_state.cams)
        for camera_id in sorted(initial_state.cams):
            camera_observations = [
                obs for obs in fixed if int(obs.cam) == int(camera_id)]
            if not camera_observations:
                raise RuntimeError(
                    f"B1 has no fixed-camera observations for cam{camera_id}")
            solved, diag = _solve_with_fk_factor(
                camera_observations, (), final_state, data,
                covariances, seed, args, use_factor=False,
                explicit_variable_keys=[("cam", int(camera_id))])
            final_state.cams[int(camera_id)] = solved.cams[int(camera_id)]
            per_camera[f"cam{camera_id}"] = diag
        stages = {
            "stage1_eih_with_fk_factor": d1,
            "stage2_e2h_per_fixed_camera": per_camera,
        }
        converged = bool(
            d1.get("success", False)
            and all(diag.get("success", False) for diag in per_camera.values()))
    else:
        raise ValueError(f"{condition.row} is not an FK-factor row")

    train_metrics = pixel_reprojection_metrics(
        relevant_train, final_state, data.robot_T, data.K_map, data.D_map,
        data.gripper)
    test_metrics = pixel_reprojection_metrics(
        relevant_test, final_state, data.robot_T, data.K_map, data.D_map,
        data.gripper)
    cube_evaluation = fit_train_only_cube_evaluation(
        final_state, data.train_obs, data.test_obs, data.visual_cubes,
        data.gripper, data.robot_T, data.K_map, data.D_map, seed, args)
    path = evaluate_paths_with_frozen_mask(
        list(data.train_obs) + list(data.test_obs),
        final_state.cams, final_state.gtc, data.robot_T,
        data.gripper, data.K_map, data.D_map, data.path_evaluation_mask)
    path.pop("predicted_by_set", None)
    return {
        "seed": int(seed),
        "converged": converged,
        "stages": stages,
        "train_reprojection": train_metrics,
        "heldout_reprojection": test_metrics,
        "cube_evaluation_reprojection": cube_evaluation,
        "heldout_path_metrics": path,
        "transforms": serialize_state(final_state),
    }


def transform_dispersion(runs: Sequence[dict]) -> dict:
    if not runs:
        return {}
    ref = runs[0]["transforms"]
    output = {}
    keys = [("T_gripper_cam", None)]
    keys.extend(("T_base_Ci", ci) for ci in sorted(ref["T_base_Ci"], key=int))
    for family, idx in keys:
        A = np.asarray(ref[family] if idx is None else ref[family][idx], float)
        values = []
        for run in runs:
            raw = run["transforms"][family] if idx is None else run["transforms"][family][idx]
            values.append(pose_delta(A, np.asarray(raw, float)))
        name = family if idx is None else f"T_base_C{idx}"
        output[name] = {
            "translation_std_mm": float(np.std([v[0] for v in values])),
            "translation_max_mm": float(np.max([v[0] for v in values])),
            "rotation_std_deg": float(np.std([v[1] for v in values])),
            "rotation_max_deg": float(np.max([v[1] for v in values])),
        }
    return output


def _make_T(rotvec: Sequence[float], translation: Sequence[float]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, float)).as_matrix()
    T[:3, 3] = np.asarray(translation, float)
    return T


def synthetic_scene() -> Tuple[List[PixelObs], PoseState, Dict[int, np.ndarray], dict, dict, int]:
    gripper = 2
    K = {ci: np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
         for ci in (0, 1, 2)}
    D = {ci: np.zeros(5) for ci in K}
    cams = {
        0: _make_T([0.0, 0.0, 0.0], [-0.12, -0.02, 0.0]),
        1: _make_T([0.0, 0.04, 0.0], [0.15, 0.01, 0.02]),
    }
    gtc = _make_T([0.015, -0.01, 0.02], [0.025, -0.01, 0.055])
    board = _make_T([0.02, -0.03, 0.01], [0.0, 0.03, 1.25])
    cubes = {
        0: _make_T([0.06, -0.02, 0.03], [-0.13, -0.08, 1.02]),
        1: _make_T([-0.03, 0.07, -0.02], [0.17, -0.03, 1.08]),
    }
    # A small exact scene keeps the mandatory preflight test fast.  The real
    # runner still uses every detected corner.
    board_obj = np.array([[0.025 * (x - 1.5), 0.025 * (y - 1.0), 0.0]
                          for y in range(3) for x in range(4)], float)
    a = 0.025
    cube_obj = np.array([
        [-a, -a, -a], [a, -a, -a], [a, a, -a], [-a, a, -a],
        [-a, -a, a], [a, -a, a], [a, a, a], [-a, a, a],
    ], float)
    robot_T, obs = {}, []
    event = 0
    for s in sorted(cubes):
        for j in range(5):
            cam_pose = _make_T(
                [0.015 * (j - 3), -0.012 * (j - 2), 0.01 * np.sin(j)],
                [-0.08 + 0.027 * j, 0.10 - 0.018 * j, 0.015 * np.cos(j)],
            )
            robot_T[event] = cam_pose @ inv_T(gtc)
            for ci in (0, 1, gripper):
                T_B_C = cam_pose if ci == gripper else cams[ci]
                for marker, target, points in (
                    ("board", board, board_obj), ("cube", cubes[s], cube_obj)):
                    pixels = project_points(inv_T(T_B_C) @ target, points, K[ci], D[ci])
                    obs.append(PixelObs(marker, ci, event, s, points.copy(), pixels))
            event += 1
    return obs, PoseState(cams, gtc, board, cubes), robot_T, K, D, gripper


def perturbed_state(state: PoseState) -> PoseState:
    out = state.clone()
    keys = ([('cam', ci) for ci in sorted(out.cams)] + [('gtc', -1), ('board', -1)]
            + [('cube', s) for s in sorted(out.cubes)])
    for key in keys:
        key_seed = int.from_bytes(
            hashlib.sha256(f"{key[0]}:{key[1]}".encode()).digest()[:4], "little")
        rng = np.random.default_rng(np.random.SeedSequence([90210, key_seed]))
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(rng.normal(0.0, np.deg2rad(0.5), 3)).as_matrix()
        delta[:3, 3] = rng.normal(0.0, 0.003, 3)
        set_state_transform(out, key, delta @ state_transform(out, key))
    return out


def solve_synthetic(condition: AblationCondition, observations, truth: PoseState,
                    robot_T, K, D, gripper: int, args,
                    fixed_cube_poses: Optional[Mapping[int, np.ndarray]] = None,
                    solver_seed: int = 0) -> Tuple[PoseState, float, dict]:
    init = perturbed_state(truth)
    if condition.fk_to_cube in {
            POSE_SOURCE_FK_FIXED, POSE_SOURCE_ALIGNED_FK_FIXED}:
        # FK hard-fixed transforms are constants, not perturbed initialization
        # variables.  Perturbing them would simulate FK noise, not solver init.
        source = truth.cubes if fixed_cube_poses is None else fixed_cube_poses
        init.cubes = {int(s): np.asarray(T, float).copy() for s, T in source.items()}
    if condition.target_set == "board":
        init.cubes = {}
    if condition.target_set == "cube":
        init.board = None
    relevant = filter_observations(observations, condition, None, gripper, init.cams)
    if condition.unified == "seq":
        spec = SEQUENTIAL_STAGE_SPECS[condition.row]
        state1, d1 = solve_stage(
            filter_observations(relevant, condition, "eih", gripper, init.cams),
            spec.stage1_free, init, robot_T, K, D, gripper, solver_seed, args,
            max_nfev=int(args.max_nfev), tol=float(args.tol))
        final, d2 = solve_stage(
            filter_observations(relevant, condition, "e2h", gripper, state1.cams),
            spec.stage2_free, state1, robot_T, K, D, gripper, solver_seed, args,
            max_nfev=int(args.max_nfev), tol=float(args.tol))
        diag = {"stage1": d1, "stage2": d2}
    else:
        final, d = solve_stage(
            relevant, UNIFIED_FREE_VARIABLES[condition.row], init,
            robot_T, K, D, gripper, solver_seed, args,
            max_nfev=int(args.max_nfev), tol=float(args.tol))
        diag = {"joint": d}
    metric = pixel_reprojection_metrics(
        relevant, final, robot_T, K, D, gripper)
    return final, float(metric["overall"]["rmse_px"]), diag


def compare_states(A: PoseState, B: PoseState, include_cubes: bool) -> dict:
    deltas = []
    for ci in sorted(set(A.cams) & set(B.cams)):
        deltas.append((f"T_base_C{ci}",) + pose_delta(A.cams[ci], B.cams[ci]))
    deltas.append(("T_gripper_cam",) + pose_delta(A.gtc, B.gtc))
    if A.board is not None and B.board is not None:
        deltas.append(("T_base_board",) + pose_delta(A.board, B.board))
    if include_cubes:
        for s in sorted(set(A.cubes) & set(B.cubes)):
            deltas.append((f"T_base_cube[{s}]",) + pose_delta(A.cubes[s], B.cubes[s]))
    return {
        "max_translation_mm": max(v[1] for v in deltas),
        "max_rotation_deg": max(v[2] for v in deltas),
        "per_transform": [{"name": v[0], "translation_mm": v[1], "rotation_deg": v[2]}
                          for v in deltas],
    }


def run_noise_free_sanity(args) -> dict:
    observations, truth, robot_T, K, D, gripper = synthetic_scene()
    by_row = {condition.row: condition for condition in MAIN_ABLATION_CONDITIONS}
    solved = {}
    for row in ("A1", "A2"):
        state, rmse, diag = solve_synthetic(
            by_row[row], observations, truth, robot_T, K, D, gripper, args)
        solved[row] = {"state": state, "rmse_px": rmse, "diagnostics": diag}
    pairs = {}
    truth_recovery = {}
    passed = True
    for seq_row, unified_row, include_cubes in (("A1", "A2", True),):
        comparison = compare_states(solved[seq_row]["state"], solved[unified_row]["state"], include_cubes)
        pair_ok = (
            solved[seq_row]["rmse_px"] <= NOISE_FREE_SANITY_TOLERANCES["reprojection_rmse_px"]
            and solved[unified_row]["rmse_px"] <= NOISE_FREE_SANITY_TOLERANCES["reprojection_rmse_px"]
            and comparison["max_translation_mm"] <= NOISE_FREE_SANITY_TOLERANCES["seq_vs_unified_translation_mm"]
            and comparison["max_rotation_deg"] <= NOISE_FREE_SANITY_TOLERANCES["seq_vs_unified_rotation_deg"]
        )
        pairs[f"{seq_row}={unified_row}"] = {
            "passed": pair_ok,
            "seq_reprojection_rmse_px": solved[seq_row]["rmse_px"],
            "unified_reprojection_rmse_px": solved[unified_row]["rmse_px"],
            **comparison,
        }
        passed = passed and pair_ok
    for row in ("A1", "A2"):
        comparison = compare_states(solved[row]["state"], truth, include_cubes=row in {"A1", "A2"})
        row_ok = (
            comparison["max_translation_mm"]
            <= NOISE_FREE_SANITY_TOLERANCES["seq_vs_unified_translation_mm"]
            and comparison["max_rotation_deg"]
            <= NOISE_FREE_SANITY_TOLERANCES["seq_vs_unified_rotation_deg"]
        )
        truth_recovery[row] = {"passed": row_ok, **comparison}
        passed = passed and row_ok
    report = {
        "passed": passed,
        "tolerances": NOISE_FREE_SANITY_TOLERANCES,
        "pairs": pairs,
        "truth_recovery": truth_recovery,
    }
    if not passed:
        raise RuntimeError("noise-free seq==U sanity gate failed: " + json.dumps(_jsonable(report)))
    return report


def detect_observations(args, meta, K_map, D_map, all_cam_ids, gripper):
    observation_manifest = getattr(args, "observation_manifest", None)
    if observation_manifest:
        allowed_events = [
            int(capture["event_id"])
            for capture in meta.get("captures", [])
            if int(capture.get("event_id", -1)) >= 0
        ]
        observations, manifest_diag = load_pixel_observations_from_manifest(
            observation_manifest,
            policy=str(getattr(
                args, "observation_filter_policy", "standard")),
            root=args.root_folder,
            intrinsics_dir=args.intrinsics_dir,
            allowed_event_ids=allowed_events,
            validate_sources=True,
            allow_relocated_root=bool(getattr(
                args, "allow_relocated_session_root", False)),
        )
        return (
            observations,
            manifest_diag.get("cube_config_source", "manifest"),
            manifest_diag,
        )
    cfg, cfg_source = resolve_cube_config_for_run(
        args.root_folder, calib_dir=args.calib_dir,
        cube_config_json=getattr(args, "cube_config", None),
        default_cfg=get_default_cube_config())
    cube = AprilTagCubeTarget(cfg)
    observations, observation_diag = load_cube_board_pixel_observations(
        args.root_folder, meta, cube, K_map, D_map, all_cam_ids, gripper,
        exclude_gripped_cube=not bool(getattr(args, "include_gripped_cube", False)),
        fixed_cube_min_corners=8,
        image_scale=float(getattr(args, "image_scale", 1.0)),
        cube_observation_policy=str(getattr(
            args, "cube_observation_policy", "core_multiface")))
    cube_reason = observation_diag["cube"]
    return observations, cfg_source, cube_reason


def apply_board_metric_scale(
        observations: Sequence[PixelObs], scale: float) -> List[PixelObs]:
    """Scale ChArUco object coordinates without changing frozen image pixels."""
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("board metric scale must be finite and positive")
    return [PixelObs(
        marker=str(observation.marker),
        cam=int(observation.cam),
        event=int(observation.event),
        set_idx=(None if observation.set_idx is None
                 else int(observation.set_idx)),
        object_points=(
            np.asarray(observation.object_points, dtype=np.float64) * scale
            if observation.marker == "board"
            else np.asarray(observation.object_points, dtype=np.float64).copy()
        ),
        image_points=np.asarray(
            observation.image_points, dtype=np.float64).copy(),
        grasp_idx=(None if observation.grasp_idx is None
                   else int(observation.grasp_idx)),
    ) for observation in observations]


def infer_board_square_length_mm(
        observations: Sequence[PixelObs]) -> float:
    """Read the frozen board metric from object points instead of a default."""
    coordinates = []
    for observation in observations:
        if observation.marker != "board":
            continue
        points = np.asarray(
            observation.object_points, dtype=np.float64).reshape(-1, 3)
        coordinates.extend(points[:, axis] for axis in (0, 1))
    if not coordinates:
        raise RuntimeError("cannot infer board square length without board corners")
    values = np.unique(np.round(np.concatenate(coordinates), 9))
    differences = np.diff(values)
    positive = differences[differences > 1e-7]
    if not len(positive):
        raise RuntimeError("frozen board object points have no metric spacing")
    spacing_m = float(np.min(positive))
    ratios = np.concatenate(coordinates) / spacing_m
    if not np.allclose(ratios, np.round(ratios), rtol=0.0, atol=1e-5):
        raise RuntimeError(
            "frozen board object points do not lie on one square grid")
    return spacing_m * 1000.0


def estimate_train_board_metric_scale(
        train_observations: Sequence[PixelObs], gripper_cam_idx: int,
        K_map, D_map) -> dict:
    """Align board metric scale to cube using train-only fixed-camera overlap.

    A uniform target-size error leaves relative-camera rotation and translation
    direction nearly unchanged but scales recovered baseline length. This
    estimator robustly averages board- and cube-derived anchor-camera transforms
    and solves one least-squares scale from their translation vectors. It never
    reads held-out observations, robot FK, Hand-Eye, or external GT.
    """
    gripper = int(gripper_cam_idx)
    fixed_cameras = sorted({
        int(observation.cam) for observation in train_observations
        if int(observation.cam) != gripper
    })
    if len(fixed_cameras) < 2:
        raise RuntimeError("board metric alignment needs at least two fixed cameras")
    anchor = int(fixed_cameras[0])
    grouped = defaultdict(dict)
    for observation in train_observations:
        camera = int(observation.cam)
        if camera in fixed_cameras and observation.marker in {"board", "cube"}:
            grouped[(str(observation.marker), int(observation.event))][camera] = observation

    fitted = {"board": {}, "cube": {}}
    fit_diagnostics = {"board": {}, "cube": {}}
    for marker in ("board", "cube"):
        for camera in fixed_cameras:
            if camera == anchor:
                continue
            candidates = []
            source_events = []
            for (target, event), by_camera in sorted(grouped.items()):
                if target != marker or anchor not in by_camera or camera not in by_camera:
                    continue
                T_anchor_target = solve_observed_pose(
                    by_camera[anchor], K_map, D_map)
                T_camera_target = solve_observed_pose(
                    by_camera[camera], K_map, D_map)
                if T_anchor_target is None or T_camera_target is None:
                    continue
                candidates.append(T_anchor_target @ inv_T(T_camera_target))
                source_events.append(int(event))
            if not candidates:
                raise RuntimeError(
                    f"no train {marker} overlap for cam{anchor}-cam{camera}")
            transform, diagnostics = cp.robust_se3_average(candidates)
            fitted[marker][camera] = transform
            fit_diagnostics[marker][str(camera)] = {
                **diagnostics,
                "source_event_ids": source_events,
            }

    board_vectors = np.concatenate([
        fitted["board"][camera][:3, 3]
        for camera in fixed_cameras if camera != anchor
    ])
    cube_vectors = np.concatenate([
        fitted["cube"][camera][:3, 3]
        for camera in fixed_cameras if camera != anchor
    ])
    denominator = float(board_vectors @ board_vectors)
    if denominator <= 1e-12:
        raise RuntimeError("degenerate board-derived fixed-camera baseline")
    scale = float((board_vectors @ cube_vectors) / denominator)
    if not 0.95 <= scale <= 1.05:
        raise RuntimeError(
            f"train-only board metric scale {scale:.6f} is outside [0.95, 1.05]")

    per_camera = {}
    for camera in fixed_cameras:
        if camera == anchor:
            continue
        board_transform = fitted["board"][camera]
        cube_transform = fitted["cube"][camera]
        before_mm, rotation_deg = pose_delta(board_transform, cube_transform)
        scaled_board = np.asarray(board_transform, dtype=np.float64).copy()
        scaled_board[:3, 3] *= scale
        after_mm, after_rotation_deg = pose_delta(scaled_board, cube_transform)
        board_vector = board_transform[:3, 3]
        camera_scale = float(
            (board_vector @ cube_transform[:3, 3])
            / max(float(board_vector @ board_vector), 1e-12))
        per_camera[str(camera)] = {
            "translation_disagreement_before_mm": float(before_mm),
            "translation_disagreement_after_mm": float(after_mm),
            "rotation_disagreement_deg": float(rotation_deg),
            "rotation_disagreement_after_deg": float(after_rotation_deg),
            "camera_specific_scale": camera_scale,
            "board_baseline_norm_mm": float(
                np.linalg.norm(board_transform[:3, 3]) * 1000.0),
            "cube_baseline_norm_mm": float(
                np.linalg.norm(cube_transform[:3, 3]) * 1000.0),
        }
    nominal_square_length_mm = infer_board_square_length_mm(
        train_observations)
    return {
        "mode": "train_only_fixed_camera_board_to_cube_metric_alignment",
        "enabled": True,
        "scale": scale,
        "nominal_square_length_mm": nominal_square_length_mm,
        "effective_square_length_mm": nominal_square_length_mm * scale,
        "anchor_camera_id": anchor,
        "fixed_camera_ids": fixed_cameras,
        "heldout_observations_used": False,
        "robot_fk_used": False,
        "external_ground_truth_used": False,
        "per_camera": per_camera,
        "fit_diagnostics": fit_diagnostics,
    }


@dataclass
class PreparedAblationData:
    meta: dict
    cube_config_source: str
    cube_detection: dict
    split: dict
    gripper: int
    K_map: dict
    D_map: dict
    robot_T: dict
    train_obs: List[PixelObs]
    test_obs: List[PixelObs]
    source_data_provenance: dict
    board_metric_scale: dict
    pose_convention: dict
    path_evaluation_mask: dict
    board_gtc: np.ndarray
    board_initial: np.ndarray
    handeye_diagnostics: dict
    visual_cubes: Dict[int, np.ndarray]
    shared_reference_state: PoseState
    shared_reference_diagnostics: dict
    fixed_cubes: Dict[int, np.ndarray]
    aligned_fk_cubes: Dict[int, np.ndarray]
    fixed_gtc_initial: np.ndarray
    alignment_artifact: dict


def prepare_ablation_data(args) -> PreparedAblationData:
    """Build the exact train/test data and train-only initialization artifacts."""
    with open(os.path.join(args.root_folder, "meta.json")) as handle:
        meta = json.load(handle)
    meta, included_set_indices = filter_meta_by_set_indices(
        meta, getattr(args, "include_sets", ""))
    if included_set_indices and not meta.get("captures"):
        raise RuntimeError(
            f"include_sets={args.include_sets!r} did not match any captures")
    protocol = capture_protocol_from_meta(meta)
    if protocol in PHASE_AWARE_CAPTURE_PROTOCOLS:
        if included_set_indices:
            raise RuntimeError(
                "phase-aware capture uses a frozen event population; "
                "--include_sets is not allowed")
        selected_captures = selected_composite_protocol_captures(meta)
        meta = dict(meta)
        meta["captures"] = selected_captures
    meta, pose_convention = apply_pose_convention_manifest(
        args.root_folder, meta)
    all_cam_ids = sorted({int(ci) for cap in meta.get("captures", [])
                          for ci in cap.get("cams", {})})
    gripper = int(meta["gripper_cam_idx"])
    K_map, D_map = {}, {}
    for ci in all_cam_ids:
        K_map[ci], D_map[ci], _ = load_intrinsics_with_depth_scale(
            args.intrinsics_dir, ci)
    robot_T = cp.load_robot_poses_from_meta(meta)
    observations, cube_cfg_source, cube_reason = detect_observations(
        args, meta, K_map, D_map, all_cam_ids, gripper)
    # placement 세트도 아니고 쥔 큐브도 아닌 큐브 관측(예: 보드 촬영 이벤트에
    # 우연히 남아 있던 큐브)은 어느 모델에도 못 넣으므로 뺀다.
    stray_cube = [obs for obs in observations
                  if obs.marker == "cube" and obs.set_idx is None and obs.grasp_idx is None]
    if stray_cube:
        print(f"[WARN] dropping {len(stray_cube)} cube observations with neither set index nor grasp id "
              f"(events {sorted({int(o.event) for o in stray_cube})[:8]}...)")
        observations = [obs for obs in observations if obs not in stray_cube]
    include_gripped = bool(getattr(args, "include_gripped_cube", False))
    if not include_gripped:
        # manifest 경로에서는 grasp_idx가 실린 채 로드되므로 여기서 확실히 뺀다.
        n_grasp = sum(1 for obs in observations if obs.grasp_idx is not None)
        if n_grasp:
            print(f"[INFO] excluding {n_grasp} gripped-cube (grasp model) observations "
                  "(pass --include_gripped_cube to keep them)")
            observations = [obs for obs in observations if obs.grasp_idx is None]
    split = build_protocol_event_split(
        observations, meta, gripper, args.test_fraction, args.split_seed,
        args.min_train_eih_cube_events)
    eligible = set(split["eligible_sets"])
    train_events = set(split["train_events"])
    test_events = set(split["test_events"])
    if split["strategy"] == COMPOSITE_PLACEMENT_SPLIT:
        selected_events = train_events | test_events
        pool = [obs for obs in observations if int(obs.event) in selected_events]
    else:
        pool = [obs for obs in observations if obs.set_idx in eligible]
        # 세트에 묶이지 않는 관측 -- 쥔 큐브(grasp+FK 모델, grasp_idx 있음)와
        # 정지 보드 -- 는 placement 세트가 아니어서 위 필터에서 통째로 빠진다
        # (예: Zeus session1 쥔 큐브 16 이벤트, session3 손목 보드 15 이벤트가
        # set_index 0). 이들은 held-out 대상이 아니므로 train 전용 보조 관측으로
        # 남긴다. held-out 이벤트에 속한 것은 넣지 않는다.
        aux_train = [obs for obs in observations
                     if obs.set_idx not in eligible and int(obs.event) not in test_events
                     and (obs.grasp_idx is not None or obs.marker == "board")]
        if aux_train:
            pool = pool + aux_train
            aux_events = {int(obs.event) for obs in aux_train}
            train_events |= aux_events
            split["train_events"] = sorted(set(split["train_events"]) | aux_events)
            split["auxiliary_train_only_events"] = sorted(aux_events)
            split["auxiliary_train_only_observations"] = {
                "gripped_cube": sum(1 for o in aux_train if o.grasp_idx is not None),
                "board": sum(1 for o in aux_train if o.marker == "board"),
            }
    train_obs = [obs for obs in pool if int(obs.event) in train_events]
    test_obs = [obs for obs in pool if int(obs.event) in test_events]
    if split["strategy"] == COMPOSITE_PLACEMENT_SPLIT:
        raise RuntimeError(
            "phase-aware placement-grouped split is active, but the final "
            "common-rig target model is not implemented in 05_calibrate.py yet; "
            "refusing to run the legacy static-board/per-set-cube model on P1/P2/P3"
        )
    if bool(getattr(args, "align_board_metric_scale", False)):
        board_metric_scale = estimate_train_board_metric_scale(
            train_obs, gripper, K_map, D_map)
        observations = apply_board_metric_scale(
            observations, board_metric_scale["scale"])
        pool = [obs for obs in observations if obs.set_idx in eligible]
        train_obs = [obs for obs in pool if int(obs.event) in train_events]
        test_obs = [obs for obs in pool if int(obs.event) in test_events]
    else:
        nominal_square_length_mm = infer_board_square_length_mm(train_obs)
        board_metric_scale = {
            "mode": "nominal_config",
            "enabled": False,
            "scale": 1.0,
            "nominal_square_length_mm": nominal_square_length_mm,
            "effective_square_length_mm": nominal_square_length_mm,
            "heldout_observations_used": False,
        }
    source_data_provenance = _source_data_provenance(
        args, all_cam_ids, pool, train_obs, test_obs)
    common_fixed_cameras = sorted({
        int(obs.cam) for obs in test_obs
        if obs.marker == "cube" and int(obs.cam) != gripper
    })
    event_roles = {
        **{int(event): "train" for event in train_events},
        **{int(event): "heldout" for event in test_events},
    }
    path_evaluation_mask = build_frozen_path_evaluation_mask(
        observations=test_obs,
        fixed_camera_ids=common_fixed_cameras,
        gripper_cam_idx=gripper,
        K_map=K_map,
        D_map=D_map,
        set_filter=sorted(eligible),
        fixed_anchor_observations=pool,
        event_roles=event_roles,
        require_e2e=bool(getattr(args, "require_e2e_path_metric", True)),
    )
    validate_frozen_path_evaluation_mask(path_evaluation_mask)

    # Build the canonical FK artifact before any board-derived quantity exists.
    # The estimator itself additionally rejects all non-eih-cube observations.
    train_meta = dict(meta)
    train_meta["captures"] = [
        cap for cap in meta.get("captures", [])
        if int(cap.get("event_id", -1)) in train_events]
    raw_fk_all = cp.load_nominal_set_cube_transforms(train_meta)
    raw_fk_source_event_by_set = {}
    for cap in train_meta["captures"]:
        set_index = get_capture_set_index(cap)
        if set_index is not None and int(set_index) not in raw_fk_source_event_by_set:
            raw_fk_source_event_by_set[int(set_index)] = int(cap["event_id"])
    aligned_fk_all, fixed_gtc_initial, artifact = \
        estimate_board_free_fk_cube_artifact(
            observations=train_obs,
            raw_fk_by_set=raw_fk_all,
            robot_T=robot_T,
            K_map=K_map,
            D_map=D_map,
            gripper_cam_idx=gripper,
            training_set_ids=sorted(eligible),
            options=SolverOptions(),
            num_inits=3,
            init_translation_mm=5.0,
            init_rotation_deg=1.0,
            raw_fk_source_event_by_set=raw_fk_source_event_by_set,
        )
    validate_fk_alignment_artifact(artifact)
    leaked_test_events = (
        set(artifact["source_observation_ids"]) & set(split["test_events"]))
    if leaked_test_events:
        raise RuntimeError(
            f"board-free FK artifact used held-out events {sorted(leaked_test_events)}")
    leaked_raw_events = {
        int(event) for event in artifact["raw_fk_source_event_by_set"].values()
        if isinstance(event, int) and int(event) in set(split["test_events"])}
    if leaked_raw_events:
        raise RuntimeError(
            f"board-free corrected-FK artifact used held-out FK events "
            f"{sorted(leaked_raw_events)}")
    missing_aligned_fk = sorted(eligible - set(aligned_fk_all))
    if missing_aligned_fk:
        raise RuntimeError(
            "board-free corrected-FK cube pose missing for sets "
            f"{missing_aligned_fk}")
    missing_raw_fk = sorted(eligible - set(raw_fk_all))
    if missing_raw_fk:
        raise RuntimeError(f"FK cube pose missing for sets {missing_raw_fk}")
    mechanical_frame_map = np.asarray(
        RAW_FK_CUBE_CENTER_TO_OBJECT, dtype=np.float64)
    fixed_cubes = {
        s: np.asarray(raw_fk_all[s], dtype=np.float64) @ mechanical_frame_map
        for s in sorted(eligible)
    }
    aligned_fk_cubes = {
        s: aligned_fk_all[s] for s in sorted(eligible)
    }

    # Board-derived quantities begin only after the canonical artifact is
    # frozen.  They initialize board-bearing rows and create a supplementary
    # leakage comparison, never the shared FK cube poses.
    eih_board = [obs for obs in train_obs
                 if obs.marker == "board" and int(obs.cam) == gripper]
    board_gtc, board_initial, handeye_diag = estimate_board_handeye_initial(
        eih_board, robot_T, K_map, D_map, gripper)
    visual_cubes = average_visual_target(
        train_obs, "cube", board_gtc, robot_T, K_map, D_map, gripper)
    missing_visual = sorted(eligible - set(visual_cubes))
    if missing_visual:
        # 그리퍼캠(eye-in-hand) 큐브 관측이 train에 없는 세트(Zeus처럼 세트당
        # 그리퍼 뷰가 1개뿐인 리그에서 그 뷰가 held-out으로 빠진 경우). 초기값만
        # raw-FK 큐브 pose로 대신 잡는다 -- 자유 변수 행(A1/A2/B*)은 이 값을
        # 재최적화하므로 결과가 FK에 묶이지는 않는다. 어떤 세트가 fallback을
        # 썼는지는 shared_reference_diag에 남긴다.
        print(f"[WARN] visual cube initialization missing for train sets {missing_visual}; "
              "initializing those sets from the raw-FK cube pose (initial value only)")
        for s in missing_visual:
            visual_cubes[s] = np.asarray(fixed_cubes[s], dtype=np.float64).copy()
    shared_reference_state, shared_reference_diag = build_shared_reference_state(
        train_obs, gripper, robot_T, K_map, D_map,
        board_gtc, board_initial, visual_cubes)
    shared_reference_diag = dict(shared_reference_diag)
    shared_reference_diag["visual_init_fallback_raw_fk_sets"] = list(missing_visual)
    return PreparedAblationData(
        meta=meta,
        cube_config_source=cube_cfg_source,
        cube_detection=cube_reason,
        split=split,
        gripper=gripper,
        K_map=K_map,
        D_map=D_map,
        robot_T=robot_T,
        train_obs=train_obs,
        test_obs=test_obs,
        source_data_provenance=source_data_provenance,
        board_metric_scale=board_metric_scale,
        pose_convention=pose_convention,
        path_evaluation_mask=path_evaluation_mask,
        board_gtc=board_gtc,
        board_initial=board_initial,
        handeye_diagnostics=handeye_diag,
        visual_cubes=visual_cubes,
        shared_reference_state=shared_reference_state,
        shared_reference_diagnostics=shared_reference_diag,
        fixed_cubes=fixed_cubes,
        aligned_fk_cubes=aligned_fk_cubes,
        fixed_gtc_initial=fixed_gtc_initial,
        alignment_artifact=artifact,
    )


def build_shared_baseline_artifact(
        args, data: PreparedAblationData,
        row_reference_states: Mapping[str, PoseState]) -> dict:
    """Serialize the exact train-only baseline shared by every condition."""
    body = {
        "artifact_schema": SHARED_BASELINE_SCHEMA,
        "dataset_root": os.path.abspath(args.root_folder),
        "intrinsics_dir": os.path.abspath(args.intrinsics_dir),
        "eligible_set_ids": [int(s) for s in data.split["eligible_sets"]],
        "train_event_ids": [int(e) for e in data.split["train_events"]],
        "heldout_event_ids": [int(e) for e in data.split["test_events"]],
        "heldout_information_used": False,
        "gripper_camera_id": int(data.gripper),
        "registered_fixed_camera_ids": sorted(
            int(ci) for ci in data.shared_reference_state.cams),
        "shared_reference_state": serialize_state(data.shared_reference_state),
        "shared_reference_state_sha256": state_sha256(
            data.shared_reference_state),
        "row_reference_states": {
            str(row): serialize_state(state)
            for row, state in sorted(row_reference_states.items())
        },
        "row_reference_state_sha256": {
            str(row): state_sha256(state)
            for row, state in sorted(row_reference_states.items())
        },
        "solver_options": canonical_solver_options(args).to_dict(),
        "frame_prune_refit": canonical_frame_prune_options(args).to_dict(),
        "initialization": {
            "num_inits": int(args.num_inits),
            "translation_mm": float(args.init_translation_mm),
            "rotation_deg": float(args.init_rotation_deg),
            "seed_zero_is_unperturbed": True,
        },
        "observation_loader": {
            "exclude_gripped_cube": not bool(getattr(args, "include_gripped_cube", False)),
            "fixed_cube_min_corners": 8,
            "cube_observation_policy": data.cube_detection[
                "observation_policy"],
            "image_scale": float(getattr(args, "image_scale", 1.0)),
            "cube_pnp_quality_contract": data.cube_detection[
                "quality_contract"],
            "board_metric_scale": data.board_metric_scale,
        },
        "cube_config_source": data.cube_config_source,
        "source_data_provenance": data.source_data_provenance,
        "pose_convention": data.pose_convention,
        "fk_alignment_artifact_sha256": data.alignment_artifact[
            "artifact_sha256"],
        "scope_note": (
            "The shared all-marker prefit initializes optimization only; "
            "held-out events are never used and row objectives still obey "
            "their declared marker/FK treatment."),
    }
    body["artifact_sha256"] = _canonical_json_sha256(body)
    return body


def validate_shared_baseline_artifact(payload: Mapping) -> None:
    if payload.get("artifact_schema") != SHARED_BASELINE_SCHEMA:
        raise ValueError("unknown Table 1 shared-baseline schema")
    unhashed = dict(payload)
    expected = str(unhashed.pop("artifact_sha256", ""))
    if not expected or _canonical_json_sha256(unhashed) != expected:
        raise ValueError("Table 1 shared-baseline SHA-256 mismatch")
    if payload.get("heldout_information_used") is not False:
        raise ValueError("shared baseline must exclude held-out information")
    validate_source_data_provenance(payload.get("source_data_provenance", {}))
    if set(payload.get("row_reference_states", {})) != set(BASELINE_ROWS):
        raise ValueError("shared baseline must contain A0-A6/B1-B3 row states")
    if payload["row_reference_states"]["A6"] != payload["row_reference_states"]["A4"]:
        raise ValueError("pending A6 must start from the byte-identical A4 baseline")


def validate_row_initialization_contract(
        row_states: Mapping[str, PoseState], shared: PoseState,
        fixed_cubes: Mapping[int, np.ndarray],
        aligned_fixed_cubes: Mapping[int, np.ndarray]) -> None:
    """Reject accidental row-specific initialization beyond the treatment."""
    for row, state in row_states.items():
        condition = next(
            item for item in MAIN_ABLATION_CONDITIONS if item.row == row)
        if sorted(state.cams) != sorted(shared.cams):
            raise ValueError(f"{row}: registered-camera baseline drift")
        for camera_id in shared.cams:
            if not np.array_equal(state.cams[camera_id], shared.cams[camera_id]):
                raise ValueError(f"{row}: T_base_C{camera_id} baseline drift")
        if not np.array_equal(state.gtc, shared.gtc):
            raise ValueError(f"{row}: T_gripper_cam baseline drift")
        if "board" in condition.target_set:
            if state.board is None or not np.array_equal(state.board, shared.board):
                raise ValueError(f"{row}: T_base_board baseline drift")
        elif state.board is not None:
            raise ValueError(f"{row}: removed board remains in the row state")
        if "cube" not in condition.target_set:
            if state.cubes:
                raise ValueError(f"{row}: removed cube remains in the row state")
            continue
        if condition.fk_to_cube == POSE_SOURCE_FK_FIXED:
            expected = fixed_cubes
        elif condition.fk_to_cube == POSE_SOURCE_ALIGNED_FK_FIXED:
            expected = aligned_fixed_cubes
        else:
            expected = shared.cubes
        if sorted(state.cubes) != sorted(expected):
            raise ValueError(f"{row}: cube-set baseline drift")
        for set_index, transform in expected.items():
            if not np.array_equal(state.cubes[int(set_index)], transform):
                raise ValueError(f"{row}: cube pose-source baseline drift at set {set_index}")


def validate_result_evaluation_contract(result: Mapping) -> None:
    """Reject output whose evaluated population can differ across rows."""
    protocol = result.get("protocol", {})
    mask = protocol.get("model_independent_path_evaluation_mask")
    if mask is None:
        raise ValueError("result is missing the frozen path-evaluation mask")
    validate_frozen_path_evaluation_mask(mask)
    mask_sha = mask["evaluation_mask_sha256"]
    expected_cross = len(mask["cross_pairs"])
    expected_e2e = len(mask["e2e_units"])
    if protocol.get("model_dependent_test_gating") is not False:
        raise ValueError("result protocol permits model-dependent test gating")
    if protocol.get("relative_pose_reporting") != \
            RELATIVE_POSE_REPORTING_CONTRACT:
        raise ValueError("result is missing the relative-pose reporting policy")
    e_cross_policy = protocol.get("metric_selection_contract", {}).get(
        "e_cross", {})
    if (e_cross_policy.get("reporting_tier") != "supplementary"
            or e_cross_policy.get(
                "may_rank_methods_before_external_gt") is not False):
        raise ValueError("e_cross was promoted above a supplementary metric")
    for row, entry in result.get("rows", {}).items():
        if entry.get("path_evaluation_mask_sha256") != mask_sha:
            raise ValueError(f"{row}: row-level evaluation-mask SHA mismatch")
        has_cube = entry.get("condition", {}).get("target_set") in {"cube", "cube+board"}
        for run in entry.get("runs", []):
            cube_eval = run.get("cube_evaluation_reprojection")
            if not cube_eval:
                raise ValueError(f"{row}: missing train-only cube evaluation")
            if cube_eval.get("artifact_schema") != TRAIN_ONLY_CUBE_EVALUATION_SCHEMA:
                raise ValueError(f"{row}: unknown cube evaluation schema")
            if cube_eval.get("camera_and_hand_eye_frozen") is not True:
                raise ValueError(f"{row}: cube evaluation refit changed calibration")
            if cube_eval.get("heldout_cube_observations_used_for_pose_fit") is not False:
                raise ValueError(f"{row}: heldout cube leaked into pose fit")
            if cube_eval.get("external_ground_truth_used") is not False:
                raise ValueError(f"{row}: cube evaluation used external GT")
            for split_name in ("train", "heldout"):
                cube = cube_eval.get(split_name, {}).get("cube")
                if cube is None or cube.get("n_corners") in (None, 0):
                    raise ValueError(
                        f"{row}: cube evaluation lacks {split_name} cube support")
            metrics = run.get("heldout_path_metrics", {})
            if metrics.get("evaluation_mask_sha256") != mask_sha:
                raise ValueError(f"{row}: run-level evaluation-mask SHA mismatch")
            if metrics.get("model_dependent_gating") is not False:
                raise ValueError(f"{row}: model-dependent path gating is forbidden")
            if metrics.get("output_dependent_pose_gate") is not None:
                raise ValueError(f"{row}: output-dependent pose threshold is forbidden")
            if has_cube:
                if metrics.get("applicable") is not True:
                    raise ValueError(f"{row}: cube-bearing row lacks path metrics")
                if int(metrics.get("n_cross_pairs", -1)) != expected_cross:
                    raise ValueError(
                        f"{row}: e_cross population differs from the frozen mask")
                if int(metrics.get("n_e2e_units", -1)) != expected_e2e:
                    raise ValueError(
                        f"{row}: e_e2e population differs from the frozen mask")
                if int(metrics.get("n_output_rejected", -1)) != 0:
                    raise ValueError(f"{row}: fitted output rejected evaluation units")
            elif metrics.get("applicable") is not False:
                raise ValueError(f"{row}: path metric must be N/A when cube is absent")


def write_outputs(result: dict, out_dir: str) -> None:
    """Write the single canonical raw Table 1 artifact.

    Human-readable tables are generated once from the canonical JSON/CSV
    pipeline.  Keeping runner-local CSV/Markdown summaries created duplicate
    result sources that could drift from the final Table 1.
    """
    validate_result_evaluation_contract(result)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ABLATION_TEST_table1_methods.json"), "w") as handle:
        json.dump(_jsonable(result), handle, indent=2)


def add_frame_prune_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training frame prune/refit")
    group.add_argument(
        "--frame-prune-refit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("After the first fit, prune high-RMSE (event,camera) training "
              "frames, refit once, and rollback unless the full training "
              "robust objective improves."),
    )
    group.add_argument("--frame-prune-mad-multiplier", type=float, default=3.0)
    group.add_argument("--frame-prune-min-rmse-px", type=float, default=4.0)
    group.add_argument("--frame-prune-max-fraction", type=float, default=0.30)
    group.add_argument(
        "--frame-prune-min-observations-per-variable", type=int, default=2)
    group.add_argument(
        "--frame-prune-min-relative-improvement", type=float, default=1e-6)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=("Unified A0-A5/B1-B3 raw-corner reprojection ablation; "
                     "A6 shares the baseline but awaits independent labels"))
    parser.add_argument("--root_folder", default=DEFAULT_SESSION_ROOT)
    parser.add_argument("--intrinsics_dir", default="intrinsics")
    parser.add_argument(
        "--cube-config", "--cube_config", dest="cube_config", default=None,
        help=("Cube geometry JSON to override the project default when "
              "detecting observations directly (i.e. when --observation-manifest "
              "is unset/empty and detectors are re-run here). Ignored when a "
              "frozen --observation-manifest is used, since that manifest "
              "already recorded its own cube_config_source at Step 04 time."))
    parser.add_argument("--calib_dir", default=None,
                        help="Default: <session>/calib_out from --root_folder.")
    parser.add_argument(
        "--out_dir",
        help=("Output directory. Default: ABLATION_TEST_result_<MMDD>/<sessionNN>/ABLATION_TEST_table1 "
              "inferred from --root_folder."),
    )
    parser.add_argument(
        "--rows", default=",".join(RUNNABLE_ROWS),
        help="Comma-separated subset of A0,A1,A2,A3,A4,A5,B1,B2,B3.")
    parser.add_argument(
        "--include_sets", default="",
        help="Use only these set_index values (comma list/ranges, e.g. 5-12).")
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--split_seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--min_train_eih_cube_events", type=int, default=3)
    parser.add_argument("--num_inits", type=int, default=3)
    parser.add_argument("--init_translation_mm", type=float, default=5.0)
    parser.add_argument("--init_rotation_deg", type=float, default=1.0)
    parser.add_argument("--max_nfev", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--rotation_scale_rad", type=float, default=1.0,
                        help="Physical rotation represented by one optimizer unit.")
    parser.add_argument("--translation_scale_m", type=float, default=1.0,
                        help="Physical translation represented by one optimizer unit.")
    parser.add_argument("--x_scale_mode", choices=["unit", "jac"], default="jac",
                        help="SciPy trust-region x_scale mode.")
    parser.add_argument("--loss", choices=["huber", "soft_l1", "linear"], default="soft_l1",
                        help=("Identical pixel loss for every row; canonical "
                              "default is smooth robust soft_l1."))
    parser.add_argument("--f_scale_px", type=float, default=2.0,
                        help="Robust-loss transition in pixels; ignored by linear loss.")
    parser.add_argument(
        "--residual-weighting",
        choices=sorted(RESIDUAL_WEIGHTING_MODES),
        default="per_corner",
        help=("Visual residual aggregation. 'per_corner' is canonical; "
              "'equal_observation_total' scales each observation block by "
              "1/sqrt(number of detected corners)."),
    )
    add_frame_prune_arguments(parser)
    parser.add_argument(
        "--image_scale", type=float, default=1.0,
        help=("Detection raster scale. Detected points are mapped back to native "
              "pixel coordinates; 1.0 preserves the production path."))
    parser.add_argument(
        "--cube-observation-policy",
        choices=("core_multiface", "legacy"),
        default="core_multiface",
        help=("core_multiface keeps only >=2-face non-planar cube images; "
              "legacy reproduces the prior corner-count policy for comparison."),
    )
    parser.add_argument(
        "--observation-manifest", "--observation_manifest",
        dest="observation_manifest",
        help=("Step 04 frozen-corner manifest. When set, detectors are not run "
              "and this exact observation population is used."),
    )
    parser.add_argument(
        "--allow-relocated-session-root", "--allow_relocated_session_root",
        dest="allow_relocated_session_root", action="store_true",
        help=("Replay a --observation-manifest captured in another checkout. "
              "Only the recorded absolute path prefix is remapped; every "
              "recorded SHA-256 is still verified against the local files."),
    )
    parser.add_argument(
        "--observation-filter-policy", "--observation_filter_policy",
        dest="observation_filter_policy",
        choices=("standard", "strict"),
        default="standard",
        help="Selection policy stored in --observation-manifest.",
    )
    parser.add_argument(
        "--align-board-metric-scale", "--align_board_metric_scale",
        dest="align_board_metric_scale", action="store_true",
        help=("Estimate one board object-coordinate scale from train-only "
              "fixed-camera board/cube relative transforms, then freeze it "
              "for train and held-out observations."),
    )
    parser.add_argument(
        "--fk_covariance_json",
        help=("Preregistered measured FK covariance. Without it, A4/B1/B2 "
              "are explicitly marked Simulation-prior preflight."))
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Prepare the authenticated shared baseline/artifacts without fitting rows.")
    parser.add_argument("--sanity_only", action="store_true")
    parser.add_argument(
        "--require-e2e-path-metric", "--require_e2e_path_metric",
        dest="require_e2e_path_metric", action="store_true", default=True,
        help="Require the held-out set's e2e cross-camera path metric (default on).")
    parser.add_argument(
        "--no-require-e2e-path-metric", "--no_require_e2e_path_metric",
        dest="require_e2e_path_metric", action="store_false",
        help=("Skip the e2e cross-camera path metric requirement. Needed for "
              "capture protocols with only one gripper-camera event per set "
              "(e.g. Zeus session2 pick-and-place): that single event cannot "
              "simultaneously satisfy per-set training AND be the held-out "
              "e2e sample, so something has to give. Off by default; when "
              "used, e2e diagnostics are simply absent from the report "
              "instead of the run refusing to proceed."))
    parser.add_argument(
        "--include_gripped_cube", action="store_true",
        help=("Also load cube_gripped=True observations (grip-target model, "
              "T_gripper_cube[grasp_idx]) instead of dropping them. Off by "
              "default to preserve prior runs' behavior; needed for datasets "
              "whose fixed-camera cube views only exist while the cube is "
              "gripped (e.g. UR3 session1)."))
    args = parser.parse_args(argv)
    return apply_session_defaults(
        args, {'calib_dir': 'calib_dir', 'out_dir': 'table1_dir',
               'observation_manifest': 'observation_manifest'})


def main(argv=None, force_baseline_only: bool = False) -> None:
    args = parse_args(argv)
    args.baseline_only = bool(args.baseline_only or force_baseline_only)
    validate_main_runner_contract()
    print("[SANITY] noise-free A1=A2")
    sanity = run_noise_free_sanity(args)
    print("  PASS")
    if args.sanity_only:
        print(json.dumps(_jsonable(sanity), indent=2))
        return

    requested = [row.strip() for row in args.rows.split(",") if row.strip()]
    unsupported = sorted(set(requested) - set(RUNNABLE_ROWS))
    if unsupported:
        raise ValueError(f"unknown or non-executable rows: {unsupported}")
    by_row = {condition.row: condition for condition in MAIN_ABLATION_CONDITIONS}
    unknown = [row for row in requested if row not in by_row]
    if unknown:
        raise ValueError(f"unknown rows: {unknown}")
    prepared = prepare_ablation_data(args)
    cube_cfg_source = prepared.cube_config_source
    cube_reason = prepared.cube_detection
    split = prepared.split
    gripper = prepared.gripper
    K_map, D_map = prepared.K_map, prepared.D_map
    robot_T = prepared.robot_T
    train_obs, test_obs = prepared.train_obs, prepared.test_obs
    path_evaluation_mask = prepared.path_evaluation_mask
    handeye_diag = prepared.handeye_diagnostics
    fixed_cubes = prepared.fixed_cubes
    artifact = prepared.alignment_artifact
    os.makedirs(args.out_dir, exist_ok=True)
    artifact_path = os.path.join(args.out_dir, "shared_board_free_fk_cube.json")
    with open(artifact_path, "w") as handle:
        json.dump(_jsonable(artifact), handle, indent=2)

    row_initials = {}
    row_initialization_diagnostics = {}
    for row in RUNNABLE_ROWS:
        state, diagnostics = make_initial_state(
            by_row[row], prepared.shared_reference_state, fixed_cubes,
            prepared.aligned_fk_cubes)
        row_initials[row] = state
        row_initialization_diagnostics[row] = diagnostics
    validate_row_initialization_contract(
        row_initials, prepared.shared_reference_state, fixed_cubes,
        prepared.aligned_fk_cubes)
    baseline_states = dict(row_initials)
    baseline_states["A6"] = row_initials["A4"].clone()
    baseline_artifact = build_shared_baseline_artifact(
        args, prepared, baseline_states)
    validate_shared_baseline_artifact(baseline_artifact)
    baseline_path = os.path.abspath(
        os.path.join(args.out_dir, "shared_train_only_baseline.json"))
    with open(baseline_path, "w") as handle:
        json.dump(_jsonable(baseline_artifact), handle, indent=2)

    if args.baseline_only:
        print(f"[BASELINE] {baseline_path}")
        print(f"[BASELINE] sha256={baseline_artifact['artifact_sha256']}")
        return

    factor_rows_requested = bool(set(requested) & {"A4", "B1", "B2"})
    if factor_rows_requested and args.fk_covariance_json:
        covariances, covariance_provenance = load_fk_covariances(
            args.fk_covariance_json, sorted(prepared.aligned_fk_cubes))
    elif factor_rows_requested:
        covariances, covariance_provenance = preflight_fk_covariances(
            sorted(prepared.aligned_fk_cubes))
    else:
        covariances, covariance_provenance = {}, {
            "path": None,
            "confirmatory_ready": False,
            "status": "not_needed_for_requested_rows",
        }

    result = {
        "protocol": {
            "dataset": args.root_folder,
            "intrinsics_dir": args.intrinsics_dir,
            "requested_set_filter": str(getattr(args, "include_sets", "")),
            "resolved_set_indices": split["eligible_sets"],
            "cube_config_source": cube_cfg_source,
            "source_data_provenance": prepared.source_data_provenance,
            "board_metric_scale": prepared.board_metric_scale,
            "pose_convention": prepared.pose_convention,
            "primary_metric": PRIMARY_METRIC,
            "reprojection_metric_contract": REPROJECTION_METRIC_CONTRACT,
            "set_equal_weight_metric_contract": (
                SET_EQUAL_WEIGHT_REPROJECTION_CONTRACT),
            "split": split,
            "backend": "canonical_corner_reprojection_v1",
            "optimization_structure": OPTIMIZATION_STRUCTURE_CONTRACT,
            "objective_terms": OBJECTIVE_TERM_CONTRACT,
            "inter_camera_coupling": INTER_CAMERA_COUPLING_CONTRACT,
            "relative_pose_reporting": RELATIVE_POSE_REPORTING_CONTRACT,
            "fk_fixed": {
                "mathematical_contract": FK_FIXED_CONTRACT,
                "rows": sorted(RAW_FK_FIXED_ROWS),
            },
            "vision_aligned_fk_fixed": {
                "mathematical_contract": VISION_ALIGNED_FK_FIXED_CONTRACT,
                "rows": sorted(ALIGNED_FK_FIXED_ROWS),
            },
            "optimization_terminology": {
                "seq": "sequential_frozen_stage",
                "U": "unified_joint_optimization",
                "forbidden_synonym_for_seq": "independent",
            },
            "one_runner_for_all_executable_rows": True,
            "visual_objective": "raw_distorted_pixel_corner_reprojection",
            "solver_options": canonical_solver_options(args).to_dict(),
            "frame_prune_refit": canonical_frame_prune_options(args).to_dict(),
            "max_nfev": int(args.max_nfev), "tol": float(args.tol),
            "num_inits": int(args.num_inits),
            "post_correction": False,
            "test_time_refit": False,
            "model_dependent_test_gating": False,
            "model_independent_path_evaluation_mask": path_evaluation_mask,
            "metric_selection_contract": {
                "heldout_reprojection": {
                    "model_dependent_gating": False,
                    "frozen_parameters": True,
                    "reported_components": (
                        "overall/marker/role/camera/marker-camera"),
                },
                "cube_evaluation_reprojection": {
                    "schema": TRAIN_ONLY_CUBE_EVALUATION_SCHEMA,
                    "target": "cube",
                    "camera_and_hand_eye_frozen": True,
                    "free_variables": ("T_base_cube_by_set",),
                    "pose_fit_observations": "train_cube_only",
                    "score_observations": "train_cube_and_heldout_cube",
                    "heldout_information_used_for_pose_fit": False,
                    "external_ground_truth_used": False,
                    "purpose": (
                        "report Train/ALL/Heldout cube RMSE on one shared cube "
                        "population even for board-only A0/B3 without letting "
                        "cube observations update camera calibration"),
                },
                "e_e2e": {
                    "model_dependent_gating": False,
                    "evaluation_mask_sha256": path_evaluation_mask[
                        "evaluation_mask_sha256"],
                },
                "e_cross": {
                    "model_dependent_gating": False,
                    "evaluation_mask_sha256": path_evaluation_mask[
                        "evaluation_mask_sha256"],
                    "role": "method_specific_heldout_consistency",
                    "reporting_tier": "supplementary",
                    "may_rank_methods_before_external_gt": False,
                },
                "e_task_pose": {
                    "available_external_ground_truth": False,
                    "allowed_current_label": TASK_POSE_PROXY_LABEL,
                    "may_be_described_as_absolute_accuracy": False,
                    "protocol": "separate_position_holdout_not_event_holdout",
                },
            },
            "comparison_component_contract": EVALUATION_COMPARISON_CONTRACT,
            "backbone_fk_used_in_all_conditions": True,
            "shared_train_only_baseline": {
                "path": baseline_path,
                "schema": SHARED_BASELINE_SCHEMA,
                "sha256": baseline_artifact["artifact_sha256"],
                "shared_reference_state_sha256": baseline_artifact[
                    "shared_reference_state_sha256"],
                "heldout_information_used": False,
                "initialization_only": True,
                "rows": list(BASELINE_ROWS),
                "executable_consumers": list(RUNNABLE_ROWS),
            },
            "fk_factor": {
                "mathematical_contract": CORRECTED_FK_FACTOR_CONTRACT,
                "covariance": covariance_provenance,
                "rows": ["A4", "B1", "B2"],
                "loss": "huber",
                "huber_f_scale": HUBER_F_SCALE,
                "visual_problem": "CornerReprojectionProblem",
                "external_ground_truth_used": False,
            },
            "shared_reference_initialization": (
                prepared.shared_reference_diagnostics),
            "board_handeye_initialization": handeye_diag,
            "observation_support_train": support_report(train_obs, gripper),
            "observation_support_test": support_report(test_obs, gripper),
            "cube_detection": cube_reason,
            "freeze_contract": {
                row: {
                    "stage1_free": list(spec.stage1_free),
                    "stage2_free": list(spec.stage2_free),
                    "stage2_frozen_from_stage1": list(spec.stage2_frozen_from_stage1),
                    "alternating_pass": False,
                }
                for row, spec in SEQUENTIAL_STAGE_SPECS.items()
            },
        },
        "noise_free_sanity": sanity,
        "shared_fk_cube_artifact": {
            "path": artifact_path,
            "sha256": artifact["artifact_sha256"],
            "consumers": sorted(FK_ALIGNMENT_SHARED_ROWS),
            "board_information_used": False,
            "heldout_information_used": False,
        },
        "pending_rows": PENDING_ROWS,
        "rows": {},
    }
    for row in requested:
        condition = by_row[row]
        print(f"[RUN] {row}: {condition.target_set}/{condition.unified}/"
              f"{condition.fk_to_cube}/{condition.fk_to_board}")
        initial_state = row_initials[row]
        init_diag = row_initialization_diagnostics[row]
        runs = []
        for seed in range(int(args.num_inits)):
            print(f"  seed={seed}")
            if row in {"A4", "B1", "B2"}:
                run = run_factor_condition_once(
                    condition, initial_state, prepared, covariances, seed, args)
            else:
                run = run_condition_once(
                    condition, initial_state, train_obs, test_obs,
                    gripper, robot_T, K_map, D_map, seed, args,
                    path_evaluation_mask, prepared.visual_cubes)
            runs.append(run)
            held = run["heldout_reprojection"].get("overall", {}).get("rmse_px")
            print(f"    converged={run['converged']} heldout_reproj={held}")
        result["rows"][row] = {
            "condition": {
                "target_set": condition.target_set,
                "unified": condition.unified,
                "optimization_label": (
                    "sequential_frozen_stage"
                    if condition.unified == "seq"
                    else "unified_joint_optimization"),
                "fk_to_cube": condition.fk_to_cube,
                "fk_to_board": condition.fk_to_board,
                "label": condition.label,
            },
            "initialization": init_diag,
            "n_registered_cams": len(initial_state.cams),
            "shared_fk_artifact_sha256": (
                artifact["artifact_sha256"] if row in FK_ALIGNMENT_SHARED_ROWS else None),
            "path_evaluation_mask_sha256": path_evaluation_mask[
                "evaluation_mask_sha256"],
            "runs": runs,
            "initialization_dispersion": transform_dispersion(runs),
        }
        write_outputs(result, args.out_dir)
    print(f"[SAVE] {args.out_dir}/ABLATION_TEST_table1_methods.json")


if __name__ == "__main__":
    main()
