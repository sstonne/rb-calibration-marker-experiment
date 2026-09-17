"""Placement holdout excludes images and optimization factors from every target."""

from __future__ import annotations

import json

import numpy as np
import pytest

from calibration_pipeline.reprojection import PixelObs
from zeus_gello_calibration import table1_zeus as table1


def _data():
    def observation(marker, camera, event, placement=None, grasp=None):
        return PixelObs(
            marker=marker, cam=camera, event=event, set_idx=placement,
            grasp_idx=grasp,
            object_points=np.zeros((4, 3)), image_points=np.zeros((4, 2)),
        )

    gripper = table1.GRIPPER
    return {
        "obs_s1": [observation("cube", 0, 0, grasp=0)],
        "obs_s2_fixed": [observation("cube", 0, 1000 + placement, placement) for placement in (1, 2)],
        "obs_s2_gripper": [observation("cube", gripper, 1000 + placement, placement) for placement in (1, 2)],
        # P2 board metadata deliberately has no set_idx: split by event ID.
        "obs_s2_board": [
            observation("board", camera, 1000 + placement)
            for placement in (1, 2) for camera in (0, gripper)
        ],
        "obs_s3": [observation("board", gripper, 2000)],
        "include_session2_board": True,
        "items_by_index": {1: {"target": [0.0] * 6}, 2: {"target": [0.0] * 6}},
        "grasp_init": np.eye(4),
        "cam_init": {0: np.eye(4)},
        "K_map": {0: np.eye(3), gripper: np.eye(3)},
        "D_map": {0: np.zeros(5), gripper: np.zeros(5)},
    }


@pytest.mark.parametrize("row", ["A0", "A1", "A2", "A4", "A5", "B1", "B2", "B3"])
def test_lopo_removes_heldout_images_pose_variables_and_fk_factors(monkeypatch, row):
    data = _data()
    initialized_events = []
    solver_calls = []

    def observed_pose(observation, _K, _D):
        initialized_events.append(observation.event)
        return np.eye(4)

    def solve(**kwargs):
        solver_calls.append(kwargs)
        return kwargs["reference_state"], {"success": True}

    monkeypatch.setattr(table1.fcm, "solve_observed_pose", observed_pose)
    monkeypatch.setattr(table1, "solve_corner_reprojection", solve)
    monkeypatch.setattr(table1, "solve_factorized_fk", solve)
    monkeypatch.setattr(table1.fcm, "per_corner_errors", lambda *_: [3.0, 4.0])

    _state, train_px, _n_obs, converged = table1.fit_row(
        row, data, 1,
        {event: np.eye(4) for event in (0, 1001, 1002, 2000)},
        np.eye(4), np.eye(4),
    )

    assert converged
    assert train_px == pytest.approx(5.0)
    assert solver_calls
    assert 1001 not in initialized_events
    for call in solver_calls:
        assert all(observation.event != 1001 for observation in call["observations"])
        assert all(key != ("cube", 1) for key in call["variable_keys_"])
        assert 1 not in call.get("fk_targets", {})
        assert 1 not in call.get("fk_covariances", {})
        if table1.ROWS[row]["fk"] in ("none", "corrected_factor"):
            assert 1 not in call["reference_state"].cubes


def test_all_fit_restores_every_placement_and_board_event(monkeypatch):
    data = _data()
    solver_calls = []

    def solve(**kwargs):
        solver_calls.append(kwargs)
        return kwargs["reference_state"], {"success": True}

    monkeypatch.setattr(table1.fcm, "solve_observed_pose", lambda *_: np.eye(4))
    monkeypatch.setattr(table1, "solve_factorized_fk", solve)
    monkeypatch.setattr(table1.fcm, "per_corner_errors", lambda *_: [0.0, 0.0])

    table1.fit_row(
        "A4", data, None,
        {event: np.eye(4) for event in (0, 1001, 1002, 2000)},
        np.eye(4), np.eye(4),
    )

    assert len(solver_calls) == 1
    call = solver_calls[0]
    assert set(call["reference_state"].cubes) == {1, 2}
    assert set(call["fk_targets"]) == {1, 2}
    for event in (1001, 1002):
        assert {observation.marker for observation in call["observations"] if observation.event == event} == {"cube", "board"}


def _write_mechanical_transform(tmp_path, transform, **metadata):
    path = tmp_path / "mechanical.json"
    path.write_text(json.dumps({
        "vision_used": False,
        "source": "independently measured flange-to-Cube datum",
        "T_flange_cube": np.asarray(transform).tolist(),
        **metadata,
    }), encoding="utf-8")
    return path


def test_mechanical_loader_accepts_vision_free_rigid_transform(tmp_path):
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = [0.001, -0.002, 0.165]
    path = _write_mechanical_transform(tmp_path, transform)

    loaded, provenance = table1.load_mechanical_transform(path)

    assert loaded == pytest.approx(transform)
    assert provenance["vision_used"] is False
    assert provenance["source"] == "independently measured flange-to-Cube datum"


@pytest.mark.parametrize("metadata", [{"vision_used": True}, {"vision_used": None}, {"source": ""}])
def test_mechanical_loader_rejects_vision_derived_or_unproven_transform(tmp_path, metadata):
    path = _write_mechanical_transform(tmp_path, np.eye(4), **metadata)

    with pytest.raises(ValueError, match="vision_used=false and source"):
        table1.load_mechanical_transform(path)


@pytest.mark.parametrize("invalid", [
    np.eye(3),
    np.diag([1.0, 1.0, -1.0, 1.0]),  # Reflection has determinant -1.
    np.diag([1.0, 2.0, 1.0, 1.0]),   # Scale is not a rotation.
    np.diag([1.0, 1.0, 1.0, 0.0]),
    np.array([[1.0, 0.0, 0.0, np.nan], [0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
])
def test_mechanical_loader_rejects_non_rigid_or_nonfinite_matrix(tmp_path, invalid):
    path = _write_mechanical_transform(tmp_path, invalid)

    with pytest.raises(ValueError, match=r"must be SE\(3\)"):
        table1.load_mechanical_transform(path)


def test_a3_uses_supplied_mechanical_transform_without_vision_offset(monkeypatch):
    data = _data()
    mechanical = np.eye(4)
    mechanical[:3, :3] = np.diag([1.0, -1.0, -1.0])
    mechanical[:3, 3] = [0.001, -0.002, 0.165]
    data["mechanical_grasp"] = mechanical
    # Make the independently fitted P1 transform visibly different.
    data["grasp_init"][:3, 3] = [0.020, 0.030, 0.040]
    data["items_by_index"][2]["target"] = [100.0, -50.0, 200.0, 30.0, 0.0, 0.0]
    calls = []

    def solve(**kwargs):
        calls.append(kwargs)
        return kwargs["reference_state"], {"success": True}

    monkeypatch.setattr(table1, "solve_corner_reprojection", solve)
    monkeypatch.setattr(table1.fcm, "per_corner_errors", lambda *_: [0.0, 0.0])

    state, _error, _n_observations, converged = table1.fit_row(
        "A3", data, 1,
        {event: np.eye(4) for event in (0, 1001, 1002, 2000)},
        np.eye(4), np.eye(4),
    )

    assert converged
    assert state.grasps[0] == pytest.approx(mechanical)
    expected_cube = table1.fcm.pose6_to_T(data["items_by_index"][2]["target"]) @ mechanical
    assert state.cubes[2] == pytest.approx(expected_cube)
    assert all(kind not in {"cube", "grasp"} for kind, _index in calls[0]["variable_keys_"])
    assert all(observation.event != 1001 for observation in calls[0]["observations"])
