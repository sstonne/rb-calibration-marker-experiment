from __future__ import annotations

import json

import numpy as np
import pytest

from calibration_pipeline.apriltag_cube import inv_T
from calibration_pipeline.reprojection import PixelObs, PoseState, project_points
from zeus_gello_calibration import table1_zeus as table1


def _transform(x=0.0, y=0.0, z=0.0):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def _two_camera_fixture():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    D = np.zeros(5, dtype=np.float64)
    object_points = np.array([
        [-0.04, -0.04, 0.00],
        [0.04, -0.04, 0.00],
        [0.04, 0.04, 0.00],
        [-0.04, 0.04, 0.00],
        [-0.04, -0.04, 0.08],
        [0.04, -0.04, 0.08],
        [0.04, 0.04, 0.08],
        [-0.04, 0.04, 0.08],
    ])
    T_base_cube = _transform(z=1.0)
    camera_poses = {0: _transform(), 1: _transform(x=0.20)}
    observations = []
    camera_cube_poses = {}
    for camera, T_base_camera in camera_poses.items():
        T_camera_cube = inv_T(T_base_camera) @ T_base_cube
        camera_cube_poses[camera] = T_camera_cube
        observations.append(PixelObs(
            marker="cube",
            cam=camera,
            event=1000,
            set_idx=0,
            object_points=object_points,
            image_points=project_points(T_camera_cube, object_points, K, D),
        ))
    state = PoseState(
        cams=camera_poses,
        gtc=np.eye(4),
        board=None,
        cubes={},
    )
    return observations, camera_cube_poses, state, T_base_cube, {0: K, 1: K}, {0: D, 1: D}


def test_cross_view_uses_bidirectional_source_only_transfer(monkeypatch) -> None:
    observations, camera_cube_poses, state, _cube, K_map, D_map = _two_camera_fixture()
    monkeypatch.setattr(
        table1,
        "solve_observed_pose",
        lambda observation, _K, _D: camera_cube_poses[int(observation.cam)],
    )

    result = table1.cross_view_transfer_stats(
        observations, state, {}, K_map, D_map)

    assert result["rmse_px"] == pytest.approx(0.0, abs=1e-10)
    assert result["n_observations"] == 2
    assert result["n_pairs"] == 1
    assert result["n_directions"] == 2
    assert result["by_pair_type"]["fixed_fixed"]["n_pairs"] == 1
    assert result["by_pair_type"]["fixed_gripper"]["n_pairs"] == 0


def test_cube_reprojection_uses_two_dimensional_corner_rmse() -> None:
    observations, _camera_cube_poses, state, cube, K_map, D_map = _two_camera_fixture()
    shifted = [
        PixelObs(
            marker=observation.marker,
            cam=observation.cam,
            event=observation.event,
            set_idx=observation.set_idx,
            object_points=observation.object_points,
            image_points=observation.image_points + np.array([3.0, 4.0]),
        )
        for observation in observations
    ]

    result = table1.cube_reprojection_stats(
        shifted, {0: cube}, state, {}, K_map, D_map)

    assert result["rmse_px"] == pytest.approx(5.0)
    assert result["sum_squared_error_px2"] == pytest.approx(400.0)
    assert result["n_corners"] == 16
    assert result["n_residual_components"] == 32


def test_cube_reprojection_pools_corners_across_unequal_placements() -> None:
    observations, _poses, state, cube, K_map, D_map = _two_camera_fixture()
    observation = observations[0]
    unequal = [
        PixelObs(
            marker="cube", cam=0, event=set_index, set_idx=set_index,
            object_points=observation.object_points[:count],
            image_points=observation.image_points[:count] + shift,
        )
        for set_index, count, shift in ((0, 2, [3.0, 4.0]), (1, 8, [0.0, 0.0]))
    ]

    result = table1.cube_reprojection_stats(
        unequal, {0: cube, 1: cube}, state, {}, K_map, D_map)

    assert result["n_corners"] == 10
    assert result["sum_squared_error_px2"] == pytest.approx(50.0)
    assert result["rmse_px"] == pytest.approx(np.sqrt(50.0 / 10.0))
    assert [item["rmse_px"] for item in result["per_set"]] == pytest.approx([5.0, 0.0])


def test_cross_view_pools_destination_corners_across_both_directions(monkeypatch) -> None:
    observations, poses, state, _cube, K_map, D_map = _two_camera_fixture()
    second = observations[1]
    observations[1] = PixelObs(
        marker="cube", cam=1, event=1000, set_idx=0,
        object_points=second.object_points[:4],
        image_points=second.image_points[:4] + [3.0, 4.0],
    )
    monkeypatch.setattr(
        table1, "solve_observed_pose",
        lambda observation, _K, _D: poses[int(observation.cam)],
    )

    result = table1.cross_view_transfer_stats(observations, state, {}, K_map, D_map)

    assert result["n_pairs"] == 1
    assert result["n_directions"] == 2
    assert result["n_corners"] == 12
    assert result["n_residual_components"] == 24
    assert result["sum_squared_error_px2"] == pytest.approx(100.0)
    assert result["rmse_px"] == pytest.approx(np.sqrt(100.0 / 12.0))


def test_cross_view_destination_pnp_does_not_change_forward_prediction(monkeypatch) -> None:
    observations, poses, state, _cube, K_map, D_map = _two_camera_fixture()
    predicted_poses = []

    def record_projection(pose, object_points, K, D):
        predicted_poses.append(np.asarray(pose).copy())
        return project_points(pose, object_points, K, D)

    monkeypatch.setattr(table1, "project_points", record_projection)
    monkeypatch.setattr(
        table1, "solve_observed_pose",
        lambda observation, _K, _D: poses[int(observation.cam)],
    )
    baseline = table1.cross_view_transfer_stats(observations, state, {}, K_map, D_map)
    assert baseline["rmse_px"] == pytest.approx(0.0, abs=1e-10)
    baseline_predictions = list(predicted_poses)
    predicted_poses.clear()

    # Changing camera B's PnP affects B -> A only. A -> B must continue to use
    # A's measured pose, even though both PnP poses are solved for the pair.
    poses[1] = _transform(x=0.10) @ poses[1]
    shifted = table1.cross_view_transfer_stats(observations, state, {}, K_map, D_map)

    assert len(predicted_poses) == 2
    assert predicted_poses[0] == pytest.approx(baseline_predictions[0])
    assert not np.allclose(predicted_poses[1], baseline_predictions[1])
    assert shifted["rmse_px"] > 0.0
    assert shifted["n_directions"] == 2


def test_fold_aggregation_pools_corners_instead_of_averaging_fold_means() -> None:
    def metric(count, squared_sum):
        return {
            "mse_px2": squared_sum / count,
            "sum_squared_error_px2": squared_sum,
            "n_corners": count,
        }

    folds = [
        {"heldout_test": {"cube_reprojection": metric(2, 50.0)}},
        {"heldout_test": {"cube_reprojection": metric(8, 0.0)}},
    ]

    result = table1.aggregate_fold_metric(folds, "heldout_test", "cube_reprojection")

    assert result["n_fold_evaluations"] == 2
    assert result["n_corners"] == 10
    assert result["sum_squared_error_px2"] == pytest.approx(50.0)
    assert result["rmse_px"] == pytest.approx(np.sqrt(5.0))


def test_cross_view_fold_aggregation_pools_pair_type_corners() -> None:
    def metric(count, squared_sum):
        return {
            "mse_px2": squared_sum / count if count else float("nan"),
            "sum_squared_error_px2": squared_sum,
            "n_corners": count,
        }

    folds = []
    for count, squared_sum in ((2, 50.0), (8, 0.0)):
        cross_view = metric(count, squared_sum)
        cross_view["by_pair_type"] = {
            "fixed_fixed": metric(count, squared_sum),
            "fixed_gripper": metric(0, 0.0),
        }
        folds.append({"train": {"cross_view": cross_view}})

    result = table1.aggregate_fold_metric(folds, "train", "cross_view")

    assert result["rmse_px"] == pytest.approx(np.sqrt(5.0))
    assert result["by_pair_type"]["fixed_fixed"]["rmse_px"] == pytest.approx(np.sqrt(5.0))
    assert result["by_pair_type"]["fixed_fixed"]["n_corners"] == 10
    assert result["by_pair_type"]["fixed_gripper"]["n_corners"] == 0
    assert np.isnan(result["by_pair_type"]["fixed_gripper"]["rmse_px"])


def test_lopo_excludes_all_cube_and_board_observations_of_heldout_placement() -> None:
    def observation(marker, event, set_index=None):
        return PixelObs(
            marker=marker, cam=0, event=event, set_idx=set_index,
            object_points=np.zeros((4, 3)), image_points=np.zeros((4, 2)),
        )

    heldout_event = table1.fcm.SESSION2_EVENT_OFFSET + 1
    train_event = table1.fcm.SESSION2_EVENT_OFFSET + 2
    data = {
        "obs_s1": [observation("cube", 0)],
        "obs_s2_fixed": [observation("cube", heldout_event, 1), observation("cube", train_event, 2)],
        "obs_s2_gripper": [observation("cube", heldout_event, 1)],
        "obs_s2_board": [observation("board", heldout_event), observation("board", train_event)],
        "obs_s3": [observation("board", 2000)],
        "include_session2_board": True,
    }

    cubes, boards = table1.split_observations(data, ("cube", "board"), drop_set=1)

    assert [item.event for item in cubes] == [0, train_event]
    assert [item.event for item in boards] == [2000, train_event]
    all_cubes, all_boards = table1.split_observations(data, ("cube", "board"))
    assert len(all_cubes) == 4
    assert len(all_boards) == 3


def test_current_zeus_data_contract_separates_a3_nominal_from_a5_corrected() -> None:
    assert table1.ROWS["A3"]["fk"] == "mechanical_fixed"
    assert table1.ROWS["A5"]["fk"] == "corrected_fixed"
    assert table1.grasp_variable_families(table1.ROWS["A3"]) == []
    assert table1.grasp_variable_families(table1.ROWS["A5"]) == []
    assert table1.grasp_variable_families(table1.ROWS["A4"]) == [
        "T_gripper_cube_by_grasp"
    ]

    mechanical = table1.mechanical_flange_cube_transform()
    assert mechanical[:3, :3] == pytest.approx(np.diag([-1.0, 1.0, -1.0]))
    assert mechanical[:3, 3] * 1000.0 == pytest.approx([0.0, 0.0, 160.0])

    contract = table1.mechanical_fk_contract()
    assert contract["role"] == "A3 FK hard fixed only"
    assert contract["vision_used"] is False
    assert contract["measured"] is False


def test_corrected_fk_training_is_not_a3_and_reports_nominal_axis_delta(tmp_path) -> None:
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(json.dumps({
        "T_gripper_cube": np.array([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.162],
            [0.0, 0.0, 0.0, 1.0],
        ]).tolist(),
        "n_captures": 16,
        "pnp_accepted_per_camera": {"fixed1": 16},
        "grasp_init_dispersion_across_cams": {
            "translation_std_mm": 0.4,
            "rotation_std_deg": 0.1,
        },
        "solve_diagnostics": {
            "train_reprojection_rmse_px": 1.0,
            "jacobian_rank_deficient": False,
        },
    }))

    contract = table1.corrected_fk_training_contract(fit_path)

    assert "A4/A5/B1/B2 corrected-FK" in contract["role"]
    assert "forbidden as A3 mechanical FK" in contract["role"]
    assert contract["robot_pose_frame"] == "T_base_flange from tool1=0"
    assert contract["n_captures"] == 16
    assert contract["translation_mm"] == pytest.approx([0.0, 0.0, 162.0])
    assert contract["rotation_deviation_from_nominal_deg"] == pytest.approx(0.0)
    assert contract["grasp_model"].endswith("regrasp repeatability not measured")


def test_external_gt_placeholder_is_empty_not_zero() -> None:
    pending = table1.external_gt_pending()

    assert pending["status"] == "pending"
    assert all(pending[key] is None for key in (
        "mean_tre_mm", "median_tre_mm", "p95_tre_mm",
        "mean_rotation_error_deg", "p95_rotation_error_deg", "failure_rate",
    ))
    assert pending["reason"]
    assert pending["failure_definition"] == "missing_or_failed_predictions_over_all_independent_GT_poses"
