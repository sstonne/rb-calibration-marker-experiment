"""Render saved Table 1 results using their schema, without rerunning calibration."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping

from calibration_pipeline.runtime import DEFAULT_SESSION_ROOT, session_paths


METHOD_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3")
CANONICAL_LABEL_OVERRIDES = {
    "A3": "FK hard fixed",
    "A4": "corrected-FK soft factor",
    "A5": "corrected-FK hard fixed (VISION-aligned)",
}
CANONICAL_POSE_OVERRIDES = {
    "estimated": "VISION",
    "raw-FK-fixed": "FK hard fixed",
    "corrected-FK-factor": "corrected-FK soft factor",
    "vision-aligned-FK-fixed": "corrected-FK hard fixed (VISION-aligned)",
}

MATRIX_SEMANTICS = {
    "T_base_Ci": (
        "T^B_Ci; fixed-camera coordinates to robot-base coordinates; "
        "final deployable extrinsic; 4x4 SE(3), translation in meters"),
    "T_gripper_cam": (
        "T^G_C; wrist-camera coordinates to robot-gripper coordinates; "
        "final deployable hand-eye transform; 4x4 SE(3), translation in meters"),
    "T_base_board": (
        "T^B_board; board coordinates to robot-base coordinates; optimized "
        "target pose, not a camera calibration deliverable"),
    "T_base_cube_by_set": (
        "T^B_cube(s); cube coordinates to robot-base coordinates for each set; "
        "optimized/fixed target pose, not a camera calibration deliverable"),
}


def _numbers(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values
            if isinstance(value, (int, float)) and math.isfinite(float(value))]


def _mean_std(values: Iterable[Any]) -> tuple[float | None, float | None]:
    numeric = _numbers(values)
    if not numeric:
        return None, None
    return mean(numeric), pstdev(numeric)


def _frame_prune_records(value: Any) -> list[dict]:
    records: list[dict] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "frame_prune_refit" and isinstance(item, Mapping):
                records.append(dict(item))
            else:
                records.extend(_frame_prune_records(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_frame_prune_records(item))
    return records


def _display_condition(method: str, row: Mapping[str, Any]) -> dict:
    condition = dict(row["condition"])
    if method in CANONICAL_LABEL_OVERRIDES:
        condition["label"] = CANONICAL_LABEL_OVERRIDES[method]
    return condition


def _validate(payload: dict, representative_seed: int) -> None:
    rows = payload.get("rows", {})
    if tuple(rows) != METHOD_ORDER:
        raise ValueError(
            f"Table 1 must contain rows {METHOD_ORDER}; got {tuple(rows)}")
    for method in METHOD_ORDER:
        runs = rows[method].get("runs", [])
        if not runs:
            raise ValueError(f"{method} has no calibration runs")
        seeds = {int(run.get("seed", -1)) for run in runs}
        if representative_seed not in seeds:
            raise ValueError(
                f"{method} lacks representative seed {representative_seed}")
        for run in runs:
            transforms = run.get("transforms", {})
            if not transforms.get("T_base_Ci") or transforms.get(
                    "T_gripper_cam") is None:
                raise ValueError(f"{method}/seed{run.get('seed')} lacks final transforms")


def _row_summary(method: str, row: dict,
                 representative_seed: int) -> dict:
    runs = row["runs"]
    train_mean, train_std = _mean_std(
        run["train_reprojection"]["overall"].get("rmse_px") for run in runs)
    heldout_mean, heldout_std = _mean_std(
        run["heldout_reprojection"]["overall"].get("rmse_px") for run in runs)
    prune = [record for run in runs
             for record in _frame_prune_records(run.get("stages", {}))]
    dispersion = row.get("initialization_dispersion", {})
    translation_max = max(_numbers(
        item.get("translation_max_mm") for item in dispersion.values()),
        default=None)
    rotation_max = max(_numbers(
        item.get("rotation_max_deg") for item in dispersion.values()),
        default=None)
    representative = next(
        run for run in runs if int(run["seed"]) == representative_seed)
    transform = representative["transforms"]
    heldout = representative["heldout_reprojection"]["overall"]
    train = representative["train_reprojection"]["overall"]
    condition = _display_condition(method, row)
    return {
        "method": method,
        "label": condition["label"],
        "targets": condition["target_set"],
        "optimization": condition["optimization_label"],
        "fk_to_cube": CANONICAL_POSE_OVERRIDES.get(
            condition["fk_to_cube"], condition["fk_to_cube"]),
        "converged_runs": sum(bool(run.get("converged")) for run in runs),
        "total_runs": len(runs),
        "train_rmse_px_mean": train_mean,
        "train_rmse_px_std": train_std,
        "heldout_rmse_px_mean": heldout_mean,
        "heldout_rmse_px_std": heldout_std,
        "train_observations": train.get("n_observations"),
        "train_corners": train.get("n_corners"),
        "heldout_observations": heldout.get("n_observations"),
        "heldout_corners": heldout.get("n_corners"),
        "solver_stages": len(prune),
        "prune_refit_attempts": sum(bool(record.get("selection", {}).get(
            "attempted")) for record in prune),
        "prune_refit_accepted": sum(bool(record.get("accepted")) for record in prune),
        "prune_refit_rollbacks": sum(bool(record.get("rolled_back")) for record in prune),
        "pruned_frames_considered": sum(int(record.get("selection", {}).get(
            "n_pruned_frames", 0)) for record in prune),
        "seed_dispersion_translation_max_mm": translation_max,
        "seed_dispersion_rotation_max_deg": rotation_max,
        "representative_seed": representative_seed,
        "fixed_camera_ids": ",".join(sorted(
            transform["T_base_Ci"], key=lambda value: int(value))),
        "has_board_pose": transform.get("T_base_board") is not None,
        "cube_pose_count": len(transform.get("T_base_cube_by_set", {})),
    }


def _matrix_artifact(payload: dict, source: Path,
                     representative_seed: int) -> dict:
    return {
        "artifact_schema": "calibration_matrices_v1",
        "source_table1": str(source.resolve()),
        "dataset": payload["protocol"].get("dataset"),
        "representative_seed": representative_seed,
        "representative_seed_contract": (
            "seed 0 is the unperturbed shared initialization; it is fixed "
            "before held-out evaluation and is not chosen by held-out score"),
        "matrix_semantics": MATRIX_SEMANTICS,
        "rows": {
            method: {
                "condition": _display_condition(method, payload["rows"][method]),
                "initialization_dispersion": payload["rows"][method].get(
                    "initialization_dispersion", {}),
                "runs": [
                    {
                        "seed": int(run["seed"]),
                        "converged": bool(run.get("converged")),
                        "train_reprojection_rmse_px": run[
                            "train_reprojection"]["overall"].get("rmse_px"),
                        "heldout_reprojection_rmse_px": run[
                            "heldout_reprojection"]["overall"].get("rmse_px"),
                        "transforms": run["transforms"],
                    }
                    for run in payload["rows"][method]["runs"]
                ],
            }
            for method in METHOD_ORDER
        },
    }


def write_report(table1_path: Path, out_dir: Path,
                 representative_seed: int = 0) -> dict:
    source_bytes = table1_path.read_bytes()
    payload = json.loads(source_bytes)
    schema = payload.get("schema")
    if schema is not None:
        from zeus_gello_calibration.report_table1 import SCHEMA, write_reports

        if schema != SCHEMA:
            raise ValueError(f"Unsupported Table 1 schema: {schema!r}")
        paths = write_reports(payload, out_dir)
        raw_json = out_dir / "ABLATION_TEST_table1_methods.json"
        if table1_path.resolve() != raw_json.resolve():
            raw_json.write_bytes(source_bytes)
        return {
            "source": str(table1_path),
            "schema": schema,
            "raw_json": str(raw_json),
            **{name: str(path) for name, path in paths.items()},
        }

    # Legacy JSON contains seed runs under protocol/rows. Its held-out split
    # keeps its original meaning; exporting it cannot turn it into LOPO data.
    _validate(payload, representative_seed)
    summaries = [
        _row_summary(method, payload["rows"][method], representative_seed)
        for method in METHOD_ORDER
    ]
    matrix_artifact = _matrix_artifact(
        payload, table1_path, representative_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "calibration_summary.csv"
    matrix_path = out_dir / "calibration_matrices.json"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)
    matrix_path.write_text(
        json.dumps(matrix_artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {
        "source": str(table1_path),
        "csv": str(csv_path),
        "matrices": str(matrix_path),
        "rows": len(summaries),
        "representative_seed": representative_seed,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_folder", default=DEFAULT_SESSION_ROOT)
    parser.add_argument(
        "--table1", help="Default: ABLATION_TEST_result_<MMDD>/<session>/ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json")
    parser.add_argument(
        "--out_dir", help="Default: input Table 1 JSON directory")
    parser.add_argument(
        "--representative_seed", type=int, default=0,
        help="Legacy JSON only: fixed representative seed, never selected by held-out score")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = session_paths(args.root_folder)
    table1_path = Path(args.table1 or paths["table1_result"])
    out_dir = Path(args.out_dir) if args.out_dir else table1_path.parent
    result = write_report(
        table1_path, out_dir, representative_seed=args.representative_seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
