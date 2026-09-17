from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from calibration_pipeline import report
from zeus_gello_calibration import report_table1 as zeus_report


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _legacy_payload():
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return {
        "protocol": {"dataset": "fixture", "split": {"unit": "event"}},
        "rows": {
            method: {
                "condition": {"label": method, "target_set": "board + cube",
                              "optimization_label": "Unified", "fk_to_cube": "estimated"},
                "runs": [{
                    "seed": 0, "converged": True,
                    "train_reprojection": {"overall": {"rmse_px": 1.25}},
                    "heldout_reprojection": {"overall": {"rmse_px": 2.5}},
                    "transforms": {"T_base_Ci": {"0": identity},
                                   "T_gripper_cam": identity},
                }],
            } for method in report.METHOD_ORDER
        },
    }


def test_v4_delegates_saved_payload_and_accepts_variable_output_keys(tmp_path, monkeypatch):
    payload = {"schema": zeus_report.SCHEMA, "rows": {}, "source_data": {"root": "capture_new"}}
    source = _write_json(tmp_path / "methods.json", payload)
    original = source.read_bytes()
    out = tmp_path / "export"
    observed = []

    def fake_write_reports(result, output_dir):
        observed.append((result, output_dir))
        output_dir.mkdir(parents=True)
        return {"markdown": output_dir / "summary.md",
                "fold_metrics": output_dir / "fold_metrics.csv"}

    monkeypatch.setattr(zeus_report, "write_reports", fake_write_reports)
    result = report.write_report(source, out)
    assert observed == [(payload, out)]
    assert result["schema"] == zeus_report.SCHEMA
    assert result["markdown"] == str(out / "summary.md")
    assert result["fold_metrics"] == str(out / "fold_metrics.csv")
    assert "matrices" not in result
    assert source.read_bytes() == original
    assert Path(result["raw_json"]).read_bytes() == original
    json.dumps(result)  # main() must be able to print Path outputs as JSON.


def test_v4_entrypoint_exports_next_to_explicit_input(tmp_path, capsys):
    source = _write_json(tmp_path / "methods.json", {"schema": zeus_report.SCHEMA, "rows": {}})
    original = source.read_bytes()
    report.main(["--table1", str(source)])
    result = json.loads(capsys.readouterr().out)
    for key in ("markdown", "csv", "fold_metrics"):
        path = Path(result[key])
        assert path.parent == tmp_path
        assert path.exists()
    assert not (tmp_path / "calibration_matrices.json").exists()
    assert Path(result["raw_json"]).read_bytes() == original
    assert source.read_bytes() == original


def test_legacy_export_preserves_original_metrics_and_matrix_artifacts(tmp_path):
    source = _write_json(tmp_path / "methods.json", _legacy_payload())
    result = report.write_report(source, tmp_path / "legacy")
    assert result["rows"] == len(report.METHOD_ORDER)
    with Path(result["csv"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["method"] for row in rows] == list(report.METHOD_ORDER)
    assert all(float(row["heldout_rmse_px_mean"]) == 2.5 for row in rows)
    assert all("heldout_test_cross_view_cube_rmse_px" not in row for row in rows)
    matrices = json.loads(Path(result["matrices"]).read_text())
    assert matrices["artifact_schema"] == "calibration_matrices_v1"
    assert tuple(matrices["rows"]) == report.METHOD_ORDER
    assert "markdown" not in result


def test_unknown_explicit_schema_is_not_exported_as_legacy(tmp_path):
    payload = {**_legacy_payload(), "schema": "table1_zeus_future_v5"}
    source = _write_json(tmp_path / "methods.json", payload)
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="Unsupported Table 1 schema"):
        report.write_report(source, out)
    assert not out.exists()
