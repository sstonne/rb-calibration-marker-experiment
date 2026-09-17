from __future__ import annotations

import csv
import json
import math

import pytest

from zeus_gello_calibration import table1_zeus as report


def _stats(rmse, count):
    return {"rmse_px": rmse, "mse_px2": rmse ** 2,
            "sum_squared_error_px2": rmse ** 2 * count, "n_corners": count,
            "n_residual_components": 2 * count, "n_observations": 2,
            "n_pairs": 1, "n_directions": 2, "n_sets": 1, "n_events": 1}


def _result():
    folds = []
    for heldout, (rmse, count) in enumerate(((2.0, 1), (4.0, 9))):
        cross = {**_stats(rmse, count), "by_pair_type": {
            "fixed_fixed": _stats(rmse, count), "fixed_gripper": _stats(rmse, count)}}
        folds.append({"set": heldout, "converged": heldout == 0,
                      "training_placement_ids": [1 - heldout], "heldout_placement_ids": [heldout],
                      "train": {"cube_reprojection": _stats(rmse, count), "cross_view": cross},
                      "heldout_test": {"cube_reprojection": _stats(rmse, count), "cross_view": cross}})
    pooled = math.sqrt(14.8)
    summary = {key: 1.23456789 + index for index, key in enumerate(report.PIXEL_COLUMNS)}
    summary["heldout_test_cross_view_cube_rmse_px"] = pooled
    summary.update(n_folds=2, n_converged=1, full_fit_converged=True)
    return {"schema": report.SCHEMA, "source_data": {"root": "/data/capture_0914"},
            "pending_rows": {"A3": "No independently specified T_flange_cube"},
            "rows": {"A2": {"status": "complete", "summary": summary, "folds": folds,
                             "all": {**folds[0]["train"], "training_placement_ids": [0, 1]},
                             "external_gt": {"status": "complete", "mean_tre_mm": 10.125,
                                             "mean_rotation_error_deg": 2.375, "p95_tre_mm": 17.625,
                                             "failure_rate": 0.125}}}}


def _csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_ten_metric_fields_stay_synchronized_across_json_csv_and_markdown(tmp_path):
    result = _result()
    paths = report.write_reports(result, tmp_path)
    csv_row = next(row for row in _csv(paths["csv"]) if row["row"] == "A2")
    markdown = paths["markdown"].read_text()
    expected = {key: result["rows"]["A2"]["summary"][key] for key in report.PIXEL_COLUMNS}
    expected.update({key: result["rows"]["A2"]["external_gt"][key] for key in report.EXTERNAL_COLUMNS})
    assert len(expected) == 10
    for key, value in expected.items():
        assert float(csv_row[key]) == value
        assert report.METRIC_COLUMNS[key] in markdown
    exact_row = "| A2<br>(board+cube+Unified+VISION) | " + " | ".join(f"**{expected[key]:.4f}**" for key in report.METRIC_COLUMNS) + " |"
    assert exact_row in markdown
    assert "heldout_px_vision" not in markdown


def test_missing_and_explicit_pending_rows_never_report_zero(tmp_path):
    result = _result()
    result["rows"]["A2"]["external_gt"] = {"status": "pending", "mean_tre_mm": 0}
    result["rows"]["A3"] = {"status": "pending", "summary": dict.fromkeys(report.PIXEL_COLUMNS, 0)}
    paths = report.write_reports(result, tmp_path)
    records = {row["row"]: row for row in _csv(paths["csv"])}
    assert list(records) == list(report.ROW_ORDER)
    assert all(records["A3"][key] == "Pending" for key in report.METRIC_COLUMNS)
    assert all(records["A0"][key] == "Pending" for key in report.METRIC_COLUMNS)
    assert all(records["A2"][key] == "Pending" for key in report.EXTERNAL_COLUMNS)
    assert records["A3"]["pending_reason"] == "No independently specified T_flange_cube"
    assert "No independently specified T_flange_cube" in paths["markdown"].read_text()


def test_pair_metrics_pool_unequal_corner_support_and_folds_preserve_counts(tmp_path):
    result = _result()
    paths = report.write_reports(result, tmp_path)
    markdown = paths["markdown"].read_text()
    expected = math.sqrt((2 ** 2 * 1 + 4 ** 2 * 9) / 10)
    assert expected != 3.0
    assert "fixed↔fixed Held-out Test px" in markdown
    assert "fixed↔gripper Train px" in markdown
    assert f"| A2 | 2.0000 | {expected:.4f} | {expected:.4f} | 2.0000 | {expected:.4f} | {expected:.4f} |" in markdown
    records = _csv(paths["fold_metrics"])
    selected = [row for row in records if row["split"] == "heldout_test"
                and row["metric"] == "cross_view" and row["pair_type"] == "overall"]
    assert [int(row["n_corners"]) for row in selected] == [1, 9]
    assert [row["converged"] for row in selected] == ["True", "False"]
    assert [json.loads(row["heldout_placement_ids"]) for row in selected] == [[0], [1]]
    assert sum(float(row["sum_squared_error_px2"]) for row in selected) == 148


def test_report_only_cli_regenerates_without_refit(tmp_path, monkeypatch):
    source = tmp_path / "result.json"
    source.write_text(json.dumps(_result()))
    original = source.read_bytes()
    out = tmp_path / "output"
    monkeypatch.setattr(report, "fit_row", lambda *args, **kwargs: pytest.fail("refit"))
    monkeypatch.setattr("sys.argv", ["table1_zeus.py", "--report-only", str(source),
                                     "--report-dir", str(out)])
    report.main()
    assert source.read_bytes() == original
    assert (out / "ABLATION_TEST_table1_methods.json").read_bytes() == original
    assert (out / "ABLATION_TEST_table1_results.csv").exists()
    assert (out / "fold_metrics.csv").exists()
    markdown = (out / "ABLATION_TEST_TABLE1_RESULTS.md").read_text()
    for contract in ("외부 입력: capture_0914", "end-to-end", "A→B", "B→A", "sqrt(mean(dx² + dy²))"):
        assert contract in markdown
    assert "P2 및 GT" not in markdown  # Do not invent a new dataset's geometry history.


def test_rejects_legacy_metrics_instead_of_mislabeling_them(tmp_path):
    with pytest.raises(ValueError, match="Expected"):
        report.write_reports({"schema": "table1_v1", "rows": {}}, tmp_path)
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"schema": "table1_zeus_future_v5", "rows": {}}))
    out = tmp_path / "export"
    with pytest.raises(ValueError, match="Expected"):
        report.regenerate_reports(source, out)
    assert not out.exists()


def test_report_only_defaults_next_to_input_and_keeps_json(tmp_path):
    source = tmp_path / "ABLATION_TEST_table1_methods.json"
    source.write_text(json.dumps({"schema": report.SCHEMA, "rows": {}}))
    original = source.read_bytes()
    paths = report.regenerate_reports(source, source.parent)
    assert all(path.parent == tmp_path and path.exists() for path in paths.values())
    assert source.read_bytes() == original


def test_report_paths_and_structure_are_portable_across_new_datasets(tmp_path):
    reports = []
    for name, prefix in (("capture_0914", "/Users/woo/work"), ("capture_1020", "/home/robot")):
        result = _result()
        result["source_data"]["root"] = f"{prefix}/rb-calibration-marker-experiment/data/{name}"
        result["evaluation_limitations"] = {"geometry": f"independent geometry for {name}"}
        paths = report.write_reports(result, tmp_path / name)
        markdown = paths["markdown"].read_text()
        assert f"data/{name}" in markdown
        assert prefix not in markdown
        assert "rb-calibration-marker-experiment/" not in markdown
        assert len(markdown.splitlines()) < 150
        assert f"independent geometry for {name}" in markdown
        reports.append(([line for line in markdown.splitlines() if line.startswith("## ")],
                        list(_csv(paths["csv"])[0]), [path.name for path in paths.values()]))
    assert reports[0] == reports[1]


def test_compatibility_renderer_uses_same_template_for_new_data(tmp_path):
    from zeus_gello_calibration.table1_zeus import write_markdown_report

    result = _result()
    result["source_data"]["root"] = "/work/rb-calibration-marker-experiment/data/capture_1020"
    requested = tmp_path / "custom.md"
    write_markdown_report(result, requested)
    standard = tmp_path / "ABLATION_TEST_TABLE1_RESULTS.md"
    assert requested.read_bytes() == standard.read_bytes()
    assert "data/capture_1020" in standard.read_text()
    assert "0914" not in standard.read_text()
