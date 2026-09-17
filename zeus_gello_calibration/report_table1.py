"""Render Table 1 v4 JSON without loading cameras or rerunning calibration.

Usage: python -m zeus_gello_calibration.report_table1 --input result.json --out-dir results
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


SCHEMA = "table1_zeus_cube_and_cross_view_v4"
ROW_ORDER = ("A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3")
DEFAULT_CONDITIONS = {
    "A0": ("board", "Sequential", "VISION"),
    "A1": ("board + cube", "Sequential", "VISION"),
    "A2": ("board + cube", "Unified", "VISION"),
    "A3": ("board + cube", "Unified", "raw-FK hard fixed"),
    "A4": ("board + cube", "Unified", "corrected-FK soft factor"),
    "A5": ("board + cube", "Unified", "corrected-FK hard fixed"),
    "B1": ("board + cube", "Sequential", "corrected-FK soft factor"),
    "B2": ("cube", "Unified", "corrected-FK soft factor"),
    "B3": ("board", "Unified", "VISION"),
}
PIXEL_COLUMNS = {
    "all_cube_rmse_px": "ALL Cube RMSE px (Train + Held-out Test)",
    "train_cube_rmse_px": "Train Cube RMSE px",
    "heldout_test_cube_rmse_px": "Held-out Test Cube RMSE px",
    "all_cross_view_cube_rmse_px": "ALL Cross-view Cube RMSE px (Train + Held-out Test)",
    "train_cross_view_cube_rmse_px": "Train Cross-view Cube RMSE px",
    "heldout_test_cross_view_cube_rmse_px": "Held-out Test Cross-view Cube RMSE px",
}
EXTERNAL_COLUMNS = {
    "mean_tre_mm": "TRE mm",
    "mean_rotation_error_deg": "Rotation Error deg",
    "p95_tre_mm": "P95 TRE mm",
    "failure_rate": "Failure Rate",
}
METRIC_COLUMNS = {**PIXEL_COLUMNS, **EXTERNAL_COLUMNS}
SUPPORT_COLUMNS = (
    "n_sets", "n_events", "n_observations", "n_pairs", "n_directions",
    "n_corners", "n_residual_components",
)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _metric(value):
    return float(value) if _finite(value) else "Pending"


def _portable(value):
    """Display repository paths without machine-specific prefixes."""
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable(item) for item in value]
    if isinstance(value, str):
        normalized = value.replace("\\", "/")
        marker = "rb-calibration-marker-experiment/"
        if marker in normalized:
            return normalized.split(marker, 1)[1]
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
            return "외부 입력: " + normalized.rstrip("/").rsplit("/", 1)[-1]
    return value


def _display(value):
    value = _portable(value)
    if value is None:
        return "Pending"
    if isinstance(value, float):
        return f"{value:.4f}" if math.isfinite(value) else "Pending"
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value, ensure_ascii=False)
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _table(lines, headers, records):
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"])
    lines.extend("| " + " | ".join(_display(value) for value in record) + " |" for record in records)
    lines.append("")


def _highlight_minimum(value, minimum):
    formatted = _display(value)
    return f"**{formatted}**" if _finite(value) and value == minimum else formatted


def _rows(result):
    """Include unavailable and unrequested baselines as explicit pending rows."""
    bodies = result.get("rows", {})
    pending = result.get("pending_rows", {})
    for row in ROW_ORDER:
        body = bodies.get(row, {})
        pending_info = pending.get(row, {})
        if isinstance(pending_info, str):
            pending_info = {"pending_reason": pending_info}
        reason = body.get("pending_reason") or pending_info.get("pending_reason") or pending_info.get("reason")
        if not body and not reason:
            reason = "이 결과 JSON에 실행 결과가 없음"
        status = body.get("status", "pending")
        yield row, body, status, reason or ""


def _conditions(row, body):
    condition = body.get("condition", {})
    target, optimization, fk = DEFAULT_CONDITIONS[row]
    if condition.get("targets"):
        target = " + ".join(condition["targets"])
    optimization = {"seq": "Sequential", "uni": "Unified"}.get(condition.get("opt"), optimization)
    fk = {
        "none": "VISION", "mechanical_fixed": DEFAULT_CONDITIONS["A3"][2],
        "corrected_factor": "corrected-FK soft factor",
        "corrected_fixed": "corrected-FK hard fixed",
    }.get(condition.get("fk"), fk)
    return target, optimization, fk


def _summary_records(result):
    records = []
    for row, body, status, reason in _rows(result):
        target, optimization, fk = _conditions(row, body)
        summary = body.get("summary", {})
        external = body.get("external_gt", {})
        record = {"row": row, "status": status, "targets": target,
                  "optimization": optimization, "fk": fk, "pending_reason": reason}
        for key in PIXEL_COLUMNS:
            record[key] = _metric(summary.get(key)) if status != "pending" else "Pending"
        for key in EXTERNAL_COLUMNS:
            record[key] = (_metric(external.get(key))
                           if status != "pending" and external.get("status") == "complete" else "Pending")
        record["external_gt_status"] = external.get("status", "pending")
        for key in ("n_folds", "n_converged", "full_fit_converged"):
            record[key] = summary.get(key, "Pending") if status != "pending" else "Pending"
        records.append(record)
    return records


def _method_label(record):
    targets = record["targets"].replace(" + ", "+")
    optimization = {"Sequential": "Seq"}.get(record["optimization"], record["optimization"])
    return f"{record['row']}<br>({targets}+{optimization}+{record['fk']})"


def _pooled(items):
    """Use sufficient statistics; never average per-fold RMSE values."""
    valid = [item for item in items if item.get("n_corners", 0) > 0
             and _finite(item.get("sum_squared_error_px2"))]
    n_corners = sum(item["n_corners"] for item in valid)
    squared = sum(item["sum_squared_error_px2"] for item in valid)
    return {"rmse_px": math.sqrt(squared / n_corners) if n_corners else None,
            "n_corners": n_corners, "sum_squared_error_px2": squared,
            "n_fold_evaluations": len(valid)}


def _pair_records(result):
    records = []
    for row, body, status, _ in _rows(result):
        for split, label in (("all", "ALL"), ("train", "Train"), ("heldout_test", "Held-out Test")):
            for pair in ("fixed_fixed", "fixed_gripper"):
                if split == "all":
                    stats = body.get("all", {}).get("cross_view", {}).get("by_pair_type", {}).get(pair, {})
                else:
                    stats = body.get("fold_summary", {}).get(f"{split}_cross_view", {}).get("by_pair_type", {}).get(pair)
                    if stats is None:
                        stats = _pooled([fold.get(split, {}).get("cross_view", {}).get("by_pair_type", {}).get(pair, {})
                                         for fold in body.get("folds", [])])
                records.append((row, label, pair.replace("_", "↔"),
                                _metric(stats.get("rmse_px")) if status != "pending" else "Pending",
                                stats.get("n_corners", "Pending") if status != "pending" else "Pending"))
    return records


def _fold_records(result):
    records = []
    for row, body, status, _ in _rows(result):
        if status == "pending":
            continue
        evaluations = [("ALL", "all", body.get("all", {}), body.get("summary", {}).get("full_fit_converged"))]
        for fold in body.get("folds", []):
            evaluations.extend((fold["set"], split, fold, fold.get("converged")) for split in ("train", "heldout_test"))
        for fold_id, split, evaluation, converged in evaluations:
            payload = evaluation if split == "all" else evaluation.get(split, {})
            for metric in ("cube_reprojection", "cross_view"):
                primary = payload.get(metric, {})
                stats_groups = [("overall", primary)]
                if metric == "cross_view":
                    stats_groups.extend((pair, primary.get("by_pair_type", {}).get(pair, {}))
                                        for pair in ("fixed_fixed", "fixed_gripper"))
                for pair, stats in stats_groups:
                    record = {"row": row, "fold": fold_id, "split": split, "metric": metric,
                              "pair_type": pair, "converged": converged if converged is not None else "Pending",
                              "training_placement_ids": json.dumps(evaluation.get("training_placement_ids", [])),
                              "heldout_placement_ids": json.dumps(evaluation.get("heldout_placement_ids", [])),
                              "rmse_px": _metric(stats.get("rmse_px")),
                              "sum_squared_error_px2": _metric(stats.get("sum_squared_error_px2"))}
                    record.update({key: stats.get(key, "Pending") for key in SUPPORT_COLUMNS})
                    records.append(record)
    return records


def _write_csv(path, fields, records):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_reports(result, output_dir):
    """Create Markdown, summary CSV and long-form per-fold CSV from v4 JSON."""
    if result.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}; got {result.get('schema')!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": output_dir / "ABLATION_TEST_TABLE1_RESULTS.md",
             "csv": output_dir / "ABLATION_TEST_table1_results.csv",
             "fold_metrics": output_dir / "fold_metrics.csv"}
    records = _summary_records(result)
    minima = {key: min((record[key] for record in records if _finite(record[key])), default=None)
              for key in METRIC_COLUMNS}
    fold_records = _fold_records(result)
    _write_csv(paths["csv"], list(records[0]), records)
    fold_fields = ["row", "fold", "split", "metric", "pair_type", "converged",
                   "training_placement_ids", "heldout_placement_ids", "rmse_px",
                   "sum_squared_error_px2", *SUPPORT_COLUMNS]
    _write_csv(paths["fold_metrics"], fold_fields, fold_records)
    source = result.get("source_data", {})
    complete = [record for record in records if _finite(record["all_cube_rmse_px"])]
    source_rows = []
    directories = source.get("session_directories", {})
    subdirs = source.get("capture_subdirectories", {})
    for phase in ("p1", "p2", "p3"):
        if phase in directories:
            source_rows.append((phase.upper(), str(Path(directories[phase]) / subdirs.get(phase, ""))))
    if not source_rows and source.get("root"):
        source_rows.append(("입력", source["root"]))
    source_rows.extend((key, source[key]) for key in ("cube_config",) if key in source)
    counts = source.get("observations", {})
    placements = result.get("split", {}).get("placement_ids", [])
    n_folds = len(placements) if placements else max((record["n_folds"] for record in complete), default=0)
    all_converged = sum(record["full_fit_converged"] is True for record in complete)
    converged_folds = sum(record["n_converged"] for record in complete if isinstance(record["n_converged"], int))
    total_folds = sum(record["n_folds"] for record in complete if isinstance(record["n_folds"], int))
    lines = ["# Table 1 결과 요약", "",
             f"**계산된 방법 {len(complete)}개 · placement {n_folds}개 · ALL 수렴 {all_converged}/{len(complete)} · fold 수렴 {converged_folds}/{total_folds}.**", "",
             "내부 주 지표는 **Held-out Test Cross-view Cube RMSE px**. 최종 물리 순위는 독립 External GT로 결정한다.", "",
             "## 1. 전체 평가지표", "",
             "각 열은 작을수록 좋으며 **최솟값**을 굵게 표시한다. 반올림 전 값으로 비교하고 동률은 모두 표시한다. 미산출 값은 Pending이며, 소수점 4자리로 표시한다.", ""]
    _table(lines, ["실험 (구성)", *METRIC_COLUMNS.values()],
           [(_method_label(record), *(_highlight_minimum(record[key], minima[key])
                                     for key in METRIC_COLUMNS)) for record in records])
    lines.extend(["TRE·Rotation Error는 평균, P95 TRE는 95백분위수다. Failure Rate는 전체 GT pose 중 예측 실패·누락 비율(0–1)이며 solver 수렴률과 다르다.", "",
                  "## 2. 비교 구성과 미산출 항목", ""])
    _table(lines, ["Row", "학습 표적", "최적화", "FK 처리", "상태"],
           [(record["row"], record["targets"], record["optimization"], record["fk"], record["status"]) for record in records])
    pending = [(record["row"], record["pending_reason"]) for record in records if record["pending_reason"]]
    if pending:
        _table(lines, ["미산출 행", "사유"], pending)
    if any(record[key] == "Pending" for record in records for key in EXTERNAL_COLUMNS):
        lines.extend(["External GT가 미산출인 행은 독립 6-DoF GT와 방법별 frozen prediction이 필요하다. P1 비전으로 추정한 flange→Cube 변환은 corrected-FK이며 raw-FK의 독립 기계값이나 External GT가 아니다.", ""])
    lines.extend(["Sequential은 gripper 단계 후 결과를 동결하고 fixed 카메라를 적합한다. Unified는 함께 적합한다. 모든 행은 P1 Cube 기반 카메라 초기값과 P3 Board 기반 hand-eye 초기값을 공유한다. board-only/cube-only는 최적화 잔차 구성을 뜻하며 초기화까지 표적을 제외한 비교는 아니다.", "",
                  "## 3. 모든 데이터에 적용하는 평가 원리", "",
                  "1. **동일 LOPO:** placement 하나의 모든 P2 Cube·Board 관측을 학습에서 제외한다. 모든 방법과 지표에 같은 split을 쓴다.",
                  "2. **ALL / Train / Test:** ALL은 전체 placement로 별도 재fit한다. Train은 각 fold의 학습 placement, Test는 제외 placement를 평가한다.",
                  "3. **Cube:** 공통 FK-reference를 투영하여 코너 오차를 구한다. FK 계열에 유리할 수 있어 보조 지표로 사용한다.",
                  "4. **Cross-view:** A만으로 PnP → B로 전달한다. A→B·B→A 모두 계산하고, 각 방향의 destination 코너는 채점에만 쓴다.",
                  "5. **동일 RMSE:** `sqrt(mean(dx² + dy²))`. 코너 제곱오차 합 ÷ 코너 수의 제곱근이며 fold RMSE를 단순 평균하지 않는다.", "",
                  "fixed↔fixed 전달에는 FK가 필요 없고, fixed↔gripper 전달에는 촬영 순간 FK가 포함된다. Train은 fold별 평가 코너를 모아 집계한다.", "",
                  "## 4. 카메라 쌍별 Cross-view", ""])
    pairs = {(row, split, pair): value for row, split, pair, value, _count in _pair_records(result)}
    pair_columns = [(split, pair) for pair in ("fixed↔fixed", "fixed↔gripper")
                    for split in ("ALL", "Train", "Held-out Test")]
    _table(lines, ["Row", *(f"{pair} {split} px" for split, pair in pair_columns)],
           [(record["row"], *(pairs[(record["row"], split, pair)] for split, pair in pair_columns)) for record in records])
    lines.extend(["전체 카메라 쌍을 합친 값은 1절 Cross-view 열이다. 쌍·방향·코너 수와 fold별 수렴 상태는 [fold_metrics.csv](fold_metrics.csv)에 있다.", "",
                  "## 5. 입력·해석 범위·재생성", ""])
    _table(lines, ["입력", "저장소 기준 경로"], source_rows)
    if counts:
        lines.extend(["관측 수: " + "; ".join(f"{key}={value}" for key, value in counts.items()) + ".", ""])
    geometry = result.get("evaluation_limitations", {}).get("geometry")
    lines.extend(["**기하·스케일의 독립성:** " + (_display(geometry) if geometry else
                  "보정 자료의 출처가 기록되지 않았다. 기하 추정까지 독립적인 end-to-end 검증으로 해석하지 않는다."), "",
                  "원본 JSON에 입력 해시·K/D·변환행렬·split을, CSV에 반올림 전 값을 보존한다. 초기화 횟수·수렴·관측 수가 다른 실행끼리는 수치만으로 우열을 확정하지 않는다.", "",
                  "산출물: [원본 JSON](ABLATION_TEST_table1_methods.json) · [요약 CSV](ABLATION_TEST_table1_results.csv) · [fold CSV](fold_metrics.csv).", "",
                  "같은 JSON을 보고서로 다시 내보내기(재fit 없음):", "",
                  "```bash", "python 06_make_report.py --table1 <결과폴더>/ABLATION_TEST_table1_methods.json", "```", ""])
    paths["markdown"].write_text("\n".join(lines), encoding="utf-8")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = json.loads(args.input.read_text(encoding="utf-8"))
    for path in write_reports(result, args.out_dir).values():
        print(path)


if __name__ == "__main__":
    main()
