"""targets/charuco_boards/*.json 이 물리 보드의 단일 소스임을 강제하는 계약 테스트.

2026-09-15 intrinsics 촬영은 9x6 legacy 보드를 6x9 non-legacy 로 지정해서 전부
실패했다. 마커는 27/27 다 검출되는데 charuco 코너만 0 개로 나오는 조용한 실패라,
여기서 배치가 실제로 적용되는지 합성 이미지로 직접 확인한다.
"""

import json
import os

import cv2
import numpy as np
import pytest

from calibration_pipeline.board_config import (
    COMPARE_KEYS,
    CONFIG_KEYS,
    charuco_board_dir,
    charuco_board_path,
    charuco_config_from_dict,
    charuco_config_to_dict,
    charuco_configs_equivalent,
    charuco_config_mismatch_keys,
    charuco_topology,
    describe_charuco_config,
    list_charuco_boards,
    load_charuco_config_from_json_file,
    resolve_charuco_config,
)
from calibration_pipeline.charuco import CharucoTarget
from calibration_pipeline.config import get_default_charuco_board_config


def test_every_board_definition_loads_and_validates():
    names = list_charuco_boards()
    assert names, f"no board definitions in {charuco_board_dir()}"
    for name in names:
        cfg, source = resolve_charuco_config(name)
        assert source == f"board_file:targets/charuco_boards/board_{name}.json"
        assert describe_charuco_config(cfg)


def test_main_board_file_matches_the_code_default():
    """11x7 보드는 config.py 와 JSON 두 곳에 있으므로 드리프트를 막는다."""
    cfg, _ = resolve_charuco_config("11x7_id5")
    assert charuco_configs_equivalent(cfg, get_default_charuco_board_config()), (
        charuco_config_mismatch_keys(get_default_charuco_board_config(), cfg))


def test_board_9x6_id90_describes_the_printed_board():
    cfg, _ = resolve_charuco_config("9x6_id90")
    assert (cfg.squares_x, cfg.squares_y) == (9, 6), (
        "인쇄물 캡션의 '6x9' 는 행x열 표기다; OpenCV 는 열x행이므로 9x6")
    assert cfg.legacy_pattern is True
    assert cfg.marker_id_start == 90
    topology = charuco_topology(cfg)
    assert topology["marker_count"] == 27
    assert topology["maximum_charuco_corners"] == 40
    assert topology["legacy_pattern_matters"] is True


def test_legacy_pattern_is_omitted_when_false_so_frozen_artifacts_still_match():
    """observations._configs_match 는 키 집합까지 비교한다. legacy_pattern 이
    생기기 전에 굳은 meta.json 과 여전히 같아야 한다."""
    assert "legacy_pattern" not in CONFIG_KEYS
    assert "legacy_pattern" in COMPARE_KEYS
    default = charuco_config_to_dict(get_default_charuco_board_config())
    assert set(default) == set(CONFIG_KEYS)
    legacy_cfg, _ = resolve_charuco_config("9x6_id90")
    assert charuco_config_to_dict(legacy_cfg)["legacy_pattern"] is True


def test_legacy_pattern_round_trips_and_is_compared():
    cfg, _ = resolve_charuco_config("9x6_id90")
    restored = charuco_config_from_dict(charuco_config_to_dict(cfg))
    assert restored.legacy_pattern is True
    assert charuco_configs_equivalent(cfg, restored)

    modern = charuco_config_from_dict(
        {k: v for k, v in charuco_config_to_dict(cfg).items()
         if k != "legacy_pattern"})
    assert modern.legacy_pattern is False
    assert charuco_config_mismatch_keys(cfg, modern) == ["legacy_pattern"]


def test_unknown_board_name_lists_what_is_available():
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_charuco_config("12x8_id0")
    message = str(excinfo.value)
    for name in list_charuco_boards():
        assert name in message


def test_cli_overrides_are_recorded_in_the_source_string():
    cfg, source = resolve_charuco_config(
        "9x6_id90", {"marker_id_start": 120, "squares_x": None})
    assert cfg.marker_id_start == 120
    assert cfg.squares_x == 9
    assert "+override(marker_id_start=120)" in source
    with pytest.raises(ValueError):
        resolve_charuco_config("9x6_id90", {"not_a_field": 1})


def _render(cfg, pixels_per_square=80, margin=40):
    target = CharucoTarget(cfg)
    size = (cfg.squares_x * pixels_per_square, cfg.squares_y * pixels_per_square)
    image = target.board.generateImage(size, marginSize=margin)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def test_charuco_target_applies_legacy_pattern_from_the_config():
    cfg, _ = resolve_charuco_config("9x6_id90")
    target = CharucoTarget(cfg)
    assert target.board.getLegacyPattern() is True
    assert CharucoTarget(get_default_charuco_board_config(
    )).board.getLegacyPattern() is False


def test_wrong_layout_detects_every_marker_but_no_corner():
    """실제로 터진 실패 모드를 고정한다: 마커는 다 잡히는데 코너가 0 개."""
    cfg, _ = resolve_charuco_config("9x6_id90")
    image = _render(cfg)

    _, _, n_correct = CharucoTarget(cfg).detect(image)[:3]
    assert n_correct == charuco_topology(cfg)["maximum_charuco_corners"]

    from dataclasses import replace
    for broken, label in (
            (replace(cfg, legacy_pattern=False), "legacy 누락"),
            (replace(cfg, squares_x=6, squares_y=9), "축 전치"),
    ):
        target = CharucoTarget(broken)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, ids, _ = target.detector.detectMarkers(gray)
        found = set() if ids is None else {int(i) for i in ids.reshape(-1)}
        assert found >= set(range(90, 117)), f"{label}: 마커는 전부 검출돼야 한다"
        assert target.detect(image)[2] == 0, f"{label}: 코너는 0 이어야 한다"


def test_board_json_files_are_documented():
    for name in list_charuco_boards():
        with open(charuco_board_path(name), "r", encoding="utf-8") as stream:
            data = json.load(stream)
        assert data.get("name") == name
        assert data.get("_comment"), f"{name}: 어떤 실물 보드인지 _comment 로 남길 것"


def test_load_from_json_file_is_fail_closed(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError):
        load_charuco_config_from_json_file(str(missing))
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_charuco_config_from_json_file(str(bad))
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(json.dumps({"squares_x": 9}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_charuco_config_from_json_file(str(incomplete))
