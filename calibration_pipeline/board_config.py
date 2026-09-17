"""Fail-closed serialization helpers for the physical ChArUco board.

OpenCV's ``squares_x``/``squares_y`` count checker squares, while the number
of detectable ChArUco chessboard corners is ``(squares_x-1)*(squares_y-1)``.
Keeping that topology with the metric lengths in every capture artifact avoids
silently reinterpreting old images after a code-default change.
"""

from __future__ import annotations

import copy
import json
import os
from typing import List, Mapping, Optional, Tuple

from calibration_pipeline.config import (
    CharucoBoardConfig,
    get_default_charuco_board_config,
    get_default_charuco_board_config_source,
)


CONFIG_KEYS = (
    "squares_x",
    "squares_y",
    "square_length_m",
    "marker_length_m",
    "dictionary_name",
    "marker_id_start",
)

# ``legacy_pattern`` is deliberately NOT in CONFIG_KEYS. Every artifact frozen
# before it existed omits the key, and observations._configs_match compares key
# sets exactly, so requiring it would invalidate those captures. Absent means
# False -- the only layout the old code could build.
OPTIONAL_CONFIG_KEYS = ("legacy_pattern",)
COMPARE_KEYS = CONFIG_KEYS + OPTIONAL_CONFIG_KEYS

BOARD_DIR_NAME = os.path.join("targets", "charuco_boards")


def clone_charuco_config(cfg: CharucoBoardConfig) -> CharucoBoardConfig:
    return copy.deepcopy(cfg)


def charuco_config_to_dict(cfg: CharucoBoardConfig) -> dict:
    """Serialize a board. ``legacy_pattern`` is emitted only when it is True so
    that a default board still hashes and compares identically to artifacts
    frozen before the field existed."""
    data = {
        "squares_x": int(cfg.squares_x),
        "squares_y": int(cfg.squares_y),
        "square_length_m": float(cfg.square_length_m),
        "marker_length_m": float(cfg.marker_length_m),
        "dictionary_name": str(cfg.dictionary_name),
        "marker_id_start": int(cfg.marker_id_start),
    }
    if bool(getattr(cfg, "legacy_pattern", False)):
        data["legacy_pattern"] = True
    return data


def charuco_config_from_dict(
        data: Mapping, base_cfg: Optional[CharucoBoardConfig] = None,
) -> CharucoBoardConfig:
    if not isinstance(data, Mapping):
        raise ValueError("ChArUco config must be a mapping")
    missing = [key for key in CONFIG_KEYS if key not in data]
    if missing:
        raise ValueError(
            "ChArUco config is incomplete; missing " + ", ".join(missing))
    cfg = clone_charuco_config(base_cfg or get_default_charuco_board_config())
    cfg.squares_x = int(data["squares_x"])
    cfg.squares_y = int(data["squares_y"])
    cfg.square_length_m = float(data["square_length_m"])
    cfg.marker_length_m = float(data["marker_length_m"])
    cfg.dictionary_name = str(data["dictionary_name"])
    cfg.marker_id_start = int(data["marker_id_start"])
    cfg.legacy_pattern = bool(data.get("legacy_pattern", False))
    validate_charuco_config(cfg)
    return cfg


def _normalized(cfg: CharucoBoardConfig) -> dict:
    result = charuco_config_to_dict(cfg)
    result["legacy_pattern"] = bool(getattr(cfg, "legacy_pattern", False))
    for key in ("square_length_m", "marker_length_m"):
        result[key] = round(float(result[key]), 12)
    return result


def charuco_configs_equivalent(
        first: CharucoBoardConfig, second: CharucoBoardConfig) -> bool:
    return _normalized(first) == _normalized(second)


def charuco_config_mismatch_keys(
        expected: CharucoBoardConfig, actual: CharucoBoardConfig) -> List[str]:
    first, second = _normalized(expected), _normalized(actual)
    return [key for key in COMPARE_KEYS if first[key] != second[key]]


def validate_charuco_config(cfg: CharucoBoardConfig) -> None:
    if int(cfg.squares_x) < 2 or int(cfg.squares_y) < 2:
        raise ValueError("ChArUco board needs at least 2x2 checker squares")
    if float(cfg.square_length_m) <= 0.0:
        raise ValueError("ChArUco square_length_m must be positive")
    if not 0.0 < float(cfg.marker_length_m) < float(cfg.square_length_m):
        raise ValueError(
            "ChArUco marker_length_m must be positive and smaller than "
            "square_length_m")
    if int(cfg.marker_id_start) < 0:
        raise ValueError("ChArUco marker_id_start must be non-negative")
    if not str(cfg.dictionary_name).startswith("DICT_"):
        raise ValueError("ChArUco dictionary_name must be an OpenCV DICT_* name")
    if not isinstance(getattr(cfg, "legacy_pattern", False), bool):
        raise ValueError("ChArUco legacy_pattern must be a bool")


def charuco_topology(cfg: CharucoBoardConfig) -> dict:
    return {
        "checker_squares_x": int(cfg.squares_x),
        "checker_squares_y": int(cfg.squares_y),
        "charuco_corner_columns": int(cfg.squares_x) - 1,
        "charuco_corner_rows": int(cfg.squares_y) - 1,
        "maximum_charuco_corners": (
            (int(cfg.squares_x) - 1) * (int(cfg.squares_y) - 1)),
        "checker_width_mm": (
            int(cfg.squares_x) * float(cfg.square_length_m) * 1000.0),
        "checker_height_mm": (
            int(cfg.squares_y) * float(cfg.square_length_m) * 1000.0),
        "marker_count": (int(cfg.squares_x) * int(cfg.squares_y)) // 2,
        "legacy_pattern": bool(getattr(cfg, "legacy_pattern", False)),
        # Only an even row count makes the legacy and post-4.6 layouts differ.
        "legacy_pattern_matters": int(cfg.squares_y) % 2 == 0,
    }


def load_charuco_config_from_meta(
        root_folder: str, *, require_frozen: bool = True,
        default_cfg: Optional[CharucoBoardConfig] = None,
) -> Tuple[CharucoBoardConfig, str]:
    cfg = clone_charuco_config(
        default_cfg or get_default_charuco_board_config())
    meta_path = os.path.join(root_folder, "meta.json")
    if not os.path.isfile(meta_path):
        if require_frozen:
            raise ValueError(f"capture metadata is missing: {meta_path}")
        return cfg, "default"
    with open(meta_path, "r", encoding="utf-8") as stream:
        meta = json.load(stream)
    data = meta.get("charuco_board_config")
    if not isinstance(data, Mapping):
        if require_frozen:
            raise ValueError(
                "meta.json has no frozen charuco_board_config; refusing to "
                "reinterpret captured images with current code defaults")
        return cfg, "default"
    return charuco_config_from_dict(data, cfg), "meta"


# --------------------------------------------------------------------------
# targets/charuco_boards/*.json -- one file per physical board
# --------------------------------------------------------------------------
# Boards get swapped between shoots, so no caller should hard-code a layout.
# Describe each printed board once as JSON here, then resolve it by name.


def _repo_relative(path: str) -> str:
    """Repo-relative path when the file lives in the repo, else absolute.

    Board sources end up frozen in meta.json, so they must not carry one
    machine's home directory.
    """
    abs_path = os.path.abspath(path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        relative = os.path.relpath(abs_path, repo_root)
    except ValueError:
        return abs_path
    if relative.startswith(os.pardir):
        return abs_path
    return relative.replace(os.sep, "/")


def charuco_board_dir() -> str:
    """Directory holding the per-board JSON definitions."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, BOARD_DIR_NAME)


def list_charuco_boards() -> List[str]:
    """Names of every board definition on disk, e.g. ``["11x7_id5", "9x6_id90"]``."""
    directory = charuco_board_dir()
    if not os.path.isdir(directory):
        return []
    names = []
    for entry in os.listdir(directory):
        if not entry.endswith(".json"):
            continue
        stem = entry[: -len(".json")]
        names.append(stem[len("board_"):] if stem.startswith("board_") else stem)
    return sorted(names)


def charuco_board_path(name: str) -> str:
    """Absolute path of the JSON defining board ``name``."""
    if not name:
        raise ValueError("board name must be non-empty")
    directory = charuco_board_dir()
    for candidate in (f"board_{name}.json", f"{name}.json"):
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"unknown ChArUco board {name!r}; available: "
        f"{', '.join(list_charuco_boards()) or '(none)'} in {directory}")


def load_charuco_config_from_json_file(
        path: str, default_cfg: Optional[CharucoBoardConfig] = None,
) -> Tuple[CharucoBoardConfig, str]:
    """Load one board definition. Fail-closed: a broken file raises."""
    if not path:
        raise ValueError("board config path must be non-empty")
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"ChArUco board config not found: {abs_path}")
    with open(abs_path, "r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError(f"ChArUco board config must be an object: {abs_path}")
    return charuco_config_from_dict(data, default_cfg), abs_path


def resolve_charuco_config(
        board: Optional[str] = None,
        overrides: Optional[Mapping] = None,
        default_cfg: Optional[CharucoBoardConfig] = None,
) -> Tuple[CharucoBoardConfig, str]:
    """Resolve the board to use for a run.

    ``board`` is a name from ``targets/charuco_boards/`` (``"9x6_id90"``) or a
    path to a JSON definition; ``None`` keeps the project default from
    ``config.py``. ``overrides`` applies per-field CLI overrides on top and is
    recorded in the returned source string so a report never claims a stock
    board that was actually tweaked.
    """
    if board:
        path = board if os.path.isfile(board) else charuco_board_path(board)
        cfg, abs_path = load_charuco_config_from_json_file(path, default_cfg)
        source = f"board_file:{_repo_relative(abs_path)}"
    else:
        cfg = clone_charuco_config(
            default_cfg or get_default_charuco_board_config())
        source = get_default_charuco_board_config_source()

    applied = []
    for key, value in dict(overrides or {}).items():
        if value is None:
            continue
        if key not in COMPARE_KEYS:
            raise ValueError(f"unknown ChArUco board override: {key}")
        setattr(cfg, key, value)
        applied.append(f"{key}={value}")
    validate_charuco_config(cfg)
    if applied:
        source += " +override(" + ",".join(sorted(applied)) + ")"
    return cfg, source


def describe_charuco_config(cfg: CharucoBoardConfig) -> str:
    """One-line human summary for logs and reports."""
    topology = charuco_topology(cfg)
    legacy = "legacy" if topology["legacy_pattern"] else "modern"
    if topology["legacy_pattern_matters"]:
        legacy += "(짝수 행이라 배치 구분 필수)"
    ids = (f"{int(cfg.marker_id_start)}.."
           f"{int(cfg.marker_id_start) + topology['marker_count'] - 1}")
    return (
        f"{int(cfg.squares_x)}x{int(cfg.squares_y)} squares "
        f"({topology['checker_width_mm']:.0f}x{topology['checker_height_mm']:.0f} mm), "
        f"square={float(cfg.square_length_m) * 1000:.1f}mm "
        f"marker={float(cfg.marker_length_m) * 1000:.1f}mm, "
        f"{cfg.dictionary_name} id {ids}, "
        f"{topology['marker_count']} markers, "
        f"max {topology['maximum_charuco_corners']} corners, {legacy}")
