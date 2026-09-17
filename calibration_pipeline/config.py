# config.py
"""Project configuration and the single source of truth for the AprilTag cube.

Units: every field ending with ``_m`` is meters.
Only edit the marker-ID block when the cube is reprinted with different IDs.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


# =============================================================================
# USER-EDITABLE MARKER IDS / SIZES - AprilTag marker cube (59mm body footprint)
# -----------------------------------------------------------------------------
# Edit only this block when reprinting tags with different IDs.
# Sizes are the outer black-border side lengths used by solvePnP. Measure that
# border on the printed tags (not the white paper/pocket) before changing them.
# Top face has two 25mm tags on the +Z protrusion, centered at y=-14mm and
# y=+14mm. Four side faces have one 51mm tag each.
#
# CAD revision (2026-09): the +Z protrusion grew from 2mm to 35mm. The 59x59x57mm
# body and its four side tags are unchanged, and the object-frame origin is kept
# at the point it has always been, so only the top-marker plane moved
# (+29.5mm -> +62.5mm). Data captured before this revision carries its own frozen
# cube_config in meta.json / the observation manifest and is unaffected.
# =============================================================================
CUBE_WIDTH_M = 0.059
CUBE_DEPTH_M = 0.059
CUBE_BODY_HEIGHT_M = 0.057
TOP_PROTRUSION_HEIGHT_M = 0.035
CUBE_OVERALL_HEIGHT_M = CUBE_BODY_HEIGHT_M + TOP_PROTRUSION_HEIGHT_M

CUBE_HALF_WIDTH_M = CUBE_WIDTH_M / 2.0
CUBE_HALF_DEPTH_M = CUBE_DEPTH_M / 2.0

# Object-frame origin, deliberately unchanged across the CAD revision: the
# center of the original 59mm envelope, which sits 1mm above the body center.
# Pinning it to the body (not to the protrusion-dependent envelope) keeps every
# body/side-marker coordinate identical to the pre-revision cube, so previously
# estimated cube poses and priors stay comparable.
CUBE_ORIGIN_ABOVE_BODY_CENTER_M = 0.001
CUBE_BODY_TOP_Z_M = CUBE_BODY_HEIGHT_M / 2.0 - CUBE_ORIGIN_ABOVE_BODY_CENTER_M
CUBE_BODY_BOTTOM_Z_M = -(CUBE_BODY_HEIGHT_M / 2.0 + CUBE_ORIGIN_ABOVE_BODY_CENTER_M)
TOP_MARKER_PLANE_Z_M = CUBE_BODY_TOP_Z_M + TOP_PROTRUSION_HEIGHT_M
SIDE_MARKER_CENTER_Z_M = -CUBE_ORIGIN_ABOVE_BODY_CENTER_M

TOP_MARKER_NEG_Y_ID = 0   # +Z protrusion top, center: (0, -14, +62.5) mm
TOP_MARKER_POS_Y_ID = 1   # +Z protrusion top, center: (0, +14, +62.5) mm
SIDE_MARKER_POS_X_ID = 2  # +X body face, center: (+29.5, 0, -1) mm
SIDE_MARKER_POS_Y_ID = 3  # +Y body face, center: (0, +29.5, -1) mm
SIDE_MARKER_NEG_X_ID = 4  # -X body face, center: (-29.5, 0, -1) mm
SIDE_MARKER_NEG_Y_ID = 5  # -Y body face, center: (0, -29.5, -1) mm

TOP_MARKER_SIZE_M = 0.025
SIDE_MARKER_SIZE_M = 0.051
# =============================================================================


@dataclass
class CubeConfig:
    """Physical marker-plane definition of the AprilTag cube.

    Object frame:
      - body: 59 x 59 x 57mm, spanning z = -29.5mm .. +27.5mm
      - +Z protrusion: 35mm above the body top, same 59 x 59mm footprint
      - overall envelope: 59 x 59 x 92mm, spanning z = -29.5mm .. +62.5mm
      - origin: center of the pre-revision 59mm envelope, i.e. 1mm above the
        body center; kept fixed so the body and side markers never move when
        the protrusion changes
      - +Z: upward; top-marker plane z = +62.5mm (body top +27.5 plus 35mm)
      - side-marker centers: z = -1mm, the mid-height of the 57mm body

    ``cube_side_m`` is the 59mm square footprint (x/y extent) used by the
    marker-plane model; it is no longer the z extent. The body/protrusion
    heights are explicit module constants so the 57+35mm construction is not
    mistaken for a solid 59mm cube.
    """

    cube_side_m: float = CUBE_WIDTH_M  # square footprint (x/y), not the height
    marker_size_m: float = SIDE_MARKER_SIZE_M  # fallback only
    dictionary_name: str = "DICT_APRILTAG_36h11"

    marker_ids: Tuple[int, ...] = (
        TOP_MARKER_NEG_Y_ID,
        TOP_MARKER_POS_Y_ID,
        SIDE_MARKER_POS_X_ID,
        SIDE_MARKER_POS_Y_ID,
        SIDE_MARKER_NEG_X_ID,
        SIDE_MARKER_NEG_Y_ID,
    )

    id_to_face: Dict[int, str] = field(default_factory=lambda: {
        TOP_MARKER_NEG_Y_ID: "+Z",
        TOP_MARKER_POS_Y_ID: "+Z",
        SIDE_MARKER_POS_X_ID: "+X",
        SIDE_MARKER_POS_Y_ID: "+Y",
        SIDE_MARKER_NEG_X_ID: "-X",
        SIDE_MARKER_NEG_Y_ID: "-Y",
    })

    marker_size_by_id: Dict[int, float] = field(default_factory=lambda: {
        TOP_MARKER_NEG_Y_ID: TOP_MARKER_SIZE_M,
        TOP_MARKER_POS_Y_ID: TOP_MARKER_SIZE_M,
        SIDE_MARKER_POS_X_ID: SIDE_MARKER_SIZE_M,
        SIDE_MARKER_POS_Y_ID: SIDE_MARKER_SIZE_M,
        SIDE_MARKER_NEG_X_ID: SIDE_MARKER_SIZE_M,
        SIDE_MARKER_NEG_Y_ID: SIDE_MARKER_SIZE_M,
    })

    marker_center_m: Dict[int, Tuple[float, float, float]] = field(default_factory=lambda: {
        TOP_MARKER_NEG_Y_ID: (0.0, -0.014, TOP_MARKER_PLANE_Z_M),
        TOP_MARKER_POS_Y_ID: (0.0, 0.014, TOP_MARKER_PLANE_Z_M),
        SIDE_MARKER_POS_X_ID: (CUBE_HALF_WIDTH_M, 0.0, SIDE_MARKER_CENTER_Z_M),
        SIDE_MARKER_POS_Y_ID: (0.0, CUBE_HALF_DEPTH_M, SIDE_MARKER_CENTER_Z_M),
        SIDE_MARKER_NEG_X_ID: (-CUBE_HALF_WIDTH_M, 0.0, SIDE_MARKER_CENTER_Z_M),
        SIDE_MARKER_NEG_Y_ID: (0.0, -CUBE_HALF_DEPTH_M, SIDE_MARKER_CENTER_Z_M),
    })

    # OpenCV returns decoded marker corners visually clockwise in the image.
    # local_corners_for() uses the matching order
    # (+u,-v), (-u,-v), (-u,+v), (+u,+v), which is clockwise when viewed from
    # outside the cube. Keep identity unless detector/model order is explicitly
    # verified to differ; printed in-plane rotation belongs in face_roll_deg.
    corner_reorder: Dict[int, Tuple[int, int, int, int]] = field(default_factory=lambda: {
        TOP_MARKER_NEG_Y_ID: (0, 1, 2, 3),
        TOP_MARKER_POS_Y_ID: (0, 1, 2, 3),
        SIDE_MARKER_POS_X_ID: (0, 1, 2, 3),
        SIDE_MARKER_POS_Y_ID: (0, 1, 2, 3),
        SIDE_MARKER_NEG_X_ID: (0, 1, 2, 3),
        SIDE_MARKER_NEG_Y_ID: (0, 1, 2, 3),
    })

    # In-plane rotation around each face normal, degrees. Validate physically.
    # Per-face in-plane roll (deg, about the face normal) of the printed marker vs
    # the nominal face axes. Calibrated from data/session02_NOUSE_session04_0814 multi-face co-observations
    # (self-calibration over roll in {0,90,180,270}): the physical cube's side tags
    # are mounted rotated by 90-degree steps. Applying these brought the inter-face
    # cube-pose rotation disagreement from median 90.72deg down to 2.20deg.
    # If the cube is reprinted/re-stickered, re-run the face-roll self-calibration.
    face_roll_deg: Dict[int, float] = field(default_factory=lambda: {
        TOP_MARKER_NEG_Y_ID: 0.0,
        TOP_MARKER_POS_Y_ID: 0.0,
        SIDE_MARKER_POS_X_ID: 90.0,
        SIDE_MARKER_POS_Y_ID: 180.0,
        SIDE_MARKER_NEG_X_ID: 270.0,
        SIDE_MARKER_NEG_Y_ID: 0.0,
    })

    # Out-of-plane recess of the marker face, meters. Default 0.0 means the
    # marker plane coincides with the CAD outer surface (d = 29.5mm).
    # The printed cube (_cube_print/..._recess0p1...) has a 0.1mm recess pocket,
    # but paper/sticker thickness usually brings the tag flush with the surface,
    # so 0.0 is the physically correct default. Set to 0.0001 only if you have
    # verified the tags actually sit 0.1mm below the surface.
    marker_inset_m: float = 0.0

    # Optional explicit marker pose override: marker_id -> 4x4 T_object_marker.
    marker_pose_4x4: Dict[int, list] = field(default_factory=dict)


def get_default_cube_config() -> CubeConfig:
    return CubeConfig()


def get_default_cube_config_source() -> str:
    return "config_py:CubeConfig"


@dataclass
class CharucoBoardConfig:
    squares_x: int = 11
    squares_y: int = 7
    square_length_m: float = 0.025
    marker_length_m: float = 0.018
    dictionary_name: str = "DICT_4X4_250"
    marker_id_start: int = 5  # 인쇄된 보드의 ArUco ID 시작값. 큐브(DICT_APRILTAG_36h11)와 다른 딕셔너리(DICT_4X4_250)라 ID 겹쳐도 무방
    # OpenCV 4.6 이전 ChArUco 인쇄 배치. squares_y 가 짝수인 보드에서만 신규 배치와
    # 달라지며, 틀리면 마커는 전부 검출되는데 코너 보간만 0개로 조용히 실패한다.
    legacy_pattern: bool = False


def get_default_charuco_board_config() -> CharucoBoardConfig:
    return CharucoBoardConfig()


def get_default_charuco_board_config_source() -> str:
    return "config_py:CharucoBoardConfig"
