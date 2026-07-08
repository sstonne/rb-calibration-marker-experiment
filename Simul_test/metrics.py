"""
SE(3) 오차 측정 헬퍼. 기존 utils_pose._rot_angle 을 재사용한다.

- rot_geodesic_deg: 두 변환의 회전 측지 오차 (도)
- trans_err_mm:     두 변환의 병진 오차 (mm)
- assert_se3_close:  GT 대비 회전/병진 허용오차 단언 (실패 시 명확한 진단 메시지)
"""

import numpy as np

from utils_pose import _rot_angle


def rot_geodesic_deg(Ta: np.ndarray, Tb: np.ndarray) -> float:
    """두 SE(3) 의 회전 측지 거리 (degree)."""
    dR = np.asarray(Ta)[:3, :3] @ np.asarray(Tb)[:3, :3].T
    return float(np.degrees(_rot_angle(dR)))


def trans_err_mm(Ta: np.ndarray, Tb: np.ndarray) -> float:
    """두 SE(3) 의 병진 오차 (mm). 내부 단위는 m 라고 가정."""
    return float(np.linalg.norm(np.asarray(Ta)[:3, 3] - np.asarray(Tb)[:3, 3]) * 1000.0)


def assert_se3_close(T: np.ndarray,
                     T_gt: np.ndarray,
                     rot_tol_rad: float = 1e-6,
                     trans_tol_m: float = 1e-6,
                     msg: str = "") -> None:
    """T 가 T_gt 와 허용오차 이내인지 단언."""
    T = np.asarray(T, dtype=np.float64)
    T_gt = np.asarray(T_gt, dtype=np.float64)
    assert T.shape == (4, 4), f"{msg} T shape {T.shape} != (4,4)"
    dr = _rot_angle(T[:3, :3] @ T_gt[:3, :3].T)
    dt = float(np.linalg.norm(T[:3, 3] - T_gt[:3, 3]))
    assert dr < rot_tol_rad and dt < trans_tol_m, (
        f"{msg} SE(3) mismatch: rot={np.degrees(dr):.3e}deg "
        f"(tol={np.degrees(rot_tol_rad):.3e}deg), "
        f"trans={dt * 1000:.3e}mm (tol={trans_tol_m * 1000:.3e}mm)"
    )
