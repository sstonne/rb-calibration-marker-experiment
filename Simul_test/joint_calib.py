"""
Joint(동시 최적화) 통합 캘리브 + Independent rigid 정합 조합 단계.

세 방식의 차이(핵심):
  Independent : 고정·그리퍼를 *따로* 캘리브 → 공유 큐브로 rigid 정합(마지막에 붙임).
  1-pass      : 공유 큐브 앵커로 한 번 정보교환 (unified_vs_independent.calib_unified).
  Joint       : 모든 관측(고정+그리퍼)을 *하나의 비선형 최소제곱*으로 bTf·gTc·큐브 동시 최적화.

Joint 목적함수 (재투영/pose 잔차):
  모든 (고정 카메라 ci, set s):   bTf_ci @ obs_f[ci,s]   ==  cube_s
  모든 (그리퍼 event e, set s):   bTg[e] @ gTc @ obs_g[e] ==  cube_s
  → 위 등식의 SE(3) 불일치(회전 3 + 병진 3)를 모든 항에서 동시에 최소화.
  미지수: bTf_ci(각 6) + gTc(6) + cube_s(각 6).  gauge: cube 를 FK 로 초기화(절대 base).
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from aruco_cube import inv_T
from utils_pose import robust_se3_average as se3_avg


# ----------------------------------------------------------------------
#  SE(3) <-> 6-vector (rvec[3] + tvec[3])  파라미터화
# ----------------------------------------------------------------------
def se3_to_vec(T):
    rv = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    return np.concatenate([rv, T[:3, 3]])


def vec_to_se3(v):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    T[:3, 3] = v[3:6]
    return T


def _se3_residual(A, B):
    """두 SE(3) 의 불일치 6-vector (회전 rotvec 3 + 병진 3). A==B 면 0."""
    E = inv_T(A) @ B
    rv = Rotation.from_matrix(E[:3, :3]).as_rotvec()
    return np.concatenate([rv, E[:3, 3]])


# ----------------------------------------------------------------------
#  Independent + 명시적 rigid 정합 (마지막 조합 단계)
#   고정·그리퍼를 따로 캘리브한 뒤, 공유 큐브로 그리퍼 서브시스템을
#   고정 서브시스템 base 에 rigid 정렬. (각자 오차는 그대로 — 정합만)
# ----------------------------------------------------------------------
def _fit_rigid(P, Q):
    """P->Q 최소제곱 rigid (R,t):  Q ≈ R P + t."""
    P = np.asarray(P); Q = np.asarray(Q)
    cP, cQ = P.mean(0), Q.mean(0)
    H = (P - cP).T @ (Q - cQ)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, cQ - R @ cP


def calib_independent_aligned(scene, train_sets, calib_independent_fn):
    """calib_independent 로 따로 캘리브 후, 공유 큐브로 그리퍼→고정 base rigid 정합.

    독립 캘리브는 고정(bTf)과 그리퍼(gTc)를 각자 다른 gauge 로 풀 수 있다.
    두 서브시스템이 같은 큐브를 base 에서 본 위치가 어긋나므로, 그 어긋남을
    rigid(R,t)로 맞춰 그리퍼 예측을 고정 base 에 정렬한다(명시적 조합 단계).
    """
    d = scene
    m = calib_independent_fn(scene, train_sets)   # {cams, gTc}
    if m["gTc"] is None:
        m["align"] = None
        return m
    # 각 train set 에서: 고정으로 본 큐브 위치 P_fix, 그리퍼로 본 위치 P_grip
    P_grip, P_fix = [], []
    for s in train_sets:
        # 고정 예측
        Tf = [m["cams"][ci] @ d["obs_fixed"][(ci, s)]
              for ci in d["cam_ids"] if (ci, s) in d["obs_fixed"]]
        # 그리퍼 예측
        Tg = [d["bTg"][e] @ m["gTc"] @ d["obs_grip"][e]
              for e in d["set_events"].get(s, [])]
        if Tf and Tg:
            P_fix.append(se3_avg(Tf)[:3, 3])
            P_grip.append(se3_avg(Tg)[:3, 3])
    if len(P_fix) >= 3:
        R, t = _fit_rigid(P_grip, P_fix)     # 그리퍼 좌표 -> 고정 base
        m["align"] = (R, t)
    else:
        m["align"] = None
    return m


def predict_cube_independent_aligned(model, scene, s):
    """독립+정합 예측: 고정 예측과 (정합된) 그리퍼 예측을 base 에서 평균."""
    d = scene
    Ts_pos = []
    Tf = [model["cams"][ci] @ d["obs_fixed"][(ci, s)]
          for ci in d["cam_ids"] if (ci, s) in d["obs_fixed"]]
    if Tf:
        Ts_pos.append(se3_avg(Tf)[:3, 3])
    if model["gTc"] is not None:
        Tg = [d["bTg"][e] @ model["gTc"] @ d["obs_grip"][e]
              for e in d["set_events"].get(s, [])]
        if Tg:
            pg = se3_avg(Tg)[:3, 3]
            if model.get("align") is not None:
                R, t = model["align"]
                pg = R @ pg + t                # 그리퍼 예측을 고정 base 로 정합
            Ts_pos.append(pg)
    if not Ts_pos:
        return None
    out = np.eye(4)
    out[:3, 3] = np.mean(Ts_pos, 0)
    return out


# ----------------------------------------------------------------------
#  Joint bundle adjustment — 모든 미지수 동시 최적화
# ----------------------------------------------------------------------
def calib_joint(scene, train_sets, bootstrap_fn):
    """모든 관측을 하나의 비선형 최소제곱으로 bTf·gTc·cube 동시 최적화.

    bootstrap_fn(scene, train_sets) -> {cams, gTc} : 초기값(1-pass 등).
    """
    d = scene
    cam_ids = d["cam_ids"]
    sets = [s for s in train_sets]

    init = bootstrap_fn(scene, train_sets)
    cams0, gTc0 = init["cams"], init["gTc"]
    if gTc0 is None or len(cams0) != len(cam_ids):
        return init

    # --- 파라미터 레이아웃: [bTf_c0..bTf_cN, gTc, cube_s0..cube_sM] (각 6) ---
    cube0 = {}
    for s in sets:
        Tf = [cams0[ci] @ d["obs_fixed"][(ci, s)]
              for ci in cam_ids if (ci, s) in d["obs_fixed"]]
        Tg = [d["bTg"][e] @ gTc0 @ d["obs_grip"][e] for e in d["set_events"].get(s, [])]
        allT = Tf + Tg
        cube0[s] = se3_avg(allT) if allT else d["FK"][s]

    p0 = []
    for ci in cam_ids:
        p0.append(se3_to_vec(cams0[ci]))
    p0.append(se3_to_vec(gTc0))
    for s in sets:
        p0.append(se3_to_vec(cube0[s]))
    p0 = np.concatenate(p0)

    n_cam = len(cam_ids)
    off_gtc = n_cam * 6
    off_cube = off_gtc + 6
    cube_idx = {s: off_cube + i * 6 for i, s in enumerate(sets)}

    def unpack(p):
        cams = {ci: vec_to_se3(p[i * 6:(i + 1) * 6]) for i, ci in enumerate(cam_ids)}
        gTc = vec_to_se3(p[off_gtc:off_gtc + 6])
        cube = {s: vec_to_se3(p[cube_idx[s]:cube_idx[s] + 6]) for s in sets}
        return cams, gTc, cube

    def residuals(p):
        cams, gTc, cube = unpack(p)
        res = []
        for s in sets:
            Cs = cube[s]
            # 고정 카메라 항: bTf_ci @ obs_f == cube_s
            for ci in cam_ids:
                if (ci, s) in d["obs_fixed"]:
                    pred = cams[ci] @ d["obs_fixed"][(ci, s)]
                    res.append(_se3_residual(pred, Cs))
            # 그리퍼 항: bTg[e] @ gTc @ obs_g[e] == cube_s
            for e in d["set_events"].get(s, []):
                pred = d["bTg"][e] @ gTc @ d["obs_grip"][e]
                res.append(_se3_residual(pred, Cs))
        return np.concatenate(res) if res else np.zeros(1)

    sol = least_squares(residuals, p0, method="lm", max_nfev=60)
    cams, gTc, cube = unpack(sol.x)
    return {"cams": cams, "gTc": gTc, "mode": "joint", "cube": cube,
            "cost": float(sol.cost)}


def calib_joint_fk_fixed(scene, train_sets, bootstrap_fn):
    """② Robot FK-based: 큐브(타깃) 위치를 FK 값으로 *고정*하고 bTf·gTc 만 최적화.

    ①(calib_joint)과 유일한 차이: 큐브를 미지수로 두지 않고 FK[s] 상수로 고정.
    → gripper-target 변환을 '카메라로 추정'하는 게 아니라 'FK 로 아는 큐브에 맞춘다'.
    """
    d = scene
    cam_ids = d["cam_ids"]
    sets = [s for s in train_sets if s in d["FK"]]

    init = bootstrap_fn(scene, train_sets)
    cams0, gTc0 = init["cams"], init["gTc"]
    if gTc0 is None or len(cams0) != len(cam_ids):
        return init

    cube_fixed = {s: d["FK"][s] for s in sets}       # 큐브 = FK (상수, 최적화 안 함)

    # 파라미터: [bTf_c0..cN, gTc] (큐브는 제외)
    p0 = [se3_to_vec(cams0[ci]) for ci in cam_ids] + [se3_to_vec(gTc0)]
    p0 = np.concatenate(p0)
    n_cam = len(cam_ids); off_gtc = n_cam * 6

    def unpack(p):
        cams = {ci: vec_to_se3(p[i * 6:(i + 1) * 6]) for i, ci in enumerate(cam_ids)}
        gTc = vec_to_se3(p[off_gtc:off_gtc + 6])
        return cams, gTc

    def residuals(p):
        cams, gTc = unpack(p)
        res = []
        for s in sets:
            Cs = cube_fixed[s]                        # FK 로 고정된 큐브
            for ci in cam_ids:
                if (ci, s) in d["obs_fixed"]:
                    res.append(_se3_residual(cams[ci] @ d["obs_fixed"][(ci, s)], Cs))
            for e in d["set_events"].get(s, []):
                pred = d["bTg"][e] @ gTc @ d["obs_grip"][e]
                res.append(_se3_residual(pred, Cs))
        return np.concatenate(res) if res else np.zeros(1)

    sol = least_squares(residuals, p0, method="lm", max_nfev=60)
    cams, gTc = unpack(sol.x)
    return {"cams": cams, "gTc": gTc, "mode": "joint_fk", "cube": cube_fixed,
            "cost": float(sol.cost)}


def predict_cube_joint_model(model, scene, s):
    """joint/1-pass 공용: 고정+그리퍼를 모두 써서 base 큐브 위치 예측."""
    d = scene
    Ts = []
    for ci in d["cam_ids"]:
        if (ci, s) in d["obs_fixed"]:
            Ts.append(model["cams"][ci] @ d["obs_fixed"][(ci, s)])
    if model["gTc"] is not None:
        for e in d["set_events"].get(s, []):
            Ts.append(d["bTg"][e] @ model["gTc"] @ d["obs_grip"][e])
    return se3_avg(Ts) if Ts else None
