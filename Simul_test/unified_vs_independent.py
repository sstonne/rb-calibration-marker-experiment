"""
C1 — Unified Hybrid Calibration:  통합(Unified) vs 독립(Independent) 비교 실험.

배경(기여점)
-----------
eye-in-hand(그리퍼 카메라)와 eye-to-hand(고정 카메라)를 *하나의 공유 타깃(큐브)* 을
통해 *같은 robot base frame* 에 공동 등록한다.

  - 독립(Independent):  고정 카메라는 A/B 로 따로, 그리퍼는 cv2.calibrateHandEye 로 따로
                        추정한 뒤 base 에서 결과를 조합. 각 오차가 독립적으로 누적됨.
  - 통합(Unified):      고정+그리퍼 카메라가 *같은 공유 큐브 앵커* 에 함께 합의(B 확장).
                        큐브 = 모든 센서(고정 obs + 그리퍼 obs·핸드아이 경유) 평균 →
                        그 큐브로 bTf_i 와 gTc 를 동시에 역산. 1-pass (gauge = FK anchor).

두 방식 모두 위에 C(잔차 ridge 학습) 보정 레이어를 얹을 수 있다.

시뮬을 쓰는 이유
--------------
GT(gTc, bTf_i, bTo)를 정확히 알기 때문에 "추정 vs 정답" 오차를 직접 잴 수 있다
(실데이터로는 불가 — 진짜 카메라·핸드아이 위치를 모름).

평가 지표 (4종, 모두 GT 대비)
----------------------------
  1. 고정 카메라 bTf 오차 (mm/deg)          — eye-to-hand 절대 정확도
  2. 그리퍼 핸드아이 gTc 오차 (mm/deg)       — eye-in-hand 절대 정확도
  3. 공유 base 정합 일관성 (mm)              — 고정·그리퍼가 같은 큐브를 base 에서
                                              얼마나 일치하게 보는가 (통합의 핵심 주장)
  4. 다운스트림 큐브 예측 (mm, holdout)      — test set 큐브 위치 예측 오차

실행:
  PYTHONPATH= python Simul_test/unified_vs_independent.py           # 전체 (30 seed)
  PYTHONPATH= python Simul_test/unified_vs_independent.py --seeds 10
"""
import sys, os, itertools, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from aruco_cube import inv_T
from utils_pose import robust_se3_average as se3_avg
from synthetic_scene import SyntheticScene
from abc_calib import (predict_cube, solve_camera_from_cube, learn_residual,
                       _resid_feature)


# ======================================================================
#  씬 + 관측 생성 (고정 카메라 + 그리퍼 카메라, 동일한 systematic bias)
# ======================================================================
def build_scene(seed=0, n_sets=10, noise_mm=6.0, noise_type="systematic", n_cams=3,
                n_events_per_set=6, gripper_tilt_deg=35.0):
    """SyntheticScene 으로 고정/그리퍼 관측·FK·로봇자세를 생성.

    반환 dict:
      cam_ids        : 고정 카메라 id
      sets           : set(큐브 placement) id
      FK[s]          : GT 큐브 base pose (base<-cube)  [완벽 prior]
      obs_fixed[ci,s]: 고정 카메라 ci 가 set s 에서 본 큐브 (camera<-cube, 노이즈)
      obs_grip[e]    : 그리퍼 카메라가 event e 에서 본 큐브 (camera<-cube, 노이즈)
      bTg[e]         : event e 로봇 자세 (base<-gripper)
      event_set[e]   : event e 가 속한 set
      set_events[s]  : set s 의 event 목록
      gt_bTf, gt_gTc : GT (평가용)
    """
    sc = SyntheticScene(seed=seed, n_fixed_cams=n_cams, n_sets=n_sets,
                        n_events_per_set=n_events_per_set, layout="realistic",
                        gripper_tilt_deg=gripper_tilt_deg)
    cam_ids = list(sc.fixed_cam_ids)
    sets = list(sc.set_indices)
    FK = {s: sc.bTo_by_set[s].copy() for s in sets}

    rb = np.random.default_rng(1000 + seed)
    # systematic bias: 큐브 base 위치(x,y) 에 선형 의존 (abc_calib.load_sim 과 동일 성격)
    #   공통 G_common (모든 센서 공유, 학습가능) + 센서별 G_sensor (불일치원)
    s_pos = noise_mm / 1000 / 0.3
    G_common = rb.normal(0, s_pos * 0.7, (3, 2))
    G_cam = {ci: rb.normal(0, s_pos * 0.7, (3, 2)) for ci in cam_ids}
    G_grip = rb.normal(0, s_pos * 0.7, (3, 2))   # 그리퍼 카메라도 자기 편향

    event_set = {e: sc.event_set[e] for e in sc.events}
    set_events = {}
    for e in sc.events:
        set_events.setdefault(event_set[e], []).append(e)

    def _bias_base(G_sensor, s):
        """set s 큐브 (x,y) 에 선형 의존하는 base frame 편향 벡터 (3,)."""
        xy = FK[s][:2, 3]
        return (G_common + G_sensor) @ xy

    def _inject(T, R_cb, bias_base_vec, rng_key):
        """관측 T(camera<-cube) 에 노이즈 주입. systematic=base bias, random=가우시안."""
        Tn = T.copy()
        if noise_type == "random":
            rng = np.random.default_rng(hash(rng_key) % (2**31))
            Tn[:3, 3] = Tn[:3, 3] + rng.normal(0, noise_mm / 1000, 3)
        else:  # systematic: base frame 편향을 camera frame 으로 옮겨 병진에 더함
            Tn[:3, 3] = Tn[:3, 3] + R_cb @ bias_base_vec
        return Tn

    # --- 고정 카메라 관측: set 당 1회 (대표 이벤트) ---
    obs_fixed = {}
    for ci in cam_ids:
        R_cb = inv_T(sc.bTf[ci])[:3, :3]
        for s in sets:
            eid = set_events[s][0]
            T = sc.T_Ci_O(ci, eid).copy()
            obs_fixed[(ci, s)] = _inject(T, R_cb, _bias_base(G_cam[ci], s),
                                         (ci, s, seed))

    # --- 그리퍼 카메라 관측: 모든 이벤트 (핸드아이용) ---
    #  그리퍼는 event 마다 자세가 다르므로 cam<-base 회전도 event 마다 다름.
    obs_grip = {}
    for e in sc.events:
        s = event_set[e]
        R_cb = inv_T(sc.bTg[e] @ sc.gTc)[:3, :3]   # base -> gripper-camera 회전
        T = sc.T_Cg_O(e).copy()
        obs_grip[e] = _inject(T, R_cb, _bias_base(G_grip, s), ("grip", e, seed))

    return dict(cam_ids=cam_ids, sets=sets, FK=FK,
                obs_fixed=obs_fixed, obs_grip=obs_grip,
                bTg={e: sc.bTg[e].copy() for e in sc.events},
                event_set=event_set, set_events=set_events,
                gt_bTf={ci: sc.bTf[ci].copy() for ci in cam_ids},
                gt_gTc=sc.gTc.copy())


# ======================================================================
#  핸드아이 gTc 솔버 (cv2.calibrateHandEye, Step3 규약)
# ======================================================================
def solve_handeye(obs_grip, bTg, events, method=cv2.CALIB_HAND_EYE_PARK):
    """그리퍼 관측(camera<-cube)들로 gTc(gripper<-camera) 복원."""
    if len(events) < 3:
        return None
    R_g2b = [bTg[e][:3, :3] for e in events]
    t_g2b = [bTg[e][:3, 3].reshape(3, 1) for e in events]
    R_t2c = [obs_grip[e][:3, :3] for e in events]
    t_t2c = [obs_grip[e][:3, 3].reshape(3, 1) for e in events]
    R_gc, t_gc = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T = np.eye(4)
    T[:3, :3] = np.asarray(R_gc).reshape(3, 3)
    T[:3, 3] = np.asarray(t_gc).reshape(3)
    return T


def grip_predict_cube(gTc, obs_grip, bTg, events):
    """그리퍼 카메라 관측을 base 로 올려 큐브 위치 예측: bTg @ gTc @ obs."""
    Ts = [bTg[e] @ gTc @ obs_grip[e] for e in events]
    return se3_avg(Ts) if Ts else None


# ======================================================================
#  독립(Independent) 캘리브 — 고정과 그리퍼를 *따로* 푼다 (정보 공유 없음)
#   - 고정 카메라: 카메라 합의(no-fk)  ← 그리퍼 정보 미사용
#   - 그리퍼:      cv2.calibrateHandEye 로 gTc 따로  ← 고정 정보 미사용
# ======================================================================
def calib_independent(scene, train_sets):
    d = scene
    cam_ids, obs_f, FK = d["cam_ids"], d["obs_fixed"], d["FK"]
    # 고정 카메라: 카메라 합의(no-fk). FK 는 gauge 초기화에만 쓰고(절대 base 기준),
    #   그 뒤 캘리브 신호는 카메라 관측 합의뿐 (원래 B 방법의 정의).
    cubeFK = {s: FK[s] for s in train_sets if s in FK}
    cams0 = solve_camera_from_cube(cubeFK, obs_f, cam_ids, train_sets)  # gauge 초기화
    cube_consensus = {s: predict_cube(cams0, obs_f, cam_ids, s) for s in train_sets}
    cube_consensus = {s: T for s, T in cube_consensus.items() if T is not None}
    cams = solve_camera_from_cube(cube_consensus, obs_f, cam_ids, train_sets)
    # 그리퍼 핸드아이 (train set 에 속한 이벤트만) — 고정 카메라와 정보 공유 안 함(독립)
    train_events = [e for s in train_sets for e in d["set_events"].get(s, [])]
    gTc = solve_handeye(d["obs_grip"], d["bTg"], train_events)
    return {"cams": cams, "gTc": gTc, "mode": "indep"}


# ======================================================================
#  통합(Unified) 캘리브 — 고정+그리퍼를 *하나의 공유 큐브 앵커* 로 함께 푼다
#
#   핵심(독립과의 유일한 차이): 큐브 앵커를 고정 카메라 관측 *과* 그리퍼 관측을
#   *함께* 평균해 만든다 → 그 공유 앵커로 bTf 와 gTc 를 동시에 역산.
#     - 독립: gTc 는 그리퍼만의 AX=XB, 고정은 고정끼리만 → 두 서브시스템 분리.
#     - 통합: 두 서브시스템이 같은 큐브 앵커에 합의 → base frame 공유·상호 보정.
#
#   *FK 는 gauge 초기화에만* 쓴다(독립과 동일). 캘리브 신호는 카메라·그리퍼 관측뿐.
#   → FK 잔차보정은 오직 +fk(C) 단계에서 양쪽에 동일하게 적용 (대칭 비교).
# ======================================================================
def calib_unified(scene, train_sets):
    d = scene
    cam_ids, obs_f, FK = d["cam_ids"], d["obs_fixed"], d["FK"]
    train_events = [e for s in train_sets for e in d["set_events"].get(s, [])]

    # 0) 부트스트랩: 고정=gauge 초기화(FK), 그리퍼=핸드아이 (초기값; 독립과 동일)
    cams0 = solve_camera_from_cube({s: FK[s] for s in train_sets if s in FK},
                                   obs_f, cam_ids, train_sets)
    gTc0 = solve_handeye(d["obs_grip"], d["bTg"], train_events)
    if gTc0 is None:
        return {"cams": cams0, "gTc": None, "mode": "unified"}

    # 1) 공유 큐브 앵커 = 고정 관측 + 그리퍼 관측을 *함께* 합의 (통합의 본질)
    cube_by_set = {}
    for s in train_sets:
        Ts = [cams0[ci] @ obs_f[(ci, s)] for ci in cam_ids if (ci, s) in obs_f]
        for e in d["set_events"].get(s, []):
            Ts.append(d["bTg"][e] @ gTc0 @ d["obs_grip"][e])   # 그리퍼도 같은 앵커로
        if Ts:
            cube_by_set[s] = se3_avg(Ts)

    # 2) 공유 앵커로 고정 카메라 bTf 역산
    cams = solve_camera_from_cube(cube_by_set, obs_f, cam_ids, train_sets)

    # 3) *같은 공유 앵커* 로 그리퍼 gTc 재추정 (고정 정보가 간접 반영됨)
    #      각 event: gTc = inv(bTg[e]) @ cube_anchor[s] @ inv(obs_grip[e])
    gTc_ests = []
    for s in train_sets:
        if s not in cube_by_set:
            continue
        for e in d["set_events"].get(s, []):
            gTc_ests.append(inv_T(d["bTg"][e]) @ cube_by_set[s] @ inv_T(d["obs_grip"][e]))
    gTc_ref = se3_avg(gTc_ests) if gTc_ests else gTc0
    return {"cams": cams, "gTc": gTc_ref, "mode": "unified"}


# ======================================================================
#  공유 예측 헬퍼 — 고정+그리퍼를 함께 써서 test 큐브를 base 에서 예측
# ======================================================================
def predict_cube_joint(model, scene, s):
    """고정 카메라 + 그리퍼 카메라를 *모두* 써서 set s 큐브 위치 예측 (공유 base)."""
    d = scene
    Ts = []
    for ci in d["cam_ids"]:
        if (ci, s) in d["obs_fixed"]:
            Ts.append(model["cams"][ci] @ d["obs_fixed"][(ci, s)])
    if model["gTc"] is not None:
        for e in d["set_events"].get(s, []):
            Ts.append(d["bTg"][e] @ model["gTc"] @ d["obs_grip"][e])
    return se3_avg(Ts) if Ts else None


# ======================================================================
#  평가 지표
# ======================================================================
def _rot_deg(A, B):
    R = A[:3, :3].T @ B[:3, :3]
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def _trans_mm(A, B):
    return float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000)


def eval_model(model, scene, train_sets, test_sets, lam=1e-3, use_C=False):
    """한 model(indep/1pass/joint)에 대해 4종 지표를 계산해 dict 반환."""
    d = scene
    out = {}
    # Indep 은 rigid 정합된 예측(그리퍼→고정 base)을 사용, 그 외는 공동 평균.
    is_indep = model.get("mode") == "indep" and "align" in model
    def _predict(s):
        return (predict_cube_independent_aligned(model, scene, s) if is_indep
                else predict_cube_joint(model, scene, s))

    # 1) 고정 카메라 bTf 오차 (GT 대비)
    ferr = [_trans_mm(model["cams"][ci], d["gt_bTf"][ci])
            for ci in d["cam_ids"] if ci in model["cams"]]
    frot = [_rot_deg(model["cams"][ci], d["gt_bTf"][ci])
            for ci in d["cam_ids"] if ci in model["cams"]]
    out["bTf_mm"] = float(np.mean(ferr)) if ferr else None
    out["bTf_deg"] = float(np.mean(frot)) if frot else None

    # 2) 그리퍼 핸드아이 gTc 오차
    if model["gTc"] is not None:
        out["gTc_mm"] = _trans_mm(model["gTc"], d["gt_gTc"])
        out["gTc_deg"] = _rot_deg(model["gTc"], d["gt_gTc"])
    else:
        out["gTc_mm"] = out["gTc_deg"] = None

    # 3) 공유 base 정합 일관성: train set 에서 (고정 예측 큐브) vs (그리퍼 예측 큐브)
    #    두 서브시스템이 같은 base 에서 큐브를 얼마나 일치하게 보는가.
    cons = []
    for s in train_sets:
        pf = predict_cube(model["cams"], d["obs_fixed"], d["cam_ids"], s)
        pg = (grip_predict_cube(model["gTc"], d["obs_grip"], d["bTg"],
                                d["set_events"].get(s, []))
              if model["gTc"] is not None else None)
        if pf is not None and pg is not None:
            cons.append(_trans_mm(pf, pg))
    out["consistency_mm"] = float(np.mean(cons)) if cons else None

    # 4) 다운스트림 큐브 예측 (holdout test set) — 고정+그리퍼 공동 예측
    #    C 옵션: train 잔차를 큐브위치 [1,x,y] 에 ridge 회귀해 보정.
    W = None
    if use_C:
        X, Y = [], []
        for s in train_sets:
            p = _predict(s)
            if p is None or s not in d["FK"]:
                continue
            t = p[:3, 3]
            X.append(_resid_feature(t)); Y.append(d["FK"][s][:3, 3] - t)
        if len(X) >= 3:
            X = np.array(X); Y = np.array(Y)
            reg = lam * np.eye(X.shape[1]); reg[0, 0] = 0.0
            W = np.linalg.solve(X.T @ X + reg, X.T @ Y)

    derr = []
    for s in test_sets:
        p = _predict(s)
        if p is None or s not in d["FK"]:
            continue
        t = p[:3, 3]
        if W is not None:
            t = t + _resid_feature(t) @ W
        derr.append(np.linalg.norm(t - d["FK"][s][:3, 3]) * 1000)
    out["downstream_mm"] = float(np.mean(derr)) if derr else None
    return out


# ======================================================================
#  실험 러너 — holdout 조합 평균
# ======================================================================
#  세 통합 수준:
#   Indep  = 고정·그리퍼 따로 캘리브 → 공유 큐브로 rigid 정합(마지막에 붙임).
#   1pass  = 공유 큐브 앵커로 한 번 정보교환 (약통합).
#   Joint  = 모든 관측을 하나의 비선형 최소제곱으로 동시 최적화 (진짜 통합).
#  각 방식에 FK 잔차보정(+fk) 유무를 곱해 다운스트림 비교.
from joint_calib import (calib_independent_aligned, calib_joint,
                         predict_cube_independent_aligned)


def _m_indep(sc, tr):
    return calib_independent_aligned(sc, tr, calib_independent)


def _m_joint(sc, tr):
    return calib_joint(sc, tr, calib_unified)   # 1-pass 로 부트스트랩 후 동시 최적화


# 캘리브 방식(비싼 부분) 3개. C(잔차보정)는 예측 단계에서만 다르므로
#   방식당 캘리브는 *한 번만* 하고 no-fk/fk 를 함께 평가한다(중복 제거).
BASE_METHODS = [
    ("Indep", _m_indep),
    ("Joint", _m_joint),
]
METHODS = [(f"{n}{sfx}", fn, use_C)          # figure/report 용 6개 이름
           for n, fn in BASE_METHODS
           for sfx, use_C in [("", False), ("+fk", True)]]
KEYS = ["bTf_mm", "gTc_mm", "consistency_mm", "downstream_mm"]


def run(seeds=30, n_sets=8, noise_mm=6.0, noise_type="systematic", train_size=6,
        n_events_per_set=6):
    acc = {name: {k: [] for k in KEYS} for name, _, _ in METHODS}
    for seed in range(seeds):
        scene = build_scene(seed=seed, n_sets=n_sets, noise_mm=noise_mm,
                            noise_type=noise_type, n_events_per_set=n_events_per_set)
        sets = scene["sets"]
        for test_sets in itertools.combinations(sets, 2):
            rest = [s for s in sets if s not in test_sets]
            for train_sets in itertools.combinations(rest, train_size):
                tr = list(train_sets)
                for base, fn in BASE_METHODS:
                    model = fn(scene, tr)                       # 캘리브 1번만
                    for sfx, use_C in [("", False), ("+fk", True)]:
                        res = eval_model(model, scene, tr, list(test_sets),
                                         use_C=use_C)            # C 유무만 다름
                        name = f"{base}{sfx}"
                        for k in KEYS:
                            if res[k] is not None:
                                acc[name][k].append(res[k])
    return acc


def report(acc, header):
    print("\n" + "=" * 74)
    print(header)
    print("=" * 74)
    hdr = f"{'방법':<12}" + "".join(f"{k:>17}" for k in KEYS)
    print(hdr)
    print("-" * 74)
    labels = {"bTf_mm": "고정bTf(mm)", "gTc_mm": "핸드아이gTc(mm)",
              "consistency_mm": "공유정합(mm)", "downstream_mm": "다운스트림(mm)"}
    print(f"{'':<12}" + "".join(f"{labels[k]:>17}" for k in KEYS))
    print("-" * 74)
    for name, _, _ in METHODS:
        row = f"{name:<12}"
        for k in KEYS:
            a = np.array(acc[name][k])
            row += (f"{a.mean():>10.3f}±{a.std():<5.2f}" if len(a) else f"{'—':>17}")
        print(row)
    print("-" * 74)
    def _mean(n, k):
        a = acc[n][k]; return np.mean(a) if a else float("nan")
    # 독립 vs 통합(Joint) 비교
    print(f"  gTc 병진: Indep {_mean('Indep','gTc_mm'):.2f} -> "
          f"Joint {_mean('Joint','gTc_mm'):.2f} mm")
    print(f"  bTf 병진: Indep {_mean('Indep','bTf_mm'):.2f} | "
          f"Joint {_mean('Joint','bTf_mm'):.2f} mm")
    print(f"  다운스트림(no-fk): Indep {_mean('Indep','downstream_mm'):.2f} -> "
          f"Joint {_mean('Joint','downstream_mm'):.2f} mm")
    di, dj = _mean('Indep', 'downstream_mm'), _mean('Joint', 'downstream_mm')
    print(f"  -> Joint 가 Indep 대비 {(di-dj)/di*100:+.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--noise", type=float, default=6.0)
    ap.add_argument("--sets", type=int, default=8)
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--dump", type=str, default=None,
                    help="결과(mean/std)를 JSON 으로 저장할 경로 (figure 용)")
    args = ap.parse_args()

    print("#" * 74)
    print("# C1 — Unified Hybrid Calibration:  통합 vs 독립  (시뮬, GT 대비 오차)")
    print(f"#   고정 카메라 3대(eye-to-hand) + 그리퍼 카메라(eye-in-hand), "
          f"큐브 {args.sets} set")
    print(f"#   train={args.train}/test=2 holdout, {args.seeds} seed 평균, "
          f"관측 노이즈 {args.noise}mm")
    print("#" * 74)
    #  실제 검출오차의 지배성분 = systematic(위치의존 체계 편향)만 사용.
    #  random 은 실제와 성격이 달라 제외.
    dump = {}
    for ntype in ("systematic",):
        acc = run(seeds=args.seeds, n_sets=args.sets, noise_mm=args.noise,
                  noise_type=ntype, train_size=args.train)
        report(acc, f"[{ntype} 노이즈 {args.noise}mm]")
        dump[ntype] = {name: {k: [float(np.mean(acc[name][k])),
                                  float(np.std(acc[name][k])),
                                  len(acc[name][k])]
                              for k in KEYS} for name, _, _ in METHODS}
    if args.dump:
        import json
        with open(args.dump, "w") as f:
            json.dump({"meta": {"seeds": args.seeds, "noise": args.noise,
                                "sets": args.sets, "train": args.train,
                                "keys": KEYS, "methods": [m[0] for m in METHODS]},
                       "data": dump}, f, indent=2)
        print(f"\n[저장] {args.dump}")


if __name__ == "__main__":
    main()
