"""
Experiment 3 — Gripper-target 변환 추정: Camera-based vs FK-based vs Camera+FK후보정.

목적
----
통합(Joint) 캘리브에서 gripper-target 변환(gTc)과 타깃 위치를 다루는 세 방식을 비교한다:

  ① Camera-based        : 큐브(타깃) 위치를 *미지수*로 두고 카메라 관측으로 함께 추정.
                          (= joint_calib.calib_joint,  gripper-target 을 카메라로 추정)
  ② Robot FK-based      : 큐브 위치를 로봇 FK 로 아는 값으로 *고정*하고 카메라·gTc 만 최적화.
                          (= joint_calib.calib_joint_fk_fixed,  타깃을 FK 로 앎)
  ③ Camera + FK 후보정  : ① 로 추정 후, FK 정답으로 학습한 위치의존 잔차(Ridge)를 후보정.
                          (= ① + 잔차보정 W)

지표 (모두 GT 대비, 시뮬이라 정답을 앎)
  [핵심] Held-out 큐브 예측 오차 (mm)  : 실전 성능 (train/test holdout)
  [진단] gTc 복원 오차 (mm/deg)         : gripper-target 변환 정확도 (세 방식의 핵심 차이)
  [진단] Camera-to-base 오차 (mm)       : 고정 카메라 위치 정확도
  [진단] Prior/Target consistency (mm)  : 카메라 추정 타깃 vs FK 타깃의 정합성
  [sweep] 데이터량 민감도 (train set 수), 노이즈 강건성 (관측 노이즈)

  (Reprojection error 는 시뮬에서 주입 노이즈를 되비출 뿐 변별력이 없어 제외.)

실행:
  PYTHONPATH= python Simul_test/exp3_gtc_estimation.py               # 메인 3방식 비교
  PYTHONPATH= python Simul_test/exp3_gtc_estimation.py --sweep set   # 데이터량 sweep
  PYTHONPATH= python Simul_test/exp3_gtc_estimation.py --sweep noise # 노이즈 sweep
"""
import sys, os, argparse, itertools
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from unified_vs_independent import (build_scene, calib_unified, predict_cube,
                                    predict_cube_joint, _resid_feature,
                                    _trans_mm, _rot_deg, grip_predict_cube)
from joint_calib import (calib_joint, calib_joint_fk_fixed,
                         predict_cube_joint_model)


# ----------------------------------------------------------------------
#  세 방식 캘리브
# ----------------------------------------------------------------------
def calib_camera_based(scene, train_sets):
    """① 큐브를 미지수로 두고 카메라로 추정 (Joint)."""
    return calib_joint(scene, train_sets, calib_unified)


def calib_fk_based(scene, train_sets):
    """② 큐브를 FK 로 고정하고 카메라·gTc 만 최적화."""
    return calib_joint_fk_fixed(scene, train_sets, calib_unified)


def learn_fk_residual(scene, model, train_sets, lam=1e-3):
    """③ 후보정용: train 에서 (카메라 예측 vs FK 정답) 잔차를 Ridge 회귀."""
    d = scene
    X, Y = [], []
    for s in train_sets:
        p = predict_cube_joint(model, scene, s)
        if p is None or s not in d["FK"]:
            continue
        t = p[:3, 3]
        X.append(_resid_feature(t)); Y.append(d["FK"][s][:3, 3] - t)
    if len(X) < 3:
        return None
    X = np.array(X); Y = np.array(Y)
    reg = lam * np.eye(X.shape[1]); reg[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + reg, X.T @ Y)


# ----------------------------------------------------------------------
#  평가 (한 model + 방식에 대해 4지표)
# ----------------------------------------------------------------------
def eval_method(scene, model, train_sets, test_sets, W=None):
    d = scene
    out = {}
    # gTc 복원
    out["gtc_mm"] = _trans_mm(model["gTc"], d["gt_gTc"]) if model["gTc"] is not None else None
    out["gtc_deg"] = _rot_deg(model["gTc"], d["gt_gTc"]) if model["gTc"] is not None else None
    # Camera-to-base 오차 (고정 카메라 GT 대비)
    ce = [_trans_mm(model["cams"][ci], d["gt_bTf"][ci])
          for ci in d["cam_ids"] if ci in model["cams"]]
    out["cam_mm"] = float(np.mean(ce)) if ce else None
    # Prior/Target consistency: 카메라 추정 타깃 vs FK 타깃 (train)
    cons = []
    for s in train_sets:
        p = predict_cube_joint(model, scene, s)
        if p is not None and s in d["FK"]:
            cons.append(_trans_mm(p, d["FK"][s]))
    out["consistency_mm"] = float(np.mean(cons)) if cons else None
    # Held-out 큐브 예측 (핵심)
    derr = []
    for s in test_sets:
        p = predict_cube_joint(model, scene, s)
        if p is None or s not in d["FK"]:
            continue
        t = p[:3, 3]
        if W is not None:
            t = t + _resid_feature(t) @ W
        derr.append(np.linalg.norm(t - d["FK"][s][:3, 3]) * 1000)
    out["heldout_mm"] = float(np.mean(derr)) if derr else None
    return out


METHODS = ["Camera-based", "FK-based", "Camera+FK-corr"]
KEYS = ["heldout_mm", "gtc_mm", "gtc_deg", "cam_mm", "consistency_mm"]


def run_once(scene, train_sets, test_sets):
    """한 train/test 분할에서 세 방식 평가."""
    res = {}
    # ① camera-based (③ 후보정도 이 위에 얹음)
    m_cam = calib_camera_based(scene, train_sets)
    res["Camera-based"] = eval_method(scene, m_cam, train_sets, test_sets)
    W = learn_fk_residual(scene, m_cam, train_sets)
    res["Camera+FK-corr"] = eval_method(scene, m_cam, train_sets, test_sets, W=W)
    # ② fk-based
    m_fk = calib_fk_based(scene, train_sets)
    res["FK-based"] = eval_method(scene, m_fk, train_sets, test_sets)
    return res


def run(seeds=20, n_sets=8, noise_mm=6.0, train_size=6):
    acc = {m: {k: [] for k in KEYS} for m in METHODS}
    for seed in range(seeds):
        scene = build_scene(seed=seed, n_sets=n_sets, noise_mm=noise_mm,
                            noise_type="systematic")
        sets = scene["sets"]
        for test_sets in itertools.combinations(sets, 2):
            rest = [s for s in sets if s not in test_sets]
            for train_sets in itertools.combinations(rest, train_size):
                res = run_once(scene, list(train_sets), list(test_sets))
                for m in METHODS:
                    for k in KEYS:
                        if res[m][k] is not None:
                            acc[m][k].append(res[m][k])
    return acc


# ----------------------------------------------------------------------
#  Sweep (데이터량 / 노이즈)
# ----------------------------------------------------------------------
def run_sweep(kind, seeds=15, n_sets=8, noise_mm=6.0):
    """kind='set': train_size 스윕 / kind='noise': noise_mm 스윕. heldout·gtc 반환."""
    if kind == "set":
        xs = [3, 4, 5, 6]
    else:
        xs = [2, 4, 6, 10, 15]
    curve = {m: {"heldout": [], "gtc": []} for m in METHODS}
    for x in xs:
        acc = {m: {"heldout": [], "gtc": []} for m in METHODS}
        for seed in range(seeds):
            nz = noise_mm if kind == "set" else x
            scene = build_scene(seed=seed, n_sets=n_sets, noise_mm=nz,
                                noise_type="systematic")
            sets = scene["sets"]
            ts = x if kind == "set" else 6
            test_combos = list(itertools.combinations(sets, 2))
            for test_sets in test_combos[:4]:            # test 조합도 제한(속도)
                rest = [s for s in sets if s not in test_sets]
                combos = list(itertools.combinations(rest, ts))
                for train_sets in combos[:1]:            # 조합 1개만(속도)
                    res = run_once(scene, list(train_sets), list(test_sets))
                    for m in METHODS:
                        if res[m]["heldout_mm"] is not None:
                            acc[m]["heldout"].append(res[m]["heldout_mm"])
                        if res[m]["gtc_mm"] is not None:
                            acc[m]["gtc"].append(res[m]["gtc_mm"])
        for m in METHODS:
            curve[m]["heldout"].append(float(np.mean(acc[m]["heldout"])) if acc[m]["heldout"] else np.nan)
            curve[m]["gtc"].append(float(np.mean(acc[m]["gtc"])) if acc[m]["gtc"] else np.nan)
    return xs, curve


# ----------------------------------------------------------------------
#  리포트
# ----------------------------------------------------------------------
def report(acc, seeds):
    print("=" * 78)
    print(" Experiment 3 — Camera-based vs FK-based vs Camera+FK후보정 (통합 Joint 기반)")
    print(f"   고정3+그리퍼1,  큐브 8set,  train=6/test=2,  systematic 6mm,  {seeds} seed")
    print("=" * 78)
    lab = {"heldout_mm": "★Held-out(mm)", "gtc_mm": "gTc(mm)", "gtc_deg": "gTc(°)",
           "cam_mm": "Cam-base(mm)", "consistency_mm": "Prior정합(mm)"}
    print(f"{'방식':<16}" + "".join(f"{lab[k]:>15}" for k in KEYS))
    print("-" * 78)
    for m in METHODS:
        row = f"{m:<16}"
        for k in KEYS:
            a = np.array(acc[m][k])
            row += f"{a.mean():>10.2f}±{a.std():<4.1f}" if len(a) else f"{'—':>15}"
        print(row)
    print("-" * 78)
    def _m(m, k): return np.mean(acc[m][k]) if acc[m][k] else float("nan")
    print(f"  ★Held-out: Camera {_m('Camera-based','heldout_mm'):.2f} | "
          f"FK {_m('FK-based','heldout_mm'):.2f} | "
          f"Camera+FK {_m('Camera+FK-corr','heldout_mm'):.2f} mm")
    print(f"  gTc 복원:  Camera {_m('Camera-based','gtc_mm'):.2f} | "
          f"FK {_m('FK-based','gtc_mm'):.2f} | "
          f"Camera+FK {_m('Camera+FK-corr','gtc_mm'):.2f} mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--sweep", choices=["set", "noise"], default=None)
    ap.add_argument("--dump", type=str, default=None)
    args = ap.parse_args()

    if args.sweep:
        xs, curve = run_sweep(args.sweep, seeds=max(args.seeds // 2, 8))
        print(f"[sweep {args.sweep}]  x={xs}")
        for m in METHODS:
            print(f"  {m:<16} heldout={['%.2f'%v for v in curve[m]['heldout']]}  "
                  f"gtc={['%.2f'%v for v in curve[m]['gtc']]}")
        if args.dump:
            import json
            json.dump({"kind": args.sweep, "xs": xs, "curve": curve},
                      open(args.dump, "w"), indent=2)
            print(f"[저장] {args.dump}")
        return

    acc = run(seeds=args.seeds)
    report(acc, args.seeds)
    if args.dump:
        import json
        out = {m: {k: [float(np.mean(acc[m][k])), float(np.std(acc[m][k]))]
                   for k in KEYS if acc[m][k]} for m in METHODS}
        json.dump({"meta": {"seeds": args.seeds}, "data": out},
                  open(args.dump, "w"), indent=2)
        print(f"\n[저장] {args.dump}")


if __name__ == "__main__":
    main()
