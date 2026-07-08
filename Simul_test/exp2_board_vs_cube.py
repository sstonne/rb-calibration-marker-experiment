"""
Experiment 2 — Planar Board vs Board+Marker Cube (순수 카메라 관측 비교, FK 미사용).

두 촬영 시나리오 (실제 운영을 반영)
----------------------------------
A) Board only:
   - 사람이 ChArUco board 를 손에 들고 여러 위치·자세로 옮긴다.
   - *최소 2대 카메라가 같은 board 를 동시에 보는* 자세에서만 촬영(사람이 눈으로 확인).
   - 이렇게 총 N_SHOTS(=30) 장 촬영. → 카메라 간 연결이 "2대씩" 사슬처럼 약하다.

B) Board + Cube:
   - board 는 바닥에 고정 → 4대 카메라가 *항상* board 를 본다(작업공간 바닥을 다 비춤).
   - cube 는 여러 위치로 이동, 6면 모두 마커라 어느 각도든 → 4대가 *항상* cube 도 본다.
   - → 매 촬영 4대가 같은 타깃을 봐서 카메라 간 연결이 강하고, cube 다면성으로 시야각 넓다.

비교: **FK(로봇 기구학) 미사용. 오직 카메라 관측 합의만으로** 두 방식을 캘리브하고
      카메라 외부파라미터 정확도(GT 대비)와 타깃(물체) 예측 정확도를 비교한다.

캘리브 방식 (FK 없음 → gauge = 카메라 0번 고정):
  - 각 촬영(shot)에서 타깃을 본 카메라들이 그 타깃을 공유한다는 제약으로,
    카메라<->타깃을 번갈아 추정(카메라 합의). base 절대기준이 없으므로 카메라 0을
    GT 로 고정(gauge)하고 나머지를 상대적으로 정렬해 GT 와 비교한다.

평가:
  - 카메라 위치 오차 (GT bTf 대비, gauge 정렬 후) — 순수 관측만으로 얼마나 정확한가
  - 타깃(물체) 예측 오차 — 캘리브된 카메라로 새 타깃 위치를 base 에서 예측
  - 관측성: 촬영당 동시 관측 카메라 수, 시야각 커버리지

실행:
  PYTHONPATH= python Simul_test/exp2_board_vs_cube.py                # 30 seed
  PYTHONPATH= python Simul_test/exp2_board_vs_cube.py --shots 30 --incidence 60
"""
import sys, os, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from aruco_cube import inv_T, rot_axis_angle
from utils_pose import robust_se3_average as se3_avg
from synthetic_scene import SyntheticScene


CUBE_FACE_NORMALS = {"+Z": np.array([0, 0, 1.0]), "+X": np.array([1.0, 0, 0]),
                     "+Y": np.array([0, 1.0, 0]), "-X": np.array([-1.0, 0, 0]),
                     "-Y": np.array([0, -1.0, 0])}
BOARD_NORMAL = {"board": np.array([0, 0, 1.0])}


# ----------------------------------------------------------------------
#  가시성: 카메라가 타깃(면들)을 입사각/FoV 로 볼 수 있나
# ----------------------------------------------------------------------
def _visible(bT_target, faces, cam_poses, inc_max_deg, fov_half_deg=45.0):
    R_bt, p_t = bT_target[:3, :3], bT_target[:3, 3]
    cos_inc, cos_fov = np.cos(np.deg2rad(inc_max_deg)), np.cos(np.deg2rad(fov_half_deg))
    out = {}
    for ci, bT_cam in cam_poses.items():
        d = p_t - bT_cam[:3, 3]; dist = np.linalg.norm(d)
        if dist < 1e-6:
            continue
        view = d / dist
        if float(bT_cam[:3, 2] @ view) < cos_fov:        # FoV
            continue
        best = max(float((R_bt @ n) @ (-view)) for n in faces.values())
        if best >= cos_inc:                               # 입사각 <= 임계
            out[ci] = view
    return out


# ----------------------------------------------------------------------
#  씬: 고정 4 카메라 (그리퍼 제외 — 이 실험은 순수 고정 카메라 관측 비교)
# ----------------------------------------------------------------------
def build_scene(seed=0, n_cams=4, inc_max_deg=60.0):
    sc = SyntheticScene(seed=seed, n_fixed_cams=n_cams, n_sets=1,
                        layout="realistic",
                        fixed_cam_radius_m=0.45, fixed_cam_height_m=0.35)
    cam_ids = list(sc.fixed_cam_ids)
    cam_poses = {ci: sc.bTf[ci].copy() for ci in cam_ids}
    center = np.zeros(3)
    return dict(sc=sc, cam_ids=cam_ids, cam_poses=cam_poses, center=center,
                gt_cam=cam_poses, inc=inc_max_deg)


# ----------------------------------------------------------------------
#  촬영(shot) 생성
# ----------------------------------------------------------------------
def make_shots_board_only(scene, n_shots=30, seed=0):
    """board 를 손에 들고 이동. 최소 2대가 볼 때만 촬영. n_shots 장 모을 때까지 시도."""
    d = scene; rng = np.random.default_rng(1000 + seed)
    shots = []
    tries = 0
    while len(shots) < n_shots and tries < n_shots * 60:
        tries += 1
        # board 를 작업공간 어딘가에 랜덤 위치·자세로 든다
        T = np.eye(4)
        T[:3, 3] = d["center"] + rng.uniform(-0.15, 0.15, 3)
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-12
        T[:3, :3] = rot_axis_angle(ax, np.deg2rad(rng.uniform(-70, 70)))
        vis = _visible(T, BOARD_NORMAL, d["cam_poses"], d["inc"])
        if len(vis) >= 2:                                 # 2대+ 동시 관측일 때만 촬영
            shots.append({"target": T, "vis": vis, "is_cube": False})
    return shots


def make_shots_board_cube(scene, n_shots=30, seed=0):
    """board 바닥 고정(항상 4대) + cube 여러 위치(항상 4대). 각 shot = board관측 + cube관측."""
    d = scene; rng = np.random.default_rng(2000 + seed)
    # board 바닥 고정: 작업공간 중심 바닥, 위를 향함(+Z) → 둘러싼 4대가 다 비춤
    T_board = np.eye(4); T_board[:3, 3] = d["center"].copy()
    shots = []
    for _ in range(n_shots):
        # cube 를 여러 위치로 (6면이라 항상 4대가 봄)
        Tc = np.eye(4)
        Tc[:3, 3] = d["center"] + rng.uniform(-0.12, 0.12, 3)
        axc = rng.normal(size=3); axc /= np.linalg.norm(axc) + 1e-12
        Tc[:3, :3] = rot_axis_angle(axc, np.deg2rad(rng.uniform(-180, 180)))
        vis_b = _visible(T_board, BOARD_NORMAL, d["cam_poses"], d["inc"])
        vis_c = _visible(Tc, CUBE_FACE_NORMALS, d["cam_poses"], d["inc"])
        shots.append({"target": T_board, "vis": vis_b, "is_cube": False,
                      "target2": Tc, "vis2": vis_c})
    return shots


# ----------------------------------------------------------------------
#  관측 생성 (camera<-target, 노이즈) + 캘리브 (FK 없음, 카메라 합의)
# ----------------------------------------------------------------------
def _obs_of_shot(shot, cam_poses, noise_mm, rng):
    """shot 의 각 타깃에 대해 관측 리스트 생성: [(cam_id, T_cam_target, target_key), ...]
       target_key 로 같은 타깃을 본 카메라끼리 묶는다."""
    obs = []
    def add(T_target, vis, key):
        for ci in vis:
            Tc = inv_T(cam_poses[ci]) @ T_target
            Tc[:3, 3] = Tc[:3, 3] + rng.normal(0, noise_mm / 1000, 3)
            obs.append((ci, Tc, key))
    add(shot["target"], shot["vis"], "b")
    if "target2" in shot:
        add(shot["target2"], shot["vis2"], "c")
    return obs


def calibrate(scene, shots, noise_mm=6.0, iters=5):
    """순수 카메라 합의 캘리브 (FK 미사용).

    각 shot 의 각 타깃마다 '그 타깃을 본 카메라들이 같은 위치를 가리킨다'는 제약.
      - 타깃 위치[shot,key] = 카메라들이 본 것의 합의
      - 카메라[ci] = 그 타깃들로 역산
    gauge: 카메라 0 을 GT 로 고정(절대기준 없으므로). 반복 수렴.
    """
    d = scene; cam_ids = d["cam_ids"]
    rng = np.random.default_rng(hash(("obs", noise_mm)) % (2**31))
    # 관측 테이블: obs_list[(shot_idx, key)] = [(ci, T_cam_target), ...]
    obs_tab = {}
    cams_seen = {ci: 0 for ci in cam_ids}
    for si, shot in enumerate(shots):
        for (ci, Tc, key) in _obs_of_shot(shot, d["cam_poses"], noise_mm, rng):
            obs_tab.setdefault((si, key), []).append((ci, Tc))
            cams_seen[ci] += 1
    calib_cams = [ci for ci in cam_ids if cams_seen[ci] >= 2]
    if len(calib_cams) < 2:
        return {}, {}

    # gauge: 카메라 0 = GT 고정, 나머지는 nominal(대략) 초기화
    gauge = calib_cams[0]
    cams = {ci: d["gt_cam"][ci].copy() for ci in calib_cams}   # 초기값 GT 근처
    rng0 = np.random.default_rng(42)
    for ci in calib_cams:
        if ci != gauge:
            cams[ci][:3, 3] += rng0.normal(0, 0.02, 3)         # 나머지만 섭동

    for _ in range(iters):
        # 1) 타깃 위치 합의 (그 타깃을 본 카메라들이 base 로 올린 것의 평균)
        tgt = {}
        for k, lst in obs_tab.items():
            Ts = [cams[ci] @ Tc for (ci, Tc) in lst if ci in cams]
            if Ts:
                tgt[k] = se3_avg(Ts)
        # 2) 카메라 역산 (그 타깃들로; gauge 카메라는 GT 고정)
        new = {}
        for ci in calib_cams:
            if ci == gauge:
                new[ci] = d["gt_cam"][ci].copy(); continue
            Ts = [tgt[k] @ inv_T(Tc) for k, lst in obs_tab.items()
                  for (cj, Tc) in lst if cj == ci and k in tgt]
            if Ts:
                new[ci] = se3_avg(Ts)
        cams = new if new else cams
    # 최종 타깃 위치도 반환
    tgt = {}
    for k, lst in obs_tab.items():
        Ts = [cams[ci] @ Tc for (ci, Tc) in lst if ci in cams]
        if Ts:
            tgt[k] = se3_avg(Ts)
    return cams, obs_tab


# ----------------------------------------------------------------------
#  평가
# ----------------------------------------------------------------------
def _trans_mm(A, B): return float(np.linalg.norm(A[:3, 3] - B[:3, 3]) * 1000)
def _rot_deg(A, B):
    R = A[:3, :3].T @ B[:3, :3]; c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def evaluate(scene, shots, cams_est, obs_tab, noise_mm=6.0):
    d = scene
    # 카메라 위치 오차 (GT 대비; gauge=카메라0 고정이라 절대비교 가능)
    cam_mm = [_trans_mm(cams_est[ci], d["gt_cam"][ci]) for ci in cams_est]
    cam_deg = [_rot_deg(cams_est[ci], d["gt_cam"][ci]) for ci in cams_est]
    # 타깃 예측 오차: 각 타깃(shot,key)을 캘리브 카메라들로 예측 vs GT
    obj_errs = []
    gt_targets = {}
    for si, shot in enumerate(shots):
        gt_targets[(si, "b")] = shot["target"]
        if "target2" in shot:
            gt_targets[(si, "c")] = shot["target2"]
    rng = np.random.default_rng(hash(("ev", noise_mm)) % (2**31))
    for k, lst in obs_tab.items():
        preds = [cams_est[ci] @ Tc for (ci, Tc) in lst if ci in cams_est]
        if preds and k in gt_targets:
            p = se3_avg(preds)[:3, 3]
            obj_errs.append(np.linalg.norm(p - gt_targets[k][:3, 3]) * 1000)
    return {
        "cam_mm": float(np.mean(cam_mm)) if cam_mm else None,
        "cam_deg": float(np.mean(cam_deg)) if cam_deg else None,
        "obj_mm": float(np.mean(obj_errs)) if obj_errs else None,
        "n_calib": len(cams_est),
    }


def observability(shots):
    """촬영당 동시 관측 카메라 수 + 시야각 커버리지."""
    simul, cover = [], []
    for shot in shots:
        vis = dict(shot["vis"])
        if "vis2" in shot:
            for ci, v in shot["vis2"].items():
                vis.setdefault(ci, v)
        simul.append(len(vis))
        dirs = list(vis.values())
        if len(dirs) >= 2:
            angs = [np.degrees(np.arccos(np.clip(dirs[i] @ dirs[j], -1, 1)))
                    for i in range(len(dirs)) for j in range(i + 1, len(dirs))]
            cover.append(np.mean(angs))
        else:
            cover.append(0.0)
    return float(np.mean(simul)), float(np.mean(cover))


# ----------------------------------------------------------------------
#  러너
# ----------------------------------------------------------------------
def run(seeds=30, n_shots=30, incidence=60.0, noise_mm=6.0):
    KS = ["simul", "coverage", "cam_mm", "cam_deg", "obj_mm", "n_calib"]
    acc = {m: {k: [] for k in KS} for m in ["board", "board+cube"]}
    for seed in range(seeds):
        scene = build_scene(seed=seed, inc_max_deg=incidence)
        shots_A = make_shots_board_only(scene, n_shots, seed)
        shots_B = make_shots_board_cube(scene, n_shots, seed)
        for mode, shots in [("board", shots_A), ("board+cube", shots_B)]:
            if not shots:
                continue
            simul, cover = observability(shots)
            acc[mode]["simul"].append(simul)
            acc[mode]["coverage"].append(cover)
            cams_est, obs_tab = calibrate(scene, shots, noise_mm)
            ev = evaluate(scene, shots, cams_est, obs_tab, noise_mm)
            for k in ["cam_mm", "cam_deg", "obj_mm", "n_calib"]:
                if ev[k] is not None:
                    acc[mode][k].append(ev[k])
    return acc


def report(acc, incidence, seeds, n_shots):
    print("=" * 70)
    print(f" Experiment 2 — Board vs Board+Cube  (FK 미사용, 순수 카메라 관측)")
    print(f"   고정 카메라 4대,  입사각 임계 {incidence:.0f}°,  {n_shots}장/방식,  {seeds} seed")
    print("=" * 70)
    def _m(d, k): return np.mean(d[k]) if d[k] else float("nan")
    b, c = acc["board"], acc["board+cube"]
    lab = {"simul": "동시관측(대)", "coverage": "시야각(°)", "n_calib": "캘리브카메라",
           "cam_mm": "카메라오차(mm)", "cam_deg": "카메라오차(°)", "obj_mm": "타깃예측(mm)"}
    print(f"{'지표':<16}{'Board only':>16}{'Board+Cube':>16}")
    print("-" * 70)
    for k in ["simul", "coverage", "n_calib", "cam_mm", "cam_deg", "obj_mm"]:
        print(f"{lab[k]:<16}{_m(b,k):>16.2f}{_m(c,k):>16.2f}")
    print("-" * 70)
    print(f"  카메라 위치 오차: board {_m(b,'cam_mm'):.2f} -> board+cube {_m(c,'cam_mm'):.2f} mm "
          f"({(_m(b,'cam_mm')-_m(c,'cam_mm'))/_m(b,'cam_mm')*100:+.0f}%)")
    print(f"  타깃 예측 오차:   board {_m(b,'obj_mm'):.2f} -> board+cube {_m(c,'obj_mm'):.2f} mm "
          f"({(_m(b,'obj_mm')-_m(c,'obj_mm'))/_m(b,'obj_mm')*100:+.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--shots", type=int, default=30)
    ap.add_argument("--incidence", type=float, default=60.0)
    ap.add_argument("--noise", type=float, default=6.0)
    ap.add_argument("--dump", type=str, default=None)
    args = ap.parse_args()
    acc = run(seeds=args.seeds, n_shots=args.shots, incidence=args.incidence,
              noise_mm=args.noise)
    report(acc, args.incidence, args.seeds, args.shots)
    if args.dump:
        import json
        out = {m: {k: [float(np.mean(acc[m][k])), float(np.std(acc[m][k]))]
                   for k in acc[m] if acc[m][k]} for m in acc}
        json.dump({"meta": {"seeds": args.seeds, "shots": args.shots,
                            "incidence": args.incidence}, "data": out},
                  open(args.dump, "w"), indent=2)
        print(f"\n[저장] {args.dump}")


if __name__ == "__main__":
    main()
