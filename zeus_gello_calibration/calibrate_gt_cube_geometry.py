#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/calibrate_gt_cube_geometry.py -- GT 큐브 마커 기하 자체 보정.

targets/gt_cube/cube_config.json 의 face_roll_deg 는 메인 큐브에서 복사한 값이라
실물 GT 큐브에 맞지 않는다(다면 관측 PnP RMSE 9~19 px). 이 스크립트는 GT 큐브가
찍힌 모든 사진(0914 session1/2/3 + 외부 GT 프레임)에서 마커를 검출해

  1단계: 측면 마커(2~5)의 in-plane roll 을 {0,90,180,270}deg 이산 탐색으로 고정하고
  2단계: 상단 마커 0,1 을 게이지로 고정한 채, 측면 마커 4개의 6-DoF 포즈와 사진별
         큐브 포즈를 함께 최적화(코너 재투영 bundle adjustment)

한 뒤 결과를 cube_config.json 의 ``marker_pose_4x4`` 로 기록한다.
(카메라 외부 파라미터는 전혀 쓰지 않는다 -- 사진마다 큐브 포즈를 따로 두므로
 카메라 내부 파라미터만 있으면 된다.)

사용법:
  python calibrate_gt_cube_geometry.py            # 진단만
  python calibrate_gt_cube_geometry.py --write    # cube_config.json 갱신(백업 남김)
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from calibration_pipeline.apriltag_cube import AprilTagCubeTarget, rodrigues_to_Rt  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402

from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label  # noqa: E402
from paths import SESSION1_DIR, SESSION2_DIR, SESSION3_DIR, ZEUS_DATA_ROOT  # noqa: E402

GT_CFG_PATH = REPO_ROOT / "targets" / "gt_cube" / "cube_config.json"
TOP_IDS = (0, 1)
SIDE_IDS = (2, 3, 4, 5)


def image_files(roots):
    out = []
    for r in roots:
        r = Path(r)
        if not r.is_dir():
            continue
        for p in sorted(r.rglob("cam_*.png")):
            if p.name.endswith("_depth.png"):
                continue
            label = p.stem[len("cam_"):]
            if label in LOCAL_CAM_IDS:
                out.append((p, LOCAL_CAM_IDS[label]))
    return out


def detect_all(files, target, K, D):
    """사진별 {marker_id: 4x2 (reordered) corners}."""
    obs = []
    for p, cid in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        corners_list, ids = target.detect(img)
        if ids is None:
            continue
        d = {}
        for c, mid in zip(corners_list, ids):
            mid = int(mid)
            if not target.model.has_marker(mid):
                continue
            d[mid] = target.model.reorder_image_corners(mid, np.asarray(c).reshape(4, 2))
        if d:
            obs.append({"path": p, "cam": cid, "corners": d})
    return obs


def pnp_rmse(model, corners_by_id, ids, K, D):
    ids = [m for m in ids if m in corners_by_id]
    if len(ids) < 2:
        return None, None
    obj = np.concatenate([model.model.marker_corners_in_rig(m) for m in ids]).reshape(-1, 1, 3)
    img = np.concatenate([corners_by_id[m] for m in ids]).reshape(-1, 1, 2)
    ok, rv, tv = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None
    proj, _ = cv2.projectPoints(obj.reshape(-1, 3), rv, tv, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(err ** 2))), (rv, tv)


def discrete_roll_search(cfg, obs, K, D):
    """측면 마커별 roll ∈ {0,90,180,270}: 이미 확정된 마커들과 같이 찍힌 사진의 PnP RMSE 중앙값 최소."""
    resolved = set(TOP_IDS)
    chosen = {}
    pending = set(SIDE_IDS)
    while pending:
        progressed = False
        for m in sorted(pending):
            rel = [o for o in obs if m in o["corners"] and any(r in o["corners"] for r in resolved)]
            if not rel:
                continue
            best = None
            for roll in (0.0, 90.0, 180.0, 270.0):
                cfg.face_roll_deg[m] = roll
                model = AprilTagCubeTarget(cfg)
                vals = []
                for o in rel:
                    ids = [m] + [r for r in resolved if r in o["corners"]]
                    v, _ = pnp_rmse(model, o["corners"], ids, K[o["cam"]], D[o["cam"]])
                    if v is not None:
                        vals.append(v)
                med = float(np.median(vals)) if vals else float("inf")
                print(f"  marker {m} roll {roll:5.1f}: median PnP RMSE {med:7.2f} px  (n={len(vals)})")
                if best is None or med < best[1]:
                    best = (roll, med)
            cfg.face_roll_deg[m] = best[0]
            chosen[m] = best
            resolved.add(m)
            pending.discard(m)
            progressed = True
            print(f"  -> marker {m}: roll {best[0]:.0f} deg (median {best[1]:.2f} px)")
        if not progressed:
            print(f"  [warn] 확정 마커와 같이 찍힌 사진이 없음: {sorted(pending)} (nominal 유지)")
            break
    return chosen


def bundle_adjust(cfg, obs, K, D, gauge_ids=TOP_IDS, f_scale=2.0, max_init_rmse=15.0, fit_top_size=False):
    """gauge_ids 마커는 nominal 고정, 나머지 마커 6-DoF + 사진별 큐브 포즈 최적화.
    fit_top_size: 상단 마커 0,1 의 인쇄 크기 배율(공통 1개)도 추정."""
    model = AprilTagCubeTarget(cfg)
    all_ids = TOP_IDS + SIDE_IDS
    free_ids = tuple(m for m in all_ids if m not in gauge_ids)
    multi, dropped = [], []
    for o in obs:
        if len(o["corners"]) < 2:
            continue
        v, pose = pnp_rmse(model, o["corners"], list(o["corners"]), K[o["cam"]], D[o["cam"]])
        if pose is None:
            continue
        if v > max_init_rmse:
            dropped.append((str(o["path"].relative_to(ZEUS_DATA_ROOT)), v, sorted(o["corners"])))
            continue
        multi.append({**o, "rv": pose[0].reshape(3), "tv": pose[1].reshape(3), "rmse0": v})
    n_img = len(multi)
    T0 = {m: model.model.marker_pose_in_rig(m) for m in all_ids}
    free_init = np.concatenate([np.r_[Rotation.from_matrix(T0[m][:3, :3]).as_rotvec(), T0[m][:3, 3]] for m in free_ids])
    x0 = np.concatenate([np.concatenate([np.r_[o["rv"], o["tv"]] for o in multi]), free_init, [1.0] if fit_top_size else []])
    local = {m: model.model.local_corners_for(m) for m in all_ids}
    off_free = 6 * n_img
    off_size = off_free + 6 * len(free_ids)

    entries = []  # (img_idx, marker_id, corners 4x2)
    for i, o in enumerate(multi):
        for m, c in o["corners"].items():
            entries.append((i, m, c))
    n_res = 8 * len(entries)

    def marker_T(x, m):
        if m not in free_ids:
            return T0[m]
        k = free_ids.index(m)
        p = x[off_free + 6 * k: off_free + 6 * k + 6]
        T = np.eye(4); T[:3, :3] = Rotation.from_rotvec(p[:3]).as_matrix(); T[:3, 3] = p[3:]
        return T

    def residuals(x):
        res = np.empty(n_res)
        Tm = {m: marker_T(x, m) for m in all_ids}
        s_top = x[off_size] if fit_top_size else 1.0
        for e, (i, m, c) in enumerate(entries):
            o = multi[i]
            lc = local[m] * (s_top if m in TOP_IDS else 1.0)
            pts = (Tm[m][:3, :3] @ lc.T).T + Tm[m][:3, 3]
            proj, _ = cv2.projectPoints(pts, x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6], K[o["cam"]], D[o["cam"]])
            res[8 * e:8 * e + 8] = (proj.reshape(4, 2) - c).ravel()
        return res

    S = lil_matrix((n_res, x0.size), dtype=int)
    for e, (i, m, _) in enumerate(entries):
        S[8 * e:8 * e + 8, 6 * i:6 * i + 6] = 1
        if m in free_ids:
            k = free_ids.index(m)
            S[8 * e:8 * e + 8, off_free + 6 * k: off_free + 6 * k + 6] = 1
        if fit_top_size and m in TOP_IDS:
            S[8 * e:8 * e + 8, off_size] = 1

    r0 = residuals(x0)
    sol = least_squares(residuals, x0, jac_sparsity=S, loss="soft_l1", f_scale=f_scale, x_scale="jac", max_nfev=300, verbose=0)
    r1 = residuals(sol.x)

    def rmse(r):
        return float(np.sqrt(np.mean(np.linalg.norm(r.reshape(-1, 2), axis=1) ** 2)))

    per_img = []
    for i, o in enumerate(multi):
        idx = [e for e, (ii, _, _) in enumerate(entries) if ii == i]
        rr = np.concatenate([r1[8 * e:8 * e + 8] for e in idx])
        per_img.append((str(o["path"].relative_to(ZEUS_DATA_ROOT)), o["rmse0"], rmse(rr), sorted(o["corners"])))
    poses = {m: marker_T(sol.x, m) for m in all_ids}
    return {"n_images": n_img, "n_marker_obs": len(entries), "rmse_before": rmse(r0), "rmse_after": rmse(r1),
            "median_after": float(np.median([r[2] for r in per_img])), "dropped": dropped,
            "per_image": per_img, "marker_pose": poses, "nominal_pose": T0, "free_ids": free_ids,
            "top_size_scale": float(sol.x[off_size]) if fit_top_size else 1.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="*", default=None, help="사진 루트들 (기본: 0914 세션 전부 + gt_frames_0914)")
    ap.add_argument("--cube-config", default=str(GT_CFG_PATH))
    ap.add_argument("--write", action="store_true", help="cube_config.json 에 marker_pose_4x4 기록")
    ap.add_argument("--skip-roll-search", action="store_true")
    ap.add_argument("--gauge", choices=["top", "side5"], default="top", help="고정(게이지) 마커: top=0,1 / side5=5")
    ap.add_argument("--fit-top-size", action="store_true", help="상단 마커 인쇄 크기 배율도 추정")
    ap.add_argument("--max-init-rmse", type=float, default=15.0, help="초기 PnP RMSE 가 이보다 큰 사진은 제외")
    ap.add_argument("--zeus-intrinsics-dir", default=str(REPO_ROOT / "intrinsics"))
    ap.add_argument("--ur3-intrinsics-dir", default=str(REPO_ROOT / "ur3_calibration" / "intrinsics"))
    ap.add_argument("--device-map", default=str(REPO_ROOT / "intrinsics" / "device_map.json"))
    ap.add_argument("--out", default=str(Path(__file__).with_name("results") / "gt_eval" / "gt_cube_geometry_calibration.json"))
    args = ap.parse_args()

    roots = args.roots or [SESSION1_DIR / "capture_replayed_0914", SESSION2_DIR / "capture_placed_0914",
                           SESSION3_DIR / "capture_replayed_0914", ZEUS_DATA_ROOT / "gt_frames_0914"]
    K, D = load_intrinsics_by_label(Path(args.zeus_intrinsics_dir), Path(args.ur3_intrinsics_dir), Path(args.device_map))
    cfg, _ = load_cube_config_from_json_file(args.cube_config)
    cfg.marker_pose_4x4 = {}  # 항상 nominal(face_roll/marker_center)에서 시작
    target = AprilTagCubeTarget(cfg)

    files = image_files(roots)
    obs = detect_all(files, target, K, D)
    n_multi = sum(1 for o in obs if len(o["corners"]) >= 2)
    print(f"사진 {len(files)}장 중 마커 검출 {len(obs)}장, 2개 이상 {n_multi}장")
    co = {}
    for o in obs:
        for a, b in itertools.combinations(sorted(o["corners"]), 2):
            co[(a, b)] = co.get((a, b), 0) + 1
    print("마커 쌍 동시 관측 수:", {f"{a}-{b}": n for (a, b), n in sorted(co.items())})

    if not args.skip_roll_search:
        print("\n[1단계] 측면 마커 roll 이산 탐색")
        discrete_roll_search(cfg, obs, K, D)
    gauge = TOP_IDS if args.gauge == "top" else (5,)
    print(f"\n[2단계] bundle adjustment (게이지 마커 {gauge} 고정, 나머지 자유{', 상단 크기 배율 추정' if args.fit_top_size else ''})")
    ba = bundle_adjust(cfg, obs, K, D, gauge_ids=gauge, max_init_rmse=args.max_init_rmse, fit_top_size=args.fit_top_size)
    print(f"  다면 사진 {ba['n_images']}장 (초기 RMSE>{args.max_init_rmse}px 제외 {len(ba['dropped'])}장), 마커 관측 {ba['n_marker_obs']}개")
    for name, v, ids in ba["dropped"]:
        print(f"    제외: {name} {v:.1f} px markers {ids}")
    print(f"  코너 재투영 RMSE: {ba['rmse_before']:.2f} px -> {ba['rmse_after']:.2f} px (사진별 중앙값 {ba['median_after']:.2f} px)")
    if args.fit_top_size:
        print(f"  상단 마커 크기 배율: {ba['top_size_scale']:.4f} (= {25.0 * ba['top_size_scale']:.2f} mm)")
    print("\n  마커별 nominal 대비 변화:")
    for m in ba["free_ids"]:
        Tn, Tf = ba["nominal_pose"][m], ba["marker_pose"][m]
        dR = Rotation.from_matrix(Tn[:3, :3].T @ Tf[:3, :3]).as_rotvec()
        dt = (Tf[:3, 3] - Tn[:3, 3]) * 1000.0
        print(f"    marker {m}: dt = {np.round(dt, 2)} mm, |dR| = {np.degrees(np.linalg.norm(dR)):.2f} deg, roll = {cfg.face_roll_deg.get(m, 0):.0f}")
    print("\n  사진별 (before -> after):")
    worst = sorted(ba["per_image"], key=lambda r: -r[2])[:8]
    for name, b, a, ids in worst:
        print(f"    {name:60s} {b:6.2f} -> {a:6.2f} px  markers {ids}")

    result = {
        "roots": [str(r) for r in roots], "n_images_detected": len(obs), "n_multi_marker_images": ba["n_images"],
        "face_roll_deg": {str(m): cfg.face_roll_deg[m] for m in TOP_IDS + SIDE_IDS},
        "rmse_before_px": ba["rmse_before"], "rmse_after_px": ba["rmse_after"],
        "marker_pose_4x4": {str(m): ba["marker_pose"][m].tolist() for m in TOP_IDS + SIDE_IDS},
        "gauge": list(gauge), "top_size_scale": ba["top_size_scale"],
        "per_image": [{"image": n, "rmse_before": b, "rmse_after": a, "markers": ids} for n, b, a, ids in ba["per_image"]],
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")

    if args.write:
        data = json.loads(Path(args.cube_config).read_text())
        bak = Path(args.cube_config).with_name(f"cube_config.json.bak_{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy(args.cube_config, bak)
        data["face_roll_deg"] = {str(m): cfg.face_roll_deg[m] for m in TOP_IDS + SIDE_IDS}
        data["marker_pose_4x4"] = {str(m): ba["marker_pose"][m].tolist() for m in TOP_IDS + SIDE_IDS}
        if args.fit_top_size:
            data["marker_size_by_id"]["0"] = round(0.025 * ba["top_size_scale"], 6)
            data["marker_size_by_id"]["1"] = round(0.025 * ba["top_size_scale"], 6)
        data["_comment"] = [c for c in data.get("_comment", []) if "face_roll_deg is COPIED" not in c] + [
            f"marker_pose_4x4 (side markers 2..5) self-calibrated on {datetime.now():%Y-%m-%d} by",
            "zeus_gello_calibration/calibrate_gt_cube_geometry.py from 0914 captures:",
            f"multi-marker corner RMSE {ba['rmse_before']:.2f} -> {ba['rmse_after']:.2f} px. Gauge markers: {list(gauge)}.",
        ]
        Path(args.cube_config).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        print(f"updated {args.cube_config} (backup {bak.name})")


if __name__ == "__main__":
    main()
