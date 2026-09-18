#!/usr/bin/env python3
"""A3 캘리브레이션(T_base_Ci)으로 고정캠 3대 depth를 base 좌표로 합쳐 정합 확인.
depth PNG는 color 해상도(1920x1080)로 정렬 저장돼 있음 -> color K로 역투영, 단위 mm."""
import sys, json, cv2, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
from scipy.spatial import cKDTree
REPO = Path("/home/sstone/rb-calibration-marker-experiment")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "zeus_gello_calibration"))
from robot.backends.zeus_client import pose6_to_T
from fit_grasp_offset import LOCAL_CAM_IDS, load_intrinsics_by_label
from session2_pick_and_place import compute_ordered_targets
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
SET = int(sys.argv[2]) if len(sys.argv) > 2 else 3
FIXED = ("039422061216", "fixed2", "fixed3")
intr = REPO / "intrinsics_1920x1080_rgbd720"
K, D = load_intrinsics_by_label(intr, REPO / "ur3_calibration/intrinsics", intr / "device_map.json")
m = json.loads((REPO / "ABLATION_TEST_result_0917/zeus_0917_1920x1080_newintr/table1_zeus/ABLATION_TEST_table1_methods.json").read_text())
T_bC = {int(k): np.array(v, float) for k, v in m["rows"]["A5"]["all"]["transforms"]["T_base_Ci"].items()}
T_fc = np.array(json.loads((REPO / "zeus_gello_calibration/fit_통합_0917_1920x1080.json").read_text())["T_gripper_cube"], float)
S2 = REPO / "zeus_gello_calibration/data/session2_floor_board_dual_cam_0909"
items = compute_ordered_targets(S2)
anchor = (pose6_to_T(items[SET]["target"]) @ T_fc)[:3, 3] * 1000  # 큐브 원점(본체 중심), base mm
top_z = anchor[2] + 64.5  # I자 플레이트 윗면
print(f"set {SET}: 큐브 중심 FK 앵커 {np.round(anchor,1)} mm, 윗면 z={top_z:.1f}")

clouds = {}
for lab in FIXED:
    cid = LOCAL_CAM_IDS[lab]
    d = cv2.imread(str(S2 / "capture_placed_0917_1920x1080" / f"{SET:03d}" / f"cam_{lab}_depth.png"), cv2.IMREAD_UNCHANGED).astype(np.float64)
    h, w = d.shape; v, u = np.mgrid[0:h, 0:w]
    ok = d > 0
    pix = np.stack([u[ok], v[ok]], axis=1).astype(np.float64)
    z = d[ok]  # mm
    norm = cv2.undistortPoints(pix.reshape(-1, 1, 2), K[cid], D[cid]).reshape(-1, 2)
    P_cam = np.column_stack([norm[:, 0] * z, norm[:, 1] * z, z]) / 1000.0
    P_base = (T_bC[cid][:3, :3] @ P_cam.T).T + T_bC[cid][:3, 3]
    P = P_base * 1000.0
    box = (np.abs(P[:, 0] - anchor[0]) < 120) & (np.abs(P[:, 1] - anchor[1]) < 120) & (P[:, 2] > anchor[2] - 45) & (P[:, 2] < anchor[2] + 90)
    clouds[lab] = P[box]
    print(f"  {lab}: 전체 {ok.sum()} pts, 큐브 주변 {box.sum()} pts")

# 윗면 평면 높이 비교 (마커 I자 플레이트 윗면, xy는 앵커 ±40mm)
print("\n윗면(z≈%.1f) 점들의 z 통계 (카메라별):" % top_z)
for lab, P in clouds.items():
    sel = (np.abs(P[:, 0] - anchor[0]) < 40) & (np.abs(P[:, 1] - anchor[1]) < 40) & (np.abs(P[:, 2] - top_z) < 12)
    if sel.sum() < 20: print(f"  {lab}: 점 부족"); continue
    zz = P[sel, 2]
    print(f"  {lab}: n={sel.sum():5d}  z 중앙값 {np.median(zz):7.1f} (앵커 대비 {np.median(zz)-top_z:+.1f}mm)  std {zz.std():.1f}mm")
# 카메라 쌍 최근접 거리 (윗면 영역)
print("\n카메라 쌍 최근접점 거리 (윗면 영역, mm):")
labs = list(clouds)
def top(P):
    s = (np.abs(P[:, 0] - anchor[0]) < 40) & (np.abs(P[:, 1] - anchor[1]) < 40) & (np.abs(P[:, 2] - top_z) < 12); return P[s]
for i in range(3):
    for j in range(i + 1, 3):
        A, B = top(clouds[labs[i]]), top(clouds[labs[j]])
        if len(A) < 20 or len(B) < 20: continue
        dnn, _ = cKDTree(B).query(A); print(f"  {labs[i][:6]}↔{labs[j][:6]}: 중앙값 {np.median(dnn):.2f}  P90 {np.percentile(dnn,90):.2f}")

# 그림: 위에서 본 것 + 옆에서 본 것
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
cols = {"039422061216": "tab:red", "fixed2": "tab:green", "fixed3": "tab:blue"}
for lab, P in clouds.items():
    idx = np.random.default_rng(0).choice(len(P), min(len(P), 15000), replace=False); Q = P[idx]
    axes[0].scatter(Q[:, 0], Q[:, 1], s=1, c=cols[lab], alpha=0.4, label=lab)
    axes[1].scatter(Q[:, 0], Q[:, 2], s=1, c=cols[lab], alpha=0.4, label=lab)
    axes[2].scatter(Q[:, 1], Q[:, 2], s=1, c=cols[lab], alpha=0.4, label=lab)
for ax, (a, b), t in zip(axes, ((0, 1), (0, 2), (1, 2)), ("위에서 (x-y)", "옆에서 (x-z)", "옆에서 (y-z)")):
    ax.set_title(t); ax.set_aspect("equal"); ax.grid(alpha=0.3); ax.legend(markerscale=8)
    ax.set_xlabel("xyz"[a] + " mm"); ax.set_ylabel("xyz"[b] + " mm")
    if b == 2: ax.axhline(top_z, c="k", ls="--", lw=0.8); ax.axhline(anchor[2] + 29.5, c="gray", ls=":", lw=0.8)
# 큐브 본체 59mm 윤곽(위에서)
c = anchor; axes[0].add_patch(plt.Rectangle((c[0] - 29.5, c[1] - 29.5), 59, 59, fill=False, ec="k", lw=1.2, ls="--"))
fig.suptitle(f"set {SET}: 고정캠 3대 depth를 A3 캘리브레이션으로 base 좌표에 합침 (점선 = FK 앵커 기준 큐브 59mm / 윗면 z)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / f"pointcloud_set{SET:02d}.png", dpi=100)
# PLY
allP = np.concatenate([np.column_stack([P, np.tile(np.array(matplotlib.colors.to_rgb(cols[l])) * 255, (len(P), 1))]) for l, P in clouds.items()])
with open(OUT / f"pointcloud_set{SET:02d}.ply", "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(allP)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    np.savetxt(f, allP, fmt="%.2f %.2f %.2f %d %d %d")
print("wrote", OUT / f"pointcloud_set{SET:02d}.png", OUT / f"pointcloud_set{SET:02d}.ply")
