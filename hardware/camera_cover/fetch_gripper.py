#!/usr/bin/env python3
"""Robotiq 2F-85 메시를 ROS 패키지에서 받아 조립하고, 나사 구멍을 찾아낸다.

Robotiq 공식 STEP 은 계정이 있어야 받을 수 있어서, CAD 에서 나온 ROS 메시를 쓴다.
  ros-industrial-attic/robotiq : robotiq_2f_85_gripper_visualization/
    meshes/visual/*.dae           (단위 mm)
    urdf/robotiq_arg2f_85_model_macro.xacro   (조인트 원점, 단위 m)

조립 결과는 148.4 x 146.4 mm 로 2F-85 스펙(폭 148mm)과 맞는다.

좌표계 (URDF base_link):
  원점 = 로봇에 붙는 체결면 중심,  +Z = 손가락 쪽
  Y = 손가락이 벌어지는 방향,      X = 몸통 납작한 옆면 (+-37.5)

베이스에서 찾은 구멍 (이 좌표계 기준):
  M4 x4    (+-27, +-6)   z 29~36.2   어깨 윗면에 열리는 블라인드 홀, 깊이 7mm
                                      (탭 드릴 Ø3.3 로 모델링돼 있다)
  M5 x4    (+-16, +-27)  z 0.2~17.8  커플링 체결. 위에서 Ø10 카운터보어

사용:
  python fetch_gripper.py            # 받아서 조립 -> gripper_2f85.stl, 구멍 목록 출력
  python fetch_gripper.py --holes    # 이미 받은 것으로 구멍만 다시 훑기
"""

import os
import subprocess
import sys

import numpy as np
import trimesh
from trimesh.intersections import mesh_plane

# 베이스는 MuJoCo Menagerie 쪽이 훨씬 자세하다(46k면 vs 34k면). ros-industrial
# 메시에는 몸통 어깨의 M4 액세서리 구멍 4개가 아예 모델링돼 있지 않았다.
MENAGERIE = "google-deepmind/mujoco_menagerie"
MENAGERIE_BASE = "robotiq_2f85_v4/assets/base.stl"   # 단위 m, 축은 X<->Y 바뀜

REPO = "ros-industrial-attic/robotiq"
BASE = "robotiq_2f_85_gripper_visualization"
PARTS = {
    "base": f"{BASE}/meshes/visual/robotiq_arg2f_85_base_link.dae",
    "outer_knuckle": f"{BASE}/meshes/visual/robotiq_arg2f_85_outer_knuckle.dae",
    "outer_finger": f"{BASE}/meshes/visual/robotiq_arg2f_85_outer_finger.dae",
    "inner_knuckle": f"{BASE}/meshes/visual/robotiq_arg2f_85_inner_knuckle.dae",
    "inner_finger": f"{BASE}/meshes/visual/robotiq_arg2f_85_inner_finger.dae",
}

# xacro 에서 읽은 조인트 원점 (m -> mm). left reflect=+1, right reflect=-1
J_OUTER_KNUCKLE = (0.0, 30.6011, 54.904)    # y 부호는 좌우로 뒤집는다
J_OUTER_FINGER = (0.0, 31.5, -4.1)
J_INNER_FINGER = (0.0, 6.1, 47.1)
J_INNER_KNUCKLE = (0.0, 12.7, 61.42)

HERE = os.path.dirname(os.path.abspath(__file__))


def fetch(cache="."):
    """gh CLI 로 raw URL 을 얻어 내려받는다."""
    paths = {}
    for name, rel in PARTS.items():
        out = os.path.join(cache, "rq_%s.dae" % name)
        if not os.path.exists(out):
            url = subprocess.check_output(
                ["gh", "api", f"repos/{REPO}/contents/{rel}", "--jq", ".download_url"],
                text=True).strip()
            subprocess.check_call(["curl", "-sSL", "-o", out, url])
        paths[name] = out
    return paths


def load(path):
    s = trimesh.load(path)
    return s.to_geometry() if hasattr(s, "to_geometry") else s


def T(t=(0, 0, 0), rz=0.0):
    M = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
    M[:3, 3] = t
    return M


def assemble(paths):
    """손가락이 열린 자세(URDF 0 자세)로 조립."""
    p = {k: load(v) for k, v in paths.items()}
    parts = [p["base"].copy()]
    for sgn, rz in ((+1, np.pi), (-1, 0.0)):        # left, right
        t_ok = T((0, -J_OUTER_KNUCKLE[1] * sgn, J_OUTER_KNUCKLE[2]), rz)
        t_of = t_ok @ T(J_OUTER_FINGER)
        t_if = t_of @ T(J_INNER_FINGER)
        t_ik = T((0, -J_INNER_KNUCKLE[1] * sgn, J_INNER_KNUCKLE[2]), rz)
        for mesh, tf in ((p["outer_knuckle"], t_ok), (p["outer_finger"], t_of),
                         (p["inner_finger"], t_if), (p["inner_knuckle"], t_ik)):
            q = mesh.copy()
            q.apply_transform(tf)
            parts.append(q)
    return trimesh.util.concatenate(parts)


def _loops(seg, ia, ib):
    pts = seg.reshape(-1, 3)[:, [ia, ib]]
    uniq, inv = np.unique(np.round(pts, 2), axis=0, return_inverse=True)
    adj = {}
    for a, b in inv.reshape(-1, 2):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen, out = set(), []
    for s in adj:
        if s in seen:
            continue
        stack, comp = [s], []
        while stack:
            v = stack.pop()
            if v in seen:
                continue
            seen.add(v)
            comp.append(v)
            stack.extend(adj[v] - seen)
        out.append(uniq[comp])
    return out


def find_holes(mesh, step=1.0):
    """세 축으로 훑으며 원형 단면이 반복되는 구간을 구멍으로 본다."""
    hits = []
    for axis in (0, 1, 2):
        lo, hi = mesh.bounds[0][axis] + 0.5, mesh.bounds[1][axis] - 0.5
        for v in np.arange(lo, hi, step):
            n = np.zeros(3); n[axis] = 1
            o = np.zeros(3); o[axis] = v
            seg = mesh_plane(mesh, plane_normal=n, plane_origin=o)
            if len(seg) == 0:
                continue
            ia, ib = [i for i in range(3) if i != axis]
            for P in _loops(seg, ia, ib):
                w = P[:, 0].max() - P[:, 0].min()
                h = P[:, 1].max() - P[:, 1].min()
                if 2.0 < w < 13 and 2.0 < h < 13 and abs(w - h) < max(1.0, 0.15 * w):
                    hits.append((axis, v, (P[:, 0].max() + P[:, 0].min()) / 2,
                                 (P[:, 1].max() + P[:, 1].min()) / 2, (w + h) / 2))
    groups = {}
    for axis, v, a, b, d in hits:
        groups.setdefault((axis, round(a), round(b), round(d, 1)), []).append(v)
    return {k: (min(v), max(v)) for k, v in groups.items() if len(v) >= 3}


def main():
    holes_only = "--holes" in sys.argv
    stl = os.path.join(HERE, "gripper_2f85.stl")
    if holes_only and os.path.exists(stl):
        g = trimesh.load(stl)
    else:
        g = assemble(fetch(HERE))
        g.export(stl)
        print("조립 -> %s" % stl)
    print("전체 크기 %.1f x %.1f x %.1f mm  (2F-85 스펙 폭 148mm)" % tuple(g.extents))

    base = load(os.path.join(HERE, "rq_base.dae")) if os.path.exists(
        os.path.join(HERE, "rq_base.dae")) else g
    print("\n베이스에서 찾은 구멍 (축 / 중심 / 지름 / 구간, mm)")
    for (axis, a, b, d), (v0, v1) in sorted(find_holes(base).items()):
        ia, ib = [i for i in range(3) if i != axis]
        print("  %s축  %s=%+6.1f %s=%+6.1f   Ø%-5.2f  %s %6.1f ~ %6.1f" % (
            "XYZ"[axis], "XYZ"[ia], a, "XYZ"[ib], b, d, "XYZ"[axis], v0, v1))


if __name__ == "__main__":
    main()
