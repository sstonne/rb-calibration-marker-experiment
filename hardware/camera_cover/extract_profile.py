#!/usr/bin/env python3
"""Intel 공식 D435 메시에서 커버 설계에 필요한 형상을 뽑아 d435_profile.json 으로 저장.

메시 출처: realsenseai/realsense-ros, realsense2_description/meshes/d435.dae
(CAD 에서 나온 것이라 나사 구멍까지 들어 있다. 단위는 m 이라 1000 배 한다.)

  python extract_profile.py d435.dae
"""
import json
import sys

import numpy as np
import trimesh


def hull2d(pts):
    """Andrew monotone chain. 반시계 방향 볼록껍질."""
    p = np.unique(np.round(pts, 4), axis=0)
    p = p[np.lexsort((p[:, 1], p[:, 0]))]

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def half(ps):
        out = []
        for q in ps:
            while len(out) >= 2 and turn(out[-2], out[-1], q) <= 0:
                out.pop()
            out.append(q)
        return out

    return np.array(half(p)[:-1] + half(p[::-1])[:-1])


def simplify(poly, n):
    """둘레를 따라 n 점으로 균등 재샘플."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    return np.c_[np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "d435.dae"
    m = trimesh.load(src).to_geometry()
    m.apply_scale(1000.0)                      # m -> mm

    # 메시 좌표계: X=긴축, Y=높이, Z=깊이(앞면 z=0, 뒤 z=-D)
    depth = float(-m.bounds[0][2])

    # 깊이 방향으로 본 실루엣. 바디가 볼록해서 볼록껍질이 곧 실루엣이다.
    sil = simplify(hull2d(m.vertices[:, :2]), 240)

    out = {
        "source": "realsenseai/realsense-ros realsense2_description/meshes/d435.dae",
        "units": "mm",
        "body": {
            "length_x": float(m.extents[0]),
            "height_y": float(m.extents[1]),
            "depth_z": depth,
        },
        # 깊이 방향 실루엣 (XY 평면, 원점 = 바디 중심). 캐비티의 기준 단면.
        "silhouette_xy": [[round(a, 3), round(b, 3)] for a, b in sil],
        # 메시에서 직접 읽은 체결부
        "m3_holes_xy": [[-22.5, 0.0], [22.5, 0.0]],   # 뒷면, Ø2.5 탭, 깊이 2.5
        "tripod_xy_depth": [0.0, -12.5, 14.9],        # 바닥면 1/4-20 Ø6.35
    }
    json.dump(out, open("d435_profile.json", "w"), indent=1)
    print("length x height x depth =",
          f"{out['body']['length_x']:.2f} x {out['body']['height_y']:.2f} x {depth:.2f} mm")
    print("실루엣 점 개수:", len(sil))
    print("d435_profile.json 저장")


if __name__ == "__main__":
    main()
