#!/usr/bin/env python3
"""RealSense D435 슬리브 커버 + Robotiq 2F-85 플랜지 암 STL 생성기.

캐비티 단면은 추정값이 아니라 Intel 공식 메시에서 뽑은 실제 실루엣이다
(extract_profile.py 가 만든 d435_profile.json). D435 바디는 양 끝이 반원인
스타디움 단면이고, 깊이 방향으로는 앞면에서 약 17mm 지점이 가장 넓다.
따라서 앞에서 밀어 넣는 한 덩어리 커버의 캐비티는 그 실루엣을 그대로 밀어낸
기둥이어야 하고, 카메라는 가장 넓은 벨트에서 커버에 밀착된다.

세 개의 STL 을 만든다.
  d435_gauge.stl   : 20분짜리 시험 출력. 끼움과 M3 구멍 위치만 확인한다.
  d435_sleeve.stl  : 카메라를 감싸는 커버 본체.
  d435_arm.stl     : 툴 플랜지(ISO 9409-1-50-4-M6)와 그리퍼 커플러 사이에 끼우는 암.

좌표계(설계 기준, 출력 방향은 export 시 회전):
  X = 카메라 긴 축, Y = 깊이(앞면 0 -> 뒤 +), Z = 높이(캐비티 바닥 0)

사용: python d435_gripper_cover.py [출력디렉터리]
"""

import json
import os
import sys

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

ENGINE = "manifold"
HERE = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ 끼움 공차
# "오차 0" 은 FDM 으로 불가능하다. 실루엣을 이만큼 바깥으로 오프셋해서 캐비티를
# 만든다. 프린터가 타이트하면 0.20, 헐거우면 0.40 으로.
CLR = 0.30

# ---------------------------------------------------------------- 커버 두께류
WALL = 2.4            # 둘레 벽
REAR_WALL = 3.0       # 뒷판 (M3 나사가 물리는 곳)
PAD_EXTRA = 2.5       # 암이 붙는 윗면에 덧붙이는 두께
REAR_HALF_X = 30.0    # 뒷판이 덮는 반폭. 바깥은 열려 있어 USB-C 케이블이 나온다
GAUGE_X = (-48.0, -12.0)  # 게이지가 잘라 낼 구간. 둥근 끝 + M3 구멍 하나가 들어간다

# 암 체결 4공 패턴
PAD_X = 20.0
PAD_Y = (7.0, 21.0)   # 앞면에서의 깊이
M3_CLR = 3.4
M3_CSK = 6.4

# --------------------------------------------------------------------- 암 제원
FLANGE_D = 50.0
FLANGE_T = 6.0
FLANGE_BC = 31.5
FLANGE_M6 = 6.6
FLANGE_DOWEL = 6.2
FLANGE_BORE = 22.0
ARM_R = 42.0
ARM_Z = 60.0
ARM_T = 8.0
ARM_W = 30.0


# ---------------------------------------------------------------- 형상 데이터
def load_profile():
    with open(os.path.join(HERE, "d435_profile.json")) as f:
        p = json.load(f)
    p["silhouette_xy"] = np.array(p["silhouette_xy"])
    return p


PROF = load_profile()
SIL = PROF["silhouette_xy"]                 # (x, 높이) 실루엣, 중심 원점
CAM_D = PROF["body"]["depth_z"]             # 25.05
CAM_H = PROF["body"]["height_y"]            # 25.00
M3_XY = PROF["m3_holes_xy"]                 # [[-22.5, 0], [22.5, 0]]
TRIPOD = PROF["tripod_xy_depth"]            # [0, -12.5, 14.9]

CY = CAM_D + 2 * CLR          # 캐비티 깊이
CZ = CAM_H + 2 * CLR          # 캐비티 높이
ZC = CZ / 2                   # 캐비티 높이 중심 (= 카메라 중심)
OY = CY + REAR_WALL           # 외형 깊이


def _hull2d(pts):
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

    return np.array(half(p)[:-1] + half(list(p[::-1]))[:-1])


def offset_poly(poly, d, n=24):
    """볼록 다각형을 바깥으로 d 만큼 오프셋 (원과의 민코프스키 합).

    각 점에 반경 d 인 원을 놓고 전체의 볼록껍질을 취하면 정확한 오프셋이 된다.
    D435 실루엣은 볼록하므로 이 방법이 그대로 성립한다.
    """
    if d <= 0:
        return poly
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    disc = np.c_[d * np.cos(a), d * np.sin(a)]
    return _hull2d((poly[:, None, :] + disc[None, :, :]).reshape(-1, 2))


def prism(poly, y0, y1, zc=0.0):
    """(x, z) 다각형을 Y 방향으로 밀어낸 기둥. 볼록 가정."""
    n = len(poly)
    verts = [(x, y0, z + zc) for x, z in poly] + [(x, y1, z + zc) for x, z in poly]
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i]]
    c0, c1 = 2 * n, 2 * n + 1
    verts += [(0.0, y0, zc), (0.0, y1, zc)]
    for i in range(n):
        j = (i + 1) % n
        faces += [[c0, j, i], [c1, n + i, n + j]]
    m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m


# ------------------------------------------------------------------ 기본 도형
def bx(size, center):
    T = np.eye(4)
    T[:3, 3] = center
    return box(extents=size, transform=T)


def span(x0, x1, y0, y1, z0, z1):
    return bx((x1 - x0, y1 - y0, z1 - z0),
              ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def _axis_T(axis, center):
    T = np.eye(4)
    if axis == "x":
        T[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]
    elif axis == "y":
        T[:3, :3] = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]
    T[:3, 3] = center
    return T


def cyl(d, length, center, axis="z", sections=64):
    return cylinder(radius=d / 2, height=length, sections=sections,
                    transform=_axis_T(axis, center))


def csk(d_big, d_small, length, center):
    """+Z 쪽이 넓은 카운터싱크 원뿔대."""
    n = 48
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    bot = np.c_[d_small / 2 * np.cos(ang), d_small / 2 * np.sin(ang),
                np.full(n, -length / 2)]
    top = np.c_[d_big / 2 * np.cos(ang), d_big / 2 * np.sin(ang),
                np.full(n, length / 2)]
    verts = np.vstack([bot, top, [[0, 0, -length / 2]], [[0, 0, length / 2]]])
    cb, ct = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    m = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    m.apply_translation(center)
    return m


# -------------------------------------------------------------------- 커버 본체
def sleeve_body(clip_x=None, with_pad=True, with_vents=True):
    cav = offset_poly(SIL, CLR)                 # 캐비티 = 실제 실루엣 + 공차
    out = offset_poly(SIL, CLR + WALL)          # 바깥 = 거기에 벽 두께
    ox = out[:, 0].max()
    top_out = out[:, 1].max() + ZC              # 바깥 윗면 높이
    pad_top = top_out + PAD_EXTRA

    solid = prism(out, 0.0, OY, zc=ZC)
    if with_pad:                                # 암이 붙을 평평한 자리
        solid = solid.union(span(-26, 26, 0.0, OY, top_out - 1.0, pad_top),
                            engine=ENGINE)

    cuts = [prism(cav, -1.0, CY, zc=ZC)]        # 앞으로 열린 캐비티

    # 뒷판은 |x| <= REAR_HALF_X 만 남긴다 (USB-C 커넥터와 케이블 통로)
    for sx in (1, -1):
        x0, x1 = (REAR_HALF_X, ox + 1) if sx > 0 else (-ox - 1, -REAR_HALF_X)
        cuts.append(span(x0, x1, CY - 0.01, OY + 1, -1.0, pad_top + 1))

    # 뒷판 M3 관통 2개 — 메시에서 직접 읽은 위치
    for hx, hy in M3_XY:
        cuts.append(cyl(M3_CLR, 20.0, (hx, CY + REAR_WALL / 2, ZC + hy), axis="y"))

    if with_vents:
        # 삼각대 구멍 접근 (바닥, 앞면에서 14.9mm) + 방열 슬롯
        cuts.append(cyl(11.0, 20.0, (TRIPOD[0], CLR + TRIPOD[2], -1.0), axis="z"))
        for sx in (-1, 1):
            for x0 in (20.0, 32.0):
                cuts.append(bx((5.0, 16.0, 20.0), (sx * x0, CLR + TRIPOD[2], -1.0)))

    if with_pad:
        # 암 체결 4공. 접시머리 M3 를 캐비티 안쪽에서 위로 꽂아 머리를 묻는다
        for sx in (-1, 1):
            for y in PAD_Y:
                cuts.append(cyl(M3_CLR, 20.0, (sx * PAD_X, y, pad_top), axis="z"))
                cuts.append(csk(M3_CSK, M3_CLR, 1.6, (sx * PAD_X, y, CZ - 0.8)))

    if clip_x is not None:                      # 게이지: 같은 단면으로 구간만 남김
        cuts.append(span(clip_x[1], ox + 1, -2, OY + 1, -5, pad_top + 1))
        cuts.append(span(-ox - 1, clip_x[0], -2, OY + 1, -5, pad_top + 1))

    for c in cuts:
        solid = solid.difference(c, engine=ENGINE)
    return solid


# ------------------------------------------------------------------------- 암
def arm():
    base = cyl(FLANGE_D, FLANGE_T, (0, 0, FLANGE_T / 2))
    radial = span(0, ARM_R, -ARM_W / 2, ARM_W / 2, FLANGE_T, FLANGE_T + ARM_T)
    axial = span(ARM_R - ARM_T, ARM_R, -ARM_W / 2, ARM_W / 2, FLANGE_T, FLANGE_T + ARM_Z)
    pad = span(ARM_R - 5.0, ARM_R, -30.0, 30.0,
               FLANGE_T + ARM_Z - 26.0, FLANGE_T + ARM_Z + 6.0)

    solid = base.union(radial, engine=ENGINE)
    solid = solid.union(axial, engine=ENGINE)
    solid = solid.union(pad, engine=ENGINE)

    cuts = [cyl(FLANGE_BORE, 40.0, (0, 0, FLANGE_T / 2))]
    for k in range(4):
        a = np.pi / 4 + k * np.pi / 2
        cuts.append(cyl(FLANGE_M6, 40.0, (FLANGE_BC / 2 * np.cos(a),
                                          FLANGE_BC / 2 * np.sin(a), FLANGE_T / 2)))
    cuts.append(cyl(FLANGE_DOWEL, 40.0, (0, FLANGE_BC / 2, FLANGE_T / 2)))

    zc = FLANGE_T + ARM_Z - 10.0
    for sy in (-1, 1):
        for dy in PAD_Y:
            y = sy * PAD_X
            z = zc + (dy - np.mean(PAD_Y))
            cuts.append(cyl(M3_CLR, 30.0, (ARM_R - 2.5, y, z), axis="x"))
            cuts.append(cyl(6.35, 2.8, (ARM_R - 3.6, y, z), axis="x", sections=6))

    for c in cuts:
        solid = solid.difference(c, engine=ENGINE)
    return solid


def to_print_orientation(mesh):
    """뒷판이 베드에 닿도록 회전. 앞 개구부가 위를 본다."""
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out, exist_ok=True)
    print(f"실루엣 출처: {PROF['source']}")
    print(f"바디 {PROF['body']['length_x']:.2f} x {CAM_H:.2f} x {CAM_D:.2f} mm\n")

    parts = {
        "d435_gauge.stl": to_print_orientation(
            sleeve_body(clip_x=GAUGE_X, with_pad=False, with_vents=False)),
        "d435_sleeve.stl": to_print_orientation(sleeve_body()),
        "d435_arm.stl": arm(),
    }
    for name, mesh in parts.items():
        mesh.export(os.path.join(out, name))
        e = mesh.extents
        print(f"  {name:16s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm   "
              f"watertight={mesh.is_watertight}  vol={mesh.volume / 1000:6.1f} cm^3")
    print(f"\n{out} 에 저장했습니다.")


if __name__ == "__main__":
    main()
