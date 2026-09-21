#!/usr/bin/env python3
"""RealSense D435 클램셸 커버 — 위/아래 두 쪽으로 카메라 표면 전체를 감싼다.

한 덩어리 슬리브(d435_gripper_cover.py)는 앞에서 밀어 넣어야 하므로 캐비티가
실루엣을 밀어낸 기둥일 수밖에 없고, 카메라는 가장 넓은 벨트에서만 닿는다.
클램셸은 그 제약이 없다. 깊이별 실제 단면(d435_profile.json 의 sections_xy)을
각각 CLR 만큼 오프셋해서 쌓기 때문에, 캐비티가 카메라 표면을 전 구간 따라간다.

쪼개는 평면은 z = 0 (카메라 높이 중앙). 33 개 단면 전부 가장 넓은 지점이 z=0
±1mm 안이라, 이 면에서 자르면 어느 쪽도 언더컷이 없어 수직으로 그냥 열린다.

  d435_shell_top.stl     : 윗쪽. 암 체결 4공이 여기 있다.
  d435_shell_bottom.stl  : 아랫쪽. 삼각대 구멍과 방열 슬롯이 여기 있다.

두 쪽은 양 끝 귀(tab)의 M3 4개로 조인다. 카메라를 잡는 뒷면 M3 2개는 쪼개는
평면 위에 정확히 놓이므로, 그 나사도 두 쪽을 함께 눌러 준다.

사용: python d435_clamshell.py [출력디렉터리]
"""

import os
import sys

import numpy as np
import trimesh

from d435_gripper_cover import (CAM_D, CAM_H, ENGINE, HERE, M3_CLR, M3_CSK,
                                M3_XY, PAD_EXTRA, PAD_X, PAD_Y, PROF,
                                REAR_HALF_X, REAR_WALL, TRIPOD, TRIPOD_D, WALL,
                                bx, csk, cyl, offset_poly, span)

# 클램셸은 슬리브와 공차를 따로 쓴다.
#
# 슬리브는 카메라를 앞에서 90mm 밀어 넣어야 해서 공차가 없으면 조립 자체가
# 안 된다. 클램셸은 위아래로 덮는 방식이라 그 제약이 없고, 오히려 0 으로 두면
# 두 쪽이 카메라를 가볍게 물어 준다. 실물 출력에서 0.30 은 과했다.
CLR = 0.0

CY = CAM_D + 2 * CLR          # 캐비티 깊이
CZ = CAM_H + 2 * CLR          # 캐비티 높이
ZC = CZ / 2                   # 쪼개는 평면 = 카메라 높이 중앙
OY = CY + REAR_WALL           # 외형 깊이

RING_N = 512          # 단면 하나를 몇 점으로 다시 샘플할지 (r_grid 와 같다)

# 두 쪽을 조이는 귀
TAB_X = (46.0, 55.5)  # 안쪽 -> 바깥쪽
TAB_Y = (0.5, 18.5)   # 렌즈 쪽으로 3.5mm 당김. 뒤쪽 USB-C 케이블 공간 확보
TAB_HALF_Z = 3.5      # 쪼갠 뒤 한 쪽당 3.5mm
TAB_SCREW_X = 50.7
TAB_SCREW_Y = (4.5, 14.5)
PIN_D = 3.0           # 정렬 다월 (3mm 봉을 따로 꽂는다)
PIN_DEPTH = 2.6       # 한 쪽당 구멍 깊이
PIN_Y = 9.5           # 나사 두 개 사이


def resample_angular(poly, n=RING_N):
    """원점에서 각도 n 방향으로 광선을 쏴 볼록 다각형의 변과 만나는 점을 구한다.

    꼭짓점의 반경을 각도로 보간하면 직선 구간에서 값이 부풀어 오른다
    (평평한 윗면의 양 끝점만 남으므로 그 사이가 직선이 아니라 부채꼴이 된다).
    변의 직선식과 직접 교차시켜야 정확하다.
    """
    t = np.linspace(-np.pi, np.pi, n, endpoint=False)
    dirs = np.c_[np.cos(t), np.sin(t)]
    a = poly
    b = np.roll(poly, -1, axis=0)
    e = b - a
    nrm = np.c_[e[:, 1], -e[:, 0]]                      # 바깥 방향 법선
    L = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = nrm / np.where(L < 1e-12, 1.0, L)
    c = np.einsum("ij,ij->i", nrm, a)                   # 변까지의 수직거리
    den = dirs @ nrm.T                                  # (n, 변)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(den > 1e-9, c[None, :] / den, np.inf)
    rr = r.min(axis=1)
    return np.c_[rr * np.cos(t), rr * np.sin(t)]


def loft(rings, ys, zc=0.0):
    """같은 점 개수의 단면들을 Y 방향으로 쌓아 닫힌 솔리드로."""
    n = len(rings[0])
    verts, faces = [], []
    for ring, y in zip(rings, ys):
        verts.extend([(p[0], y, p[1] + zc) for p in ring])
    for k in range(len(rings) - 1):
        a, b = k * n, (k + 1) * n
        for i in range(n):
            j = (i + 1) % n
            faces += [[a + i, a + j, b + j], [a + i, b + j, b + i]]
    for k, first in ((0, True), (len(rings) - 1, False)):
        c = len(verts)
        verts.append((0.0, ys[k], zc))
        base = k * n
        for i in range(n):
            j = (i + 1) % n
            faces.append([c, base + j, base + i] if first else [c, base + i, base + j])
    m = trimesh.Trimesh(vertices=np.array(verts), faces=np.array(faces), process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(m)
    if m.volume < 0:
        m.invert()
    return m


def _stacks():
    """깊이별 단면을 안쪽(캐비티)/바깥쪽(외형) 두 벌로 만들어 돌려준다.

    링 사이를 직선으로 이으면 배부른 구간이 안쪽으로 깎여 카메라를 파고든다.
    그래서 각 링의 반경을 이웃 구간의 최대값으로 올린다. 양 끝이 모두 구간
    최대 이상이면 그 사이를 이은 직선도 구간 최대 이상이므로, 카메라가
    캐비티 밖으로 나오지 않는 것이 보장된다.
    """
    npz = np.load(os.path.join(HERE, PROF["sections_file"]))
    depths = npz["depths"].astype(float)
    raw = npz["r_grid"].astype(float)                    # (깊이, 각도)
    n_ang = raw.shape[1]
    ang = np.linspace(-np.pi, np.pi, n_ang, endpoint=False)

    # 각 깊이에서 실제 단면을 CLR 만큼 제대로(법선 방향) 오프셋
    r_in = np.empty_like(raw)
    for k, row in enumerate(raw):
        poly = np.c_[row * np.cos(ang), row * np.sin(ang)]
        r_in[k] = np.hypot(*resample_angular(offset_poly(poly, CLR), n_ang).T)

    # 구간 최대값으로 올려 포함을 보장
    r_in = np.maximum.reduce([r_in,
                              np.roll(r_in, 1, axis=0),
                              np.roll(r_in, -1, axis=0)])

    inner, outer = [], []
    for row in r_in:
        poly = np.c_[row * np.cos(ang), row * np.sin(ang)]
        inner.append(poly)
        outer.append(resample_angular(offset_poly(poly, WALL), n_ang))
    ys = CLR + depths                      # 카메라 깊이 -> 설계 Y
    return ys, inner, outer


def shell():
    """쪼개기 전의 껍데기 한 덩어리."""
    ys, inner, outer = _stacks()

    # 캐비티: 앞으로 -1 까지 곧게 빼 열어 두고, 뒤는 CY 까지
    in_rings = [inner[0]] + inner + [inner[-1]]
    in_ys = [-1.0] + list(ys) + [CY]
    cavity = loft(in_rings, in_ys, zc=ZC)

    # 외형: 앞면 0 에서 시작해 뒤판 두께까지
    out_rings = [outer[0]] + outer + [outer[-1], outer[-1]]
    out_ys = [0.0] + list(ys) + [CY, OY]
    solid = loft(out_rings, out_ys, zc=ZC)

    ox = max(r[:, 0].max() for r in outer)

    # 두 쪽을 조일 귀
    for sx in (-1, 1):
        x0, x1 = sorted((sx * TAB_X[0], sx * TAB_X[1]))
        solid = solid.union(span(x0, x1, TAB_Y[0], TAB_Y[1],
                                 ZC - TAB_HALF_Z, ZC + TAB_HALF_Z), engine=ENGINE)

    cuts = [cavity]
    # 뒷판은 가운데만 남겨 USB-C 를 연다
    for x0, x1 in ((REAR_HALF_X, ox + 20), (-ox - 20, -REAR_HALF_X)):
        cuts.append(span(x0, x1, CY - 0.01, OY + 1, -20, ZC + 40))
    # 카메라를 잡는 뒷면 M3 2개 (쪼개는 평면 위에 있다)
    for hx, hy in M3_XY:
        cuts.append(cyl(M3_CLR, 20.0, (hx, CY + REAR_WALL / 2, ZC + hy), axis="y"))
    # 귀의 M3 4개
    for sx in (-1, 1):
        for y in TAB_SCREW_Y:
            cuts.append(cyl(M3_CLR, 40.0, (sx * TAB_SCREW_X, y, ZC), axis="z"))
        # 정렬 다월 구멍. 양쪽에 구멍만 내고 3mm 봉(필라멘트 토막)을 꽂는다.
        # 한쪽에 핀을 세우면 그 쪽이 핀 두 개로만 베드에 서서 출력이 안 된다.
        cuts.append(cyl(PIN_D + 0.25, 2 * PIN_DEPTH,
                        (sx * TAB_SCREW_X, PIN_Y, ZC), axis="z"))

    for c in cuts:
        solid = solid.difference(c, engine=ENGINE)
    return solid


def split(solid):
    """z = ZC 에서 위/아래로 자른다."""
    big = 200.0
    top = solid.intersection(
        bx((big, big, big), (0, 0, ZC + big / 2)), engine=ENGINE)
    bottom = solid.intersection(
        bx((big, big, big), (0, 0, ZC - big / 2)), engine=ENGINE)
    return top, bottom


def finish_top(top):
    """암이 붙는 자리와 정렬 핀 구멍."""
    top_out = top.bounds[1][2]
    pad_top = top_out + PAD_EXTRA
    top = top.union(span(-26, 26, 1.0, OY - 1.0, top_out - 1.0, pad_top), engine=ENGINE)

    cuts = []
    for sx in (-1, 1):
        for y in PAD_Y:                       # 암 체결 4공 (안쪽에서 접시머리)
            cuts.append(cyl(M3_CLR, 30.0, (sx * PAD_X, y, pad_top), axis="z"))
            cuts.append(csk(M3_CSK, M3_CLR, 1.8, (sx * PAD_X, y, CZ + 0.8),
                            wide_down=True))
        pass
    for c in cuts:
        top = top.difference(c, engine=ENGINE)
    return top


def finish_bottom(bottom):
    """삼각대 구멍, 방열 슬롯, 너트 자리, 정렬 핀."""
    cuts = [cyl(TRIPOD_D, 30.0, (TRIPOD[0], CLR + TRIPOD[2], -10.0), axis="z")]
    for sx in (-1, 1):
        for x0 in (20.0, 32.0):
            cuts.append(bx((5.0, 16.0, 30.0), (sx * x0, CLR + TRIPOD[2], -10.0)))
        for y in TAB_SCREW_Y:                 # M3 육각 너트 자리 (아래에서 끼움)
            cuts.append(cyl(6.35, 2.7, (sx * TAB_SCREW_X, y, ZC - TAB_HALF_Z + 1.35),
                            axis="z", sections=6))
    for c in cuts:
        bottom = bottom.difference(c, engine=ENGINE)

    return bottom


def lay_flat(mesh, flip):
    """맞대는 면이 베드에 닿도록. 두 쪽 다 서포트 없이 출력된다."""
    m = mesh.copy()
    m.apply_translation([0, 0, -ZC])
    if flip:
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def clean_export(mesh, path):
    """내보내기 전 정리. 부울 결과에 겹친 면이 남아 STL 이 비다양체가 되곤 한다."""
    m = mesh.copy()
    # STL 은 float32 라, 1e-6 mm 쯤 떨어진 정점들이 다시 읽을 때 합쳐지면서
    # 면이 4개 붙은 모서리가 생긴다. 내보내기 전에 미리 같은 눈금으로 스냅한다.
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.export(path)
    return m


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else HERE
    os.makedirs(out, exist_ok=True)
    print("클램셸 생성 중...")
    top, bottom = split(shell())
    parts = {
        "d435_shell_top.stl": lay_flat(finish_top(top), flip=False),
        "d435_shell_bottom.stl": lay_flat(finish_bottom(bottom), flip=True),
    }
    for name, mesh in parts.items():
        mesh = clean_export(mesh, os.path.join(out, name))
        e = mesh.extents
        print(f"  {name:22s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm   "
              f"watertight={mesh.is_watertight}  vol={mesh.volume / 1000:6.1f} cm^3")
    print(f"\n{out} 에 저장했습니다.")


if __name__ == "__main__":
    main()
