#!/usr/bin/env python3
"""RealSense D435 슬리브 커버 + Robotiq 2F-85 플랜지 암 STL 생성기.

세 개의 STL 을 만든다.
  d435_gauge.stl   : 15분짜리 시험 출력. 본체 끼움과 M3 구멍 위치만 확인한다.
  d435_sleeve.stl  : 카메라를 감싸는 커버 본체.
  d435_arm.stl     : 툴 플랜지(ISO 9409-1-50-4-M6)와 그리퍼 커플러 사이에 끼우는 암.

좌표계(설계 기준, 출력 방향은 export 시 회전):
  X = 카메라 긴 축(좌우), Y = 깊이(앞 0 -> 뒤 +), Z = 높이(바닥 0 -> 위 +)
  카메라 앞면(렌즈면)이 Y=0, 뒷면이 Y=CAM_D.

사용: python d435_gripper_cover.py [출력디렉터리]
"""

import os
import sys

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

ENGINE = "manifold"

# ---------------------------------------------------------------- 카메라 제원
CAM_L = 90.0          # 긴 축 길이
CAM_H = 25.0          # 높이
CAM_D = 25.0          # 깊이(앞뒤)
CAM_R = 4.0           # 앞면 네 모서리 필렛 반경. 실제값보다 크게 잡아야 안전하다
                      # (캐비티 필렛이 실제보다 작으면 모서리가 걸려 아예 안 들어간다)
M3_SPACING = 45.0     # 뒷면 M3 구멍 중심간 거리
M3_Z = 12.5           # 카메라 바닥면에서 M3 구멍 중심까지 높이  <-- 캘리퍼로 확인할 값
M3_DEPTH = 2.5        # 나사산 깊이 (이보다 긴 나사를 쓰면 안 됨)

# ------------------------------------------------------------------ 끼움 공차
# "오차 0" 은 FDM 으로 불가능하다. 한 면당 이만큼 띄우고 프린터 실측으로 조정한다.
CLR = 0.30            # 조이면 0.20, 헐거우면 0.40

# ---------------------------------------------------------------- 커버 두께류
WALL = 2.4            # 위/아래/양끝 벽
REAR_WALL = 3.0       # 뒷판 (M3 나사가 물리는 곳)
TOP_PAD = 4.5         # 암이 붙는 윗면 국부 두께
REAR_HALF_X = 30.0    # 뒷판이 덮는 반폭. 이 바깥은 열려 있어 USB-C 케이블이 나온다
END_WALL_Y = 15.0     # 양끝 벽이 앞에서부터 덮는 깊이. 뒤쪽 모서리는 열어 둔다

# 암 체결 4공 패턴 (커버 윗면 기준)
PAD_X = 20.0          # ±20
PAD_Y = (7.0, 21.0)   # 앞면에서의 거리
M3_CLR = 3.4          # M3 관통 구멍
M3_CSK = 6.4          # 접시머리 카운터싱크 지름

# --------------------------------------------------------------------- 암 제원
FLANGE_D = 50.0       # ISO 9409-1-50-4-M6 플랜지 지름
FLANGE_T = 6.0        # 샌드위치 판 두께 (M6 볼트가 이만큼 길어져야 한다)
FLANGE_BC = 31.5      # M6 볼트 피치원 지름
FLANGE_M6 = 6.6       # M6 관통
FLANGE_DOWEL = 6.2    # 위치결정 핀 구멍
FLANGE_BORE = 22.0    # 가운데 배선 구멍
ARM_R = 42.0          # 툴 축에서 암 바깥면까지 (= 커버 윗면이 닿는 반경)
ARM_Z = 60.0          # 플랜지면에서 카메라 중심까지 툴축 방향 거리
ARM_T = 8.0           # 암 두께
ARM_W = 30.0          # 암 폭

# 파생값
CX = CAM_L / 2 + CLR          # 캐비티 반폭
CY = CAM_D + 2 * CLR          # 캐비티 깊이 (앞은 열림)
CZ = CAM_H + 2 * CLR          # 캐비티 높이
OX = CX + WALL                # 외형 반폭
OY = CY + REAR_WALL           # 외형 깊이
OZ_LO = -WALL                 # 바닥 외면
OZ_HI = CZ + TOP_PAD          # 윗면 외면


def bx(size, center):
    """중심과 크기로 박스 하나."""
    T = np.eye(4)
    T[:3, 3] = center
    return box(extents=size, transform=T)


def span(x0, x1, y0, y1, z0, z1):
    """세 축 구간으로 박스 하나."""
    return bx((x1 - x0, y1 - y0, z1 - z0),
              ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def cyl(d, length, center, axis="z"):
    T = np.eye(4)
    if axis == "x":
        T[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]
    elif axis == "y":
        T[:3, :3] = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]
    T[:3, 3] = center
    return cylinder(radius=d / 2, height=length, sections=64, transform=T)


def cone(d_big, d_small, length, center, axis="z"):
    """카운터싱크용 원뿔대. d_big 쪽이 axis 의 + 방향."""
    m = trimesh.creation.cone  # 사용 안 함: 수동 생성
    n = 64
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rb, rs = d_big / 2, d_small / 2
    bot = np.c_[rs * np.cos(ang), rs * np.sin(ang), np.full(n, -length / 2)]
    top = np.c_[rb * np.cos(ang), rb * np.sin(ang), np.full(n, length / 2)]
    verts = np.vstack([bot, top, [[0, 0, -length / 2]], [[0, 0, length / 2]]])
    cb, ct = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [[i, j, n + j], [i, n + j, n + i], [cb, j, i], [ct, n + i, n + j]]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=True)
    T = np.eye(4)
    if axis == "x":
        T[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])[:3, :3]
    elif axis == "y":
        T[:3, :3] = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])[:3, :3]
    T[:3, 3] = center
    mesh.apply_transform(T)
    return mesh


def rprism(x_half, z0, z1, y0, y1, r):
    """XZ 단면이 둥근 사각형이고 Y 방향으로 밀어낸 형상.

    D435 는 앞에서 봤을 때 90x25 사각형의 네 모서리가 둥글다. 그 단면을 그대로
    깊이 방향으로 밀어낸 것이 카메라 바디이고, 캐비티도 같은 단면으로 판다.
    """
    r = min(r, x_half - 0.1, (z1 - z0) / 2 - 0.1)
    yc, yl = (y0 + y1) / 2, y1 - y0
    parts = [
        span(-x_half, x_half, y0, y1, z0 + r, z1 - r),
        span(-x_half + r, x_half - r, y0, y1, z0, z1),
    ]
    for sx in (-1, 1):
        for z in (z0 + r, z1 - r):
            parts.append(cyl(2 * r, yl, (sx * (x_half - r), yc, z), axis="y"))
    return trimesh.boolean.union(parts, engine=ENGINE)


def sleeve_body(half_x, with_pad=True, with_vents=True):
    """반폭 half_x 인 커버 하나. 게이지는 half_x 만 줄여서 같은 단면을 쓴다."""
    ox = half_x + WALL
    cx = half_x

    r_in = CAM_R + CLR          # 캐비티 모서리 (카메라 실제 필렛보다 크게)
    r_out = r_in + WALL         # 바깥 모서리는 벽 두께만큼 더 크다

    # 바깥 덩어리 (앞면은 아예 벽이 없다)
    solid = rprism(ox, OZ_LO, OZ_HI, 0.0, OY, r_out)

    cuts = []
    # 카메라가 들어갈 캐비티 — 앞쪽으로 뚫려 있다
    cuts.append(rprism(cx, 0.0, CZ, -1.0, CY, r_in))
    # 뒷판은 |x| <= REAR_HALF_X 만 남긴다 (USB-C 케이블 통로)
    if half_x > REAR_HALF_X:
        cuts.append(span(REAR_HALF_X, ox + 1, CY - 1, OY + 1, OZ_LO - 1, OZ_HI + 1))
        cuts.append(span(-ox - 1, -REAR_HALF_X, CY - 1, OY + 1, OZ_LO - 1, OZ_HI + 1))
    # 양끝 벽은 앞에서 END_WALL_Y 까지만. 뒤쪽 모서리는 열어 둔다
    cuts.append(span(cx, ox + 1, END_WALL_Y, OY + 1, OZ_LO - 1, OZ_HI + 1))
    cuts.append(span(-ox - 1, -cx, END_WALL_Y, OY + 1, OZ_LO - 1, OZ_HI + 1))

    # 뒷판 M3 관통 2개
    for sx in (-1, 1):
        x = sx * M3_SPACING / 2
        if abs(x) < cx:
            cuts.append(cyl(M3_CLR, 20.0, (x, CY + REAR_WALL / 2, CLR + M3_Z), axis="y"))

    if with_vents:
        # 삼각대 구멍 접근 + 방열 슬롯
        cuts.append(cyl(11.0, 20.0, (0.0, CLR + CAM_D / 2, OZ_LO - 1), axis="z"))
        for sx in (-1, 1):
            for x0 in (20.0, 32.0):
                x = sx * x0
                if abs(x) + 2.5 < cx:
                    cuts.append(bx((5.0, 16.0, 20.0), (x, CLR + CAM_D / 2, OZ_LO)))

    if with_pad:
        # 암 체결 4공: 안쪽에서 접시머리 M3 를 꽂아 머리를 캐비티 면과 같게 묻는다
        for sx in (-1, 1):
            for y in PAD_Y:
                x = sx * PAD_X
                if abs(x) + 4 < cx:
                    cuts.append(cyl(M3_CLR, 20.0, (x, y, CZ + TOP_PAD / 2), axis="z"))
                    cuts.append(cone(M3_CSK, M3_CLR, 1.6, (x, y, CZ - 0.8), axis="z"))

    for c in cuts:
        solid = solid.difference(c, engine=ENGINE)
    return solid


def arm():
    """플랜지 샌드위치 판 + L 빔 + 카메라 체결 패드."""
    # 베이스 원판: 툴 축 = Z, 플랜지면 z=0 ~ FLANGE_T
    base = cyl(FLANGE_D, FLANGE_T, (0, 0, FLANGE_T / 2))

    # 반경 방향 빔 (+X) 과 툴축 방향 빔
    radial = span(0, ARM_R, -ARM_W / 2, ARM_W / 2, FLANGE_T, FLANGE_T + ARM_T)
    axial = span(ARM_R - ARM_T, ARM_R, -ARM_W / 2, ARM_W / 2, FLANGE_T, FLANGE_T + ARM_Z)

    # 카메라 패드: 암 바깥면 x=ARM_R 에 커버 윗면이 닿는다
    pad = span(ARM_R - 5.0, ARM_R, -30.0, 30.0,
               FLANGE_T + ARM_Z - 26.0, FLANGE_T + ARM_Z + 6.0)

    solid = base.union(radial, engine=ENGINE)
    solid = solid.union(axial, engine=ENGINE)
    solid = solid.union(pad, engine=ENGINE)

    cuts = [cyl(FLANGE_BORE, 40.0, (0, 0, FLANGE_T / 2))]
    for k in range(4):
        a = np.pi / 4 + k * np.pi / 2
        cuts.append(cyl(FLANGE_M6, 40.0,
                        (FLANGE_BC / 2 * np.cos(a), FLANGE_BC / 2 * np.sin(a), FLANGE_T / 2)))
    cuts.append(cyl(FLANGE_DOWEL, 40.0, (0, FLANGE_BC / 2, FLANGE_T / 2)))

    # 카메라 체결 4공 + M3 육각너트 포켓 (안쪽에서 끼운다)
    zc = FLANGE_T + ARM_Z - 10.0
    for sy in (-1, 1):
        for dy in PAD_Y:
            y = sy * PAD_X
            z = zc + (dy - np.mean(PAD_Y))
            cuts.append(cyl(M3_CLR, 30.0, (ARM_R - 2.5, y, z), axis="x"))
            # 너트 포켓: AF 5.5, 두께 2.6, 육각
            nut = cyl(6.35, 2.8, (ARM_R - 5.0 + 1.4, y, z), axis="x")
            nut = trimesh.creation.cylinder(radius=3.18, height=2.8, sections=6)
            T = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
            T[:3, 3] = (ARM_R - 5.0 + 1.4, y, z)
            nut.apply_transform(T)
            cuts.append(nut)

    for c in cuts:
        solid = solid.difference(c, engine=ENGINE)
    return solid


def to_print_orientation(mesh):
    """뒷판이 베드에 닿도록 X 축 -90도 회전. 앞 개구부가 위를 본다."""
    m = mesh.copy()
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.apply_translation([0, 0, -m.bounds[0][2]])
    return m


def report(name, mesh):
    e = mesh.extents
    print(f"  {name:16s} {e[0]:6.1f} x {e[1]:6.1f} x {e[2]:6.1f} mm   "
          f"watertight={mesh.is_watertight}  vol={mesh.volume/1000:6.1f} cm^3")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out, exist_ok=True)

    print("생성 중...")
    parts = {
        "d435_gauge.stl": to_print_orientation(
            sleeve_body(half_x=30.0, with_pad=False, with_vents=False)),
        "d435_sleeve.stl": to_print_orientation(sleeve_body(half_x=CX)),
        "d435_arm.stl": arm(),
    }
    for name, mesh in parts.items():
        mesh.export(os.path.join(out, name))
        report(name, mesh)
    print(f"\n{out} 에 저장했습니다.")


if __name__ == "__main__":
    main()
