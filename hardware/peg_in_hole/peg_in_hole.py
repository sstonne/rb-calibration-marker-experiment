"""peg-in-hole 실험용 peg / hole 블록 STL 생성.

단위는 mm. 기본값: peg 지름 25.0 x 길이 100.0,
hole 지름 25.0 / 26.0 / 27.0 (peg 대비 틈 0.0 / 1.0 / 2.0 mm).

boolean 백엔드(manifold3d)나 shapely 없이도 돌아가도록
삼각형을 직접 쌓아 watertight 메시를 만든다.

원을 N각형으로 근사할 때 생기는 오차를 한쪽으로만 몰아준다.
  - peg  : 꼭짓점을 원 위에 둔다(내접) -> 실제 지름 <= 정격 지름
  - hole : 변을 원에 접하게 둔다(외접) -> 실제 지름 >= 정격 지름
즉 근사 오차가 끼워맞춤을 더 빡빡하게 만드는 방향으로는 절대 가지 않는다.
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh

SEG = 256  # 원 분할 수. 4의 배수여야 블록 모서리가 정확히 살아난다.


def _ring(radius, z, theta):
    """반지름 radius, 높이 z 인 원형 점열."""
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta),
                            np.full_like(theta, z)])


def _square_ring(half, z, theta):
    """중심에서 각도 theta 로 쏜 광선이 한 변 2*half 인 정사각형과 만나는 점열."""
    c, s = np.cos(theta), np.sin(theta)
    t = half / np.maximum(np.abs(c), np.abs(s))
    return np.column_stack([t * c, t * s, np.full_like(theta, z)])


def _stitch(a_start, b_start, n, flip=False):
    """정점 인덱스 a_start..+n 고리와 b_start..+n 고리를 사각형 띠로 잇는다."""
    i = np.arange(n)
    j = (i + 1) % n
    a0, a1 = a_start + i, a_start + j
    b0, b1 = b_start + i, b_start + j
    faces = np.vstack([np.column_stack([a0, b0, b1]),
                       np.column_stack([a0, b1, a1])])
    return faces[:, ::-1] if flip else faces


def _fan(center, ring_start, n, flip=False):
    """정점 하나(center)와 고리 하나를 삼각형 부채꼴로 잇는다. 평면 뚜껑용."""
    i = np.arange(n)
    j = (i + 1) % n
    faces = np.column_stack([np.full(n, center), ring_start + i, ring_start + j])
    return faces[:, ::-1] if flip else faces


def _build(rings, seg, closes, points=(), fans=()):
    """고리(+단일 점)들을 이어 붙여 하나의 메시로 만든다.

    rings  : (seg, 3) 배열들의 리스트
    closes : (i, j, flip) 목록 -> 고리 i 와 j 를 사각형 띠로 잇는다
    points : 단일 정점 목록 (부채꼴 뚜껑의 중심)
    fans   : (점 인덱스, 고리 인덱스, flip) 목록
    """
    verts = np.vstack(list(rings) + [np.asarray(pt, float).reshape(1, 3) for pt in points])
    ring_off = [k * seg for k in range(len(rings))]
    pt_off = len(rings) * seg
    faces = [_stitch(ring_off[i], ring_off[j], seg, flip) for i, j, flip in closes]
    faces += [_fan(pt_off + pi, ring_off[ri], seg, flip) for pi, ri, flip in fans]
    mesh = trimesh.Trimesh(vertices=verts, faces=np.vstack(faces), process=True)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def make_peg(diameter=25.0, length=100.0, tip_chamfer=1.0, seg=SEG):
    """peg: 지름 diameter, 길이 length 원기둥. 삽입되는 끝(z=0)에 45도 모따기.

    모따기는 정격 지름/길이 안쪽으로만 깎으므로 외형 치수는 그대로다.
    """
    r = diameter / 2.0                     # 내접: 꼭짓점이 정격 원 위
    theta = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    c = float(tip_chamfer)
    assert 0.0 <= c < min(r, length / 2.0)

    rings = [
        _ring(r - c, 0.0, theta),   # 0 바닥 (모따기 시작)
        _ring(r, c, theta),         # 1 모따기 끝
        _ring(r, length, theta),    # 2 윗면
    ]
    closes = [(0, 1, False), (1, 2, False)]   # 모따기 면 + 원기둥 벽
    points = [(0.0, 0.0, 0.0), (0.0, 0.0, length)]  # 바닥/윗면 중심
    fans = [(0, 0, False), (1, 2, False)]
    return _build(rings, seg, closes, points, fans)


def make_hole_block(hole_diameter, block=70.0, thickness=25.0,
                    entry_chamfer=1.0, seg=SEG):
    """가운데를 관통하는 구멍이 있는 사각 블록.

    block         : 한 변 길이 (mm)
    thickness     : 두께 = 구멍 깊이 (mm)
    entry_chamfer : 윗면 입구 45도 모따기 (mm). 0 이면 직선 구멍.
                    모따기 아래 구간은 정확히 hole_diameter 로 유지된다.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    # 외접: 다각형의 내접원이 정격 지름 -> 구멍이 정격보다 작아지지 않는다
    r = (hole_diameter / 2.0) / np.cos(np.pi / seg)
    half = block / 2.0
    c = float(entry_chamfer)
    assert r + c < half, "구멍 + 모따기가 블록보다 크다"

    rings = [
        _square_ring(half, 0.0, theta),        # 0 바닥 외곽
        _square_ring(half, thickness, theta),  # 1 윗면 외곽
        _ring(r, 0.0, theta),                  # 2 바닥 구멍
        _ring(r, thickness - c, theta),        # 3 모따기 시작
        _ring(r + c, thickness, theta),        # 4 윗면 구멍(모따기 끝)
    ]
    closes = [
        (0, 1, False),   # 바깥 벽
        (2, 0, False),   # 바닥면 (구멍 -> 외곽)
        (1, 4, False),   # 윗면 (외곽 -> 구멍)
        (4, 3, False),   # 입구 모따기
        (3, 2, False),   # 직선 구멍 벽
    ]
    return _build(rings, seg, closes)


def _report(name, mesh, path):
    ext = mesh.extents
    print(f"{name:28s} watertight={str(mesh.is_watertight):5s} "
          f"bbox={ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} mm  "
          f"vol={mesh.volume / 1000.0:8.2f} cm^3  -> {path.name}")


def save_preview(meshes, path):
    """부품들을 나란히 놓은 미리보기 PNG (matplotlib, 헤드리스)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(13, 4.2))
    for k, (name, mesh) in enumerate(meshes):
        ax = fig.add_subplot(1, len(meshes), k + 1, projection="3d")
        tri = mesh.vertices[mesh.faces]
        ax.add_collection3d(Poly3DCollection(tri, facecolors="#8fb8de",
                                             edgecolors="#6a93b5", linewidths=0.05,
                                             shade=True))
        lo, hi = mesh.bounds
        ctr, span = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1
        ax.set_xlim(ctr[0] - span, ctr[0] + span)
        ax.set_ylim(ctr[1] - span, ctr[1] + span)
        ax.set_zlim(ctr[2] - span, ctr[2] + span)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=24, azim=-58)
        ax.set_title(name, fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"preview -> {path.name}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--peg-diameter", type=float, default=25.0)
    p.add_argument("--peg-length", type=float, default=100.0)
    p.add_argument("--peg-chamfer", type=float, default=1.0)
    p.add_argument("--holes", type=float, nargs="+", default=[25.0, 26.0, 27.0])
    p.add_argument("--block", type=float, default=70.0, help="블록 한 변 (mm)")
    p.add_argument("--thickness", type=float, default=25.0, help="블록 두께 = 구멍 깊이")
    p.add_argument("--entry-chamfer", type=float, default=1.0,
                   help="구멍 입구 모따기 (mm). 0 이면 직선 구멍")
    p.add_argument("--preview", action="store_true", help="미리보기 PNG 저장")
    p.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    a = p.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    made = []

    peg = make_peg(a.peg_diameter, a.peg_length, a.peg_chamfer)
    path = a.outdir / f"peg_d{a.peg_diameter:.1f}_l{a.peg_length:.0f}.stl"
    peg.export(path)
    _report("peg", peg, path)
    made.append((f"peg  D{a.peg_diameter:g} x L{a.peg_length:g}", peg))

    for d in a.holes:
        m = make_hole_block(d, a.block, a.thickness, a.entry_chamfer)
        path = a.outdir / f"hole_d{d:.1f}.stl"
        m.export(path)
        _report(f"hole d={d:.1f} (틈 {d - a.peg_diameter:+.1f})", m, path)
        made.append((f"hole D{d:g}  (clearance {d - a.peg_diameter:+g} mm)", m))

    if a.preview:
        save_preview(made, a.outdir / "preview.png")


if __name__ == "__main__":
    main()
