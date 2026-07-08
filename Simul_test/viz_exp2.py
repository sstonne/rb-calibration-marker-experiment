"""
Experiment 2 figure — Board vs Board+Cube 관측성 비교.

exp2_board_vs_cube.py --dump 으로 저장한 JSON 을 읽어 4지표 막대그래프로.
실행:
  PYTHONPATH= python Simul_test/viz_exp2.py
  PYTHONPATH= python Simul_test/viz_exp2.py --recompute --seeds 30
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
JSON_DEFAULT = os.path.join(FIG_DIR, "exp2_board_vs_cube_data.json")

# 지표: (key, 제목, 단위, 방향)  방향 up=높을수록좋음, down=낮을수록좋음
# 상단 행 = 관측성(observability), 하단 행 = 실제 캘리브 결과 (FK 미사용)
PANELS = [
    ("simul",    "Simultaneous views\n(cameras / shot)",     "cameras", "up"),
    ("coverage", "Viewpoint coverage\n(view-angle spread)",  "deg",     "up"),
    ("cam_mm",   "Camera pose error\n(vs GT)",               "mm",      "down"),
    ("obj_mm",   "Target(object) prediction\n(vs GT)",       "mm",      "down"),
]
MODE_ORDER = ["board", "board+cube"]
MODE_LABEL = {"board": "Board only", "board+cube": "Board + Cube"}
MODE_COLOR = {"board": "#c44e52", "board+cube": "#4c72b0"}


def make_figure(blob, out="fig_exp2_board_vs_cube.png", show=False):
    meta, data = blob["meta"], blob["data"]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(4.1 * len(PANELS), 4.6))
    slots = list(axes)

    for i, (ax, (key, title, unit, direction)) in enumerate(zip(slots, PANELS)):
        means = [data[m][key][0] for m in MODE_ORDER]
        stds = [data[m][key][1] for m in MODE_ORDER]
        # zero_obs 는 비율 → % 로 표시
        scale = 100.0 if key == "zero_obs" else 1.0
        means = [v * scale for v in means]; stds = [v * scale for v in stds]
        xs = np.arange(len(MODE_ORDER))
        ax.bar(xs, means, yerr=stds, capsize=5,
               color=[MODE_COLOR[m] for m in MODE_ORDER],
               edgecolor="black", linewidth=0.7, alpha=0.9, width=0.6)
        for x, m, s in zip(xs, means, stds):
            ax.text(x, m + s + max(means) * 0.02 + 1e-6, f"{m:.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(xs)
        ax.set_xticklabels([MODE_LABEL[m] for m in MODE_ORDER], fontsize=9)
        top = max(m + s for m, s in zip(means, stds))
        ax.set_ylim(0, top * 1.25 if top > 0 else 1)
        ax.set_ylabel("%" if key == "zero_obs" else unit, fontsize=9)
        arrow = "↑ better" if direction == "up" else "↓ better"
        # 앞 2개 = 관측성(회색), 뒤 2개 = 캘리브 결과(초록) — 제목 배경색으로 구분
        is_calib = i >= 2
        bg = "#e3f2e8" if is_calib else "#eeeeee"
        ax.set_title(f"{title}\n({arrow})", fontsize=10, fontweight="bold",
                     bbox=dict(boxstyle="round,pad=0.4", fc=bg, ec="none"))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 2 — Planar Board only  vs  Board + Marker Cube   (FK not used, camera-only)\n"
        f"4 fixed cameras,  incidence ≤ {meta['incidence']:.0f}°,  "
        f"{meta['shots']} shots/method,  {meta['seeds']} seeds       "
        "gray = observability,  green = calibration result",
        fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, out)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[저장] {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=JSON_DEFAULT)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.recompute or not os.path.exists(args.json):
        import exp2_board_vs_cube as e2
        acc = e2.run(seeds=args.seeds)
        out = {m: {k: [float(np.mean(acc[m][k])), float(np.std(acc[m][k]))]
                   for k in acc[m]} for m in acc}
        blob = {"meta": {"seeds": args.seeds, "incidence": 60.0, "sets": 8},
                "data": out}
        json.dump(blob, open(args.json, "w"), indent=2)
        print(f"[계산·저장] {args.json}")
    else:
        blob = json.load(open(args.json))
    make_figure(blob, show=args.show)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
