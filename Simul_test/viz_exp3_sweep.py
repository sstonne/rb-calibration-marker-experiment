"""
Experiment 3 sweep 곡선 figure — 노이즈(또는 데이터량) 대비 세 방식 오차.

exp3_gtc_estimation.py --sweep noise --dump 으로 저장한 JSON 을 읽어 곡선 그래프.
실행:
  PYTHONPATH= python Simul_test/viz_exp3_sweep.py --json .../exp3_noise_sweep_data.json
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
COLORS = {"Camera-based": "#4c72b0", "FK-based": "#c44e52",
          "Camera+FK-corr": "#55a868"}
MARK = {"Camera-based": "o", "FK-based": "s", "Camera+FK-corr": "^"}


def make_figure(blob, out="fig_exp3_noise_sweep.png", show=False):
    kind = blob["kind"]
    xs = blob["xs"]
    curve = blob["curve"]
    xlabel = ("observation noise (mm)" if kind == "noise"
              else "train set count")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, metric, title in [(axes[0], "heldout", "Held-out cube prediction"),
                              (axes[1], "gtc", "Hand-eye gTc recovery")]:
        for m in ["Camera-based", "FK-based", "Camera+FK-corr"]:
            ys = curve[m][metric]
            ax.plot(xs, ys, marker=MARK[m], color=COLORS[m], lw=2, ms=7, label=m)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("error (mm)")
        ax.set_title(f"{title}\n(↓ better)", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    fig.suptitle(
        f"Experiment 3 — {('noise' if kind=='noise' else 'data-amount')} robustness:  "
        "Camera-based  vs  FK-based  vs  Camera+FK-correction\n"
        "unified (Joint) calibration,  systematic noise,  8 sets  (lower = better)",
        fontsize=11, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
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
    ap.add_argument("--json", default=os.path.join(FIG_DIR, "exp3_noise_sweep_data.json"))
    ap.add_argument("--out", default="fig_exp3_noise_sweep.png")
    args = ap.parse_args()
    blob = json.load(open(args.json))
    make_figure(blob, out=args.out)


if __name__ == "__main__":
    main()
