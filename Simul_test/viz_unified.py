"""
C1 — Unified vs Independent 비교 figure 생성.

unified_vs_independent.py --dump 으로 저장한 JSON 을 읽어 막대그래프 4패널(4지표)로.
주 결과는 systematic 노이즈(실제 검출오차의 지배성분). random 은 대조군.

실행:
  PYTHONPATH= python Simul_test/viz_unified.py                       # JSON 있으면 그걸로
  PYTHONPATH= python Simul_test/viz_unified.py --recompute --seeds 30 # 새로 계산+저장
"""
import os, sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
JSON_DEFAULT = os.path.join(FIG_DIR, "unified_vs_indep_data.json")

# 지표 라벨 (한글 폰트 문제 회피 위해 영문 병기)
KEY_LABELS = {
    "bTf_mm":         "Fixed cam bTf\n(eye-to-hand, mm)",
    "gTc_mm":         "Hand-eye gTc\n(eye-in-hand, mm)",
    "consistency_mm": "Shared-base\nconsistency (mm)",
    "downstream_mm":  "Downstream cube\nprediction (mm)",
}
METHOD_ORDER = ["Indep", "Indep+fk", "Joint", "Joint+fk"]
METHOD_COLORS = {"Indep": "#c44e52", "Indep+fk": "#e17c7f",
                 "Joint": "#4c72b0", "Joint+fk": "#7a9bcc"}


def _load(path):
    with open(path) as f:
        return json.load(f)


def make_figure(blob, out="fig_unified_vs_indep.png", show=False):
    meta = blob["meta"]
    data = blob["data"]
    keys = meta["keys"]
    ntypes = list(data.keys())            # ["systematic", "random"]

    # systematic 을 왼쪽(주 결과)으로 정렬
    ntypes = sorted(ntypes, key=lambda n: 0 if n == "systematic" else 1)

    fig, axes = plt.subplots(len(ntypes), len(keys),
                             figsize=(4.3 * len(keys), 4.0 * len(ntypes)),
                             squeeze=False)

    for ri, ntype in enumerate(ntypes):
        for ci, key in enumerate(keys):
            ax = axes[ri][ci]
            means = [data[ntype][m][key][0] for m in METHOD_ORDER]
            stds = [data[ntype][m][key][1] for m in METHOD_ORDER]
            xs = np.arange(len(METHOD_ORDER))
            bars = ax.bar(xs, means, yerr=stds, capsize=4,
                          color=[METHOD_COLORS[m] for m in METHOD_ORDER],
                          edgecolor="black", linewidth=0.6, alpha=0.9)
            for x, m, s in zip(xs, means, stds):
                ax.text(x, m + s + max(means) * 0.02, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels(METHOD_ORDER, rotation=20, ha="right", fontsize=8)
            ax.set_ylim(0, max(m + s for m, s in zip(means, stds)) * 1.28)
            ax.grid(axis="y", alpha=0.3)
            if ri == 0:
                ax.set_title(KEY_LABELS[key], fontsize=10, fontweight="bold")
            if ci == 0:
                if len(ntypes) > 1:
                    tag = "SYSTEMATIC" if ntype == "systematic" else "random (control)"
                    ax.set_ylabel(f"[{tag}]\nerror (mm)", fontsize=9)
                else:
                    ax.set_ylabel("error (mm)", fontsize=9)

    fig.suptitle(
        "C1 Unified Calibration:  Independent (separate + align)  vs  Joint (bundle adjustment)\n"
        f"eye-to-hand ×3 + eye-in-hand,  {meta['sets']} sets, "
        f"train={meta['train']}/test=2,  {meta['seeds']} seeds,  systematic noise {meta['noise']}mm  "
        "(lower = better)",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, out)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[저장] {path}")
    if show:
        plt.show()
    plt.close(fig)
    return path


def make_tilt_sweep(tilts=(15, 25, 35, 50, 70, 90), seeds=15, noise_mm=6.0,
                    out="fig_unified_tilt_sweep.png", show=False):
    """그리퍼 EE 틸트(±deg = 자세 다양성)에 따른 핸드아이 gTc 오차:
       독립 vs 통합. 자세 다양성이 낮을수록(왼쪽) 통합의 이점이 커짐을 보인다."""
    import unified_vs_independent as uvi
    indep_tm, indep_rd, uni_tm, uni_rd = [], [], [], []
    for tilt in tilts:
        it, ir, ut, ur = [], [], [], []
        for seed in range(seeds):
            sc = uvi.build_scene(seed=seed, n_sets=8, noise_mm=noise_mm,
                                 noise_type="systematic", n_events_per_set=6,
                                 gripper_tilt_deg=tilt)
            tr = sc["sets"]                       # 전체 set 사용(캘리브 자체가 관심)
            mi = uvi.calib_independent(sc, tr)
            mu = uvi.calib_unified(sc, tr)
            if mi["gTc"] is not None:
                it.append(uvi._trans_mm(mi["gTc"], sc["gt_gTc"]))
                ir.append(uvi._rot_deg(mi["gTc"], sc["gt_gTc"]))
            if mu["gTc"] is not None:
                ut.append(uvi._trans_mm(mu["gTc"], sc["gt_gTc"]))
                ur.append(uvi._rot_deg(mu["gTc"], sc["gt_gTc"]))
        indep_tm.append(np.mean(it)); indep_rd.append(np.mean(ir))
        uni_tm.append(np.mean(ut)); uni_rd.append(np.mean(ur))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.array(tilts)
    for ax, yi, yu, ttl, unit in [
        (axes[0], indep_tm, uni_tm, "Hand-eye gTc translation error", "mm"),
        (axes[1], indep_rd, uni_rd, "Hand-eye gTc rotation error", "deg")]:
        ax.plot(x, yi, "o-", color="#c44e52", lw=2, label="Independent (grip-only AX=XB)")
        ax.plot(x, yu, "s-", color="#4c72b0", lw=2, label="Unified (shared cube anchor)")
        ax.set_xlabel("gripper EE tilt range  ±deg  (pose diversity →)")
        ax.set_ylabel(f"gTc error ({unit})")
        ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
        ax.axvline(35, color="gray", ls="--", alpha=0.6)
        ax.text(35, ax.get_ylim()[1]*0.92, " realistic\n (±35°)", fontsize=8, color="gray")
    fig.suptitle(
        "Hand-eye gTc vs pose diversity:  Unified compensates limited EE tilt\n"
        f"systematic noise {noise_mm}mm,  8 sets,  {seeds} seeds  (lower = better)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
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
    ap.add_argument("--tilt-sweep", action="store_true",
                    help="tilt sweep figure 도 생성")
    ap.add_argument("--only-tilt", action="store_true",
                    help="tilt sweep figure 만 생성")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.only_tilt:
        make_tilt_sweep(show=args.show)
        return

    if args.recompute or not os.path.exists(args.json):
        import unified_vs_independent as uvi
        dump = {}
        for ntype in ("systematic",):
            acc = uvi.run(seeds=args.seeds, noise_type=ntype)
            dump[ntype] = {name: {k: [float(np.mean(acc[name][k])),
                                      float(np.std(acc[name][k])),
                                      len(acc[name][k])]
                                  for k in uvi.KEYS} for name, _, _ in uvi.METHODS}
        blob = {"meta": {"seeds": args.seeds, "noise": 6.0, "sets": 8, "train": 6,
                         "keys": uvi.KEYS, "methods": [m[0] for m in uvi.METHODS]},
                "data": dump}
        with open(args.json, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"[계산·저장] {args.json}")
    else:
        blob = _load(args.json)

    make_figure(blob, show=args.show)
    if args.tilt_sweep:
        make_tilt_sweep(show=args.show)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))
    main()
