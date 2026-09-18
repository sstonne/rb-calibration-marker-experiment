#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/viz_slip_joint.py -- eval_slip_joint.py 결과(3대 joint 강체 판정 슬립) 시각화.

세트별 held→released / held→picked 이동량(mm, 축별 + 크기), 회전(deg), 3대 공동 fit 잔차(px).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--slip-json", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="")
a = ap.parse_args()
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False

d = json.loads(Path(a.slip_json).read_text())
ps = d["per_set"]; sets = [p["set"] for p in ps]
ev = [("released", "held → released (놓은 뒤)", "tab:blue"), ("picked", "held → picked (다시 잡은 뒤)", "tab:orange")]

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
# (1) 이동 크기 mm
ax = axes[0, 0]; w = 0.38
for i, (k, lab, c) in enumerate(ev):
    v = [p[k]["norm_mm"] if p.get(k) else np.nan for p in ps]
    ax.bar(np.array(sets) + (i - 0.5) * w, v, w, color=c, label=lab)
s = d["summary"]
ax.axhline(s["released"]["mm_median"], c="tab:blue", ls="--", lw=0.8); ax.axhline(s["picked"]["mm_median"], c="tab:orange", ls="--", lw=0.8)
ax.set_title(f"이동 크기 (3대 joint, mm) — 중앙값 {s['released']['mm_median']:.2f} / {s['picked']['mm_median']:.2f}, 최대 {max(s['released']['mm_max'], s['picked']['mm_max']):.2f}")
ax.set_xlabel("세트"); ax.set_ylabel("mm"); ax.legend(); ax.grid(alpha=0.3, axis="y")
# (2) 회전 deg
ax = axes[0, 1]
for i, (k, lab, c) in enumerate(ev):
    v = [p[k]["deg"] if p.get(k) else np.nan for p in ps]
    ax.bar(np.array(sets) + (i - 0.5) * w, v, w, color=c, label=lab)
ax.set_title(f"회전 (deg) — 중앙값 {s['released']['deg_median']:.3f} / {s['picked']['deg_median']:.3f}")
ax.set_xlabel("세트"); ax.set_ylabel("deg"); ax.legend(); ax.grid(alpha=0.3, axis="y")
# (3) 축별 mm (base 좌표)
ax = axes[1, 0]
for k, lab, c in ev:
    M = np.array([p[k]["mm"] if p.get(k) else [np.nan] * 3 for p in ps])
    for j, (axn, mk) in enumerate(zip("xyz", "os^")):
        ax.plot(sets, M[:, j], marker=mk, ls="-", lw=0.8, color=c, alpha=0.5 + 0.25 * (j == 1), label=f"{lab.split(' (')[0]} {axn}")
ax.axhline(0, c="k", lw=0.6)
ax.set_title("축별 이동 (base 좌표, mm)"); ax.set_xlabel("세트"); ax.set_ylabel("mm"); ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
# (4) fit 잔차 px
ax = axes[1, 1]
for k, lab, c in (("held", "held", "gray"), ("released", "released", "tab:blue"), ("picked", "picked", "tab:orange")):
    v = [p["fit_px"].get(k, np.nan) for p in ps]
    ax.plot(sets, v, marker="o", color=c, label=lab)
ax.set_title("3대 공동 큐브 pose fit 잔차 (px) — 강체 가정이 맞는지"); ax.set_xlabel("세트"); ax.set_ylabel("px"); ax.legend(); ax.grid(alpha=0.3)
for r in axes.flat: r.set_xticks(sets)
fig.suptitle(a.title or f"session2 슬립: 고정캠 3대 joint 강체 판정 (row {d['row']})", fontsize=14)
fig.tight_layout(); fig.savefig(a.out, dpi=110)
print("wrote", a.out)
