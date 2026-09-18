#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zeus_gello_calibration/viz_slip_overlay.py -- session2 held/released/picked 세 장을 큐브 부분만 잘라 겹쳐 본다.

세트별 그림 1장: 고정캠 3대 × 2행.
  위  행: 세 장을 같은 투명도(1/3씩)로 겹친 것. 밀림이 있으면 마커 경계가 겹으로 보임.
  아래 행: 색 채널 합성 (held=빨강, released=초록, picked=파랑). 세 장이 완전히 같으면 회색, 어긋나면 경계에 색 테두리가 남음.
잘라내는 위치는 held 사진의 큐브 코너 검출 결과(중심 ±r)로 정한다.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from calibration_pipeline.apriltag_cube import AprilTagCubeTarget  # noqa: E402
from calibration_pipeline.cube_config import load_cube_config_from_json_file  # noqa: E402
from eval_joint_relative import FIXED_LABELS  # noqa: E402
from paths import SESSION2_DIR  # noqa: E402
TAGS = ("held", "released", "picked")

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--session2-capture-subdir", required=True)
ap.add_argument("--slip-json", default=None, help="eval_slip_joint.py 결과 (제목용)")
ap.add_argument("--sets", default=None)
ap.add_argument("--out-dir", required=True)
a = ap.parse_args()
plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
cfg, _ = load_cube_config_from_json_file(REPO_ROOT / "targets/gt_cube/cube_config.json"); cube = AprilTagCubeTarget(cfg)
slip = {r["set"]: r for r in json.loads(Path(a.slip_json).read_text())["per_set"]} if a.slip_json else {}
root = SESSION2_DIR / a.session2_capture_subdir
sets = [int(s) for s in a.sets.split(",")] if a.sets else sorted(int(p.name) for p in (root / "held").iterdir() if p.name.isdigit())
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

for k in sets:
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for j, lab in enumerate(FIXED_LABELS):
        imgs = {}
        for t in TAGS:
            p = root / t / f"{k:03d}" / f"cam_{lab}.png"
            if p.is_file(): imgs[t] = cv2.imread(str(p))
        if "held" not in imgs:
            for ax in axes[:, j]: ax.axis("off")
            continue
        corners, ids = cube.detect(imgs["held"])
        if ids is None:
            for ax in axes[:, j]: ax.imshow(cv2.cvtColor(imgs["held"], cv2.COLOR_BGR2RGB)); ax.set_title(f"{lab}: 검출 없음"); ax.axis("off")
            continue
        pts = np.concatenate([np.asarray(c).reshape(-1, 2) for c in corners]); cx, cy = pts.mean(0)
        r = int(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])) * 0.7 + 30)
        H, W = imgs["held"].shape[:2]; x0, x1 = max(0, int(cx - r)), min(W, int(cx + r)); y0, y1 = max(0, int(cy - r)), min(H, int(cy + r))
        crops = {t: cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).astype(np.float32) / 255 for t, im in imgs.items()}
        blend = np.mean([crops[t] for t in TAGS if t in crops], axis=0)
        gray = {t: cv2.cvtColor(crops[t], cv2.COLOR_RGB2GRAY) for t in crops}
        chan = np.stack([gray.get(t, np.zeros_like(gray["held"])) for t in TAGS], axis=-1)
        axes[0, j].imshow(blend); axes[0, j].set_title(f"{lab}  투명 겹침 ({' + '.join(t for t in TAGS if t in crops)})"); axes[0, j].axis("off")
        axes[1, j].imshow(chan); axes[1, j].set_title(f"{lab}  채널 합성 (held=R, released=G, picked=B; 어긋나면 색 테두리)"); axes[1, j].axis("off")
    s = slip.get(k, {}); t = f"set {k}: held / released / picked 세 장 겹침"
    for tag, nm in (("released", "놓은 뒤"), ("picked", "다시 잡은 뒤")):
        if tag in s: t += f"   |   {nm} joint |Δ|={s[tag]['norm_mm']:.2f}mm {s[tag]['deg']:.2f}°"
    fig.suptitle(t, fontsize=14); fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(out / f"slip_overlay_set{k:02d}.png", dpi=100); plt.close(fig)
    print("wrote", out / f"slip_overlay_set{k:02d}.png")
