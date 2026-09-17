"""Board-independent RGB capture and offline intrinsic image loading."""

import json
import time
from pathlib import Path

import cv2


class IntrinsicsImages:
    def __init__(self, directory, serial_to_idx, *, create=False):
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "capture_manifest.json"
        mapping = {str(s): int(i) for s, i in serial_to_idx.items()}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            saved = self.manifest.get("serial_to_idx") or {}
            # 새로 연결된 카메라가 device_map에 추가된 경우만 허용: 기존 serial->idx는 그대로여야 한다.
            if (self.manifest.get("version") != 1
                    or any(mapping.get(s) != i for s, i in saved.items())
                    or len(set(mapping.values())) != len(mapping)):
                raise ValueError("Raw image camera mapping does not match --intr_dir")
            added = {s: i for s, i in mapping.items() if s not in saved}
            if added:
                self.manifest["serial_to_idx"] = {**saved, **added}
                if create:
                    self._save()
        elif create:
            if self.directory.exists() and any(self.directory.iterdir()):
                raise ValueError("Nonempty --images_dir has no capture_manifest.json")
            self.directory.mkdir(parents=True, exist_ok=True)
            self.manifest = {
                "version": 1, "mode": "raw_intrinsics", "created_at_epoch": time.time(),
                "serial_to_idx": mapping, "cameras": {},
            }
            self._save()
        else:
            raise ValueError(f"Missing {self.manifest_path}; run --capture_only first")

    def _save(self):
        self.manifest["updated_at_epoch"] = time.time()
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def check_camera(self, cam_idx, serial, width, height, fps, *, create=False):
        expected = {
            "serial": str(serial), "color_w": int(width),
            "color_h": int(height), "fps": int(fps),
        }
        if self.manifest["serial_to_idx"].get(str(serial)) != int(cam_idx):
            raise ValueError(f"Raw image serial does not match cam{cam_idx}")
        record = self.manifest["cameras"].get(str(cam_idx))
        if record is None:
            if create:
                self.manifest["cameras"][str(cam_idx)] = {**expected, "frames": []}
                self._save()
            return
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Raw image serial/resolution/FPS mismatch for cam{cam_idx}")

    def frames(self, cam_idx):
        return self.manifest["cameras"].get(str(cam_idx), {}).get("frames", [])

    def count(self, cam_idx):
        return sum(frame["enabled"] for frame in self.frames(cam_idx))

    def image_path(self, frame):
        path = (self.directory / frame["file"]).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("Raw image path escapes --images_dir")
        return path

    def capture(self, cam_idx, color):
        record = self.manifest["cameras"][str(cam_idx)]
        if color.shape[:2] != (record["color_h"], record["color_w"]):
            raise ValueError(f"Camera frame resolution changed for cam{cam_idx}")
        relative = Path(f"cam{cam_idx}") / f"view_{time.time_ns()}.png"
        path = self.directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        if not cv2.imwrite(str(path), color):
            raise OSError(f"Could not save {path}")
        record["frames"].append({
            "file": relative.as_posix(), "captured_at_epoch": time.time(), "enabled": True,
        })
        self._save()
        return path

    def undo(self, cam_idx):
        for frame in reversed(self.frames(cam_idx)):
            if frame["enabled"]:
                frame["enabled"] = False
                self._save()
                return True
        return False


def collect_raw_for_camera(cam, images, cam_idx):
    win = f"cam{cam_idx} RAW intrinsics"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print(f"[cam{cam_idx}] RAW: SPACE=원본 저장, c/Enter=다음 카메라, "
          "u=직전 샘플 제외, q=종료 (저장한 원본 유지)")
    print(f"[cam{cam_idx}] 기존 저장: {images.count(cam_idx)}장")
    try:
        while True:
            color, _, _ = cam.get_latest()
            if color is not None:
                vis = color.copy()
                label = f"cam{cam_idx} RAW   saved: {images.count(cam_idx)}"
                cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(win, vis)
            key = cv2.waitKey(1 if color is not None else 30) & 0xFF
            if key == ord("q"):
                return "abort"
            if key == ord("s"):
                return "skip"
            if key in (ord("c"), 13, 10):
                return "done"
            if key == ord("u"):
                images.undo(cam_idx)
                print(f"[cam{cam_idx}] RAW 대상: {images.count(cam_idx)}장 (원본 파일 유지)")
            if key == ord(" ") and color is not None:
                path = images.capture(cam_idx, color)
                print(f"[SAVE] cam{cam_idx} RAW #{images.count(cam_idx)}: {path}")
    finally:
        cv2.destroyWindow(win)


def load_image_views(images, cam_idx, target, min_corners):
    record = images.manifest["cameras"].get(str(cam_idx))
    accepted, details = [], []
    if record is None:
        return accepted, details
    for frame in record["frames"]:
        detail = {"file": frame["file"], "charuco_corners": 0}
        details.append(detail)
        if not frame["enabled"]:
            detail["status"] = "excluded_by_user"
            continue
        color = cv2.imread(str(images.image_path(frame)), cv2.IMREAD_COLOR)
        if color is None:
            detail["status"] = "unreadable"
            continue
        if color.shape[:2] != (record["color_h"], record["color_w"]):
            detail["status"] = "resolution_mismatch"
            continue
        try:
            corners, ids, count, _, _ = target.detect(color)
        except cv2.error as error:
            detail.update(status="detection_error", error=str(error))
            continue
        detail["charuco_corners"] = int(count)
        if corners is None or ids is None or count < min_corners:
            detail["status"] = "too_few_corners"
            continue
        if (hasattr(target.board, "checkCharucoCornersCollinear")
                and target.board.checkCharucoCornersCollinear(ids)):
            detail["status"] = "collinear_corners"
            continue
        detail["status"] = "detected"
        accepted.append((corners, ids))
    return accepted, details
