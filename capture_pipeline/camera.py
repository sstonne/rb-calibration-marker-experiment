# camera.py
"""
RealSense camera wrapper with threaded capture.
Supports multiple cameras simultaneously.

Each camera keeps a short ring buffer with a host-monotonic receipt timestamp
and the RealSense device timestamp. Cross-camera matching uses the host clock;
device timestamps are not assumed to share an epoch across independent devices.
so callers can retrieve the frame closest to a target timestamp via
`get_at()`, enabling software synchronization across multiple cameras.

Color exposure/gain/white balance are locked at `start()` (see
`_lock_color_photometry`): auto-exposure converges during warmup, then the
converged values are frozen for the session. Calibration frames must be
photometrically identical — a re-converging AE changes corner subpixel accuracy
between captures and can stretch exposure into motion blur. The locked values
are exposed as `cam.color_photometry` for session metadata.

`start()` is resilient: on "Frame didn't arrive" timeout it performs a
hardware reset of the device (by serial), waits for USB re-enumeration,
rebuilds the pipeline, and retries up to 3 times. Warmup uses a longer
10-second timeout to tolerate slow first-frame delivery on multi-camera
USB hubs.

Use `RealSenseCamera.reset_all_devices()` once at process startup to
hardware-reset every connected RealSense before opening pipelines —
this clears bad state left over from a prior crash/segfault and is
especially important for D435 on chained USB hubs, which otherwise
hangs in `pipeline.start()` with "Frame didn't arrive within 10000".
"""

import threading
import time
from collections import deque
from typing import Optional, Tuple, Dict

import numpy as np
import pyrealsense2 as rs


class RealSenseCamera:
    """Thread-safe RealSense camera capture with a short frame ring buffer."""

    def __init__(
        self,
        serial: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 15,
        use_color: bool = True,
        use_depth: bool = False,
        align_depth_to_color: bool = True,
        warmup_frames: int = 10,
        buffer_size: int = 45,
        lock_color_exposure: bool = True,
        color_exposure_us: Optional[float] = None,
        color_gain: Optional[float] = None,
        color_white_balance: Optional[float] = None,
        ae_settle_frames: int = 30,
    ):
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.use_color = bool(use_color)
        self.use_depth = bool(use_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.warmup_frames = int(warmup_frames)
        # 45프레임 = 15fps 기준 3초. 8프레임(0.53초)이면 카메라 한 대가 잠깐만
        # 뒤처져도 get_at() 이 맞는 시각의 프레임을 못 찾아 span 이 폭발한다.
        self.buffer_size = max(1, int(buffer_size))

        self.lock_color_exposure = bool(lock_color_exposure)
        self.color_exposure_us = color_exposure_us
        self.color_gain = color_gain
        self.color_white_balance = color_white_balance
        self.ae_settle_frames = int(ae_settle_frames)
        # Populated by _lock_color_photometry(); belongs in session metadata so a
        # re-shoot can reproduce the same photometric setting.
        self.color_photometry: Dict[str, object] = {"locked": False}

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._buf: deque = deque(maxlen=self.buffer_size)

        self._build_pipeline()

    def _build_pipeline(self):
        """(Re)create pipeline/config/align — needed after hardware_reset()."""
        self.pipeline = rs.pipeline()
        self.cfg = rs.config()
        self.cfg.enable_device(self.serial)

        if self.use_color:
            self.cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.use_depth:
            self.cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        self.align = (
            rs.align(rs.stream.color)
            if (self.use_depth and self.align_depth_to_color and self.use_color)
            else None
        )

    def _color_sensor(self):
        """The RGB sensor of this device, or None (depth-only run / query failed)."""
        try:
            dev = self.pipeline.get_active_profile().get_device()
        except Exception:
            return None
        for sensor in dev.query_sensors():
            try:
                name = sensor.get_info(rs.camera_info.name)
            except Exception:
                continue
            if "RGB" in name or "Color" in name:
                return sensor
        return None

    def _lock_color_photometry(self):
        """Freeze color exposure / gain / white balance for the whole session.

        RealSense defaults to auto-exposure, which re-converges every time the
        robot moves and changes what the camera sees. Calibration needs the
        opposite: one photometric setting held constant across every capture, so
        that corner subpixel accuracy does not vary frame to frame and a dim view
        cannot stretch the exposure into motion blur.

        Values are not hardcoded — a good exposure depends on the room. Instead
        auto-exposure runs during warmup, and whatever it converged to is read
        back and locked. Explicit constructor values override that per option.

        Never raises: an unlockable camera is worth a warning, not a dead run.
        The outcome lands in ``self.color_photometry`` either way.
        """
        sensor = self._color_sensor()
        if sensor is None:
            self.color_photometry = {"locked": False, "reason": "no color sensor"}
            return

        locked, failed = {}, {}
        # (auto option, value option, explicit override) — auto is disabled first
        # so the subsequent set_option is not immediately overwritten by AE/AWB.
        plan = (
            (rs.option.enable_auto_exposure, rs.option.exposure, self.color_exposure_us, "exposure_us"),
            (None, rs.option.gain, self.color_gain, "gain"),
            (rs.option.enable_auto_white_balance, rs.option.white_balance, self.color_white_balance, "white_balance"),
        )
        for auto_opt, val_opt, override, label in plan:
            try:
                if not sensor.supports(val_opt):
                    continue
                # Read the converged value before switching auto off.
                value = float(sensor.get_option(val_opt)) if override is None else float(override)
                if auto_opt is not None and sensor.supports(auto_opt):
                    sensor.set_option(auto_opt, 0)
                sensor.set_option(val_opt, value)
                locked[label] = float(sensor.get_option(val_opt))
            except Exception as exc:
                failed[label] = "{}: {}".format(type(exc).__name__, exc)

        self.color_photometry = {
            "locked": bool(locked) and not failed,
            "values": locked,
            "source": "explicit" if self.color_exposure_us is not None else "auto_converged",
        }
        if failed:
            self.color_photometry["failed"] = failed
            print(f"[WARN] cam {self.serial}: could not lock {list(failed)}; "
                  f"those stay on auto and will drift during the session.")
        if locked:
            desc = "  ".join(f"{k}={v:g}" for k, v in sorted(locked.items()))
            print(f"[INFO] cam {self.serial}: color locked  {desc}")

    def _hardware_reset(self, wait_s: float = 6.0):
        """Hardware-reset device by serial and wait for USB re-enumeration."""
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.get_info(rs.camera_info.serial_number) == self.serial:
                    dev.hardware_reset()
                    break
        except Exception:
            pass

        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                serials = [
                    d.get_info(rs.camera_info.serial_number)
                    for d in rs.context().query_devices()
                ]
                if self.serial in serials:
                    time.sleep(0.8)  # extra settle time after re-enumeration
                    return
            except Exception:
                continue

    @staticmethod
    def list_devices() -> Dict[str, str]:
        """Return {serial: name} for all connected RealSense devices."""
        ctx = rs.context()
        out = {}
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            name = dev.get_info(rs.camera_info.name)
            out[serial] = name
        return out

    @staticmethod
    def reset_all_devices(wait_s: float = 6.0) -> None:
        """Hardware-reset every connected RealSense and wait for USB re-enumeration.

        Call once at process startup before opening any pipeline. Required after
        a prior crash/segfault left a device in a bad state — RealSense Viewer
        masks this by resetting on open, but pyrealsense2 pipelines do not.
        """
        try:
            ctx = rs.context()
            devs_before = list(ctx.query_devices())
            serials_before = [
                d.get_info(rs.camera_info.serial_number) for d in devs_before
            ]
            for dev in devs_before:
                try:
                    dev.hardware_reset()
                except Exception:
                    pass
            print(f"[INFO] Hardware-reset {len(serials_before)} RealSense device(s); "
                  f"waiting for re-enumeration...")
        except Exception as e:
            print(f"[WARN] reset_all_devices: enumerate/reset failed: {e}")
            return

        deadline = time.time() + wait_s
        while time.time() < deadline:
            time.sleep(0.5)
            try:
                serials_now = [
                    d.get_info(rs.camera_info.serial_number)
                    for d in rs.context().query_devices()
                ]
                if all(s in serials_now for s in serials_before):
                    time.sleep(1.0)  # extra settle time for stereo modules (D435)
                    print(f"[INFO] All {len(serials_before)} device(s) re-enumerated.")
                    return
            except Exception:
                continue
        print("[WARN] reset_all_devices: timeout waiting for re-enumeration; "
              "continuing anyway.")

    def start(self, max_attempts: int = 3, warmup_timeout_ms: int = 10000):
        last_err = None
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    print(f"[WARN] cam {self.serial}: retrying after hardware reset "
                          f"(attempt {attempt + 1}/{max_attempts})")
                    self._hardware_reset()
                    self._build_pipeline()
                self.pipeline.start(self.cfg)
                for _ in range(self.warmup_frames):
                    self.pipeline.wait_for_frames(timeout_ms=warmup_timeout_ms)
                if self.use_color and self.lock_color_exposure:
                    # Auto-exposure needs more than the warmup frames to converge;
                    # locking too early freezes a mid-ramp value. Skipped when the
                    # caller supplied an explicit exposure — nothing to converge to.
                    if self.color_exposure_us is None:
                        for _ in range(self.ae_settle_frames):
                            self.pipeline.wait_for_frames(timeout_ms=warmup_timeout_ms)
                    self._lock_color_photometry()
                self._running = True
                self._thread = threading.Thread(target=self._loop, daemon=True)
                self._thread.start()
                return
            except RuntimeError as e:
                last_err = e
                print(f"[WARN] cam {self.serial}: start failed "
                      f"(attempt {attempt + 1}/{max_attempts}): {e}")
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
        raise RuntimeError(
            f"Camera {self.serial} failed to start after {max_attempts} attempts "
            f"(last error: {last_err}). Try unplugging/replugging USB or check "
            f"`rs-enumerate-devices` output."
        )

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _loop(self):
        while self._running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                if self.align is not None:
                    frames = self.align.process(frames)

                color = frames.get_color_frame() if self.use_color else None
                depth = frames.get_depth_frame() if self.use_depth else None

                if self.use_color and color is None:
                    continue

                device_ts_ms = None
                timestamp_domain = None
                if color is not None:
                    device_ts_ms = float(color.get_timestamp())
                    try:
                        timestamp_domain = str(color.get_frame_timestamp_domain())
                    except Exception:
                        timestamp_domain = None
                elif depth is not None:
                    device_ts_ms = float(depth.get_timestamp())
                    try:
                        timestamp_domain = str(depth.get_frame_timestamp_domain())
                    except Exception:
                        timestamp_domain = None

                # Comparable across all camera threads in this process. Independent
                # RealSense hardware clocks can differ by minutes after reset.
                host_ts_ms = time.monotonic() * 1000.0

                color_arr = None if color is None else np.asanyarray(color.get_data()).copy()
                depth_arr = None if depth is None else np.asanyarray(depth.get_data()).copy()

                with self._lock:
                    self._buf.append(
                        (host_ts_ms, device_ts_ms, timestamp_domain, color_arr, depth_arr)
                    )
            except Exception:
                time.sleep(0.005)

    def get_latest(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Return color/depth and comparable host-monotonic receipt time in ms."""
        with self._lock:
            if not self._buf:
                return None, None, None
            host_ts, _device_ts, _domain, c, d = self._buf[-1]
        return (None if c is None else c.copy()), (None if d is None else d.copy()), host_ts

    def get_latest_with_timestamps(self):
        """Return color/depth plus host and device clock diagnostics."""
        with self._lock:
            if not self._buf:
                return None, None, None, None, None
            host_ts, device_ts, domain, c, d = self._buf[-1]
        return (
            None if c is None else c.copy(),
            None if d is None else d.copy(),
            host_ts,
            device_ts,
            domain,
        )

    def get_at(self, target_ts_ms: float
               ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        """Return the buffered frame whose host timestamp is closest to `target_ts_ms`.

        Used for software-synchronized multi-camera capture: pick a reference
        timestamp across cameras and have each camera return its closest frame.
        """
        with self._lock:
            if not self._buf:
                return None, None, None
            best = min(
                self._buf,
                key=lambda x: abs(x[0] - target_ts_ms) if x[0] is not None else float("inf"),
            )
            host_ts, _device_ts, _domain, c, d = best
        return (None if c is None else c.copy()), (None if d is None else d.copy()), host_ts

    def get_at_with_timestamps(self, target_ts_ms: float):
        """Like :meth:`get_at`, also returning device timestamp/domain for audit."""
        with self._lock:
            if not self._buf:
                return None, None, None, None, None
            best = min(
                self._buf,
                key=lambda x: abs(x[0] - target_ts_ms) if x[0] is not None else float("inf"),
            )
            host_ts, device_ts, domain, c, d = best
        return (
            None if c is None else c.copy(),
            None if d is None else d.copy(),
            host_ts,
            device_ts,
            domain,
        )
