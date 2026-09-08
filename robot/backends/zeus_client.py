#!/usr/bin/env python3
"""robot/backends/zeus_client.py — thin TCP client for server/zeus_server.py. PYTHON 3.

zeus_server.py has no task logic; it only executes primitives (ping / get_state /
movel / movej / grip / stop) sent as one JSON object per line over TCP and
replies with one JSON object per line. This module is the client-side half of
that protocol — planning, perception and target computation belong in the
callers (e.g. measure_grasp_accuracy.py), not here.

Pose = [x, y, z, rz, ry, rx] in MM / DEG (i611 Position argument order).
Rotation convention: R = Rz(rz) @ Ry(ry) @ Rx(rx) (extrinsic ZYX), translation
in millimetres. pose6_to_T / T_to_pose6 below implement that convention; it has
been checked against session04 meta captures to <1e-6 deg (see
live_marker_pose.py / grasp_target.py, which use the identical math).

    from robot.backends.zeus_client import ZeusClient

    with ZeusClient("192.168.0.23") as rb:
        state = rb.get_state()                  # {'pose', 'joints', 'gripper'}
        rb.movel([x, y, z, rz, ry, rx], lin_speed=40)
        held = rb.grip("close", timeout_s=3.0)   # see grip() docstring — NOT held=reached
"""
from __future__ import annotations

import json
import socket
from typing import Optional, Sequence

import numpy as np

DEFAULT_PORT = 12350  # server/zeus_server.py PORT


class ZeusError(RuntimeError):
    """A {"ok": false, ...} reply from the server, or a broken connection."""


# ── pose <-> 4x4 (shared convention with grasp_target.py / live_marker_pose.py) ──
def _R(axis: str, t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    if axis == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def pose6_to_T(pose6: Sequence[float]) -> np.ndarray:
    """[x,y,z,rz,ry,rx] (mm,deg) -> 4x4 homogeneous transform (metres)."""
    x, y, z, rz, ry, rx = [float(v) for v in pose6]
    T = np.eye(4)
    T[:3, :3] = _R("Z", np.deg2rad(rz)) @ _R("Y", np.deg2rad(ry)) @ _R("X", np.deg2rad(rx))
    T[:3, 3] = np.array([x, y, z]) / 1000.0
    return T


def T_to_pose6(T: np.ndarray) -> np.ndarray:
    """4x4 (metres) -> [x,y,z,rz,ry,rx] (mm,deg). Inverse of pose6_to_T."""
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    sy = -R[2, 0]
    ry = np.arcsin(np.clip(sy, -1, 1))
    if abs(sy) < 0.9999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.array([T[0, 3] * 1000, T[1, 3] * 1000, T[2, 3] * 1000,
                     np.rad2deg(rz), np.rad2deg(ry), np.rad2deg(rx)])


def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


class ZeusClient:
    """One TCP connection to zeus_server.py. Not thread-safe."""

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout: float = 10.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = ""

    def __enter__(self) -> "ZeusClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect(self) -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        self._sock = s
        self._buf = ""

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._send("bye")
        except Exception:
            pass
        try:
            self._sock.close()
        finally:
            self._sock = None

    def _send(self, op: str, **kwargs) -> dict:
        if self._sock is None:
            raise ZeusError("not connected — call connect() first")
        req = {"op": op}
        req.update(kwargs)
        self._sock.sendall((json.dumps(req) + "\n").encode())
        while "\n" not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ZeusError(f"connection closed by server while waiting for '{op}' reply")
            self._buf += chunk.decode(errors="ignore")
        line, self._buf = self._buf.split("\n", 1)
        reply = json.loads(line)
        if not reply.get("ok", False):
            raise ZeusError(f"{op} failed: {reply.get('err')}")
        return reply

    def ping(self) -> bool:
        return bool(self._send("ping").get("ok"))

    def get_state(self) -> dict:
        """{'pose': [x,y,z,rz,ry,rx] mm/deg, 'joints': [6] deg, 'gripper': [d,c,b,a]}"""
        r = self._send("get_state")
        return {"pose": [float(v) for v in r["pose"]],
                "joints": [float(v) for v in r["joints"]],
                "gripper": r["gripper"]}

    def get_pose6(self) -> np.ndarray:
        return np.asarray(self.get_state()["pose"], dtype=float)

    def get_T_base_tool(self) -> np.ndarray:
        """Current TCP pose (tool-1, see zeus_server.py's settool(1,...,97.5,...)) as 4x4."""
        return pose6_to_T(self.get_pose6())

    def movel(self, pose6: Sequence[float], lin_speed: float = 60.0,
              overlap: float = 0.0, acc: Optional[float] = None,
              pose_speed: Optional[float] = None) -> None:
        kwargs = {"pose": [float(v) for v in pose6], "lin_speed": float(lin_speed),
                  "overlap": float(overlap)}
        if acc is not None:
            kwargs["acc"] = float(acc)
        if pose_speed is not None:
            kwargs["pose_speed"] = float(pose_speed)
        self._send("movel", **kwargs)

    def movej(self, joints: Sequence[float], jnt_speed: float = 10.0,
              overlap: float = 0.0, acc: Optional[float] = None) -> None:
        kwargs = {"joints": [float(v) for v in joints], "jnt_speed": float(jnt_speed),
                  "overlap": float(overlap)}
        if acc is not None:
            kwargs["acc"] = float(acc)
        self._send("movej", **kwargs)

    def grip(self, state: str, timeout_s: float = 3.0) -> bool:
        """state: 'open' | 'close'. Returns the server's `reached` flag verbatim.

        CAUTION, this is easy to read backwards: on 'close', reached=True means the
        fingers closed all the way shut (nothing between them — an EMPTY grasp),
        while reached=False means they stalled against an object before fully
        closing, i.e. something IS held. Use `grip_holds_object()` for the sane
        boolean instead of reading `reached` directly after a close.
        """
        assert state in ("open", "close")
        r = self._send("grip", state=state, timeout_s=float(timeout_s))
        return bool(r["reached"])

    def grip_holds_object(self, timeout_s: float = 3.0) -> bool:
        """Close the gripper and return True iff something is now held (stalled close)."""
        reached = self.grip("close", timeout_s=timeout_s)
        return not reached

    def stream_start(self) -> None:
        """Turn on the i611 SDK's prefetch/queue mode (asyncm(1)) so subsequent
        movel/movej calls return once queued rather than once the robot
        physically stops. Pair with overlap>0. Always follow with
        stream_stop() before disconnecting."""
        self._send("stream_start")

    def stream_stop(self) -> None:
        """Flush the queued motion (join()) and turn prefetch back off
        (asyncm(2)), restoring the default synchronous movel/movej behaviour."""
        self._send("stream_stop")

    def stop(self) -> None:
        self._send("stop")
