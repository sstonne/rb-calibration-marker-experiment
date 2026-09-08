#!/usr/bin/env python3
"""GELLO(Dynamixel, ID 1~7)로 실제 ZEUS(i611) 로봇을 관절각 미러링으로 조종한다.

ur3_test/control/gello_ur3_real_teleop.py 와 같은 "joint twin" 미러링 방식
(GELLO 관절각 그대로 따라가기, Cartesian FK/IK 불필요)이지만, ZEUS 쪽 제어
메커니즘이 UR3와 근본적으로 달라서 그대로 못 옮긴다:

  - UR3: rtde_control.speedJ()로 관절 "속도"를 계속 흘려보냄 (non-blocking,
    30Hz로 계속 새 값 덮어써도 부드럽게 이어짐).
  - ZEUS: server/zeus_server.py의 movej/movel은 기본적으로 i611 SDK의
    rb.move()/rb.line()을 그대로 부르는 "절대 목표 지점" 명령이고 **blocking**
    이다 (server/c1.py의 verify_robot_still()이 이걸 전제로 함).

    다만 i611 SDK(로컬에 있는 i611_MCS.py 확인 결과)에는 실제로 논블로킹
    메커니즘이 있다 - rb.asyncm(1)("program prefetching" ON)을 켜면 이후의
    move()/line() 호출은 물리적으로 다 멈출 때까지가 아니라 "명령이 큐에
    들어가는 즉시"(대략 직전 명령이 overlap 블렌드 구간에 진입하는 시점)
    리턴한다 - i611_MCS.py의 asyncm()/join() 문서가 그렇게 설명한다. 이게
    zeus_server.py에 새로 추가한 stream_start/stream_stop op이다:
      - stream_start: rb.asyncm(1) - 이후 movej/movel이 "큐잉되면 즉시 리턴"
      - stream_stop:  rb.join() 으로 큐 flush 후 rb.asyncm(2)로 원복
    (기본 상태, stream_start 안 부르면 movej/movel은 여전히 완전 blocking -
    c1.py 등 다른 스크립트가 그 전제에 의존하므로 기본값은 안 건드림.)

    이 스크립트는 ENTER로 연결되는 순간 stream_start를 켜고, 그 뒤로는 별도
    스레드가 "직전 movej 호출이 리턴하는 즉시 그 시점의 최신 GELLO 목표로
    다시 movej를 부르는" 루프를 돈다 - asyncm(1) 덕분에 이 리턴이 "물리적으로
    완전히 도착"이 아니라 "큐에 들어감"이라 UR3의 30Hz 스트리밍에 훨씬
    가까워질 것으로 기대한다. --overlap(신규 movej 파라미터, mm 단위 명목)을
    같이 줘서 구간 사이에 완전정지 없이 이어붙이게 한다.

    **다만 정확한 큐 깊이/백프레셔 동작, movej(관절 PTP)에서도 overlap이
    movel과 똑같이 블렌딩되는지, 그리고 stream_start 상태에서 stop(motion_skip)
    이 큐에 쌓인 동작을 실제로 즉시 멈추는지는 전부 미검증이다** (i611_extend.py
    가 로컬에 없어 motion_skip 구현을 못 봤다). 처음 테스트 절차를 반드시
    지킬 것:
      1. --dry-run으로 로직만 확인
      2. 실기, --overlap 0 (스트리밍만 켜고 블렌딩 없음)으로 저속 테스트 -
         이것만으로도 예전보다 나아지는지 확인 (asyncm(1)의 순수 효과)
      3. --overlap을 조금씩 올리며 비교
      4. GELLO를 뽑아서 watchdog이 실제로 로봇을 즉시 세우는지 저속에서 확인
         (stream_start 상태에서 stop이 안 먹으면 큐에 남은 동작이 계속 실행될
         위험이 있음 - 비상정지 펜던트에 항상 손이 닿는 상태로 할 것)

    그리퍼는 ZEUS가 공압 on/off라 GELLO 그리퍼 델타를 열림/닫힘 이진값으로
    문턱값 처리해서, 상태가 바뀔 때만 grip 명령을 보낸다 (grip은 asyncm 큐와
    무관한 별도 I/O라 여전히 blocking - 빈번하게 부르면 관절 추종이 멈춘다).

사용법 (ENTER로 연결/해제 토글) - gello_ur3_real_teleop.py와 동일한 흐름:
  1. 실행하면 대기 상태로 시작 - 로봇에 어떤 명령도 안 나감
  2. GELLO를 지금 ZEUS가 서있는 자세랑 최대한 비슷하게 손으로 맞춰놓기
  3. ENTER - 그 순간 GELLO 자세 = ZEUS 현재 관절각으로 매핑을 확정(latch)하고 추종 시작
  4. 다시 ENTER - 정지(stop, motion_skip), 로봇은 마지막 자세에서 멈춤
  5. Ctrl+C - 완전 종료

--control-mode (2026-09-04 실기 테스트 결과로 추가):
  실측 결과 GELLO ID1~6 <-> ZEUS 관절 인덱스는 항등(순서 그대로)이 맞고,
  1~3번(어깨/팔꿈치)은 2번만 부호가 반대였다(--joint-sign 기본값에 이미 반영).
  반면 4~6번(손목)은 GELLO와 ZEUS의 관절 축 배치 자체가 달라서 조인트
  미러링으로 방향을 맞다고 검증할 방법이 없었다. 그래서:
    - joint(기본): 여전히 6축 전부 조인트 미러링 (movej) - 손목 방향은 안 맞을
      수 있음을 알고 쓸 것.
    - position: 4~6번(손목) 입력을 아예 버리고, 1~3번 delta로 GELLO의 표준
      UR3 DH FK를 계산해 XYZ만 movel로 움직인다. 방향(rz,ry,rx)은 ENTER 순간
      값에 고정 - 손목을 아예 못 믿으니 안 건드리는 쪽을 택함. GELLO FK 결과의
      x/y/z 축이 ZEUS의 x/y/z와 그대로 대응한다는 보장은 없어서(둘의 base
      frame이 실제로 정렬돼 있어야 함) --xyz-map/--xyz-sign으로 축 순서/부호를
      다시 맞출 수 있게 했다 - 전부 미검증 항등 기본값이니 저속으로 GELLO 팔을
      움직여보며 ZEUS가 의도한 축으로 가는지 확인 후 조정할 것. GELLO 링크
      길이가 표준 UR3 DH 추정치(위 상수)와 다르면 --position-scale로 배율
      보정.

안전장치:
  - 기준(latch) 대비 --max-joint-delta-deg 넘게 벗어나는 목표는 클램프
  - GELLO 통신이 --watchdog-timeout 이상 끊기면 자동으로 정지하고 연결 해제
  - --dry-run: 실제 movej/movel/grip 명령 없이 로그만 출력 (첫 테스트는 반드시 이걸로)

실행 전 로봇 주변 충돌 가능성을 확인하고, 티치펜던트/비상정지에 손이 닿는
상태에서 사용할 것. --jnt-speed/--overlap 기본값(10, 20)은 2026-09-07
실기 테스트로 정한 값이니, 처음 보는 환경/로봇이면 다시 낮춰서 시작할 것.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import dynamixel_sdk as dxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.backends.zeus_client import ZeusClient, ZeusError  # noqa: E402

ARM_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_ID = 7
ALL_IDS = ARM_IDS + [GRIPPER_ID]

ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
TICKS_PER_REV = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_REV

GELLO_POLL_HZ = 50.0
GELLO_POLL_DT = 1.0 / GELLO_POLL_HZ

# server/zeus_server.py의 read_gripper()/GRIP_*_SENSE와 동일한 배선 규약
# ([d,c,b,a] 순서). get_state()['gripper']를 해석해 ENTER 연결 시 GELLO
# 트리거를 안 맞춰도(정확히는 살짝 안 맞아도) 실제 그리퍼 상태부터 시작하게
# 하기 위함 - UR3(Robotiq)는 이 센서 판독이 없어서 못 하던 것.
GRIP_OPEN_SENSE = ["0", "1", "0", "0"]
GRIP_CLOSE_SENSE = ["0", "0", "0", "1"]


def sensed_gripper_state(gripper_bits: list[str]) -> str | None:
    """[d,c,b,a] 센서 비트 -> 'open'/'close'/None(전이 중이라 판별 불가)."""
    if gripper_bits == GRIP_OPEN_SENSE:
        return "open"
    if gripper_bits == GRIP_CLOSE_SENSE:
        return "close"
    return None


# ── --control-mode position 용: GELLO(UR twin) 팔(1~3번, 어깨/팔꿈치)의
# 위치(XYZ)만 계산하는 FK ──
# 실측 결과 GELLO ID1~6 <-> ZEUS 관절 인덱스는 그대로(항등) 맞고 2번만 부호가
# 반대였다. 반면 4/5/6(손목)은 GELLO와 ZEUS의 관절 축 배치 자체가 달라서
# 조인트 미러링으로 검증이 안 됐다 - 그래서 이 모드는 손목 입력을 아예 안 쓰고
# (항상 delta=0으로 고정), ZEUS 방향(rz,ry,rx)도 latch 값에 고정한 채 XYZ
# 위치만 GELLO 1~3번 관절의 표준 UR DH FK로 계산해서 movel로 보낸다.
#
# **표준 UR3 DH 파라미터(alpha/a/d, mm)를 썼다 - GELLO가 정확히 어느 UR 모델의
# 축소 복제인지, 그리고 이 수치가 실물과 맞는지는 미검증이다. 위치 스케일이
# 안 맞으면 --position-scale로 보정하고, 방향 자체가 이상하면(예: GELLO를
# 앞으로 미는데 ZEUS가 옆으로 감) --xyz-map/--xyz-sign으로 축을 재배치할 것.**
_UR_ALPHA = [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
_UR_A_MM = [0.0, -243.65, -213.25, 0.0, 0.0, 0.0]
_UR_D_MM = [151.9, 0.0, 0.0, 112.35, 85.35, 81.9]


def _rz4(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _rx4(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


def _tz4(d: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = d
    return T


def _tx4(a: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = a
    return T


def gello_fk(theta_rad: np.ndarray) -> np.ndarray:
    """관절각(rad, 6개) -> 4x4 FK. 위치 단위는 mm (a/d를 mm로 줬으므로)."""
    T = np.eye(4)
    for i in range(6):
        T = T @ _rz4(theta_rad[i]) @ _tz4(_UR_D_MM[i]) @ _tx4(_UR_A_MM[i]) @ _rx4(_UR_ALPHA[i])
    return T


GELLO_ZERO_POS_MM = gello_fk(np.zeros(6))[:3, 3]


def gello_position_delta_mm(dq_arm_deg: np.ndarray) -> np.ndarray:
    """dq_arm(latch 기준, ZEUS 인덱스/부호로 이미 보정된 관절 delta, deg) 중
    1~3번(어깨/팔꿈치)만 써서 GELLO 팔이 그만큼 움직였을 때의 XYZ 위치 변화량(mm)을
    구한다. 4~6번(손목)은 축 배치가 달라 못 믿으므로 항상 0으로 고정."""
    theta = np.zeros(6)
    theta[:3] = np.radians(dq_arm_deg[:3])
    p = gello_fk(theta)[:3, 3]
    return p - GELLO_ZERO_POS_MM

MAX_STEP_TICKS_DEFAULT = 400
MAX_STEP_TICKS_GRIP = 300
FILTER_ALPHA_DEFAULT = 0.7
FILTER_ALPHA_GRIP = 0.6


def step_limit_for(dxl_id: int) -> int:
    return MAX_STEP_TICKS_GRIP if dxl_id == GRIPPER_ID else MAX_STEP_TICKS_DEFAULT


def alpha_for(dxl_id: int) -> float:
    return FILTER_ALPHA_GRIP if dxl_id == GRIPPER_ID else FILTER_ALPHA_DEFAULT


def wrapped_tick_delta(cur: int, ref: int) -> int:
    """cur-ref를 (-2048, 2048] 범위로 감아서(최단 경로) 반환한다."""
    d = cur - ref
    return ((d + TICKS_PER_REV // 2) % TICKS_PER_REV) - TICKS_PER_REV // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=57600)
    parser.add_argument("--robot-ip", default="192.168.0.23")
    parser.add_argument("--robot-port", type=int, default=12350)
    parser.add_argument("--control-mode", choices=["joint", "position"], default="joint",
                        help="joint(기본): 관절각 그대로 미러링, movej. "
                             "position: 손목(4~6번) 입력은 무시하고, 1~3번(어깨/팔꿈치) "
                             "관절 delta로 GELLO FK를 계산해 XYZ만 움직이고 방향(rz,ry,rx)은 "
                             "latch 값에 고정한 채 movel로 보냄 (4~6번은 GELLO/ZEUS 축 배치가 "
                             "달라 조인트 미러링 검증이 안 돼서 실기 테스트 결과 이걸로 전환)")
    parser.add_argument("--lin-speed", type=float, default=20.0,
                        help="position 모드 movel의 lin_speed(mm/s). 낮게 시작할 것")
    parser.add_argument("--pose-speed", type=float, default=20.0,
                        help="position 모드 movel의 회전 속도(%%, 서버 상한 50) - 방향은 "
                             "안 바뀌므로 거의 안 쓰이지만 서버가 필수로 받음")
    parser.add_argument("--xyz-map", default="0,1,2",
                        help="GELLO FK 위치(x,y,z) -> ZEUS 위치(x,y,z) 축 매핑 (미검증 "
                             "항등 기본값). GELLO를 특정 방향으로 밀었는데 ZEUS가 다른 "
                             "축으로 움직이면 조정 (예: GELLO가 Y로 움직였는데 ZEUS의 X가 "
                             "바뀌면 이 축을 앞으로)")
    parser.add_argument("--xyz-sign", default="1,1,1",
                        help="ZEUS 위치 x,y,z 각각의 부호 (미검증 항등 기본값)")
    parser.add_argument("--position-scale", type=float, default=1.0,
                        help="GELLO FK 위치 delta(mm)에 곱하는 배율 - GELLO 링크 길이가 "
                             "표준 UR3 DH 추정치와 달라 스케일이 안 맞으면 조정")
    parser.add_argument("--joint-map", default="0,1,2,3,4,5",
                        help="GELLO ID1~6 -> ZEUS joint index 매핑 (실기 확인: 항등이 맞음 - "
                             "2026-09-04 로그로 6축 전부 검증됨)")
    parser.add_argument("--joint-sign", default="1,-1,1,1,1,1",
                        help="팔 관절 6축 방향 부호, 쉼표구분 6개 (실기 확인: 2번만 반대 - "
                             "2026-09-04 확인. joint 모드에서만 쓰임; position 모드도 1~3번 "
                             "입력엔 이 부호가 그대로 적용됨)")
    parser.add_argument("--grip-sign", type=float, default=-1.0)
    parser.add_argument("--grip-ticks-to-full", type=float, default=570.0,
                        help="GELLO 그리퍼 모터가 이만큼 틱 움직이면 완전히 닫힘 취급")
    parser.add_argument("--grip-close-threshold-pct", type=float, default=50.0,
                        help="이 %% 넘게 닫히면 ZEUS 그리퍼에 close 명령 (ZEUS는 on/off라 "
                             "비례 제어 불가, 문턱값으로 이진화)")
    parser.add_argument("--grip-timeout", type=float, default=2.0,
                        help="grip 명령의 blocking 대기 상한(초) - 이 동안 관절 추종이 멈춘다")
    parser.add_argument("--jnt-speed", type=float, default=10.0,
                        help="movej jnt_speed (i611 단위, 서버 상한 30). 느리면 낮춰서 시작할 것")
    parser.add_argument("--overlap", type=float, default=20.0,
                        help="movej 블렌딩 파라미터(mm 단위 명목, 서버 상한 20 - 기본값이 "
                             "이미 상한값). movej에서 실제로 얼마나 블렌딩되는지는 여전히 "
                             "체감으로 확인해가며 조정할 것")
    parser.add_argument("--acc", type=float, default=0.3, help="acctime/dacctime (초)")
    parser.add_argument("--max-joint-delta-deg", type=float, default=45.0,
                        help="기준 자세 대비 이 각도(deg) 넘게 벗어나는 목표는 클램프 "
                             "(하드웨어 글리치/오조작 방어, UR3보다 훨씬 보수적인 기본값)")
    parser.add_argument("--watchdog-timeout", type=float, default=0.3,
                        help="GELLO 데이터가 이 시간(초) 이상 안 갱신되면 자동 정지+연결해제")
    parser.add_argument("--status-interval", type=float, default=1.0,
                        help="[status] 로그 출력 주기(초) - 축 매핑/부호 캘리브레이션할 땐 "
                             "0.3 정도로 줄이면 로그에서 축별 타이밍 구분이 쉬워짐")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 movej/grip 명령을 보내지 않고 로그만 출력 (첫 테스트 필수)")
    parser.add_argument("--disable-gripper", action="store_true",
                        help="GELLO 그리퍼 트리거를 완전히 무시하고 grip 명령을 전혀 보내지 않음 "
                             "(캘리브레이션 촬영 등 팔만 움직이고 싶을 때, 트리거 오조작 방지용)")
    return parser.parse_args()


@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    enabled: bool = False
    shutdown: threading.Event = field(default_factory=threading.Event)
    dq_arm_deg: np.ndarray = field(default_factory=lambda: np.zeros(6))
    dq_leader_raw_deg: np.ndarray = field(default_factory=lambda: np.zeros(6))
    grip_pct: float = 0.0
    last_update_time: float = 0.0
    n_glitch: int = 0


def read_all_positions(group_read) -> dict[int, int]:
    group_read.txRxPacket()
    positions = {}
    for dxl_id in ALL_IDS:
        if group_read.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
            raw = group_read.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
            if raw >= 2**31:
                raw -= 2**32
            positions[dxl_id] = raw
    return positions


def gello_reader_worker(state: SharedState, args: argparse.Namespace) -> None:
    """GELLO를 --port GELLO_POLL_HZ로 계속 읽어 state.dq_arm_deg / grip_pct를 갱신한다.

    ZEUS로 보내는 속도와 무관하게 독립적으로 빠르게 돈다 - control_worker가
    movej 응답을 기다리는 동안에도 최신 GELLO 목표가 계속 최신으로 유지되게
    하기 위함 (응답이 오는 즉시 그 순간 최신값을 바로 보낼 수 있도록).
    """
    port_handler = dxl.PortHandler(args.port)
    if not port_handler.openPort() or not port_handler.setBaudRate(args.baudrate):
        print(f"[error] failed to open {args.port} @ {args.baudrate}")
        state.shutdown.set()
        return
    packet_handler = dxl.PacketHandler(2.0)
    for dxl_id in ALL_IDS:
        packet_handler.write1ByteTxRx(port_handler, dxl_id, ADDR_TORQUE_ENABLE, 0)
    group_read = dxl.GroupSyncRead(port_handler, packet_handler,
                                    ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
    for dxl_id in ALL_IDS:
        if not group_read.addParam(dxl_id):
            print(f"[error] GroupSyncRead addParam failed for id={dxl_id}")
            state.shutdown.set()
            return

    joint_map = [int(x) for x in args.joint_map.split(",")]
    joint_sign = [float(x) for x in args.joint_sign.split(",")]
    assert len(joint_map) == 6 and len(joint_sign) == 6

    last_good: dict[int, int] = {}
    filtered_delta: dict[int, float] = {}
    ref_ticks: dict[int, int] | None = None
    was_enabled = False
    n_read_fail = 0
    last_fail_print = 0.0

    try:
        while not state.shutdown.is_set():
            loop_start = time.monotonic()
            positions = read_all_positions(group_read)

            with state.lock:
                enabled = state.enabled

            if len(positions) < len(ALL_IDS):
                n_read_fail += 1
                if loop_start - last_fail_print > 1.0:
                    missing = set(ALL_IDS) - set(positions)
                    print(f"[leader] read fail (missing ids={missing}, total={n_read_fail})")
                    last_fail_print = loop_start
                time.sleep(max(0.0, GELLO_POLL_DT - (time.monotonic() - loop_start)))
                continue

            for dxl_id in ALL_IDS:
                step_limit = step_limit_for(dxl_id)
                prev = last_good.get(dxl_id)
                if prev is not None and abs(wrapped_tick_delta(positions[dxl_id], prev)) > step_limit:
                    with state.lock:
                        state.n_glitch += 1
                    positions[dxl_id] = prev
                else:
                    last_good[dxl_id] = positions[dxl_id]

            if enabled and not was_enabled:
                ref_ticks = dict(positions)
                filtered_delta = {dxl_id: 0.0 for dxl_id in ALL_IDS}
                print("\n[bridge] CONNECTED - GELLO 기준 latch")
            was_enabled = enabled

            if enabled and ref_ticks is not None:
                for dxl_id in ALL_IDS:
                    alpha = alpha_for(dxl_id)
                    raw_delta = wrapped_tick_delta(positions[dxl_id], ref_ticks[dxl_id])
                    prev_f = filtered_delta.get(dxl_id, raw_delta)
                    filtered_delta[dxl_id] = alpha * raw_delta + (1.0 - alpha) * prev_f

                dq_leader = np.array([filtered_delta[ARM_IDS[i]] * DEG_PER_TICK
                                      for i in range(6)])
                dq_arm = np.zeros(6)
                for zeus_i, gello_i in enumerate(joint_map):
                    dq_arm[zeus_i] = dq_leader[gello_i] * joint_sign[zeus_i]

                grip_delta_ticks = filtered_delta[GRIPPER_ID] * args.grip_sign
                grip_pct = float(np.clip(grip_delta_ticks / args.grip_ticks_to_full, 0.0, 1.0)) * 100.0

                with state.lock:
                    state.dq_arm_deg = dq_arm
                    state.dq_leader_raw_deg = dq_leader  # GELLO ID1~6 원본, map/sign 적용 전
                    state.grip_pct = grip_pct
                    state.last_update_time = loop_start

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, GELLO_POLL_DT - elapsed))
    finally:
        port_handler.closePort()


def control_worker(state: SharedState, args: argparse.Namespace) -> None:
    """ZEUS로 실제 명령을 보내는 단일 스레드.

    zeus_server는 연결을 하나만 받고 요청/응답이 1:1 blocking이라 (movej도
    grip도), 모든 로봇 명령은 반드시 이 스레드 하나에서만 순차적으로 나가야
    한다 - 병렬로 다른 스레드가 같은 ZeusClient를 건드리면 프로토콜이 깨진다.
    """
    max_delta = args.max_joint_delta_deg
    xyz_map = [int(x) for x in args.xyz_map.split(",")]
    xyz_sign = np.array([float(x) for x in args.xyz_sign.split(",")])
    assert len(xyz_map) == 3 and len(xyz_sign) == 3

    zeus: ZeusClient | None = None
    if not args.dry_run:
        zeus = ZeusClient(args.robot_ip, args.robot_port)
        zeus.connect()
        print(f"[zeus] connected to {args.robot_ip}:{args.robot_port}")
    else:
        print("[zeus] dry-run: 실제 연결 생략")
    if args.disable_gripper:
        print("[bridge] --disable-gripper: GELLO 그리퍼 트리거 무시, grip 명령 안 보냄")

    zeus_ref_joints: np.ndarray | None = None
    zeus_ref_pose6: np.ndarray | None = None
    gripper_state = "open"  # 마지막으로 ZEUS에 실제로 보낸 그리퍼 상태
    was_enabled = False
    n_sent = 0
    n_sent_at_last_print = 0
    last_status_print = 0.0
    last_send_time: float | None = None

    try:
        while not state.shutdown.is_set():
            with state.lock:
                enabled = state.enabled
                dq_arm = state.dq_arm_deg.copy()
                dq_leader_raw = state.dq_leader_raw_deg.copy()
                grip_pct = state.grip_pct
                last_update_time = state.last_update_time

            watchdog_tripped = (enabled and was_enabled
                                and (time.monotonic() - last_update_time) > args.watchdog_timeout)
            if watchdog_tripped:
                print("\n[bridge] WATCHDOG: leader data stalled, forcing disable")
                with state.lock:
                    state.enabled = False
                enabled = False
                if zeus is not None:
                    try:
                        zeus.stop()
                    except ZeusError as exc:
                        print(f"[zeus] stop failed: {exc}")

            if enabled and not was_enabled:
                if args.dry_run:
                    zeus_ref_joints = np.zeros(6)
                    zeus_ref_pose6 = np.zeros(6)
                    gripper_state = "open" if grip_pct < args.grip_close_threshold_pct else "close"
                else:
                    zeus_state = zeus.get_state()
                    zeus_ref_joints = np.array(zeus_state["joints"])
                    zeus_ref_pose6 = np.array(zeus_state["pose"])
                    # GELLO 트리거를 정확히 못 맞췄어도 실제 센서 상태부터 시작한다
                    # (0%=open으로 무조건 가정하던 UR3/Robotiq 방식보다 안전 -
                    # 여긴 그리퍼 열림/닫힘 센서가 있어서 참값을 알 수 있다).
                    sensed = sensed_gripper_state(zeus_state["gripper"])
                    if sensed is None:
                        gripper_state = "open" if grip_pct < args.grip_close_threshold_pct else "close"
                        print(f"[bridge] WARNING: 그리퍼 센서가 열림/닫힘 둘 다 아님 "
                              f"(전이 중? bits={zeus_state['gripper']}) - GELLO 문턱값으로 추정: "
                              f"{gripper_state}")
                    else:
                        gripper_state = sensed
                    try:
                        zeus.stream_start()
                    except ZeusError as exc:
                        print(f"[zeus] stream_start failed, falling back to blocking "
                              f"movej: {exc}")
                print(f"[bridge] ZEUS 기준 관절각(deg)={np.round(zeus_ref_joints, 1)} "
                      f"gripper={gripper_state}")
                if (grip_pct < args.grip_close_threshold_pct) != (gripper_state == "open"):
                    print("[bridge] WARNING: GELLO 그리퍼 트리거가 실제 ZEUS 그리퍼 상태랑 "
                          "안 맞는 것 같음 - 다음에 GELLO 트리거를 살짝만 움직여도 즉시 "
                          "grip 명령이 나갈 수 있음 (의도한 게 아니면 트리거를 반대쪽으로 "
                          "맞춘 뒤 재연결 권장)")
            elif not enabled and was_enabled:
                if zeus is not None:
                    try:
                        zeus.stop()
                    except ZeusError as exc:
                        print(f"[zeus] stop failed: {exc}")
                    try:
                        zeus.stream_stop()
                    except ZeusError as exc:
                        print(f"[zeus] stream_stop failed: {exc}")
                print("\n[bridge] DISCONNECTED - 정지")
            was_enabled = enabled

            if not enabled or zeus_ref_joints is None:
                time.sleep(0.02)
                continue

            dq_arm = np.clip(dq_arm, -max_delta, max_delta)

            if not args.disable_gripper:
                want_state = "open" if grip_pct < args.grip_close_threshold_pct else "close"

                if want_state != gripper_state:
                    if args.dry_run:
                        print(f"[dry-run gripper] {want_state} (grip={grip_pct:.0f}%)")
                    else:
                        try:
                            zeus.grip(want_state, timeout_s=args.grip_timeout)
                        except ZeusError as exc:
                            print(f"[zeus] grip failed: {exc}")
                    gripper_state = want_state
                    # 그리퍼 명령도 blocking이라 이번 사이클은 여기서 끝내고 다음
                    # 루프에서 최신 GELLO 목표로 관절 이동을 재개한다.
                    continue

            if args.control_mode == "position":
                p_gello_mm = gello_position_delta_mm(dq_arm)  # 1~3번만 반영, 4~6번 무시
                dp_zeus_mm = xyz_sign * p_gello_mm[xyz_map] * args.position_scale
                target_xyz = zeus_ref_pose6[:3] + dp_zeus_mm
                target_pose6 = np.concatenate([target_xyz, zeus_ref_pose6[3:6]])  # 방향 고정
                log_target = np.round(target_pose6, 1)
            else:
                target_joint = zeus_ref_joints + dq_arm
                log_target = np.round(target_joint, 1)

            if args.dry_run:
                kind = "movel(position)" if args.control_mode == "position" else "movej"
                print(f"[dry-run] {kind} target={log_target} grip={gripper_state}")
                time.sleep(0.05)  # dry-run: 실제 blocking 대신 대략적인 페이싱만 흉내
            else:
                try:
                    if args.control_mode == "position":
                        zeus.movel(target_pose6.tolist(), lin_speed=args.lin_speed,
                                  overlap=args.overlap, acc=args.acc,
                                  pose_speed=args.pose_speed)
                    else:
                        zeus.movej(target_joint.tolist(), jnt_speed=args.jnt_speed,
                                  overlap=args.overlap, acc=args.acc)
                except ZeusError as exc:
                    print(f"\n[zeus] move failed, disabling: {exc}")
                    with state.lock:
                        state.enabled = False
                    continue

            n_sent += 1
            now = time.monotonic()
            # 직전 전송~이번 전송 사이 실제 걸린 시간 - 이게 크면(수백ms 이상)
            # movej/movel 호출 자체가 오래 blocking된다는 뜻이고, 반대로 이게
            # 작은데도 "방향 바꾸면 늦게 반응"하는 게 느껴진다면 asyncm(1) 큐에
            # 몇 스텝이 밀려있다는 뜻일 가능성이 높다 (아직 미검증인 부분).
            send_dt_ms = None if last_send_time is None else (now - last_send_time) * 1000.0
            last_send_time = now
            if now - last_status_print > args.status_interval:
                rate_hz = (n_sent - n_sent_at_last_print) / max(1e-6, now - last_status_print)
                send_dt_str = "n/a" if send_dt_ms is None else f"{send_dt_ms:.0f}ms"
                print(f"[status] sent={n_sent} rate={rate_hz:.1f}Hz "
                      f"last_send_dt={send_dt_str} "
                      f"gello_raw(deg,ID1-6)={np.round(dq_leader_raw, 1)} "
                      f"target={log_target} grip={gripper_state}")
                last_status_print = now
                n_sent_at_last_print = n_sent
    except Exception as exc:  # noqa: BLE001
        print(f"\n[CONTROL ERROR] {type(exc).__name__}: {exc}", flush=True)
        with state.lock:
            state.enabled = False
    finally:
        if zeus is not None:
            try:
                zeus.stop()
            except Exception:
                pass
            try:
                zeus.stream_stop()  # no-op-ish if stream_start was never called
            except Exception:
                pass
            try:
                zeus.close()
            except Exception:
                pass


def main() -> int:
    args = parse_args()
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN 모드 - 실제 로봇에 명령 안 보냄")
        print("=" * 60)
    else:
        print("=" * 60)
        print("실제 ZEUS 로봇을 움직입니다. 주변 충돌 가능성을 확인하고")
        print("비상정지에 손이 닿는 상태에서 진행하세요.")
        print("overlap이 movej에서 실제로 블렌딩되는지 미검증입니다 - ")
        print("처음엔 --overlap 0으로 기준 동작을 확인하세요.")
        print("=" * 60)

    state = SharedState()
    reader = threading.Thread(target=gello_reader_worker, args=(state, args),
                              name="gello-zeus-reader", daemon=True)
    controller = threading.Thread(target=control_worker, args=(state, args),
                                  name="gello-zeus-control", daemon=True)
    reader.start()
    controller.start()

    print(f"\n[joint-map] {args.joint_map}  [joint-sign] {args.joint_sign}  "
          f"[grip-sign] {args.grip_sign}")
    print(f"[limits] jnt_speed={args.jnt_speed}, overlap={args.overlap}, "
          f"max_joint_delta={args.max_joint_delta_deg} deg")
    print("ENTER = 연결/해제 토글, Ctrl+C = 종료")
    print("연결 전 체크리스트 (UR3 GELLO 연동과 동일한 이유):")
    print("  1) GELLO 팔을 ZEUS 현재 관절 자세와 최대한 비슷하게 손으로 맞춰놓기")
    print("     (팔은 센서로 확인 불가 - 안 맞추면 연결 즉시 그 오차만큼 순간 이동 시도함)")
    print("  2) GELLO 그리퍼 트리거도 ZEUS 그리퍼의 실제 열림/닫힘 상태와 맞춰놓기")
    print("     (그리퍼는 연결 순간 실제 센서로 자동 보정하지만, 트리거가 반대로 맞춰져 "
          "있으면 그 직후 살짝만 움직여도 바로 grip 명령이 나감 - WARNING 뜨면 확인)")
    print("  3) 위 2개를 맞춘 상태에서 ENTER -> 그 순간 자세로 latch, 추종 시작")

    try:
        while not state.shutdown.is_set():
            input()
            if state.shutdown.is_set():
                break
            with state.lock:
                state.enabled = not state.enabled
                status = "연결" if state.enabled else "해제"
            print(f"[toggle] -> {status}")
    except (KeyboardInterrupt, EOFError):
        print("\n[bridge] stopped by user")
    finally:
        state.shutdown.set()
        reader.join(timeout=3.0)
        controller.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
