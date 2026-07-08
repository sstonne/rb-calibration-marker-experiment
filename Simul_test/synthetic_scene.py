"""
SyntheticScene: 캘리브레이션 알고리즘 검증용 합성 ground-truth 생성기.

핵심 아이디어
-------------
GT 변환(gTc, bTf_i, bTo, 이벤트별 bTg)을 직접 정해두고, 검증된 변환 체인으로
각 카메라가 보는 큐브 포즈 T_C_O 를 유도한다. 노이즈가 0이면 솔버는 GT 를
부동소수점 한계까지 정확히 복원해야 한다.

명명 규약 (목적지-from-출발지, T_A_C = T_A_B @ T_B_C)
- T_C_O    : camera-from-object   (카메라 ← 큐브)   == aruco_cube.rodrigues_to_Rt 의 "Object->Camera"
- bTf[ci]  : base-from-fixedcam_i  (T_base_Ci)
- bTo      : base-from-object      (큐브 앵커, 이벤트 전체에서 고정)
- bTg[e]   : base-from-gripper     (로봇 자세, robot_T)
- gTc      : gripper-from-camera   (핸드아이 미지수)

검증된 체인 (Step3_calibration.py:642, 763, 1745)
- 고정 카메라:   T_Ci_O    = inv_T(bTf[ci]) @ bTo                  ->  bTf[ci] @ T_Ci_O == bTo
- 그리퍼 카메라: T_Cg_O(e) = inv_T(gTc) @ inv_T(bTg[e]) @ bTo      ->  bTg[e] @ gTc @ T_Cg_O(e) == bTo

노이즈 훅
---------
pose_noise=dict(rot_deg=, trans_mm=) 를 주면 관측 T_C_O 에 가우시안 섭동을 준다.
기본은 noise-free. 테스트 본문은 노이즈 유무와 무관하게 동일하다.
"""

import numpy as np

from aruco_cube import inv_T, rot_axis_angle, ArucoCubeModel
from config import get_default_cube_config
from robot_comm import euler_deg_to_matrix


def _rand_se3(rng: np.random.Generator,
              t_range_m: float = 0.5,
              ang_range_deg: float = 180.0) -> np.ndarray:
    """랜덤 강체 변환 (랜덤 축-각 회전 + 랜덤 병진)."""
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    ang = np.deg2rad(rng.uniform(-ang_range_deg, ang_range_deg))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rot_axis_angle(axis, ang)
    T[:3, 3] = rng.uniform(-t_range_m, t_range_m, size=3)
    return T


def _look_at(cam_pos, target, world_up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    base 좌표계에서 cam_pos 에 있는 카메라가 target 을 바라보는 bTf (base<-camera) 반환.

    카메라 광축(+z_cam)이 target 을 향하도록 회전을 구성한다 (OpenCV 카메라 규약:
    +z 가 보는 방향, +x 오른쪽, +y 아래).
    """
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = target - cam_pos                       # 광축: 카메라 -> 타겟
    z = z / (np.linalg.norm(z) + 1e-12)
    up = np.asarray(world_up, dtype=np.float64)
    x = np.cross(up, z)                          # 오른쪽
    if np.linalg.norm(x) < 1e-6:                 # 광축이 up 과 평행하면 대체 up 사용
        up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)                           # 아래 (OpenCV: +y down)
    R = np.column_stack([x, y, z])              # base<-camera 회전 (열 = 카메라축의 base 표현)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = cam_pos
    return T


def matrix_to_euler_6dof(T: np.ndarray):
    """
    robot_comm.euler_deg_to_matrix 의 정확한 역변환.
    4x4 SE(3) -> [x_mm, y_mm, z_mm, rz_deg, ry_deg, rx_deg]  (ZYX, R=Rz@Ry@Rx, t는 m->mm).

    계약: euler_deg_to_matrix(*matrix_to_euler_6dof(T)) == T  (행렬 왕복 ~1e-9).
    gimbal lock(ry≈±90°)에서는 각도값이 다른 분기로 나올 수 있으나 복원 행렬은 정확.
    """
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    cy = float(np.hypot(R[0, 0], R[1, 0]))
    if cy > 1e-9:
        rz = np.arctan2(R[1, 0], R[0, 0])
        ry = np.arctan2(-R[2, 0], cy)
        rx = np.arctan2(R[2, 1], R[2, 2])
    else:  # gimbal lock: cos(ry)≈0 -> rz 를 rx 로 접음
        rz = 0.0
        ry = np.arctan2(-R[2, 0], cy)
        rx = np.arctan2(-R[1, 2], R[1, 1])
    return [float(t[0] * 1000.0), float(t[1] * 1000.0), float(t[2] * 1000.0),
            float(np.degrees(rz)), float(np.degrees(ry)), float(np.degrees(rx))]


class SyntheticScene:
    def __init__(self,
                 seed: int = 0,
                 n_fixed_cams: int = 3,
                 n_events: int = 12,
                 fixed_cam_ids=(1, 2, 3),
                 gripper_cam_idx: int = 0,
                 cube_cfg=None,
                 pose_noise=None,
                 layout: str = "random",
                 obs_bias=None,
                 # 멀티 placement (set) 파라미터:
                 n_sets: int = 1,                   # 큐브 placement 개수 (set)
                 n_events_per_set=None,             # set 당 그리퍼 view 수 (멀티셋일 때)
                 set_cube_jitter_m: float = 0.08,   # 재배치 간 큐브 중심 변동
                 set_cube_rot_jitter_deg: float = 25.0,  # 재배치 간 큐브 회전 변동
                 # realistic 배치 파라미터 (실험 셋업 근사):
                 fixed_cam_height_m: float = 0.2,   # 고정 카메라 공통 z 높이 (실제 ~20cm)
                 fixed_cam_radius_m: float = 0.4,   # 작업공간 중심에서의 수평 거리
                 workspace_center_m=(0.0, 0.0, 0.0),  # 작업공간(큐브) 중심
                 gripper_tilt_deg: float = 35.0,    # realistic: EE 다운워드 기준 ±틸트 범위
                                                    #   (핸드아이 gTc observability 에 직접 영향)
                 # 선택적 2D 투영 테스트용 카메라 내부 파라미터:
                 image_size=(640, 480),
                 fx=600.0, fy=600.0, cx=320.0, cy=240.0,
                 dist=(0.0, 0.0, 0.0, 0.0, 0.0)):
        self.rng = np.random.default_rng(seed)
        self.cfg = cube_cfg or get_default_cube_config()
        self.model = ArucoCubeModel(self.cfg)
        self.marker_ids = list(self.cfg.marker_ids)

        fixed_cam_ids = list(fixed_cam_ids)[:n_fixed_cams]
        if len(fixed_cam_ids) < n_fixed_cams:
            # 요청한 고정 카메라 수가 기본 ID 목록보다 많으면 자동 확장 (gripper id 회피)
            nxt = max(fixed_cam_ids + [gripper_cam_idx]) + 1
            while len(fixed_cam_ids) < n_fixed_cams:
                fixed_cam_ids.append(nxt)
                nxt += 1
        self.fixed_cam_ids = fixed_cam_ids
        self.gripper_cam_idx = int(gripper_cam_idx)
        self.pose_noise = pose_noise
        self.obs_bias = obs_bias
        self.layout = str(layout)
        self.gripper_tilt_deg = float(gripper_tilt_deg)

        # --- GT 변환 (솔버가 복원해야 할 미지수들). 큐브(bTo)만 set 마다 재배치, 나머지는 전역. ---
        self.gTc = _rand_se3(self.rng, t_range_m=0.1, ang_range_deg=180.0)
        if self.layout == "realistic":
            self.bTf, bTo0 = self._realistic_fixed_cams(
                fixed_cam_height_m, fixed_cam_radius_m, np.asarray(workspace_center_m, float))
            center = np.asarray(workspace_center_m, float)
        elif self.layout == "random":
            self.bTf = {ci: _rand_se3(self.rng, t_range_m=0.8) for ci in self.fixed_cam_ids}
            bTo0 = _rand_se3(self.rng, t_range_m=0.3)
            center = bTo0[:3, 3].copy()
        else:
            raise ValueError(f"unknown layout: {layout!r} (use 'random' or 'realistic')")

        # --- 큐브 placement (set): set 0 = bTo0, 나머지는 bTo0 에 jitter 를 준 재배치 ---
        self.n_sets = int(max(n_sets, 1))
        self.set_indices = list(range(self.n_sets))
        self.bTo_by_set = {0: bTo0}
        for s in range(1, self.n_sets):
            T = bTo0.copy()
            ax = self.rng.normal(size=3); ax /= (np.linalg.norm(ax) + 1e-12)
            T[:3, :3] = rot_axis_angle(ax, np.deg2rad(self.rng.uniform(
                -set_cube_rot_jitter_deg, set_cube_rot_jitter_deg))) @ bTo0[:3, :3]
            T[:3, 3] = center + self.rng.uniform(-set_cube_jitter_m, set_cube_jitter_m, size=3)
            self.bTo_by_set[s] = T
        self.bTo = self.bTo_by_set[0]   # 별칭 (단일셋이면 기존과 동일)

        # --- ChArUco 보드: 작업공간에 고정된 별도 타겟 (그리퍼 카메라 eye-in-hand 의 주 신호).
        #     실제 시스템처럼 베이스에 고정(bTboard)되어 있고, 그리퍼 카메라가 여러 자세에서 관측. ---
        self.bTboard = _rand_se3(self.rng, t_range_m=0.2, ang_range_deg=180.0)

        # --- 이벤트 수 결정 + 이벤트→set 매핑 ---
        if self.n_sets == 1:
            total_events = n_events
        else:
            per = int(n_events_per_set) if n_events_per_set else 4
            total_events = self.n_sets * per
        self.event_set = {}

        # --- M 개 로봇 자세 (이벤트). 6-DOF 로 샘플 후 robot_comm 규약으로 4x4 변환 ---
        # realistic: EE(그리퍼)가 아래(작업공간)를 향하도록 제한 — 실제로 EE 는
        #            -z(아래쪽 바닥)로 뒤집히지 않고 위에서 내려다보는 자세만 취함.
        #            -> 기본 자세 = 아래 향함(rx≈180°), 거기에 작은 틸트(±tilt)만 추가.
        # random:    완전 ±180° (배치무관 정확성 검증용, 더 엄격).
        self.events = list(range(total_events))
        self.bTg = {}
        self.pose6 = {}
        per_set = max(total_events // self.n_sets, 1)
        for eid in self.events:
            # 이벤트를 set 에 균등 배정 (멀티셋이면 placement 별 그룹)
            self.event_set[eid] = self.set_indices[min(eid // per_set, self.n_sets - 1)]
            if self.layout == "realistic":
                # 작업공간 위쪽에서 내려다보는 위치 (z>0, 큐브 근처)
                x, y = self.rng.uniform(-150.0, 150.0, size=2)   # mm, 작업공간 위
                z = self.rng.uniform(200.0, 400.0)              # mm, 항상 위쪽
                tilt = self.gripper_tilt_deg                     # 다운워드 기준 ±틸트
                rz = self.rng.uniform(-180.0, 180.0)            # 광축 회전은 자유
                ry = self.rng.uniform(-tilt, tilt)
                rx = 180.0 + self.rng.uniform(-tilt, tilt)      # 180°=아래 향함
            else:
                x, y, z = self.rng.uniform(-300.0, 300.0, size=3)     # mm
                rz, ry, rx = self.rng.uniform(-180.0, 180.0, size=3)  # deg
            self.pose6[eid] = [float(x), float(y), float(z),
                               float(rz), float(ry), float(rx)]
            self.bTg[eid] = euler_deg_to_matrix(x, y, z, rz, ry, rx)

        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        self.D = np.array(dist, dtype=np.float64)
        self.image_size = image_size

    # ------------------------------------------------------------------
    # realistic 배치: 고정 카메라를 동일 z 높이에서 작업공간을 둘러싸도록
    # ------------------------------------------------------------------
    def _realistic_fixed_cams(self, height_m, radius_m, center):
        """
        실험 셋업 근사:
          - 고정 카메라 N대를 같은 z 높이(height_m)에서 작업공간 중심을 둘러싸도록 원형 배치
          - 각 카메라는 작업공간 중심(큐브)을 바라봄 (look-at)
          - 큐브는 작업공간 중심에 위치 (약간의 랜덤 자세만 부여)
        반환: (bTf dict, bTo)
        """
        n = len(self.fixed_cam_ids)
        bTf = {}
        # 카메라들이 완전 동일평면이면 일부 기하가 약해지므로, 실제처럼 둘러싸되
        # 작은 z 변동(±2cm)만 허용 — "거의 동일 z 높이" 근사.
        for k, ci in enumerate(self.fixed_cam_ids):
            theta = 2.0 * np.pi * k / max(n, 1)
            jitter_z = self.rng.uniform(-0.02, 0.02)
            pos = center + np.array([radius_m * np.cos(theta),
                                     radius_m * np.sin(theta),
                                     height_m + jitter_z])
            bTf[ci] = _look_at(pos, center)
        # 큐브: 작업공간 중심 + 작은 자세 변동 (테이블 위 큐브)
        bTo = _rand_se3(self.rng, t_range_m=0.0, ang_range_deg=180.0)
        bTo[:3, 3] = center + np.array([0.0, 0.0, 0.0])
        return bTf, bTo

    # ------------------------------------------------------------------
    # GT 로부터 카메라가 보는 큐브 포즈 유도 (검증된 체인)
    # ------------------------------------------------------------------
    def _bTo_at(self, eid) -> np.ndarray:
        """이벤트 eid 가 속한 set 의 GT 큐브 placement. eid=None 이면 set 0."""
        if eid is None:
            return self.bTo_by_set[self.set_indices[0]]
        return self.bTo_by_set[self.event_set[int(eid)]]

    def _bTo_observed(self, eid) -> np.ndarray:
        """카메라가 '관측'하는 큐브 = GT 에 systematic bias(base frame) 적용.
        bias 없으면 GT 와 동일. prior(set_cube_center)는 GT 를 쓰므로 둘이 갈린다."""
        return self._bias_base() @ self._bTo_at(eid)

    def T_Ci_O(self, ci: int, eid=None) -> np.ndarray:
        """고정 카메라 ci 가 '관측'하는 큐브 포즈 (systematic bias 포함).
        (무인자 호출은 set 0 — 기존 자기검증 테스트 하위호환)"""
        return inv_T(self.bTf[ci]) @ self._bTo_observed(eid)

    def T_Cg_O(self, eid: int) -> np.ndarray:
        """그리퍼 카메라가 이벤트 eid 에서 '관측'하는 큐브 포즈 (systematic bias 포함)."""
        return inv_T(self.gTc) @ inv_T(self.bTg[eid]) @ self._bTo_observed(eid)

    def T_Cg_board(self, eid: int) -> np.ndarray:
        """그리퍼 카메라가 이벤트 eid 에서 보는 ChArUco 보드 포즈 (camera<-board).

        보드는 베이스에 고정(bTboard) -> 큐브와 동일한 AX=XB 구조:
        bTg[e] @ gTc @ T_Cg_board(e) == bTboard (이벤트 무관 상수).
        """
        return inv_T(self.gTc) @ inv_T(self.bTg[eid]) @ self.bTboard

    # ------------------------------------------------------------------
    # 노이즈 주입
    # ------------------------------------------------------------------
    def _perturb_se3(self, T: np.ndarray, rot_deg: float, trans_mm: float) -> np.ndarray:
        axis = self.rng.normal(size=3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        dR = rot_axis_angle(axis, np.deg2rad(self.rng.normal(0.0, rot_deg)))
        dt = self.rng.normal(0.0, trans_mm / 1000.0, size=3)
        Tn = T.copy()
        Tn[:3, :3] = dR @ T[:3, :3]
        Tn[:3, 3] = T[:3, 3] + dt
        return Tn

    def _bias_base(self) -> np.ndarray:
        """systematic bias 를 base frame 의 4x4 로 (모든 관측이 base 에서 같은 방향으로 쏠림).
        camera frame 에 주면 카메라마다 좌표계가 달라 base 에서 상쇄되므로 base frame 에 준다."""
        if not self.obs_bias:
            return np.eye(4)
        rot_deg = float(self.obs_bias.get("rot_deg", 0.0))
        trans_mm = float(self.obs_bias.get("trans_mm", 0.0))
        Tb = np.eye(4)
        Tb[:3, 3] = np.array([trans_mm / 1000.0, 0.0, 0.0])  # base +x 로 일정
        if rot_deg:
            Tb[:3, :3] = rot_axis_angle(np.array([0.0, 0.0, 1.0]), np.deg2rad(rot_deg))
        return Tb

    def _maybe_noise(self, T: np.ndarray):
        """관측 T_C_O 에 랜덤 노이즈 주입 후 (T_noisy, err_mean) 반환.
        (systematic bias 는 관측 생성 단계에서 base frame 으로 이미 적용됨)"""
        T = np.asarray(T, dtype=np.float64)
        if not self.pose_noise:
            return T, 0.01  # 0 이 아닌 작은 값 (weight 안정화)
        T_n = self._perturb_se3(T,
                                float(self.pose_noise.get("rot_deg", 0.0)),
                                float(self.pose_noise.get("trans_mm", 0.0)))
        # err_mean 을 노이즈에 비례하게 두어 MAD/weighting 로직이 동작하도록
        err = 0.01 + float(self.pose_noise.get("trans_mm", 0.0))
        return T_n, err

    # ------------------------------------------------------------------
    # pnp_obs / meta 조립
    # ------------------------------------------------------------------
    def _obs(self, T_C_O, used_ids=None, source="multi"):
        if used_ids is None:
            used_ids = self.marker_ids
        T_n, err = self._maybe_noise(np.asarray(T_C_O, dtype=np.float64))
        cand = {
            "T_C_O": T_n,
            "err_mean": float(err),
            "n_points": int(4 * len(used_ids)),
            "used_ids": [int(x) for x in used_ids],
            "source": str(source),
        }
        obs = dict(cand)
        obs["_candidates"] = [dict(cand)]  # noise-free 시 일관된 candidate 1개
        return obs

    def build_pnp_obs(self, include_gripper: bool = True, fixed_cam_per_set: bool = True):
        """pnp_obs[cam_id][event_id] -> 관측 dict.

        fixed_cam_per_set=True (실제 구조): 큐브가 한 set 에 고정이므로 고정 카메라는
            set 당 1번만 관측(각 set 의 첫 이벤트에만). 그리퍼는 핸드아이용으로 모든 이벤트 관측.
        fixed_cam_per_set=False (구버전): 고정 카메라도 모든 이벤트 관측(큐브 재관측).
        """
        pnp = {ci: {} for ci in self.fixed_cam_ids}
        if fixed_cam_per_set:
            # set 별 대표 이벤트 1개 (그 set 의 첫 이벤트)
            rep = {}
            for eid in self.events:
                s = self.event_set[eid]
                if s not in rep:
                    rep[s] = eid
            for ci in self.fixed_cam_ids:
                for s, eid in rep.items():
                    pnp[ci][eid] = self._obs(self.T_Ci_O(ci, eid))
        else:
            for ci in self.fixed_cam_ids:
                for eid in self.events:
                    pnp[ci][eid] = self._obs(self.T_Ci_O(ci, eid))
        if include_gripper:
            pnp[self.gripper_cam_idx] = {
                eid: self._obs(self.T_Cg_O(eid)) for eid in self.events
            }
        return pnp

    def build_charuco_obs(self):
        """charuco_obs[event_id] = {"T_cam_board":4x4, "reproj":float, "n_corners":int}.

        실제 Step3 의 ChArUco 핸드아이 경로(refine_gripper_event_base_transforms_with_board_anchor,
        _evaluate_handeye 의 use_charuco 분기)가 소비하는 스키마 그대로.
        """
        obs = {}
        for eid in self.events:
            T = self.T_Cg_board(eid)
            T_n, err = self._maybe_noise(T)
            obs[eid] = {
                "T_cam_board": T_n,
                "reproj": float(err),       # 재투영 오차 (px) 가중치용
                "n_corners": 40,            # 11x7 보드의 대략적 내부 코너 수
            }
        return obs

    def build_meta(self, emit_set_prior: bool = True):
        """meta dict (Step3 솔버가 읽는 최소 스키마).

        set_index 는 placement(set) 별로 부여(이벤트별 아님).
        emit_set_prior=True 면 set_cube_center_6dof(= GT 큐브 placement, 완벽 prior) 방출.
        """
        caps = []
        for eid in self.events:
            s = self.event_set[eid]
            cap = {
                "event_id": int(eid),
                "set_index": int(s),
                "robot_pose_matrix_4x4": self.bTg[eid].tolist(),
                "robot_pose_6dof": [float(v) for v in self.pose6[eid]],
                "cams": {str(ci): {} for ci in self.fixed_cam_ids},
            }
            if emit_set_prior:
                # 실제로는 로봇 FK 추정값; 시뮬은 GT 를 정확히 알므로 GT placement 를 prior 로.
                cap["set_cube_center_6dof"] = matrix_to_euler_6dof(self.bTo_by_set[s])
            caps.append(cap)
        return {"captures": caps}

    def gt_T_base_Ci(self):
        return {ci: self.bTf[ci].copy() for ci in self.fixed_cam_ids}

    def gt_T_B_O_by_event(self):
        return {eid: self._bTo_at(eid).copy() for eid in self.events}

    def gt_T_B_O_by_set(self):
        return {s: self.bTo_by_set[s].copy() for s in self.set_indices}
