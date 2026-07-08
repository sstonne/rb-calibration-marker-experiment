# Calibration Simulation Experiments (Exp1 / Exp2 / Exp3)

멀티 카메라 + 로봇 핸드아이 + ArUco 큐브 캘리브레이션의 세 가지 설계 선택을 **순수 SE(3)
기하 시뮬레이션**(렌더링·IsaacSim 불필요, ground-truth 를 알기 때문에 정답 대비 오차를 직접
측정)으로 검증한다. 시뮬에서 검증한 이 방식들을 이후 **실데이터로도 동일하게 테스트**한다.

> 노이즈는 **systematic**(큐브 base 위치 (x,y)에 선형 의존하는 위치의존 편향 — 렌즈왜곡·
> intrinsic 잔차·작업공간 휨 등 실제 검출오차의 지배성분)만 사용. 매번 랜덤한 가우시안이
> 아니라 **위치가 같으면 같은 방향으로 틀리는** 실제와 같은 성격.

---

## 세 실험 요약

| # | 실험 | 무엇을 비교 | 핵심 결과 | 문서 · 그림 |
|---|---|---|---|---|
| **1** | Unified vs Independent | eye-in-hand + eye-to-hand 를 **따로 캘리브 후 조합** vs **하나로 동시 최적화(bundle adjustment)** | 통합이 핸드아이 gTc 23→0.17mm, held-out 4.65→0.35mm | [C1_UNIFIED_EXPLAINED.md](C1_UNIFIED_EXPLAINED.md) · `fig_unified_vs_indep.png` |
| **2** | Board vs Cube | 평면 ChArUco board only vs board + marker cube (**FK 미사용, 순수 카메라 관측**) | cube 추가로 카메라 위치 8.3→4.7mm(-43%), 물체예측 -34% | [EXP2_BOARD_VS_CUBE_EXPLAINED.md](EXP2_BOARD_VS_CUBE_EXPLAINED.md) · `fig_exp2_board_vs_cube.png` |
| **3** | gTc estimation | gripper-target 변환을 **미지수로 추정** vs **FK 로 고정** vs **추정+FK후보정** | FK 고정이 노이즈에 취약, Camera+FK후보정이 전 구간 최고 | [EXP3_GTC_ESTIMATION_EXPLAINED.md](EXP3_GTC_ESTIMATION_EXPLAINED.md) · `fig_exp3_noise_sweep.png` |

---

## 설치

```bash
# conda 환경 예시 (rb-calib)
pip install numpy scipy matplotlib opencv-contrib-python

# 주의: opencv-python 이 아니라 opencv-contrib-python (ArUco 모듈이 contrib 에 있음).
```

의존 패키지: `numpy`, `scipy`, `matplotlib`, `opencv-contrib-python` (>=4.7).

---

## 실행

시스템 ROS pytest 플러그인 충돌을 피하려면 두 환경변수를 준다:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=` (프로젝트 루트에서 실행).

### Experiment 1 — Unified vs Independent
```bash
PYTHONPATH= python Simul_test/unified_vs_independent.py --seeds 30   # 결과 표 + JSON
PYTHONPATH= python Simul_test/viz_unified.py                         # 막대 figure
PYTHONPATH= python Simul_test/viz_unified.py --tilt-sweep            # (선택) tilt sweep
```

### Experiment 2 — Board vs Cube
```bash
PYTHONPATH= python Simul_test/exp2_board_vs_cube.py --seeds 30 \
    --dump Simul_test/figures/exp2_board_vs_cube_data.json
PYTHONPATH= python Simul_test/viz_exp2.py                            # figure
```

### Experiment 3 — gTc estimation
```bash
PYTHONPATH= python Simul_test/exp3_gtc_estimation.py --seeds 20                 # 3방식 표
PYTHONPATH= python Simul_test/exp3_gtc_estimation.py --sweep noise --seeds 12 \
    --dump Simul_test/figures/exp3_noise_sweep_data.json                        # 노이즈 sweep
PYTHONPATH= python Simul_test/viz_exp3_sweep.py                                 # 곡선 figure
```

> Joint(bundle adjustment) 최적화가 무거워 30-seed 전체는 수 분~십수 분 걸린다.
> 빠른 확인은 `--seeds` 를 6~10 으로 줄이면 된다.

---

## 파일 구성

### 시뮬 엔진 (공통)
| 파일 | 역할 |
|---|---|
| `synthetic_scene.py` | GT 변환(gTc·bTf·bTo)을 정하고 검증된 체인으로 관측 생성 |
| `metrics.py` | 회전(deg)/병진(mm) 오차 헬퍼 |

### 실험별 코드
| 파일 | 실험 | 역할 |
|---|---|---|
| `unified_vs_independent.py` | Exp1 | 씬·관측 생성 + Independent/Joint 캘리브 + 평가 |
| `joint_calib.py` | Exp1/3 | Joint bundle adjustment, FK-fixed joint, rigid 정합 |
| `viz_unified.py` | Exp1 | figure |
| `exp2_board_vs_cube.py` | Exp2 | board/cube 관측·순수 카메라 캘리브·평가 |
| `viz_exp2.py` | Exp2 | figure |
| `exp3_gtc_estimation.py` | Exp3 | 세 방식 비교 + sweep |
| `viz_exp3_sweep.py` | Exp3 | 곡선 figure |

### 루트 의존성 (프로젝트 루트, import 로 재사용)
`aruco_cube.py` (inv_T·rot_axis_angle·ArucoCubeModel), `utils_pose.py`
(robust_se3_average), `config.py` (get_default_cube_config), `robot_comm.py`
(euler_deg_to_matrix).

### 검증 테스트 (기존)
`test_*.py` — 시뮬 생성기·수식·파이프라인 정확성 검증 (pytest).

---

## 명명 규약

- 변환은 "목적지-from-출발지": `T_A_C = T_A_B @ T_B_C`.
- `bTf` = base←fixed camera, `gTc` = gripper←camera (핸드아이), `bTo` = base←object(큐브).
- **Joint / bundle adjustment** = 모든 미지수(카메라·gTc·큐브)를 하나의 비선형 최소제곱으로
  동시 최적화 (학습 아님, 추정).
- **+fk / FK-correction** = 캘리브 후 남은 위치의존 잔차를 **선형 회귀(Ridge)** 로 학습해
  최종 예측을 후보정 (supervision = 로봇 FK 가 아는 큐브 위치).
