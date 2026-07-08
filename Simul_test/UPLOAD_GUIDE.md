# GitHub 업로드 가이드 — 시뮬 실험 3종

세 실험(Exp1/2/3)을 GitHub 에 올려 이후 실데이터 테스트의 기준으로 쓰기 위한 정리.

---

## 1. 올려야 할 파일 (필수)

### 실험 코드 (Simul_test/)
```
Simul_test/synthetic_scene.py            # 시뮬 엔진 (공통)
Simul_test/metrics.py                    # 오차 헬퍼 (공통)
Simul_test/unified_vs_independent.py     # Exp1
Simul_test/joint_calib.py                # Exp1/3 (bundle adjustment)
Simul_test/viz_unified.py                # Exp1 figure
Simul_test/exp2_board_vs_cube.py         # Exp2
Simul_test/viz_exp2.py                   # Exp2 figure
Simul_test/exp3_gtc_estimation.py        # Exp3
Simul_test/viz_exp3_sweep.py             # Exp3 figure
```

### 문서 (Simul_test/)
```
Simul_test/EXPERIMENTS_README.md              # 세 실험 인덱스 (진입점)
Simul_test/C1_UNIFIED_EXPLAINED.md            # Exp1 설명
Simul_test/EXP2_BOARD_VS_CUBE_EXPLAINED.md    # Exp2 설명
Simul_test/EXP3_GTC_ESTIMATION_EXPLAINED.md   # Exp3 설명
```

### 결과 그림 + 데이터 (Simul_test/figures/)
```
Simul_test/figures/fig_unified_vs_indep.png
Simul_test/figures/fig_unified_tilt_sweep.png        # (선택)
Simul_test/figures/unified_vs_indep_data.json
Simul_test/figures/fig_exp2_board_vs_cube.png
Simul_test/figures/exp2_board_vs_cube_data.json
Simul_test/figures/fig_exp3_noise_sweep.png
Simul_test/figures/exp3_noise_sweep_data.json
```

### 프로젝트 루트 의존성 (실험 코드가 import)
```
aruco_cube.py       # inv_T, rot_axis_angle, ArucoCubeModel
utils_pose.py       # robust_se3_average
config.py           # get_default_cube_config
robot_comm.py       # euler_deg_to_matrix
requirements.txt    # 의존 패키지
.gitignore
```

### 검증 테스트 (기존, 선택적 포함)
```
Simul_test/test_*.py       # 시뮬 생성기/수식/파이프라인 정확성 (pytest)
Simul_test/conftest.py
Simul_test/README.md       # 기존 시뮬 스위트 설명
```

---

## 2. 올리면 안 되는 것 (.gitignore 로 제외됨)

```
__pycache__/  *.pyc                     # 파이썬 캐시
Simul_test/_scratch_*.py                # 실험용 임시 스크립트
Simul_test/_real_step3_scratch/         # 실데이터 캐시 (용량 큼)
calb_data_*/                            # 실데이터 캡처 (용량 큼)
Simul_test/figures/fig_abc_*.png        # (선택) 이전 ABC 실험 그림 — 무관하면 제외
Simul_test/figures/fig_ablation_*.png   # (선택) 이전 ablation 그림
```

> `.gitignore` 를 프로젝트 루트에 이미 추가함. scratch·캐시·실데이터는 자동 제외된다.

---

## 3. 업로드 명령

```bash
cd /home/sstone/rb-calibration-marker-experiment

# 새 브랜치에서 작업 권장
git checkout -b simul-experiments

# 필수 파일 스테이징 (scratch 는 .gitignore 로 자동 제외)
git add .gitignore requirements.txt
git add aruco_cube.py utils_pose.py config.py robot_comm.py
git add Simul_test/synthetic_scene.py Simul_test/metrics.py
git add Simul_test/unified_vs_independent.py Simul_test/joint_calib.py Simul_test/viz_unified.py
git add Simul_test/exp2_board_vs_cube.py Simul_test/viz_exp2.py
git add Simul_test/exp3_gtc_estimation.py Simul_test/viz_exp3_sweep.py
git add Simul_test/EXPERIMENTS_README.md Simul_test/C1_UNIFIED_EXPLAINED.md
git add Simul_test/EXP2_BOARD_VS_CUBE_EXPLAINED.md Simul_test/EXP3_GTC_ESTIMATION_EXPLAINED.md
git add Simul_test/figures/fig_unified_vs_indep.png Simul_test/figures/unified_vs_indep_data.json
git add Simul_test/figures/fig_exp2_board_vs_cube.png Simul_test/figures/exp2_board_vs_cube_data.json
git add Simul_test/figures/fig_exp3_noise_sweep.png Simul_test/figures/exp3_noise_sweep_data.json

# 커밋 전 무엇이 올라가는지 반드시 확인
git status
git commit -m "Add calibration simulation experiments (Exp1/2/3)"
git push -u origin simul-experiments
```

---

## 4. 실데이터 테스트로 넘어갈 때 (다음 단계)

시뮬에서 검증한 세 방식을 실데이터에 그대로 적용하려면:

| 시뮬 함수 | 실데이터에서 바꿀 것 |
|---|---|
| `synthetic_scene.py` 의 관측 생성 | 실제 `meta.json` 의 `T_cam_cube` / `T_cam_board` 관측으로 교체 |
| GT (gt_cam, gt_gTc, FK) | 실데이터엔 GT 없음 → 재현성/일관성 지표로 대체 평가 |
| 캘리브 함수 (calib_joint 등) | **그대로 재사용** (입력만 실데이터 관측으로) |

즉 **캘리브·평가 로직은 그대로 두고, 관측 소스만 시뮬→실데이터로 교체**하면 된다.
`abc_calib.py` 의 `load_real()` 이 실데이터 meta.json 파싱 예시를 제공한다.
