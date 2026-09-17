# Calibration → Table 1 실행 순서

메인 실행 파일은 저장소 root의 `01_...py`부터 `05_...py`까지다. 01~05가
입력 준비와 calibration을 수행한다. ABLATION_TEST 결과 표(Markdown·요약 CSV·fold CSV)는 [zeus_gello_calibration/table1_zeus.py](zeus_gello_calibration/table1_zeus.py) 하나가 계산과 보고서 생성을 모두 맡는다. 저장된 JSON으로 보고서만 다시 만들 때는 `--report-only <JSON>`을 쓴다.
Cross-target·marker-system·OpenCV baseline은 calibration 완료에 필요하지 않아
`tools/`의 선택 평가로 분리했다.

다섯 파일은 모두 얇은 진입점이고 실제 구현은 `capture_pipeline/`·`calibration_pipeline/`
안에 있다. 각 단계가 **무슨 식을 푸는지**와 **어느 모듈·함수가 그 일을 하는지**는
아래 [단계별 원리와 구현 위치](#단계별-원리와-구현-위치)에 정리했고, 같은 내용이
각 스크립트 상단 docstring에도 들어 있다. 코드를 고칠 때는 그 표에서 대상 모듈을
찾은 뒤 해당 파일의 docstring부터 읽는다.

## 촬영 프로토콜과 현재 구현 상태

새 촬영의 유일한 가이드와 명령은
[zeus_gello_calibration/PIPELINE.md](zeus_gello_calibration/PIPELINE.md)에 둔다.
PC 촬영 진입점은 `03_capture.py` 하나이고 새 데이터는 모두
`zeus_gello_calibration/data/` 아래에 저장한다. 이 문서에서는 촬영 명령을 중복하지
않으며, 기존 `data/session02_NOUSE_session04_0814`는 과거 결과 재현과 진단 입력으로만 유지한다.

## 현재 Session04 / legacy 명령 순서

아래 명령은 현재 코드와 기존 Session04 artifact를 재현하는 순서다. 새 최종 촬영
명령은 이 legacy 블록과 섞지 않고 아래 `최종 촬영 흐름` 절에서 관리한다.

```bash
# 01 — RealSense factory intrinsic과 depth calibration 저장
python3 01_export_intrinsics.py \
  --out_dir intrinsics \
  --color_w 1280 --color_h 720 --fps 15

# 02 — ChArUco로 color intrinsic 정밀 보정
python3 02_calibrate_intrinsics.py \
  --intr_dir intrinsics \
  --min_views 12 \
  --save_images

# 04 — 저장 영상 전체 재검출 및 관측 manifest 고정
python3 04_filter_observations.py \
  --session-root data/session02_NOUSE_session04_0814/calib_train \
  --intrinsics-dir intrinsics

# 05 — A0~A5/B1~B3 calibration + frame-prune/refit/rollback
python3 05_calibrate.py \
  --root_folder data/session02_NOUSE_session04_0814/calib_train \
  --intrinsics_dir intrinsics \
  --include_sets 0-12 \
  --split_seed 20260731 \
  --num_inits 3 \
  --observation-manifest data/session02_NOUSE_session04_0814/calib_out/capture_filter/Step2b_observation_manifest.json \
  --out_dir ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/ABLATION_TEST_table1

```

## 단계별 입력 · 과정 · 결과

| 단계 | 입력 | 과정 | 결과 |
| --- | --- | --- | --- |
| 01 `export_intrinsics` | 연결된 RealSense, 스트림 설정 | serial 순으로 camera ID를 고정하고 factory intrinsic/extrinsic 및 depth scale을 읽는다 | `intrinsics/device_map.json`, `depth_scales.json`, `cam*.npz` |
| 02 `calibrate_intrinsics` | 01 결과, ChArUco board | 다양한 위치의 view로 OpenCV color intrinsic calibration을 수행한다 | 갱신된 `cam*.npz`, `factory_backup/`, `charuco_capture/` |
| 03 `capture` (현재 legacy) | 02 결과, board/cube, 카메라, robot FK | `A_placement/B_eyetohand` block에서 동기화 및 marker quality gate를 통과한 event를 저장한다 | `data/session<NN>_<설명>_<MMDD>/calib_train/meta.json`, RGB/depth 이미지 |
| 04 `filter_observations` | 03 세션, 고정 K/D | 모든 RGB를 다시 검출하고 관측 정책을 적용해 native-pixel corner와 원본 SHA-256을 고정한다 | `Step2b_observation_manifest.json`, QA CSV, overlay, `CAPTURE_FILTER.md` |
| 05 `calibrate` | 04 manifest, K/D, `meta.json`, robot FK | event 단위 train/held-out 분리, 공통 초기화, 9개 조건 fit, `frame-prune → refit → rollback`, held-out 평가 | `ABLATION_TEST_table1_methods.json`, 두 shared artifact |

## 단계별 원리와 구현 위치

root의 `01_...py` ~ `05_...py`는 얇은 진입점이고 실제 동작은 패키지 모듈에 있다.
각 파일 상단 docstring에 같은 내용이 더 자세히 적혀 있으므로, 코드를 고칠 때는
아래 표에서 대상 모듈을 찾고 해당 파일의 docstring부터 읽는다.

### 01 — factory intrinsic 덤프

핀홀 + Brown–Conrady 모델의 상수를 SDK에서 **읽어 옮길 뿐 추정하지 않는다**.

```
x_n = X/Z ,  y_n = Y/Z
r² = x_n² + y_n²
x_d = x_n(1 + k1r² + k2r⁴ + k3r⁶) + 2p1x_ny_n + p2(r² + 2x_n²)
y_d = y_n(1 + k1r² + k2r⁴ + k3r⁶) + p1(r² + 2y_n²) + 2p2x_ny_n
[u v 1]ᵀ = K [x_d y_d 1]ᵀ ,  K = [[fx,0,cx],[0,fy,cy],[0,0,1]]
Z[m] = d_raw × depth_scale
```

D415/D435의 color `D`는 공장에서 전부 0으로 보고된다. 이는 "왜곡 없음"이 아니라
"공장 미제공"이라는 뜻이며, 실제 왜곡은 02에서 추정해 덮어쓴다.

| 모듈 | 함수 | 역할 |
| --- | --- | --- |
| `capture_pipeline/export_intrinsics.py` | `main()` | 장치 열거 → 스트림 개시 → npz/json 기록 |
| | `_intr_to_KD()` | pyrealsense2 intrinsics → `(K 3×3, D N×1)` |

### 02 — ChArUco color intrinsic 보정

V개 뷰에서 `K, D`와 뷰별 자세를 동시에 추정한다.

```
(K*, D*, {R_v,t_v}*) = argmin Σ_v Σ_k ‖ π(K,D,R_v,t_v,X_k) − x_vk ‖²
RMS = sqrt( (1/N) Σ_v Σ_k ‖ π(·) − x_vk ‖² )   [px]
```

정면 뷰만 모으면 `fx/fy`와 `t_z`가 서로 상쇄되어(scale–depth ambiguity) `K`가
분리되지 않고, 화면 가장자리를 덮지 않으면 `k1, k2`가 관측되지 않는다. 수집
루프가 커버리지와 선명도를 화면에 띄우는 이유다.

| 모듈 | 함수 | 역할 |
| --- | --- | --- |
| `capture_pipeline/calibrate_intrinsics.py` | `main()` | device_map 로드 후 카메라별 루프 |
| | `collect_for_camera()` | 라이브 뷰 수집, SPACE 수동 그랩 |
| | `_obj_img_from_charuco()` | 검출 결과 → `(3D X_k, 2D x_k)` 대응쌍 |
| | `_run_calib()` | `cv2.calibrateCamera` 호출 — 위 argmin이 풀리는 지점 |
| | `calibrate_intrinsics()` | 이상 뷰 제거 후 재보정 |
| | `overwrite_color_intrinsics()` | `color_K/color_D`만 교체, depth 필드 보존 |

### 03 — 동기 촬영

고정카메라 경로와 그리퍼카메라 경로가 같은 표적을 동시에 보면 닫힌 루프가 생긴다.

```
T^B_Ci · T^Ci_O(e)  =  T^B_G(e) · T^G_Cg · T^Cg_O(e)
```

이는 hand-eye의 표준형 `AX = XB`와 같은 구조이며 `T^G_Cg`와 `T^B_Ci`를 함께
결정한다. 회전축이 서로 다른 자세가 여럿 있어야 해가 유일해지므로, **장수가 아니라
자세 다양성**이 촬영의 핵심이다. 이 단계의 마커 pose는 진단용이며 calibration에
쓰이지 않는다.

| 모듈 | 함수 | 역할 |
| --- | --- | --- |
| `capture_pipeline/capture.py` | `main()` | 인자 해석, 카메라/로봇 연결, 촬영 루프 |
| | `wait_for_start_command_capture()` | 서버 start 신호 대기, 이벤트 진행 |
| | `load_device_map()` / `load_intrinsics()` | `serial→cam_idx`, `(K, D, depth_scale)` |
| | `estimate_per_marker_poses()` | 마커별 PnP (표시용, calibration 미사용) |
| | `evaluate_transport_integrity()` | 프레임 누락·불일치 검사 |
| | `load_and_validate_rig_geometry()` | rig 형상과 `rig_id` 일치 확인 |
| | `canonical_json_sha256()` / `file_sha256()` | meta·영상 출처 해시 고정 |

### 04 — 재검출과 관측 동결

표적별 PnP를 풀고 잔차로 관측을 선별한다.

```
T^C_O = argmin_T Σ_k ‖ π(K,D,T,X_k) − x_k ‖²
RMSE  = sqrt( (1/N) Σ_k ‖ π(K,D,T,X_k) − x_k ‖² )   [px]
```

큐브는 RMSE보다 **기하 조건을 먼저** 본다. 한 면만 보이면 표적점이 한 평면에 놓여
PnP가 평면 축퇴에 빠지고, 이때는 재투영오차가 낮아도 깊이·기울기가 사실상 정해지지
않는다(평면 호모그래피 이중해). core 조건은 서로 다른 면 2개 이상 + 비평면 +
양의 깊이 해 존재다. 임계값 기본값은 standard `RMSE ≤ 3.0px`, strict `RMSE ≤ 2.0px`
· `inlier ≥ 0.9`이며, 판정은 `selected / recovered / quarantine / rejected`로 나뉜다.

| 모듈 | 함수 | 역할 |
| --- | --- | --- |
| `calibration_pipeline/filter_observations.py` | `main()` / `run_filter()` | 전체 파이프라인 |
| | `_cube_records()` / `_board_records()` | 재검출과 PnP, 면 개수·평면성 진단 |
| | `_core_support()` | 평면 축퇴 방어 (core 조건 판정) |
| | `_cube_policy_decision()` | RMSE·inlier 임계 적용, 탈락 사유 생성 |
| | `_disposition()` | 4단계 분류 |
| | `_image_provenance()` / `_sha256()` | 원본 영상 SHA-256 고정 |
| | `_draw_review_overlay()` | 사람 검토용 오버레이 |

### 05 — Table 1 calibration

모든 행이 **같은 목적함수 하나**를 풀고, 달라지는 것은 (1) 넣는 관측과
(2) 자유변수 목록(freeze mask)뿐이다. 잔차는 corner 재투영 오차 한 종류이며
카메라–카메라 잔차도, 표적 자세 사전 잔차도 없다.

```
r_k = π( K_c, D_c, (T^B_Cc(e))⁻¹ · T^B_O , X_k ) − x_k        [px]

고정카메라 i  : T^B_Cc(e) = T^B_Ci                (이벤트 무관 상수)
그리퍼카메라 g : T^B_Cc(e) = T^B_G(e) · T^G_Cg     (T^B_G(e)는 FK 고정 입력)

minimize Σ_k ρ(r_k)     SciPy TRF, soft_l1, f_scale = 2px, x_scale='jac'
z = (r/f)² ,  ρ(z) = 2(√(1+z) − 1) ,  cost = 0.5·f²·Σρ(z)
RMSE_px = sqrt( (1/2N) Σ_k ((u−û)² + (v−v̂)²) )     ← 분모 2N: corner당 스칼라 2개
```

행별 자유변수:

| 행 | 자유변수 |
| --- | --- |
| A2 / A4 | `T_base_Ci`, `T_gripper_cam`, `T_base_board`, `T_base_cube_by_set` |
| A3 / A5 | `T_base_Ci`, `T_gripper_cam`, `T_base_board` — cube 자세는 상수 고정 |
| B2 | `T_base_Ci`, `T_gripper_cam`, `T_base_cube_by_set` — board 없음 |
| B3 | `T_base_Ci`, `T_gripper_cam`, `T_base_board` — cube 없음 |
| A0 / A1 / B1 | 순차 2단계: stage1 eye-in-hand → stage2 eye-to-hand (앞 단계 고정) |

A3는 컨트롤러 FK, A5는 train VISION으로 만든 corrected-FK로 큐브 자세를 하드 고정한다.
고정은 "상태에는 있으나 자유변수 목록에서 빠짐"이고 잔차 항이 생기지 않는다.
A4/B1/B2는 반대로 corrected-FK를 soft factor 잔차 블록으로 추가한다.

| 모듈 | 함수 / 상수 | 역할 |
| --- | --- | --- |
| `calibration_pipeline/table1.py` | `main()` | 인자 해석, 준비, 행×seed 루프 |
| | `prepare_ablation_data()` | 관측 로드, event 단위 split, FK 정렬 산출물 |
| | `build_shared_reference_state()` | 전 행 공유 초기 상태 1개 구성 |
| | `make_initial_state()` | 행별 freeze·고정 자세 특수화 |
| | `run_condition_once()` | 한 행 한 seed 적합 (unified / sequential) |
| | `run_factor_condition_once()` | corrected-FK soft factor 행(A4/B1/B2) 전용 경로 |
| | `fit_train_only_cube_evaluation()` | held-out cube 평가용 자세 적합 |
| | `canonical_solver_options()` | 전 행 공통 solver 설정 |
| `calibration_pipeline/reprojection.py` | `solve_corner_reprojection()` | 위 minimize가 풀리는 지점 |
| | `project_points()` | `π(·)` — `cv2.projectPoints` 래퍼 |
| | `robust_least_squares_cost()` | soft_l1 / huber / linear 비용 |
| | `PixelObs` | 관측 하나(표적·카메라·이벤트·3D/2D 점) |
| `calibration_pipeline/schema.py` | `UNIFIED_FREE_VARIABLES` | 위 freeze mask 표의 실체 |
| | `SEQUENTIAL_STAGE_SPECS` | 순차 행의 stage별 자유변수 |
| | `RAW_FK_CUBE_CENTER_TO_OBJECT` | A3용 사전등록 좌표 변환(추정 아님) |
| `calibration_pipeline/fk_alignment.py` | `estimate_board_free_fk_cube_artifact()` | board 없이 train eye-in-hand cube corner만으로 `T^G_Cg`와 FK–큐브 델타 추정. held-out 이벤트가 섞이면 예외 |
| `calibration_pipeline/fk_factor.py` | — | corrected-FK soft factor 잔차 블록 |
| `calibration_pipeline/observations.py` | — | manifest → `PixelObs` |
| `calibration_pipeline/evaluation.py` | — | 재투영 지표 집계 |
| `calibration_pipeline/path_evaluation.py` | — | cross-view / cam-common 일관성 |

## 최종 촬영 흐름

최종 protocol의 P1 15 / P2 20 / P3 10 구성, pose JSON 생성·검증, robot server 실행,
`03_capture.py` 명령, 저장 metadata와 완료 판정은 모두
[Zeus 최종 촬영 파이프라인](zeus_gello_calibration/PIPELINE.md)을 따른다.
실제 rig geometry와 45 pose teaching, 새 큐브 grip/place 높이 재실측, 저속 dry run을
통과하기 전에는 본 촬영을 시작하지 않는다.

현재 final protocol은 03 촬영과 04 frozen-observation 생성까지 실행 가능하다. 05에는
P2 placement-grouped split과 P3 독립 그룹 검사가 들어갔지만, P1에서 함께 움직이는
board/cube를 하나의 common-rig pose로 푸는 phase-aware residual 모델은 아직 구현 전이다.
따라서 `composite_rig_45_v2` 데이터의 05는 위 문서의 완료 표시 전까지 실행하지 않는다.

## 각 calibration 행렬은 언제 나오는가

행렬 표기 `T_A_B`는 **B 좌표의 점을 A 좌표로 변환**한다. 모든 4×4 SE(3)
행렬의 translation 단위는 meter다.

| 시점 | 행렬 | 생성 방식 | 저장 위치 / 최종 여부 |
| --- | --- | --- | --- |
| 01 | `color_K`, `color_D`, `depth_K`, `depth_D`, `R_depth_to_color`, `t_depth_to_color` | RealSense factory calibration을 읽음 | `intrinsics/cam*.npz`; intrinsic 초기값 |
| 02 | refined `color_K`, `color_D` | ChArUco view로 재추정 | 같은 `cam*.npz` 갱신; **05에서 고정 사용** |
| 03 | event별 `T_base_gripper`, set별 cube-center FK pose(보정 전) | robot controller FK/기록값 | `calib_train/meta.json`; optimizer 입력이며 calibration 결과 아님 |
| 04 | board/cube PnP pose | 검출 품질과 positive-depth 확인을 위한 임시 solvePnP | 최종 calibration 행렬로 전달하지 않음 |
| 05 공통 초기화 | `shared_reference_state`, `row_reference_states` | train 관측만 사용한 PnP/robust pose 초기화 | `shared_train_only_baseline.json`; optimizer 시작점 |
| 05 FK 정렬 | `T_gripper_cam`, `T_fk_cube_center_to_tag_object`, `raw_fk_pose_by_set`, `aligned_fk_pose_by_set` | train-only board-free FK–cube alignment | `shared_board_free_fk_cube.json`; A4/A5/B1/B2 입력 |
| 05 각 행·seed 종료 | `T_base_Ci`, `T_gripper_cam`, `T_base_board`, `T_base_cube_by_set` | raw-corner reprojection fit 후 prune/refit 결과가 개선되면 채택, 아니면 첫 fit으로 rollback | `ABLATION_TEST_table1_methods.json → rows.<행>.runs[*].transforms`; **최종값** |

최종 배포 대상은 `T_base_Ci = T^B_Ci`와 `T_gripper_cam = T^G_C`다.
`T_base_board`와 `T_base_cube_by_set`은 카메라들을 같은 좌표계로 묶는 target pose라서
camera calibration 배포 파일과 구분한다. 대표 행렬은 held-out 점수로 고르지 않고,
사전에 고정된 unperturbed initialization인 seed 0을 표시한다.

## 05 결과를 읽는 정확한 위치

```text
ABLATION_TEST_table1_methods.json
└── rows
    └── A0 ... A5, B1 ... B3
        └── runs[seed 0, 1, 2]
            ├── converged
            ├── stages.*.frame_prune_refit
            ├── train_reprojection.overall.rmse_px
            ├── heldout_reprojection.overall.rmse_px
            ├── cube_evaluation_reprojection.heldout.cube.rmse_px
            └── transforms
                ├── T_base_Ci.{0,1,3}
                ├── T_gripper_cam
                ├── T_base_board
                └── T_base_cube_by_set.<set>
```

`frame_prune_refit.accepted=true`이면 제거 후 재적합 행렬이 최종값이다.
`rolled_back=true`이면 전체 train robust objective가 개선되지 않아 제거 전 행렬이 최종값이다.
둘 다 정상 종료이며, 행별 시도/채택/rollback 수는 위 JSON의 `frame_prune_refit`에서 확인한다.

## 선택 평가 — calibration 완료 후 필요할 때만

```bash
# 동일 frozen split의 cross-view camera consistency 평가
python3 tools/evaluate_cross_target.py --root_folder data/session02_NOUSE_session04_0814/calib_train

# board-only / cube-only / both marker-system end-to-end 비교
python3 tools/compare_markers.py --root_folder data/session02_NOUSE_session04_0814/calib_train

# VISION OpenCV fixed-camera relative-pose 기준선
python3 tools/opencv_baseline.py --root_folder data/session02_NOUSE_session04_0814/calib_train
```

## 다른 checkout에서 재실행할 때

`Step2b_observation_manifest.json`은 촬영한 머신의 절대 경로를 기록한다. 다른
checkout에서 그대로 재실행하면 session root 불일치로 05가 중단된다. 이때
`--allow-relocated-session-root`를 05·`evaluate_cross_target`·`compare_markers`에
함께 준다. 이 플래그는 기록된 경로 접두사만 현재 checkout으로 옮기며, meta.json,
intrinsics, 모든 영상의 SHA-256 검증은 그대로 수행한다. 즉 무결성 계약은
경로 문자열이 아니라 해시가 계속 담당한다.

보조 평가 두 개는 `--include_sets`와 `--split_seed`를 05와 동일하게 주어야 한다.
기본값(`5-12`)으로 실행하면 split이 달라져 `reconstructed split does not match
stored results`로 중단된다.

이 세 평가는 05의 calibration 행렬을 만드는 필수 단계가 아니다. 과거의 통합
최종 Table 1 Markdown/HTML을 갱신할 때만 `tools/sync_table1_canonical_data.py`를 사용한다.

## 현재 Session04 결과 위치

- 모든 행·seed의 정확한 행렬: `ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/ABLATION_TEST_table1/calibration_matrices.json`
- 행별 수렴·오차·prune 요약: `ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/ABLATION_TEST_table1/calibration_summary.csv`
- 계산 원본: `ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json`
- 최종 ABLATION_TEST와 평가지표: `ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/ABLATION_TEST_table1/ABLATION_TEST_TABLE1_RESULTS.md`

Session04 현재 결과는 9개 행×3개 seed 모두 수렴했다. 최종 표의 Train Cube RMSE와
heldout은 `cube_evaluation_reprojection`의 cube-only 값으로 통일했으며,
camera/Hand–Eye는 frozen하고 train cube로 set별 evaluation pose만 맞춘다. Row별
marker 모집단이 다른 solver Train RMSE는 계산 원본에만 남고 최종 비교표에는 쓰지
않는다. 현재 보고서 생성 시점의
robot-base 절대 정확도는 계산하지 않고, 다음주
Independent External GT 태스크에서 Translation Error, Rotation Error, P95,
Failure Rate를 산출한다.

COLMAP/MATLAB과 point cloud는 현재 calibration 완료 범위에서 제외하고 후속
작업으로 유지한다. Robot task/외부 GT는 다음주 예정 태스크로 분리한다.
