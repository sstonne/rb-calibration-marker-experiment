# 촬영 캠페인 프로토콜 (Track A/B/C 통합)

> 상태: 실행 전 초안. 촬영 **전에** 4절의 사전 등록 값을 채우고 커밋해야 한다.
> 대상: 현재 열려 있는 촬영 및 외부 GT 실험.

## 0. 왜 한 번에 하는가

지금 열려 있는 세 문제는 서로 달라 보이지만 **모두 "같은 것을 반복해서 찍고 독립적으로 재는"
동일한 촬영 동작을 요구한다.**

| Track | 푸는 문제 | 없으면 막히는 것 |
| --- | --- | --- |
| **A** | Corner localization covariance 실측 | Board–Cube 잔차의 근본 원인. `corner_weighting_ablation`이 지목한 유일한 다음 수단 |
| **B** | FK covariance 실측 | A4/B1/B2가 preflight를 못 벗어남. "Ours-core" label이 근거를 못 얻음 |
| **C** | 외부 GT + 로봇 task | 절대 정확도. 교수님 피드백 5묶음 중 유일하게 손도 못 댄 항목 |

A와 B는 **같은 반복 촬영 세션**에서 동시에 얻는다. C는 jig가 추가로 필요하지만 같은 캠페인에
붙일 수 있다. 따로 세 번 나가면 세 번의 셋업 오차가 생긴다.

## 1. 공통 셋업 (모든 Track)

- 고정카메라 4대와 그리퍼카메라의 배치를 캠페인 내내 **절대 바꾸지 않는다.** 한 번이라도
  건드리면 그 시점 이후는 별도 세션으로 기록한다.
- Intrinsic은 재추정하지 않고 `intrinsics/cam*.npz`를 그대로 쓴다. 바꿔야 한다면 캠페인 전에
  바꾸고 해시를 기록한다.
- 조명을 고정한다. Track A의 결과가 입사각·노출에 민감하므로 조명 변화는 교란이 된다.
- 매 세션 시작·종료에 동일 기준 자세를 1회씩 찍어 드리프트를 확인한다.

## 2. Track A — Corner localization covariance

**목적:** camera / marker / 입사각 구간별 corner 검출 오차의 공분산을 실측해, whitened
reprojection의 가중치로 사용한다.

`corner_weighting_ablation` 결론: 남은 Board–Cube 충돌의 원인은 intrinsic도 치수도 corner
ordering도 아니고 **영상 조건에 따라 달라지는 target-dependent corner localization bias**다.
Session04에는 held-out 누수 없이 이를 추정할 반복 관측이 없다. 그래서 새로 찍어야 한다.

**촬영:**

1. 로봇과 표적을 **완전히 정지**시킨 상태에서 같은 장면을 연속 `N_rep = 30`장 촬영한다.
   이것이 "같은 자세의 반복"이며, 여기서 나오는 corner 산포가 곧 검출 잡음이다.
2. 이를 아래 층(stratum)마다 반복한다. 층을 섞지 말고 층마다 따로 기록한다.
   - 카메라: 고정 4대 + 그리퍼 1대
   - 표적: board, cube
   - 입사각: 정면(0–20°), 중간(20–45°), 경사(45–70°)
   - 거리: 근(최소 작업거리), 중, 원(최대 작업거리)
3. 층당 최소 1자세, 가능하면 2자세. 층 수 × 자세 수 × 30장이 총 촬영량이다.

**분석 산출물:** 층별 2×2 corner 공분산(px²)과 층 판정 규칙. 자세는 정지해 있으므로 자세 추정
없이 **corner 좌표의 표본 공분산**만으로 얻는다. 여기서 자세를 추정해 잔차를 쓰면 모델 오차가
섞이므로 하지 않는다.

**주의:** 이 covariance는 **train 관측에만** 적용한다. held-out에 적용하면 평가가 모델 가정을
공유하게 된다.

## 3. Track B — FK covariance

**목적:** `protocol_templates/fk_covariance_TEMPLATE.json`을 실측값으로 채워
`--fk_covariance_json`으로 넘긴다. 그러면 A4/B1/B2가 Simulation prior preflight를 벗어난다.

**스키마 요구사항(템플릿에 명시된 것):**

- `twist_order`: `[rx_rad, ry_rad, rz_rad, tx_m, ty_m, tz_m]` — **rad와 metre**, deg/mm 아님
- `n_repeats`: **최소 7회** 독립 물리 반복 (6D 표본 공분산이 full-rank가 될 수 있는 최소값).
  실제로는 7은 하한이므로 **20회 이상**을 권고한다.
- `shared_covariance_6x6`: positive-definite
- `preregistered_before_blind_test: true` — 외부 GT를 보기 **전에** 확정해야 한다
- `blind_external_gt_used: false`

**촬영/측정:** 동일 목표 자세로 **로봇을 매번 다른 경로로 접근시켜** 정지시킨 뒤 자세를 기록한다.
같은 경로로만 반복하면 backlash가 상쇄돼 covariance를 과소추정한다. Track A의 반복 촬영과 같은
세션에서 수행하되, Track A는 "로봇 정지 + 반복 촬영", Track B는 "로봇 재접근 + 1회 촬영"으로
**동작이 다르다.** 섞지 않는다.

## 4. Track C — 외부 GT

**목적:** A4, A2, A5, A3가 예측한 $T^B_{cube}$를 캘리브레이션 영상·robot FK·corrected-FK와 독립적으로 측정한 $T^B_{cube,GT}$와 비교한다. `calibration_pipeline/external_gt.py`가 이미 구현돼 있고 **독립 GT 데이터만 없다.** `protocol_templates/external_gt_eval_manifest_TEMPLATE.json`을 채우면 바로 채점된다.

### 4.1 권장 측정계 구성

가장 좋은 방법은 laser tracker, optical tracker 또는 측정용 arm으로 로봇 base 기준점과 cube rigid body를 동시에 측정하는 것이다. Tracker 좌표계를 $W$라고 하면 다음을 독립적으로 얻어야 한다.

$$
T^B_{cube,GT}=(T^W_B)^{-1}T^W_{cube}
$$

- $T^W_B$: 로봇 base에 고정한 최소 3개의 비공선 기준점 또는 정밀 datum으로 session마다 측정한다.
- $T^W_{cube}$: cube에 부착한 측정용 rigid body로 측정하고, rigid body→cube object frame offset은 CMM/정밀 jig로 한 번 별도 측정한다.
- calibration에 쓰는 RGB 카메라, controller FK, A4 covariance, A5의 $\Delta_{train}$으로 GT를 만들면 안 된다.
- tracker가 없으면 base에 고정한 kinematic nest의 6-DoF pose를 CMM으로 먼저 측정하고 cube를 여러 nest에 반복 장착할 수 있다. 눈금자만으로 translation만 재면 rotation GT가 없으므로 최종 실험으로 부족하다.

### 4.1b 눈금 큐브 translation sanity check

8/3 피드백의 "눈금 큐브 재파지로 x/y/z 오차를 실측"은 **gross translation/frame error를
잡는 pilot check**로는 유효하지만, 그 자체로 최종 6-DoF external GT가 되지는 않는다.
따라서 다음 계약으로만 사용한다.

1. A2/A3/A4/A5의 prediction 파일을 먼저 생성하고 SHA-256을 동결한다.
2. 로봇 base에 고정한 눈금 fixture 또는 kinematic nest를 사용해 cube center의 signed
   `x/y/z` translation error(mm)를 기록한다.
3. 각 pose는 최소 3회 이상 재장착하고, 접근 방향을 바꾼 반복을 포함한다. 한 방향 접근만
   반복하면 backlash/contact bias를 과소평가할 수 있다.
4. rotation은 `measured=false`로 기록한다. rotation을 별도 측정하지 않았다면 TRE/ADD/6-DoF
   ranking에 사용하지 않는다.
5. 이 sanity check의 목적은 축 방향, scale, frame sign, 큰 systematic offset을 찾는 것이다.
   A2/A4 최종 우열은 4.1의 독립 6-DoF GT 또는 4.3의 paired robot task에서만 확정한다.

### 4.2 권장 표본 수와 배치

- 코드 최소값은 독립 camera-installation session 2개지만, 최종 실험은 **5 sessions × 30 blind poses = 150 poses**를 권장한다.
- session마다 카메라를 재설치하거나 최소한 mounting repeatability가 반영되도록 독립적으로 setup하고 calibration을 새로 수행한다.
- 30 pose는 workspace 중앙/가장자리, 근거리/원거리, 낮은/높은 입사각을 preregistered strata로 균등 배치한다.
- 각 pose에서 모든 방법은 동일한 frozen RGB·intrinsic·PnP 관측을 사용하고, calibration seed는 GT를 보기 전에 `run_index=0`으로 고정한다.
- 측정 순서는 무작위화하고 session 시작/종료 기준 pose로 tracker와 camera drift를 기록한다.

**순서가 중요하다.** `README.md` §17이 지정한 순서를 지킨다.

1. **먼저** 독립 측정계에서 동일 자세 왕복 반복으로 **robot/fixture/GT-system repeatability를 측정한다.**
   이 값이 곧 GT 불확실성의 하한이며, 템플릿의 `gt_uncertainty_floor`에 들어간다.
   이걸 모르면 이후 어떤 오차도 "로봇 탓인지 캘리브레이션 탓인지" 구분할 수 없다.
2. **그 다음** 독립 측정한 target pose로 FK absolute error와 GT measurement uncertainty를 계산한다.
3. 모든 고정카메라와 그리퍼카메라가 **같은 새 위치**의 board/cube를 동시에 저장하는 blind
   capture를 추가한다 (`README.md` §17-1).
4. 별도 세션에서 peg-in-hole 또는 grasp success/접촉 위치 오차를 측정한다. 이때 **인식
   알고리즘과 target은 모든 방법에 고정한다** (§17-3).

### 4.3 Peg-in-hole / grasp task-trial 기록

[`robot_task_trial_manifest_TEMPLATE.json`](robot_task_trial_manifest_TEMPLATE.json)을 복사해
촬영 전에 success 판정, 독립 XYZ 접촉 오차 측정법, 허용 margin을 채우고 커밋한다.
각 `session_id × pair_id`는 같은 target pose/stratum에서 모든 방법을 정확히 한 번씩 실행하는
paired unit이다. 실행하지 못한 attempt도 삭제하지 말고 `success=false`와 `failure_reason`으로
남긴다. 성공 attempt에는 독립적으로 잰 signed `x/y/z` contact error(mm)가 모두 필요하다.
방법 실행 순서는 pair마다 무작위화하고 `execution_order`에 기록한다.

평가기는 방법별 success rate와 Wilson 95% 구간, 성공 trial의 XYZ bias/absolute error와
Euclidean contact-error mean/P95를 출력한다. 두 방법 모두 성공한 pair의 접촉 오차 차이와
한 방법만 성공한 pair 수도 함께 보존한다. 한 session 결과는 pilot이며, 두 개 이상의 독립
session이 있어야 `confirmatory_ready=true`가 된다.

**촬영 전에 반드시 채워야 하는 사전 등록 값** (현재 전부 `null`이다):

```
margins.rotation_deg              margins.p95_tre_mm
margins.failure_rate              margins.worst_stratum_p95_tre_mm
margins.add_mm                    gt_uncertainty_floor.translation_mm
                                  gt_uncertainty_floor.rotation_deg
```

`bootstrap`은 `repetitions: 10000`, `seed: 20260806`으로 이미 고정돼 있다. 바꾸지 않는다.

**blind 계약:** calibration artifact와 A2/A3/A4/A5의 예측을 `python3 -m calibration_pipeline.blind_prediction`으로 GT를 **읽지 않고** 생성한 뒤 SHA-256과 함께 동결한다. 그 다음 채점 시점에만 GT를 잠금 해제한다. 예측 파일을 만든 뒤 GT를 보고 calibration, seed, 관측 제외 규칙 또는 예측을 다시 만들면 계약 위반이다.

GT와 prediction 파일은 모두 `base_cube_pose_predictions_v1` 스키마를 사용한다. GT의 각 pose는 `status: ok`, `T_base_cube: 4x4 matrix`, `strata: [...]`를 가져야 하며 prediction 실패는 누락하지 않고 `status: failure`로 남긴다. GT 파일은 `protocol_templates/external_gt_pose_TEMPLATE.json`, 전체 채점 manifest는 `protocol_templates/external_gt_eval_manifest_TEMPLATE.json`에서 시작한다.

## 5. 촬영 후 실행 순서

```bash
# 1) 새 세션의 frozen-corner manifest 생성
python3 04_filter_observations.py --session-root data/<새세션>/calib_train --intrinsics-dir intrinsics

# 2) 내부 체인 (경로 인자 불필요 — --root_folder에서 유도된다)
COMMON="--root_folder data/<새세션>/calib_train --include_sets 0-12 \
  --min_train_eih_cube_events 3 --split_seed 20260731 --observation-filter-policy standard"
python3 05_calibrate.py $COMMON --num_inits 3 \
  --fk_covariance_json protocol_templates/fk_covariance_<날짜>.json   # Track B 산출물

# 선택 평가: calibration 완료에는 필요하지 않음
python3 tools/evaluate_cross_target.py $COMMON --num_inits 3
python3 tools/compare_markers.py       $COMMON --num_inits 3
python3 tools/opencv_baseline.py       $COMMON

# 3) A2/A3/A4/A5 blind prediction을 동일 run_index=0으로 생성하고 동결
python3 -m calibration_pipeline.blind_prediction \
  --table1_result_json ABLATION_TEST_result_0909/<새세션>/ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json \
  --method A4 --run_index 0 --blind_root data/<새세션>/blind_test \
  --intrinsics_dir intrinsics --output data/<새세션>/predictions/A4.json
# 위 명령의 --method/--output만 A2, A3, A5로 바꾸어 반복한다.

# 4) 예측 파일 SHA-256을 기록한 뒤 GT 잠금을 해제하고 paired 채점
python3 -m calibration_pipeline.external_gt \
  --manifest protocol_templates/external_gt_eval_manifest_<날짜>.json

# 5) paired peg-in-hole/grasp trial 평가
python3 -m calibration_pipeline.task_trial \
  --manifest protocol_templates/robot_task_trial_manifest_<날짜>.json \
  --output_dir ABLATION_TEST_result_0909/<새세션>/robot_task_trial
```

## 6. 촬영 전 체크리스트

- [ ] Track A 층 목록과 층당 자세 수를 확정해 커밋했다
- [ ] Track B의 `n_repeats`와 접근 경로 다양화 방식을 확정했다
- [ ] Track C의 `margins`와 `gt_uncertainty_floor`를 **숫자로** 채워 커밋했다
- [ ] Robot task의 success 정의·접촉 오차 측정법·margin을 채워 커밋했다
- [ ] 위 세 가지를 **GT를 보기 전에** 커밋했다 (사전 등록)
- [ ] 카메라 배치·조명·intrinsic을 고정하고 해시를 기록했다
- [ ] OpenCV 버전을 기록했다 — 4.12와 4.13은 ChArUco corner가 다르다

마지막 항목은 실제로 물린 적이 있다. 2026-09-01에 cv2가 4.12.0 → 4.13.0으로 올라가면서 board
관측 8개의 corner가 바뀌었고, Table 1 held-out RMSE가 0.17~0.52% 움직였다. 캠페인 중간에
업그레이드하면 앞뒤 데이터를 합칠 수 없다.
