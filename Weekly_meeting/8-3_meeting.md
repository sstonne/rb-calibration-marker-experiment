# 8/3 미팅 피드백 반영 현황 (2026-09-04 확인)

원본: [`8-3_meeting.txt`](8-3_meeting.txt) (1,584줄). 발언 시각 기준으로 피드백 19건을
추출하고 현재 코드와 Session04 산출물에서 직접 확인했다. 상태는 **코드와 실행 계약 기준**이며,
2026-09-02에 새 frame-prune 코드로 Session04 canonical 결과까지 다시 생성했고,
2026-09-03에 내부 지표 기반 보고 범위와 다음주 External GT 태스크 표현을 갱신했으며,
2026-09-04에 #17 robot-base point-cloud 진단 산출물을 추가했다.

**요약: ✅ 반영 11건 · ⚠️ 부분 반영 8건 · ❌ 미반영 0건**

> **현재 실행 제약:** 현재 저장된 Session04 데이터만으로는 독립 물리 GT를 계산할 수 없다.
> 따라서 #6 Track A와 #14–#16은 코드·스키마·분석기 준비까지만 완료로 보고하고,
> 실측 GT 평가는 다음주 예정 태스크로 둔다. Session04 aligned depth는 #17의
> robot-base point-cloud 진단에는 사용했지만, 독립 물리 GT가 아니므로 robot task
> 정확도의 대체 증거로 사용하지 않는다.
>
> **현재 마일스톤:** 캘리브레이션 파이프라인 완성까지를 현재 범위로 한다. COLMAP/MATLAB
> external baseline 실행은 후속 단계로 미루고, Independent External GT는 다음주 예정
> 태스크로 진행한다.

## 피드백 유형별 묶음

각 피드백은 발표에서 길게 나열하지 않고, 아래 **7개 분류** 중 하나로 먼저 묶어서 설명한다.
한 피드백은 하나의 primary 분류에만 배정하고, 세부 표에서 실제 코드·데이터 반영 상태를 추적한다.

| 분류 | 포함 피드백 | 핵심 질문 | 현재 처리 |
| --- | --- | --- | --- |
| C1. 관측 품질 / 이상치 | #1, #2, #18 | 검출·이상치 처리가 실제 이미지 단위로 되었는가? | image-level frame-prune, refit/rollback, overlay QA로 증명 |
| C2. 목적함수 / FK 사용 | #3, #4, #9, #19 | 최적화가 무엇을 최소화하고 FK는 어떤 역할인가? | camera-to-camera residual 없음, visual/FK block 분리, A2/A4/A5 역할 분리 |
| C3. 평가 공정성 / 지표 | #5, #8, #10, #11, #12 | 특정 방법에 유리한 지표만 고른 것은 아닌가? | heldout cube, External cube GT, cross-view camera consistency로 단일화 |
| C4. 기여도 / ablation 포지셔닝 | #13 | VISION/independent row를 기여로 볼 것인가? | A1/B1은 제안 방법이 아니라 A2/A4 효과 분리 baseline으로 제한 |
| C5. 큰 오차 원인 진단 | #6 | 10 mm / 10 px 수준의 큰 수치가 어디서 오는가? | cube geometry/corner ordering/refinement 수정 후 board-cube conflict를 추적 |
| C6. 외부 구현 대조 | #7 | 코드 문제인지 데이터 문제인지 공개 구현으로 대조했는가? | OpenCV reference와 frozen external baseline package 완료, MATLAB/COLMAP 실행은 후속 |
| C7. 물리 GT / 로봇 작업 검증 | #14, #15, #16, #17 | 내부 residual이 실제 robot-base 정확도로 이어지는가? | Track C schema/evaluator 준비, robot-base point-cloud diagnostic 구현, Independent External GT와 robot task는 다음주 예정 |

## 피드백 재해석 원칙

피드백은 그대로 정답으로 수용하지 않고, **문제의식은 반영하되 현재 코드·데이터와
맞지 않는 구현 방식은 반박하거나 제한**한다. 특히 다음 항목은 이미 개선된 상태지만,
발표에서는 아래처럼 설명해야 한다.

| 번호 | 피드백의 문제의식 | 그대로 따르면 생기는 문제 | 현재 반영 방식 |
| ---: | --- | --- | --- |
| #3 | reprojection error와 카메라 간 관계를 명확히 하라 | camera-to-camera residual을 objective에 직접 넣으면 shared target 관측을 중복 사용할 수 있다 | 메인 optimizer에는 camera-to-camera residual 0개. shared target pose 변수로만 카메라를 결합한다고 설명한다. |
| #8 | mm 지표도 의미가 있다 | external GT 없는 mm를 절대 robot-base 정확도처럼 해석할 위험이 있다 | Cross-view pixel transfer와 Cam-common Obj-Cam mm/deg를 cube-only 내부 일관성 보조 지표로 제한한다. |
| #13 | FK 없는 독립 방식을 기여로 둘 필요가 있는가 | A1/B1을 제안 방법처럼 보이면 논문 기여도가 흐려진다 | A1/B1은 contribution이 아니라 A2/A4 효과를 분리하는 ablation baseline으로 둔다. |
| #19 | FK를 어떻게 쓰고 무엇을 제안 방법으로 둘지 정하라 | 내부 px 최저값만 보고 final winner를 정하면 external GT 전 claim이 과해진다 | A0~A5/B1~B3를 최종 후보로 유지하고, A5도 GT 전 frozen이면 최종 후보로 비교한다. |

| # | 시각 | 피드백 | 분류 | 상태 | 근거 / 남은 것 |
| ---: | --- | --- | --- | :---: | --- |
| 1 | 00:27 | 이상치는 **이미지 레벨로 프레임을 지워라** | C1 관측 품질 / 이상치 | ✅ | [`select_frame_prune_subset`](calibration_pipeline/reprojection.py#L592)이 같은 `(event_id, camera_id)`의 board/cube 관측 전체를 한 frame으로 묶어 MAD 기준으로 최대 30% 제거한다. 이 코드로 Session04 canonical 결과를 재생성했다. |
| 2 | 02:16–04:21 | Robust loss로 영향만 줄이는 것과 제거 후 **extrinsic을 다시 푸는 것**은 다르다 | C1 관측 품질 / 이상치 | ✅ | [`run_frame_prune_refit`](calibration_pipeline/table1.py#L575)이 `fit → frame-prune → refit → rollback`을 수행한다. 제거 전 train 전체의 동일 robust objective가 개선될 때만 refit을 채택하며 held-out은 선택·재적합·판정에 쓰지 않는다. Canonical 42개 solver stage 중 15개가 prune/refit을 실행했고, 모두 full-train cost가 증가해 1차 결과로 정상 rollback됐다. |
| 3 | 06:18–10:17 | 3-2에서 reprojection error를 쓰는가? 카메라 간 관계도 목적함수에 넣는가? | C2 목적함수 / FK 사용 | ✅ | 메인 optimizer의 camera-to-camera residual은 0개이고, 카메라는 공유 target pose 변수로 결합된다. VISION 행은 1항, corrected-FK soft factor 행은 visual+FK 2항이다. [`ABLATION_TEST_TABLE1_RESULTS.md`](ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/ABLATION_TEST_table1/ABLATION_TEST_TABLE1_RESULTS.md) |
| 4 | 26:44 | 카메라가 많으면 관측 수가 많아 **FK 항이 묻힌다** | C2 목적함수 / FK 사용 | ✅ | Objective Block Diagnostics에 visual/FK residual 수와 cost를 분리했다. 기존 결과의 FK cost fraction은 A4 `0.122%`, B1 `0.145%`, B2 `1.139%`이며 이를 parameter 영향력 비율로 해석하지 말라는 제한도 기록했다. 추가로 A4 FK covariance scale sensitivity를 실행해 corrected-FK soft factor가 완전히 무시된다고 보기는 어렵다는 점을 확인했다. A4의 최종 우월성은 External cube GT로 판정한다. [`FK_FACTOR_SENSITIVITY.md`](ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/fk_factor_sensitivity/FK_FACTOR_SENSITIVITY.md) |
| 5 | 33:20 | FK 경로에 유리한 지표만 보고하는 것처럼 보이면 안 된다 | C3 평가 공정성 / 지표 | ✅ | 최종 표는 External cube GT, heldout cube RMSE, cross-view pixel transfer, Cam-common Obj-Cam consistency만 사용한다. 내부 camera consistency가 좋아도 “우리 방법이 물리적으로 우수”라고 결론 내리지 않는다. |
| 6 | 36:37–38:18 | **전체 수치가 너무 크다**(약 10 mm / 10 px). 원인을 찾아라 | C5 큰 오차 원인 진단 | ⚠️ | Cube geometry/corner ordering/refinement 오류를 수정해 direct-PnP target conflict를 `17.299 → 10.808 mm`로 낮췄다. 즉시 triage에서 재현성, leave-one event, cube 관측 하한, same-support 조건을 추가 확인했고, event 54 제거 후에도 `9.599 mm`가 남아 단일 프레임 문제는 아니었다. 현재 진단은 cube sparse observation과 target-dependent effective scale/localization bias, 제한된 intrinsic coverage가 섞인 체계 오차다. 확정에는 Track A 반복 촬영이 필요하다. [`BOARD_CUBE_RELATIVE_POSE.md`](data/session02_NOUSE_session04_0814/calib_out/verify/board_cube_relative_pose/BOARD_CUBE_RELATIVE_POSE.md), [`BOARD_CUBE_CONFLICT_TRIAGE.md`](data/session02_NOUSE_session04_0814/calib_out/verify/board_cube_relative_pose/BOARD_CUBE_CONFLICT_TRIAGE.md) |
| 7 | 38:18, 1:47:22–1:55:53 | 같은 사진을 **공개 구현**에도 넣어 데이터 문제인지 코드 문제인지 확인하라 | C6 외부 구현 대조 | ⚠️ | OpenCV PnP 기반 독립 fixed-camera 기준선을 구현·실행했고, 추가로 외부 MATLAB/COLMAP/public baseline adapter가 같은 frozen RGB·2D/3D corner·intrinsic·split만 쓰도록 neutral input package를 생성했다. 현재 package는 154 observations / 4625 corners, 이미지 81개와 intrinsics 4개 SHA-256을 검증한다. 실제 MATLAB/COLMAP 실행은 후속 단계지만, 입력 계약은 완료됐다. [`OPENCV_RELATIVE_BASELINE.md`](ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/opencv_relative_baseline/OPENCV_RELATIVE_BASELINE.md), [`EXTERNAL_BASELINE_PACKAGE.md`](ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/external_baseline_package/EXTERNAL_BASELINE_PACKAGE.md) |
| 8 | 43:32–47:33 | **mm 지표를 왜 뺐나.** 절대값은 아니어도 이 세팅에서는 의미 있다 | C3 평가 공정성 / 지표 | ✅ | Cam-common Obj-Cam consistency를 mm/deg로 보고하되, External cube GT 정확도와 구분한다. |
| 9 | 44:26 | RGB-D depth를 캘리브레이션 계산에 쓰는가? | C2 목적함수 / FK 사용 | ✅ | 최종 Table 1 solver는 RGB의 3D-object/2D-pixel 대응과 고정 intrinsic만 사용한다. 센서 depth는 목적함수에 들어가지 않는다. |
| 10 | 55:32–56:01 | Train/test가 같은 이미지의 코너를 나눠 가지면 안 된다 | C3 평가 공정성 / 지표 | ✅ | 전사에서 event가 이미 분리됐음을 확인한 항목이다. 현재도 event-grouped, set-stratified split이며 같은 event의 모든 카메라·코너는 한쪽에만 속한다. [`schema.py`](calibration_pipeline/schema.py#L416) |
| 11 | 1:01:06–1:03:31 | 마커 구성이 다른 행끼리 직접 비교하지 말고 공통 지표를 써라 | C3 평가 공정성 / 지표 | ✅ | 최종 비교는 모든 row를 같은 External cube GT pose list와 heldout cube target에서 비교한다. |
| 12 | 1:11:11–1:15:46 | 고정카메라만 보는 공통 지표 외에 **손목 카메라 포함 지표**도 필요하다 | C3 평가 공정성 / 지표 | ✅ | Cross-view pixel transfer와 Cam-common Obj-Cam consistency에 fixed-gripper camera pair를 포함해 combined 지표로 집계한다. |
| 13 | 1:33:26–1:34:18 | FK 없는 “독립” 방식을 굳이 기여로 둘 필요가 있는가 | C4 기여도 / ablation 포지셔닝 | ⚠️ | A1은 제안 방법이 아니라 A2와 동일 cube+board 관측에서 Sequential→Unified 효과만 분리하는 ablation으로 유지 중이다. 교수님도 “해도 되지만 굳이 기여는 아니다”라는 취지였으므로 삭제보다 **기여 방법으로 부르지 않는 것**이 현재 판단이다. 논문 지면에서 행을 뺄지는 최종 표 편집 결정이다. |
| 14 | 1:36:51–1:37:24 | 핵심은 카메라 간 오차보다 **로봇 그리퍼의 작업 정확도**다 | C7 물리 GT / 로봇 작업 검증 | ⚠️ | Blind prediction·외부 GT 채점 코드와 Track C 계약은 준비됐다. 독립 GT/재파지/작업 데이터는 다음주 예정 태스크에서 수집·평가한다. [`CAPTURE_CAMPAIGN_PROTOCOL.md`](protocol_templates/CAPTURE_CAMPAIGN_PROTOCOL.md#4-track-c--외부-gt) |
| 15 | 1:37:24–1:39:01 | **눈금 큐브** 재파지로 x/y/z 오차를 실측하라 | C7 물리 GT / 로봇 작업 검증 | ⚠️ | Track C에 독립 pose/GT 측정 절차를 정의했고, 추가로 눈금 큐브는 6-DoF GT가 아니라 translation/frame gross-error sanity check로만 쓰는 계약을 분리했다. 다음주 실측 때 prediction SHA-256 동결 후 signed `x/y/z` error를 기록한다. [`CAPTURE_CAMPAIGN_PROTOCOL.md`](protocol_templates/CAPTURE_CAMPAIGN_PROTOCOL.md#41b-눈금-큐브-translation-sanity-check) |
| 16 | 1:41:41–1:42:47 | **peg-in-hole 또는 grasp success rate/정밀도**를 평가하라 | C7 물리 GT / 로봇 작업 검증 | ⚠️ | Paired task-trial schema와 evaluator를 구현했다. 모든 방법의 동일 pair 기록을 강제하고 success rate/Wilson 95% CI, XYZ contact error, P95, paired 차이를 출력한다. 실측 robot trial은 다음주 외부 GT/robot-task 일정에서 재개한다. [`task_trial.py`](calibration_pipeline/task_trial.py) |
| 17 | 1:42:47–1:44:28 | **point cloud 정합을 로봇 관점**에서 표현하라 | C7 물리 GT / 로봇 작업 검증 | ⚠️ | 각 Table 1 비교실험 row의 transform과 aligned depth를 사용해 selected board/cube surface depth를 robot-base frame으로 올리는 diagnostic을 구현했다. 대표 이벤트 24/54/72에 대해 A0–A5/B1–B3 전체 row의 XY/XZ/YZ projection과 depth-to-model-plane residual을 산출한다. A2는 board `9.042 mm` RMSE, cube `8.580 mm` RMSE다. 단, 이는 depth 기반 정성/진단 자료이며 external GT나 robot-contact accuracy가 아니다. [`ROBOT_BASE_POINTCLOUD_DIAGNOSTIC.md`](ABLATION_TEST_result_0909/session02_NOUSE_session04_0814/robot_base_pointcloud/ROBOT_BASE_POINTCLOUD_DIAGNOSTIC.md) |
| 18 | 1:40:07–1:40:33 | 오버레이로 마커 검출 정확도를 정성 확인하라 | C1 관측 품질 / 이상치 | ✅ | Cube 재검출 overlay 도구가 있다. 전사 취지대로 이는 **검출 QA**이며 로봇 작업 정확도 지표로 승격하지 않는다. [`render_cube_redetection_overlays.py`](tools/render_cube_redetection_overlays.py) |
| 19 | 1:47:12 | FK를 어떻게 쓸지, 무엇을 제안 방법으로 둘지 정해야 한다 | C2 목적함수 / FK 사용 | ⚠️ | 현재 결정은 **A0~A5/B1~B3 한 벌을 최종 후보로 유지**하는 것이다. A5는 External GT 공개 전에 방법과 artifact가 frozen이면 최종 후보로 비교 가능하다. 최종 물리 순위는 다음주 Independent External cube GT 이후에만 확정한다. |

## 남은 작업의 성격과 순서

1. **완료:** 새 frame-prune 코드로 Table 1 전체와 보조 진단을 재실행하고 CSV/Markdown/HTML을 동기화했다. 추가 실험은 [`ADD_EXPERIMENTS_SUMMARY.md`](ADD_EXPERIMENTS_SUMMARY.md)에 한 표로 정리했다.
2. **완료:** #4 FK 항 영향도는 cost fraction만 보지 않고 A4 covariance scale sensitivity까지 추가해 External GT 전 보조 근거로 분리했다.
3. **현재 범위:** 캘리브레이션 입력 검증, 최적화, frame-prune/refit/rollback, 평가 산출물과 재현성 검증만 마무리한다.
4. **부분 완료/후속 — external baseline:** #7의 OpenCV reference와 external baseline frozen-input package는 완료했다. MATLAB multiview/COLMAP adapter 실행은 같은 package를 입력으로 후속 진행한다.
5. **완료/진단 — point cloud:** #17은 현재 A0–A5/B1–B3 비교 row 전체와 aligned depth로 robot-base point-cloud projection과 depth-to-plane metric까지 구현했다. 단, 이 결과는 물리 GT를 대체하지 않는다.
6. **다음주 예정 — robot task/GT:** #6 Track A, #14–#16 Track C와 #17의 robot-contact/physical GT 검증은 코드 scaffold를 유지하고 Independent External GT 수집/평가로 재개한다.

## 판정

기존 문서는 frame-prune 구현 전 상태와 현재 저장소를 섞어 기록했고, OpenCV baseline과 external-GT
scaffold를 누락했으며, 공개 baseline 대비 우위도 잘못 해석했다. 위 표가 현재 코드 기준의 수정된
tracking이다. 가장 큰 미완료 항목은 여전히 **독립 물리 GT를 사용한 robot task 정확도 실험**이며,
이는 다음주 예정 태스크로 분리한다.
