# 회의 피드백 반영 점검 — 0914

**평가 지표와 비교축은 상당 부분 반영됐다. 그러나 이상치 제거·재적합, FK 영향도 진단, 독립 물리 검증까지 완료된 것은 아니다.**

이 문서는 실제 `capture_*_0914` 결과를 기준으로 한다. [8-3_meeting.md](../../8-3_meeting.md)의 Session04 완료 기록과 [9-2_meeting.md](../../9-2_meeting.md)의 0909 코드 점검은 당시 상태이며, 0914 실행 완료의 증거로 재사용하지 않는다.

## 1. 피드백과 현재 결과

| 피드백 출처 | 판정 | 0914에서 확인한 반영 내용과 한계 |
| --- | --- | --- |
| 8/3 #5·#8·#10–12, 9/2 §2·§7·§9·§11: 공통 Cube 평가, 손목 카메라, held-out | **핵심 반영** | 모든 방법에 같은 placement split과 Cube corner 평가. Cube·Cross-view 각각 ALL/Train/Test, fixed↔fixed·fixed↔gripper·전체를 산출한다. 양방향 단일 카메라 PnP 전달과 같은 RMSE 식을 사용하며 `heldout_px_vision`은 최종 표에 없다. #8의 Cam-common mm/deg 보조표는 이번 요청의 픽셀·External GT 표에 포함하지 않았다. |
| 8/3 #3·#9·#13·#19, 9/2 §1·§3·§8·§10: 목적함수와 FK 비교축 | **부분 반영** | Sequential/Unified, VISION/raw-FK/corrected-FK, board-only A0/B3를 구분한다. RGB corner 잔차와 FK soft factor를 사용하며 Cross-view는 평가용이다. **A3는 독립 장착 변환이 없어 Pending**. A1/B1은 효과 분리용 비교군이며 최종 기여 방법을 확정하지 않았다. |
| 8/3 #1·#2·#4·#6·#18: 이미지 이상치, 재적합, FK 항 영향 | **부분 반영** | 검출 선별과 robust solver는 사용한다. 그러나 Zeus 실행에는 **이미지 단위 prune→refit→rollback이 연결되지 않았고**, visual/FK cost 분해·공분산 민감도 결과도 없다. 기하·intrinsics 보정 기록은 있지만 체계 오차의 독립 검증까지 끝난 것은 아니다. 기존 Session04 결과와 구분해야 한다. |
| 8/3 #7·#14–17, 9/2 §4·§6·§12.7–12.8: 물리 정확도와 기여 검증 | **0914 결과로 미검증** | 독립 GT의 TRE·회전·P95·실패율은 모두 Pending이다. 반복 재파지/작업 성공률, 외부 구현 대조, robot-base point-cloud, 촬영 수·카메라 수·다른 로봇 비교도 이 실행에 포함되지 않는다. |
| 9/2 §5·§6·§12.6–12.8: 공정한 데이터 구성과 재현성 | **부분 반영** | 출처·설정·split·행렬을 JSON에 저장하고 같은 JSON에서 표를 생성한다. 다만 이동 보드의 대등한 재촬영까지 검증한 것은 아니며, **기하·scale에 P2/GT 영상이 사용돼 기하를 고정한 조건부 일반화**다. 모든 행은 P1 Cube·P3 Board 기반 초기값을 공유하므로 표적 제거는 목적함수 잔차 기준이다. |

근거: [zeus_gello_calibration/table1_zeus.py](../../zeus_gello_calibration/table1_zeus.py)의 `ROWS`, `fit_row`, `evaluate_fold`, `cross_view_transfer_stats`; [INPUT_AUDIT.md](INPUT_AUDIT.md) §2–5.

## 2. 새 데이터에도 고정할 계산 원리

1. **같은 분할:** P2 placement를 하나씩 제외한다. 그 placement의 모든 카메라·Cube·Board를 calibration에서 제외하고, P1/P3는 별도 학습 자료로 유지한다. 데이터가 늘면 fold 수만 늘어난다.
2. **같은 ALL 정의:** 모든 placement로 calibration을 별도 fit하고 전체 P2 Cube를 평가한다. Train/Test 점수의 평균이 아니다. Train은 각 fold의 학습 placement, Test는 제외 placement를 평가한다.
3. **같은 Cross-view:** A 영상만으로 Cube PnP → 카메라 간 변환 → B corner 채점. B→A도 계산한다. 각 방향의 destination 영상은 그 방향의 pose 추정에 사용하지 않는다. fixed↔gripper 전달에는 촬영 순간 Robot FK가 포함된다.
4. **같은 단위와 집계:** `RMSE px = sqrt(Σ(dx² + dy²) / 평가 corner 수)`. fold별 RMSE를 단순 평균하지 않는다. 모든 방법에 같은 평가 관측을 적용하고 사용 corner·카메라 쌍·수렴 수를 함께 확인한다.
5. **같은 해석:** 내부 주 지표는 Held-out Test Cross-view다. 공통 FK-reference Cube RMSE는 보조 지표이며, 최종 순위는 독립 External GT로만 정한다. 미확보 값은 사유와 함께 `Pending`으로 유지한다.

9/2 §2의 “Cross-view는 전체만 평가”는 §11의 재질문과 현재 요구에 따라 **ALL/Train/Test 분리로 수정된 항목**이다. 과거 설명을 그대로 유지하는 것이 피드백 반영은 아니다.

## 3. 결과물이 달랐던 이유와 고정할 산출물

0909에는 [05_calibrate.py](../../05_calibrate.py)·[06_make_report.py](../../06_make_report.py)의 legacy 실행과 별도 진단이 포함됐다. 0914는 [zeus_gello_calibration/table1_zeus.py](../../zeus_gello_calibration/table1_zeus.py)·[zeus_gello_calibration/report_table1.py](../../zeus_gello_calibration/report_table1.py) 경로다. **행 이름이 같아도 실행기·split·초기화·prune 설정이 다르면 같은 실험이 아니다.**

새 Zeus 데이터에는 같은 실행기·지표 정의·출력 이름을 사용하고 입력 경로와 설정을 기록한다. 이제 `06_make_report.py --table1 <JSON>`도 v4 JSON이면 같은 간략 보고서 생성기를 호출한다. 보고서 길이는 줄여도 아래 원본은 보존한다. 재실행 명령은 [README.md](README.md)에 있다.

| 산출물 — `ABLATION_TEST_table1/` 내부 | 역할 |
| --- | --- |
| [ABLATION_TEST_TABLE1_RESULTS.md](ABLATION_TEST_table1/ABLATION_TEST_TABLE1_RESULTS.md) | 실험 조건·요청한 10개 지표·해석 한계의 간략한 표 |
| [ABLATION_TEST_table1_results.csv](ABLATION_TEST_table1/ABLATION_TEST_table1_results.csv) | 방법별 지표 원래 정밀도 |
| [fold_metrics.csv](ABLATION_TEST_table1/fold_metrics.csv) | ALL/fold별 지표·카메라 쌍 구분·평가 표본 수 |
| [ABLATION_TEST_table1_methods.json](ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json) | 계산 원본·입력 출처·설정·split·변환행렬 |

현재는 **Zeus 결과 형식과 평가 원리는 재사용할 수 있지만, legacy 실행의 모든 진단까지 자동으로 같아진 상태는 아니다.** 출력 형식 통일과 solver 절차 통일을 구분해야 한다.

## 4. 아직 필요한 검증

1. Zeus solver에 학습 이미지 단위 prune/refit/rollback과 visual/FK cost 기록을 연결한 뒤 전체 비교를 다시 실행한다.
2. 같은 데이터에서 FK 공분산 민감도와 반복 촬영 변동폭을 확인한다. 현재 `2 mm / 0.30 deg`는 실측 공분산이 아니다.
3. A3용 비전 독립 `T_flange_cube`와 독립 External GT를 확보해 미산출 행·열을 채운다. `robot.json`의 로봇 pose만으로 두 정보를 대체할 수 없다.
4. 전체 처리 과정의 일반화를 주장하려면 기하·scale도 별도 자료로 고정하거나 fold의 train 자료만으로 추정한다.

**다음 1분:** 위 §1의 판정표를 보고 발표에서는 “평가 지표 정비 완료, 물리 정확도와 추가 진단은 미검증”으로 범위를 구분한다.
