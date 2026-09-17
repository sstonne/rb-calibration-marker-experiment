# Table 1 결과 요약

**계산된 방법 8개 · placement 15개 · ALL 수렴 8/8 · fold 수렴 120/120.**

내부 주 지표는 **Held-out Test Cross-view Cube RMSE px**. 최종 물리 순위는 독립 External GT로 결정한다.

## 1. 전체 평가지표

각 열은 작을수록 좋으며 **최솟값**을 굵게 표시한다. 반올림 전 값으로 비교하고 동률은 모두 표시한다. 미산출 값은 Pending이며, 소수점 4자리로 표시한다.

| 실험 (구성) | ALL Cube RMSE px (Train + Held-out Test) | Train Cube RMSE px | Held-out Test Cube RMSE px | ALL Cross-view Cube RMSE px (Train + Held-out Test) | Train Cross-view Cube RMSE px | Held-out Test Cross-view Cube RMSE px | TRE mm | Rotation Error deg | P95 TRE mm | Failure Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0<br>(board+Seq+VISION) | 2.8597 | 2.8316 | 2.8361 | 5.9880 | 5.9816 | 5.9961 | Pending | Pending | Pending | Pending |
| A1<br>(board+cube+Seq+VISION) | 2.0714 | 2.0635 | 2.0886 | 4.2845 | 4.2723 | 4.3252 | Pending | Pending | Pending | Pending |
| A2<br>(board+cube+Unified+VISION) | 1.9813 | 1.9708 | 1.9738 | 4.3757 | 4.3553 | 4.3723 | Pending | Pending | Pending | Pending |
| A3<br>(board+cube+Unified+raw-FK hard fixed) | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| A4<br>(board+cube+Unified+corrected-FK soft factor) | 1.9573 | 1.9480 | 1.9523 | 4.3730 | 4.3529 | 4.3696 | Pending | Pending | Pending | Pending |
| A5<br>(board+cube+Unified+corrected-FK hard fixed) | **1.1286** | **1.1339** | **1.1835** | **3.8405** | **3.8442** | **3.8672** | Pending | Pending | Pending | Pending |
| B1<br>(board+cube+Seq+corrected-FK soft factor) | 2.0184 | 2.0130 | 2.0330 | 4.4062 | 4.3928 | 4.4218 | Pending | Pending | Pending | Pending |
| B2<br>(cube+Unified+corrected-FK soft factor) | 2.0178 | 1.9946 | 2.0048 | 4.1524 | 4.1107 | 4.1851 | Pending | Pending | Pending | Pending |
| B3<br>(board+Unified+VISION) | 2.8595 | 2.8315 | 2.8360 | 5.9881 | 5.9814 | 5.9959 | Pending | Pending | Pending | Pending |

TRE·Rotation Error는 평균, P95 TRE는 95백분위수다. Failure Rate는 전체 GT pose 중 예측 실패·누락 비율(0–1)이며 solver 수렴률과 다르다.

## 2. 비교 구성과 미산출 항목

| Row | 학습 표적 | 최적화 | FK 처리 | 상태 |
| --- | --- | --- | --- | --- |
| A0 | board | Sequential | VISION | complete |
| A1 | board + cube | Sequential | VISION | complete |
| A2 | board + cube | Unified | VISION | complete |
| A3 | board + cube | Unified | raw-FK hard fixed | pending |
| A4 | board + cube | Unified | corrected-FK soft factor | complete |
| A5 | board + cube | Unified | corrected-FK hard fixed | complete |
| B1 | board + cube | Sequential | corrected-FK soft factor | complete |
| B2 | cube | Unified | corrected-FK soft factor | complete |
| B3 | board | Unified | VISION | complete |

| 미산출 행 | 사유 |
| --- | --- |
| A3 | 선택한 Cube의 비전 미사용 T_flange_cube 입력이 없음. robot.json pose는 T_base_flange이며 flange-to-Cube 장착 변환이 아님. 다른 Cube의 nominal 기계 변환은 재사용하지 않음. |

External GT가 미산출인 행은 독립 6-DoF GT와 방법별 frozen prediction이 필요하다. P1 비전으로 추정한 flange→Cube 변환은 corrected-FK이며 raw-FK의 독립 기계값이나 External GT가 아니다.

Sequential은 gripper 단계 후 결과를 동결하고 fixed 카메라를 적합한다. Unified는 함께 적합한다. 모든 행은 P1 Cube 기반 카메라 초기값과 P3 Board 기반 hand-eye 초기값을 공유한다. board-only/cube-only는 최적화 잔차 구성을 뜻하며 초기화까지 표적을 제외한 비교는 아니다.

## 3. 모든 데이터에 적용하는 평가 원리

1. **동일 LOPO:** placement 하나의 모든 P2 Cube·Board 관측을 학습에서 제외한다. 모든 방법과 지표에 같은 split을 쓴다.
2. **ALL / Train / Test:** ALL은 전체 placement로 별도 재fit한다. Train은 각 fold의 학습 placement, Test는 제외 placement를 평가한다.
3. **Cube:** 공통 FK-reference를 투영하여 코너 오차를 구한다. FK 계열에 유리할 수 있어 보조 지표로 사용한다.
4. **Cross-view:** A만으로 PnP → B로 전달한다. A→B·B→A 모두 계산하고, 각 방향의 destination 코너는 채점에만 쓴다.
5. **동일 RMSE:** `sqrt(mean(dx² + dy²))`. 코너 제곱오차 합 ÷ 코너 수의 제곱근이며 fold RMSE를 단순 평균하지 않는다.

fixed↔fixed 전달에는 FK가 필요 없고, fixed↔gripper 전달에는 촬영 순간 FK가 포함된다. Train은 fold별 평가 코너를 모아 집계한다.

## 4. 카메라 쌍별 Cross-view

| Row | fixed↔fixed ALL px | fixed↔fixed Train px | fixed↔fixed Held-out Test px | fixed↔gripper ALL px | fixed↔gripper Train px | fixed↔gripper Held-out Test px |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | 6.6028 | 6.6036 | 6.6207 | 5.4325 | 5.4189 | 5.4310 |
| A1 | 4.4650 | 4.4482 | 4.4781 | 4.1308 | 4.1226 | 4.1957 |
| A2 | 4.5629 | 4.5334 | 4.5365 | 4.2161 | 4.2037 | 4.2331 |
| A3 | Pending | Pending | Pending | Pending | Pending | Pending |
| A4 | 4.5645 | 4.5354 | 4.5370 | 4.2096 | 4.1975 | 4.2275 |
| A5 | 3.7256 | 3.7273 | 3.7456 | 3.9322 | 3.9375 | 3.9641 |
| B1 | 4.6501 | 4.6316 | 4.6413 | 4.1958 | 4.1870 | 4.2335 |
| B2 | 4.2338 | 4.1611 | 4.1429 | 4.0844 | 4.0689 | 4.2195 |
| B3 | 6.6031 | 6.6033 | 6.6205 | 5.4325 | 5.4187 | 5.4308 |

전체 카메라 쌍을 합친 값은 1절 Cross-view 열이다. 쌍·방향·코너 수와 fold별 수렴 상태는 [fold_metrics.csv](fold_metrics.csv)에 있다.

## 5. 입력·해석 범위·재생성

| 입력 | 저장소 기준 경로 |
| --- | --- |
| P1 | zeus_gello_calibration/data/session1_handheld_fixed_cam_0909/capture_replayed_0917_1280x720 |
| P2 | zeus_gello_calibration/data/session2_floor_board_dual_cam_0909/capture_placed_0917_1280x720 |
| P3 | zeus_gello_calibration/data/session3_wrist_motion_gripper_cam_0909/capture_replayed_0917_1280x720 |
| cube_config | targets/gt_cube/cube_config.json |

관측 수: p1_cube=44; p2_fixed_cube=40; p2_gripper_cube=15; p3_board=14; p3_cube=59; p3_cube_set=14; p2_board=52; total=224.

**기하·스케일의 독립성:** 보정 자료의 출처가 기록되지 않았다. 기하 추정까지 독립적인 end-to-end 검증으로 해석하지 않는다.

원본 JSON에 입력 해시·K/D·변환행렬·split을, CSV에 반올림 전 값을 보존한다. 초기화 횟수·수렴·관측 수가 다른 실행끼리는 수치만으로 우열을 확정하지 않는다.

산출물: [원본 JSON](ABLATION_TEST_table1_methods.json) · [요약 CSV](ABLATION_TEST_table1_results.csv) · [fold CSV](fold_metrics.csv).

같은 JSON을 보고서로 다시 내보내기(재fit 없음):

```bash
python zeus_gello_calibration/table1_zeus.py --report-only <결과폴더>/ABLATION_TEST_table1_methods.json
```
