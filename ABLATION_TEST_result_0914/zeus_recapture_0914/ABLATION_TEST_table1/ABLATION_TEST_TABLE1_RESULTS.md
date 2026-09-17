# Table 1 결과 요약

**계산된 방법 8개 · placement 15개 · ALL 수렴 8/8 · fold 수렴 120/120.**

내부 주 지표는 **Held-out Test Cross-view Cube RMSE px**. 최종 물리 순위는 독립 External GT로 결정한다.

## 1. 전체 평가지표

각 열은 작을수록 좋으며 **최솟값**을 굵게 표시한다. 반올림 전 값으로 비교하고 동률은 모두 표시한다. 미산출 값은 Pending이며, 소수점 4자리로 표시한다.

| 실험 (구성) | ALL Cube RMSE px (Train + Held-out Test) | Train Cube RMSE px | Held-out Test Cube RMSE px | ALL Cross-view Cube RMSE px (Train + Held-out Test) | Train Cross-view Cube RMSE px | Held-out Test Cross-view Cube RMSE px | TRE mm | Rotation Error deg | P95 TRE mm | Failure Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0<br>(board+Seq+VISION) | 2.3720 | 2.3610 | 2.3616 | 4.1014 | 4.0996 | 4.1015 | Pending | Pending | Pending | Pending |
| A1<br>(board+cube+Seq+VISION) | 2.2392 | 2.2361 | 2.2304 | 4.2611 | 4.3183 | 4.3499 | Pending | Pending | Pending | Pending |
| A2<br>(board+cube+Unified+VISION) | 2.1382 | 2.1390 | 2.1488 | 4.0263 | 4.0737 | 4.0989 | Pending | Pending | Pending | Pending |
| A3<br>(board+cube+Unified+raw-FK hard fixed) | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| A4<br>(board+cube+Unified+corrected-FK soft factor) | 2.1282 | 2.1294 | 2.1419 | 4.0156 | 4.0624 | **4.0881** | Pending | Pending | Pending | Pending |
| A5<br>(board+cube+Unified+corrected-FK hard fixed) | **1.7801** | **1.7845** | **1.8893** | 5.1449 | 5.1845 | 5.1992 | Pending | Pending | Pending | Pending |
| B1<br>(board+cube+Seq+corrected-FK soft factor) | 2.1530 | 2.1505 | 2.1614 | **4.0077** | **4.0605** | 4.0945 | Pending | Pending | Pending | Pending |
| B2<br>(cube+Unified+corrected-FK soft factor) | 2.4890 | 2.4934 | 2.5060 | 4.3002 | 4.3257 | 4.3738 | Pending | Pending | Pending | Pending |
| B3<br>(board+Unified+VISION) | 2.3720 | 2.3610 | 2.3616 | 4.1011 | 4.0996 | 4.1014 | Pending | Pending | Pending | Pending |

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
| A3 | 0914 GT Cube의 비전 미사용 T_flange_cube가 저장되어 있지 않음. robot.json 79개에는 robot pose/joints 등이 있으나 flange-to-Cube 장착 변환은 없음. 0909 main Cube의 160 mm + Ry(180 deg)는 재사용하지 않음. INPUT_AUDIT.md 참조. |

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
| A0 | 4.3708 | 4.3708 | 4.3679 | 3.8801 | 3.8768 | 3.8829 |
| A1 | 4.7721 | 4.8403 | 4.8479 | 3.8191 | 3.8663 | 3.9214 |
| A2 | 4.4407 | 4.4968 | 4.5245 | 3.6738 | 3.7135 | 3.7365 |
| A3 | Pending | Pending | Pending | Pending | Pending | Pending |
| A4 | 4.4251 | 4.4805 | 4.5095 | 3.6675 | 3.7068 | 3.7295 |
| A5 | 5.7423 | 5.7878 | 5.7980 | 4.6302 | 4.6645 | 4.6836 |
| B1 | 4.4226 | 4.4857 | 4.5174 | 3.6545 | 3.6982 | 3.7346 |
| B2 | 4.7427 | 4.7733 | 4.8090 | 3.9238 | 3.9447 | 4.0048 |
| B3 | 4.3706 | 4.3708 | 4.3679 | 3.8798 | 3.8767 | 3.8827 |

전체 카메라 쌍을 합친 값은 1절 Cross-view 열이다. 쌍·방향·코너 수와 fold별 수렴 상태는 [fold_metrics.csv](fold_metrics.csv)에 있다.

## 5. 입력·해석 범위·재생성

| 입력 | 저장소 기준 경로 |
| --- | --- |
| P1 | zeus_gello_calibration/data/session1_handheld_fixed_cam_0909/capture_replayed_0914 |
| P2 | zeus_gello_calibration/data/session2_floor_board_dual_cam_0909/capture_placed_0914 |
| P3 | zeus_gello_calibration/data/session3_wrist_motion_gripper_cam_0909/capture_replayed_0914 |
| cube_config | targets/gt_cube/cube_config.json |

관측 수: p1_cube=41; p2_fixed_cube=39; p2_gripper_cube=15; p3_board=15; p2_board=58; total=168.

**기하·스케일의 독립성:** Cube 기하·scale은 P2 및 GT 촬영 영상을 포함한 전체 자료로 보정됐다. 이번 LOPO는 고정된 기하 조건에서 calibration을 평가하며, 기하 추정까지 분리한 독립 end-to-end 일반화 검증은 아니다.

원본 JSON에 입력 해시·K/D·변환행렬·split을, CSV에 반올림 전 값을 보존한다. 초기화 횟수·수렴·관측 수가 다른 실행끼리는 수치만으로 우열을 확정하지 않는다.

산출물: [원본 JSON](ABLATION_TEST_table1_methods.json) · [요약 CSV](ABLATION_TEST_table1_results.csv) · [fold CSV](fold_metrics.csv).

같은 JSON을 보고서로 다시 내보내기(재fit 없음):

```bash
python zeus_gello_calibration/table1_zeus.py --report-only <결과폴더>/ABLATION_TEST_table1_methods.json
```
