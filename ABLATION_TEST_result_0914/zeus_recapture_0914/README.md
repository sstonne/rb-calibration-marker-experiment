# Zeus 0914 재촬영 — placement 단위 비교실험

[회의 피드백 반영 요약](FEEDBACK_CHECK.md): 반영된 평가 원리와 남은 검증을 구분한다.

[상세 결과](ABLATION_TEST_table1/ABLATION_TEST_TABLE1_RESULTS.md) ·
[요약 CSV](ABLATION_TEST_table1/ABLATION_TEST_table1_results.csv) ·
[fold별 CSV](ABLATION_TEST_table1/fold_metrics.csv) ·
[원시 결과·행렬 JSON](ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json) ·
[입력 확인 근거](INPUT_AUDIT.md)

## 입력과 비교 구성

실제 0914 재촬영 15 placement와 168개 관측을 사용한다.
P1 Cube 41개, P2 Cube 54개·Board 58개, P3 gripper Board 15개다.
P3 fixed Board 관측 41개는 기존 0914 구성과 동일하게 제외한다.
GT 큐브 마커 모델과 cam0의 20260915 intrinsic override를 사용하며,
촬영 폴더·설정 파일 해시·실제로 사용한 K/D는 결과 JSON에 저장한다.

A0/A1/A2/A4/A5/B1/B2/B3는 기존 Zeus Table 1 정의에 따라 계산한다.
Sequential은 gripper 단계 이후 결과를 동결하고 fixed 카메라를 푸는 순서다.
기존 `METHODS_ANALYSIS.md`의 두 그룹을 완전히 독립 적합하는 `독립_no-fk`와는
다른 방법이므로 해당 세 방법 표의 숫자와 동일할 필요가 없다.
A3는 별도 GT 큐브에 맞는 비전 미사용 `T_flange_cube`가 확인될 때 계산한다.

## 평가 계약

1. 모든 방법과 Cube/Cross-view에 같은 15-fold leave-one-placement-out split을 적용한다. 제외 placement의 P2 Cube와 Board 영상을 모두 calibration에서 제외한다. P1·P3는 보조 학습 데이터다.
2. ALL은 전체 placement로 별도 calibration을 fit한 뒤 전체 P2 Cube를 평가한다. Train/Test 수치를 평균한 값이 아니다.
3. Cross-view는 같은 촬영의 A 한 대로 PnP한 pose를 B로 전달한다. 양방향을 모두 평가하며 destination corner는 해당 방향의 오차 계산에만 사용한다. fixed↔gripper에는 촬영 순간 FK가 들어간다.
4. 모든 픽셀 RMSE는 corner별 `dx²+dy²` 합계를 corner 수로 나눈 뒤 제곱근을 취한다. Train은 모든 fold의 학습 corner 평가를 모은다. Cube는 모든 방법에 공통 P1 corrected-FK reference를 사용하며 보조 지표다.
5. 내부 주 지표는 Held-out Test Cross-view다. 현재 큐브 geometry·scale이 전체 촬영으로 사전 보정되어 있으므로 이 결과는 **고정된 geometry 조건의 calibration LOPO**다. geometry 학습까지 독립적인 검증 또는 독립 External GT 최종 순위로 해석하지 않는다.

## 재실행

저장소 루트에서 실행한다. Python 환경에 NumPy, SciPy, OpenCV가 필요하다.

```bash
python3 zeus_gello_calibration/table1_zeus.py \
  --session1-capture-subdir capture_replayed_0914 \
  --session2-capture-subdir capture_placed_0914 \
  --session3-capture-subdir capture_replayed_0914 \
  --fit-json zeus_gello_calibration/pass1_grasp_offset_replayed_0914.json \
  --cube-config targets/gt_cube/cube_config.json \
  --s3-gripper-only --include-session2-board --workers 4 \
  --out ABLATION_TEST_result_0914/zeus_recapture_0914/ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json \
  --report-dir ABLATION_TEST_result_0914/zeus_recapture_0914/ABLATION_TEST_table1
```

현재 8개 방법은 각각 ALL 1회와 LOPO 15회를 계산한다. 동일 초기화에서 1회씩
실행하며, legacy session11의 3-seed 평균과 구분한다. 전체 실행은 이 환경에서 약 5분이다.
A3의 독립 기계 변환이 확보되면 `--mechanical-transform-json <path>`를 추가한다.
JSON에는 `T_flange_cube`(4×4, 이동 단위 m), `vision_used: false`, `source`를 기록한다.

기존 결과를 재fit하지 않고 같은 형식의 표만 재생성하려면 다음을 실행한다.

```bash
python3 zeus_gello_calibration/table1_zeus.py \
  --report-only ABLATION_TEST_result_0914/zeus_recapture_0914/ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json
```
