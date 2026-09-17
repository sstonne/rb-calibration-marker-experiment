# session11 (Zeus 0909, legacy 변환) — 01~06 파이프라인 실행 기록

> 실제 0914 재촬영의 ALL/Train/Held-out Test Cube·Cross-view 결과는
> [별도 상세 보고서](../zeus_recapture_0914/ABLATION_TEST_table1/ABLATION_TEST_TABLE1_RESULTS.md)에 있다.
> 아래는 0909 촬영의 legacy event-split 실행 기록이다.

데이터: `data/session11_zeus_handheld_floor_wrist_meta_0909/` (Zeus session1/2/3 0909
재촬영을 `zeus_gello_calibration/convert_to_meta.py`로 변환, 61 events).

실행:

```bash
S=data/session11_zeus_handheld_floor_wrist_meta_0909
python 04_filter_observations.py --session-root $S/calib_train --intrinsics-dir $S/intrinsics --include-gripped-cube
python 05_calibrate.py --root_folder $S/calib_train --intrinsics_dir $S/intrinsics \
  --observation-manifest $S/calib_out/capture_filter/Step2b_observation_manifest.json \
  --min_train_eih_cube_events 0 --split_seed 20260731 --num_inits 3 \
  --out_dir ABLATION_TEST_result_0914/session11_zeus_handheld_floor_wrist_meta_0909/ABLATION_TEST_table1
python 06_make_report.py --root_folder $S/calib_train \
  --table1 ABLATION_TEST_result_0914/session11_zeus_handheld_floor_wrist_meta_0909/ABLATION_TEST_table1/ABLATION_TEST_table1_methods.json \
  --out_dir ABLATION_TEST_result_0914/session11_zeus_handheld_floor_wrist_meta_0909/ABLATION_TEST_table1
```

## 결과 (06 `calibration_summary.csv`, 3 seed 평균, 27/27 수렴)

| Row | 구성 | Train px | Held-out px | Train obs |
|---|---|---:|---:|---:|
| A0 | board / sequential | 0.625 | 0.435 | 153 |
| A1 | board+cube / sequential / VISION | 1.297 | 2.581 | 169 |
| A2 | board+cube / unified / VISION | 0.629 | 2.095 | 169 |
| A3 | board+cube / unified / raw-FK hard fixed (nominal 160 mm) | 0.694 | 0.884 | 169 |
| A4 | board+cube / unified / corrected-FK soft factor (UR3 상수 2 mm / 0.3°) | 0.635 | 4.017 | 169 |
| A5 | board+cube / unified / VISION-aligned FK hard fixed | 7.923 | 24.673 | 169 |
| B1 | board+cube / sequential / corrected-FK soft factor | 7.489 | 7.578 | 169 |
| B2 | cube only / unified / corrected-FK soft factor | 0.723 | 1.750 | 16 |
| B3 | board / unified | 0.625 | 0.435 | 153 |

## 이 실행이 우리 `fit_calibration_methods.py` 결과와 다른 점 (반드시 읽을 것)

1. **session1(쥔 큐브, grasp+FK 모델)이 학습에 들어가지 않았다.** 05(table1)에는 grasp 모델
   (`T_gripper_cube[grasp]`)이 구현되어 있지 않아(`--include_gripped_cube` 플래그만 있고 reference
   state에 grasp 변수를 만드는 코드가 없음) gripped 관측을 넣으면 "grasp transform unavailable"로
   멈춘다. 이 실행은 gripped 관측을 제외한 UR3 legacy 프로토콜 구성(보드 + 놓인 큐브)이다. 따라서
   A2 ≠ 우리 통합_no-fk(session1 포함). 고정캠은 보드(고정캠·그리퍼캠 공유)와 놓인 큐브로만 base에
   묶인다.
2. **Zeus session2는 세트당 촬영 이벤트가 1개**(한 자세에서 4대 동시)라 05의 event-stratified
   split이 성립하지 않아, 변환기가 각 placement를 같은 로봇 자세의 두 이벤트(고정캠 3대 / 그리퍼캠
   1대)로 나눴다(`--placement-event-mode fixed_gripper_split`). held-out은 "train 이벤트로 잡은
   큐브 pose를 test 이벤트 카메라가 맞추는가"(사실상 고정캠↔그리퍼캠 cross-view)이며, 우리 문서의
   leave-one-set-out held-out과 정의가 다르다. 세트마다 어느 쪽이 test인지는 seed로 정해진다.
3. **eligible 세트 9/15**: 04의 표준 정책이 그리퍼캠의 윗면-only(평면 2마커) 큐브 관측을 격리
   (`noncore_planar_multimarker`)해서 6세트(1,2,8,11,14,15)는 그리퍼 큐브 이벤트가 0 → split에서
   제외. `--min_train_eih_cube_events 0`(기본 3)이 필요했다.
4. **A3의 raw-FK**는 place 명령 flange pose @ 플랜지 +z 160 mm(조립 nominal, 비전 미사용). 우리
   문서의 raw-fk(`FK @ session1 비전 fit T_gripper_cube`)와 앵커가 다르다.
5. 그리퍼캠 큐브 뷰가 train에 없는 세트는 큐브 초기값을 raw-FK로 대신 잡았다
   (`shared_reference_diag.visual_init_fallback_raw_fk_sets`, 초기값만).
6. A4/A5/B1이 크게 나쁜 것은 UR3 상수(2 mm / 0.3°, Huber 3σ) 및 board-free FK 정합 artifact가
   "세트당 그리퍼 뷰 1개" 데이터에 맞지 않기 때문으로 보인다 — 별도 진단 필요.

세부 정의는 `zeus_gello_calibration/METHODS_ANALYSIS.md`(우리 파이프라인), 프로토콜은
`CAPTURE_PROTOCOL.md` 참고.
