# Zeus 촬영·실행 명령어 — 0917 촬영분

2026-09-17 촬영분을 **찍는 명령어**와 **찍은 뒤 돌리는 명령어**를 한곳에 모았다.
최종 결과는 [`ABLATION_TEST_result_0917/REPORT_0917.md`](../ABLATION_TEST_result_0917/REPORT_0917.md).

- 로봇: Zeus i611
- 카메라: 고정 3대(`039422061216`, `fixed2`, `fixed3`) + 그리퍼 1대(`752112070297`)
- 표적: GT 큐브(59 mm, AprilTag 36h11 6면) + 바닥 ChArUco 보드(11×7, 25/18 mm)

---

## 0. 세션 3개가 각각 무엇인가

| 세션 | 무엇을 하나 | 무엇을 구하나 |
|---|---|---|
| **session1** | 로봇이 큐브를 **쥔 채** 공중에서 16자세, 고정캠 3대가 촬영 | **T_flange_cube** (플랜지→큐브) |
| **session2** | 로봇이 큐브를 집어 바닥 15곳에 **놓는다**. 놓을 때마다 진단 3장 + 본 촬영 | **캘리브레이션 본 데이터** (held-out 평가도 여기서) |
| **session3** | 큐브를 마지막 자리에 둔 채 **손목만** 15자세로 돌려 그리퍼캠으로 보드 촬영 | **T_gripper_cam** (그리퍼캠 hand-eye) |

세 세션 모두 사진마다 로봇 FK를 같이 저장한다.

---

## 1. 촬영

⚠️ **1920×1080 / 848×480 / 640×360 은 color만 그 해상도이고 depth는 1280×720이다.**
`--depth-width 1280 --depth-height 720` 을 **반드시 같이** 줘야 한다. 안 주면 `RuntimeError`로 막힌다.

```bash
cd zeus_gello_calibration
```

### 1920×1080 — **최종 보고서가 쓴 촬영분**

```bash
DM=../intrinsics_1920x1080_rgbd720/device_map.json

# session1 (큐브 쥐고 16자세)
python replay_and_recapture.py --session 1 --execute --no-step \
  --output-subdir capture_replayed_0917 --width 1920 --height 1080 \
  --depth-width 1280 --depth-height 720 --device-map $DM --regrasp-joints

# session2 (pick-and-place 15곳)
python session2_pick_and_place.py --execute --no-step \
  --out-root data/session2_floor_board_dual_cam_0909/capture_placed_0917 \
  --width 1920 --height 1080 --depth-width 1280 --depth-height 720 --device-map $DM

# session3 (손목 모션 15자세)
python replay_and_recapture.py --session 3 --execute --no-step \
  --output-subdir capture_replayed_0917 --width 1920 --height 1080 \
  --depth-width 1280 --depth-height 720 --device-map $DM
```

### 1280×720

```bash
DM=../intrinsics_1280x720/device_map.json     # depth 옵션 불필요 (color=depth=1280x720)

python replay_and_recapture.py --session 1 --execute --no-step \
  --output-subdir capture_replayed_0917 --width 1280 --height 720 --device-map $DM --regrasp-joints

python session2_pick_and_place.py --execute --no-step \
  --out-root data/session2_floor_board_dual_cam_0909/capture_placed_0917 \
  --width 1280 --height 720 --device-map $DM

python replay_and_recapture.py --session 3 --execute --no-step \
  --output-subdir capture_replayed_0917 --width 1280 --height 720 --device-map $DM
```

### 848×480

```bash
DM=../intrinsics_848x480_rgbd720/device_map.json

python replay_and_recapture.py --session 1 --execute --no-step \
  --output-subdir capture_replayed_0917 --width 848 --height 480 \
  --depth-width 1280 --depth-height 720 --device-map $DM --regrasp-joints

python session2_pick_and_place.py --execute --no-step \
  --out-root data/session2_floor_board_dual_cam_0909/capture_placed_0917 \
  --width 848 --height 480 --depth-width 1280 --depth-height 720 --device-map $DM

python replay_and_recapture.py --session 3 --execute --no-step \
  --output-subdir capture_replayed_0917 --width 848 --height 480 \
  --depth-width 1280 --depth-height 720 --device-map $DM
```

### 640×360

⚠️ **640×480이 아니라 640×360이다.** 그리퍼캠이 누락돼 최종 보고서에서는 제외했다.

```bash
DM=../intrinsics_640x360_rgbd720/device_map.json
# 위와 같은 3개 명령, --width 640 --height 360 --depth-width 1280 --depth-height 720
```

### 외부 GT 촬영 (별도)

```bash
python capture_frames.py --shots 5 --width 1920 --height 1080 \
  --depth-width 1280 --depth-height 720 --device-map $DM
```

로봇을 지정 좌표로 보내 큐브를 놓고, **z +300 mm 올라가서 촬영**한다. 좌표는 [`zeus 외부gt.txt`](../zeus%20외부gt.txt) 참조.

### 옵션 뜻

| 옵션 | 뜻 |
|---|---|
| `--execute` | 실제로 로봇을 움직인다 (없으면 계획만 출력) |
| `--no-step` | 단계마다 Enter 안 받고 연속 실행. **처음 실행은 빼고 하라** |
| `--regrasp-joints` | session1에서 매번 큐브를 **다시 쥔다** (재파지 반복성 확인용) |
| `--device-map` | 카메라 번호↔시리얼 매핑. 해상도별 intrinsics 폴더 안에 있다 |

---

## 2. 촬영 후 실행 — 한 줄이면 끝난다

```bash
cd zeus_gello_calibration

python run_zeus_calibration.py --tag 0917_1920x1080 \
  --zeus-intrinsics-dir ../intrinsics_1920x1080_rgbd720
```

`--tag T` 를 주면 session1/3은 `capture_replayed_T`, session2는 `capture_placed_T` 폴더를 자동으로 찾는다.

**다른 해상도도 같은 형식이다.**

```bash
python run_zeus_calibration.py --tag 0917_1280x720 --zeus-intrinsics-dir ../intrinsics_1280x720
python run_zeus_calibration.py --tag 0917_848x480  --zeus-intrinsics-dir ../intrinsics_848x480_rgbd720
python run_zeus_calibration.py --tag 0917_640x360  --zeus-intrinsics-dir ../intrinsics_640x360_rgbd720
```

### 이 한 줄이 돌리는 4단계

| 단계 | 스크립트 | 입력 | 하는 일 | 내놓는 것 |
|---|---|---|---|---|
| 1 | `fit_grasp_offset.py` | session1 | **T_flange_cube** 적합 | `results/fits/fit_통합_<tag>.json` |
| 2 | `table1_zeus.py` | session1+2+3 **전부** | **A0~B3 방식별** `T_base_Ci`, `T_gripper_cam`. 놓은 자리 1곳씩 빼는 **LOPO 15 fold** | `table1_zeus/ABLATION_TEST_table1_methods.json` |
| 3 | `eval_heldout_and_consistency.py` | 2의 결과 | **joint held-out mm/deg** ← 정확도 기준 지표 | `heldout_and_consistency_<tag>.json` |
| 4 | `eval_joint_relative.py` | 2의 결과 | session1/2의 **절대·상대** mm/px | `joint_relative_eval_<tag>.json` |

로그는 각각 `log_1_fit.txt` ~ `log_4_relative.txt`.

> ### ⚠️ 최종 외부 파라미터를 꺼내는 위치
> `table1_zeus/ABLATION_TEST_table1_methods.json` 의 **row A5** → `all.transforms.T_base_Ci`
>
> **행 이름이 두 군데에서 다르다.** 실행기의 **row A5**가 보고서의 **A3**(FtC FK fixed)다. 헷갈리지 말 것.

### 일부만 다시 돌리기

```bash
--skip-fit        # 1단계 생략 (기존 fit json 재사용)
--skip-table1     # 2단계 생략
--skip-heldout    # 3단계 생략
--skip-relative   # 4단계 생략
```

### 안에서 고정되는 규칙 (건드릴 필요 없음)

- 큐브 기하: `targets/gt_cube/cube_config.json` (GT 큐브, 자체보정값, **마커 스케일 1.0**)
- `--s3-gripper-only` : session3 보드는 그리퍼캠만 사용
- `--include-session2-board` : session2 보드 관측 포함 (고정캠 base 앵커)
- 큐브 관측 정책 `legacy` (그리퍼캠 단면 관측 허용)
- session3 **큐브** 관측도 포함

이 옵션들이 실행기 안에 박혀 있어서, **데이터셋이 달라도 같은 방식으로 계산된다.**

### 결과가 쌓이는 곳

기본값은 **`ABLATION_TEST_result_<오늘 날짜 MMDD>/zeus_<tag>/`** 다.

```
ABLATION_TEST_result_<MMDD>/zeus_<tag>/
├── SUMMARY.md                                  ← 끝나면 여기에 요약이 나온다
├── table1_zeus/ABLATION_TEST_table1_methods.json   ← 최종 외부 파라미터 (row A5)
├── heldout_and_consistency_<tag>.json
├── joint_relative_eval_<tag>.json
└── log_1_fit.txt ~ log_4_relative.txt
```

`--results-root` 로 폴더를 따로 지정할 수도 있다. 최종 보고서의 1920 결과는
`ABLATION_TEST_result_0917/zeus_0917_1920x1080_newintr/` 에 있다(내부 파라미터 재보정 후 재실행분).

---

## 3. 보고서에 들어간 추가 분석

`run_zeus_calibration.py` 밖에서 따로 돌린 것들이다.

| 무엇 | 스크립트 | 보고서 위치 |
|---|---|---|
| 그리퍼 **열기·재파지 슬립** | `eval_slip_joint.py` | 5장 |
| **촬영 수를 줄였을 때** 정확도 | `eval_data_ratio_a3.py` | 4.3절 |
| **GT 큐브 기하 자체보정** | `calibrate_gt_cube_geometry.py` | 6.1절 |
| 슬립 시각화 | `viz_slip_joint.py`, `viz_slip_overlay.py` | 5장 그림 |
| 점군 확인 | `eval_pointcloud_workspace_by_method.py` | `pointcloud/` |

---

## 4. 최종 결과 (1920×1080, row A3)

| 지표 | 값 |
|---|---:|
| **joint held-out 위치** | **0.62 mm** |
| joint held-out 회전 | 0.24° |
| held-out Cube 재투영 | 1.767 px |
| 카메라 합의 (pairwise) | 2.32 mm |
| session2 상대 오차 | 1.04 mm |
| 열기·재파지 슬립 (중앙값) | 0.1 mm |
| T_flange_cube z | 160.9 ± 0.06 mm |

**A3 = FtC FK fixed** (session1로 구한 T_flange_cube × 로봇 FK로 큐브 위치를 고정하고 카메라만 최적화) 가 가장 좋았다.

---

## 5. 재현할 때 주의할 것

1. **depth 해상도 옵션을 빼먹지 말 것.** 1920/848/640은 반드시 `--depth-width 1280 --depth-height 720`.
2. **`--no-step` 은 검증 후에.** 처음 돌릴 때는 빼고, 비상정지에 손이 닿는 상태로.
3. **큐브 config의 마커 스케일은 1.0이다.** 2026-09-17에 0.99에서 되돌렸다(0.99는 실측이 아니라 데이터 스윕값이었음).
4. **내부 파라미터 폴더를 태그와 맞출 것.** 해상도가 다르면 K가 다르다.
5. ⚠️ **1920 최종분의 내부 파라미터 스냅샷은 확인이 필요하다.** `log_1_fit.txt`가 기록한 fx(예: `039422061216` 1368.1)와 현재 `intrinsics_1920x1080_rgbd720/intrinsics_by_serial/`의 값(1373.4)이 0.5 % 정도 다르다. 실행 직후 내부 파라미터를 다시 만든 것으로 보인다. **똑같이 재현하려면 당시 스냅샷이 필요하다.**
