# 0914 재촬영 입력과 미산출 지표 확인

## 1. 실제 입력: 168개 관측, 15개 placement

이 결과는 09-14 재촬영인 `capture_replayed_0914` / `capture_placed_0914`를 사용한다. 같은 상위 폴더에 있는 0909 촬영이나 `session11_*_0909` 변환 결과와는 입력이 다르다. 실제 로더 실행에서 다음 구성을 확인했다.

| 입력 | 관측 수 | 촬영 경로 |
| --- | ---: | --- |
| P1 고정카메라·파지 Cube | 41 | [session1/capture_replayed_0914](../../zeus_gello_calibration/data/session1_handheld_fixed_cam_0909/capture_replayed_0914/) |
| P2 놓인 Cube | 고정 39 + 그리퍼 15 | [session2/capture_placed_0914](../../zeus_gello_calibration/data/session2_floor_board_dual_cam_0909/capture_placed_0914/) |
| P2 Board | 고정 43 + 그리퍼 15 | 위 P2의 동일 사진 |
| P3 그리퍼카메라·Board | 15 | [session3/capture_replayed_0914](../../zeus_gello_calibration/data/session3_wrist_motion_gripper_cam_0909/capture_replayed_0914/) |
| 합계 | **168** | **P2 placement 0–14** |

P2의 Cube와 Board, 모든 카메라는 `event_id = 1000 + placement_id`를 공유한다. Held-out placement의 P2 Board도 해당 fold의 calibration에서 제외한다. `--s3-gripper-only`에 따라 P3 고정카메라 Board 41개는 제외한다. Cube 기하는 [GT cube config](../../targets/gt_cube/cube_config.json), P1 초기값은 [0914 grasp fit](../../zeus_gello_calibration/pass1_grasp_offset_replayed_0914.json), cam0 내부 파라미터는 [20260915 override](../../intrinsics/overrides/039422061216_20260915.npz)를 사용한다. [로더](../../zeus_gello_calibration/fit_calibration_methods.py)의 `source_data_provenance`는 선택 경로·파일 해시·실제 K/D·관측 수를 보존한다.

## 2. robot.json 79개에서 확인한 값

0914 `robot.json` **79개 전체**를 확인했다: P1 16개, P2 parking/held/released 각 15개, P3 15개, GT 사진 3개. 최상위 키의 합집합은 다음과 같다.

```text
timestamp, pose, joints, capture_index, note, pose_convention,
gripper, step_index, capture_tag, replayed_from
```

`T_flange_cube`, Cube 중심 pose, tool offset, 활성 tool ID는 저장돼 있지 않다. 대표 placement 000의 기록은 다음과 같다. `pose` 단위는 mm/deg, 순서는 `[x,y,z,rz,ry,rx]`다.

| 기록 | 저장 pose | 의미 |
| --- | --- | --- |
| [held/000](../../zeus_gello_calibration/data/session2_floor_board_dual_cam_0909/capture_placed_0914/held/000/robot.json) | `[-188.899, 287.527, 178.751, -150.503121, 0, -180.000020]` | 놓는 자리에서 그리퍼를 열기 전 |
| [released/000](../../zeus_gello_calibration/data/session2_floor_board_dual_cam_0909/capture_placed_0914/released/000/robot.json) | 위 held와 동일 | 같은 자리에서 그리퍼를 연 뒤 |
| [parking/000](../../zeus_gello_calibration/data/session2_floor_board_dual_cam_0909/capture_placed_0914/000/robot.json) | `[-80.785, 270.964, 603.407, -120.546997, 4.803105, 161.178127]` | calibration 사진을 찍은 카메라 위치 |

15개 placement 전체의 held/released z는 **178.749–178.751 mm**이며, 각 pair의 z 차이는 **0 mm**다. 이 값은 로봇이 같은 높이를 유지했다는 기록이며 Cube 원점까지의 거리를 직접 측정한 값은 아니다.

## 3. A3에 필요한 flange→Cube 변환

현재 [zeus_server.py](../../server/zeus_server.py)의 `main`(154행)과 [zeus_gello.py](../../server/zeus_gello.py)의 `main`(340행)은 `settool(1,0,0,0,0,0,0)`을 설정한다. 따라서 현재 촬영 코드가 사용하는 pose 규약은 `T_base_flange`다. 다만 개별 robot.json에는 실제 실행 당시 tool 설정을 별도로 기록하지 않았다.

[session2_pick_and_place.py](../../zeus_gello_calibration/session2_pick_and_place.py)의 `compute_ordered_targets`(133행)는 저장된 x/y/yaw와 고정 z178.75, pitch0, roll180으로 **place 명령 flange pose**를 만든다. [gt_pick_test.py](../../zeus_gello_calibration/gt_pick_test.py)의 `build_pick_target`(125행)은 178.75를 “이 높이면 잡힌다”는 검증값으로 설명하며 Cube 치수와 구별한다.

```text
T_base_cube = T_base_flange(place) @ T_flange_cube
```

held 기록으로 왼쪽 계산에 필요한 `T_base_flange(place)`는 복원할 수 있다. **A3에 필요한 비전 독립 `T_flange_cube`는 이 robot 기록에 없다.** 별도 [jog 스크립트](../../server/zeus_jog_onboard.py)(450행)와 [align 스크립트](../../server/align_pose_onboard.py)(130행)에 +97.5 mm tool 설정은 있지만, 그 값과 이번 GT Cube의 원점을 연결하는 실측 변환은 없다. 기존 main Cube의 160 mm 가정이나 P1 VISION fit의 162.0425 mm를 이번 GT Cube의 독립 실측값으로 대체 표기할 수 없다.

미확보 정보는 **이번 파지 상태에서 flange와 GT Cube object frame 사이의 4×4 강체변환**, 축·단위·측정 출처, 그리고 촬영 당시 활성 tool 규약이다. 이 정보는 별도 기계 측정·검증된 datum 또는 독립 Cube pose와 flange pose의 동시 측정으로 확정돼야 한다.

## 4. External GT 네 지표의 현재 범위

[GT 사진 원본](../../zeus_gello_calibration/data/gt_frames_0914/20260914_232234/)은 3 trial이며 각 robot.json의 `pose`와 `joints`는 `null`이다. [기존 설명](../../zeus_gello_calibration/METHODS_ANALYSIS.md)은 그리퍼로 Cube를 물리 정렬한 뒤 읽은 flange pose를 GT로 사용한다. 이는 카메라 검출과 별도지만 Robot FK와 독립된 `T_base_cube_GT` 측정은 아니다.

[gt_eval_offline.py](../../zeus_gello_calibration/gt_eval_offline.py)(179–198행)는 `dz = estimate_z - (GT_flange_z - 해당 방법의 fitted T_gripper_cube_z)`를 계산한다. 따라서 기준 z 자체가 방법별 추정값에 의존한다. 회전은 yaw 차이 `drz`만 저장하며, 검출 실패는 결과 row에 남기지 않고 건너뛴다. 그리퍼카메라 포함 결과에는 촬영 pose를 `GT flange pose + z300mm`로 가정한 조건도 있다.

| 요청 지표 | 현재 정식 산출을 막는 정보 |
| --- | --- |
| TRE mm | 모든 방법과 독립적인 같은 `T_base_cube_GT` |
| Rotation Error deg | yaw만이 아닌 전체 3D 회전 GT·prediction |
| P95 TRE mm | 위 독립 TRE의 trial별 표본 |
| Failure Rate | 전체 평가 trial 목록과 실패·누락을 보존한 prediction 상태 |

정식 [external_gt.py](../../calibration_pipeline/external_gt.py)는 Failure Rate를 전체 GT pose 중 prediction 실패·누락 비율로 정의한다. 오차 허용치 초과율과는 다른 정의다. [평가 manifest 템플릿](../../protocol_templates/external_gt_eval_manifest_TEMPLATE.json)의 비교 margin·GT 불확실성도 아직 null이다. 기존 3-trial `xyz/drz` 숫자를 독립 External GT 네 열에 옮겨 쓰지 않는다.

## 5. 전체 촬영으로 보정한 Cube 기하의 영향

[기하 보정 artifact](../../zeus_gello_calibration/gt_cube_geometry_calibration.json)의 `roots`는 0914 P1/P2/P3와 `gt_frames_0914`를 포함한다. 검출 이미지 267장 중 다중 마커 이미지 179장이 기록돼 있다. [현재 Cube config](../../targets/gt_cube/cube_config.json)는 이 자료로 보정한 `marker_pose_4x4`를 포함하고, 전역 scale 0.99 역시 0914 unified fit과 P2 camera 일치도를 보고 선택했다고 기록한다.

이번 LOPO는 **이미 고정된 Cube 기하·intrinsics를 조건으로 calibration placement를 제외하는 평가**다. 각 fold에서 제외한 사진도 앞선 기하 보정에 관여했으므로, target 기하 추정까지 포함해 완전히 보지 않은 데이터로 일반화한 결과라고 해석할 수 없다. GT 사진도 기하 보정 입력에 포함돼 기존 GT 사진을 완전한 blind test로 볼 수 없다.

전체 처리 과정의 독립 일반화를 검증하려면 별도 자료로 측정·고정한 Cube 기하와 scale, 또는 fold마다 train 자료만으로 만든 기하가 필요하다. 최종 물리 순위에는 그와 별도로 calibration RGB·Robot FK와 독립적인 6-DoF GT 및 모든 시도 상태가 필요하다.
