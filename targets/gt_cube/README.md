# GT 검증용 큐브

GT 검증 실험 전용의 **별도 물체**다. 메인 큐브(`calibration_pipeline/config.py`)와
마커 6장은 완전히 동일하고, 본체 형상과 상단 마커 2장의 위치만 다르다.

## 파일

| 파일 | 내용 |
|---|---|
| [`cube_config.json`](cube_config.json) | 마커 모델. 파이프라인이 읽는 유일한 파일 |
| [`cube_solid.json`](cube_solid.json) | 물리 형상(mm 박스 5개). 작도 전용, 캘리브레이션 미사용 |

도면은 [`ABLATION_TEST_result_0909/gt_cube/`](../../ABLATION_TEST_result_0909/gt_cube/) 에 있다.

## 형상

전체 엔벨로프 **59(X) × 87(Y) × 94(Z) mm** = 본체 59 + 윗 돌출부 35 (라이저 25 + I자 플레이트 10), 아래에서 위로 3단.

| 단 | 크기 (X×Y×Z) | z 구간 |
|---|---|---|
| 본체 | 59 × 59 × 59 | −29.5 … +29.5 |
| 라이저 | 31 × 59 × 25 | +29.5 … +54.5 |
| I자 플레이트 | 아래 참조 × 10 | +54.5 … +64.5 |

I자 플레이트는 ±Y 블록 31×31 두 개와 그 사이를 잇는 목 15(X)×25(Y)로 이루어진다
(Y 방향 31 + 25 + 31 = 87). 블록은 본체 밖으로 **±Y 각 14 mm 캔틸레버**로 나온다.

## 좌표계

원점은 **본체 59³의 중심**, +Z 위쪽. 메인 큐브의 "구 59 mm 엔벨로프 중심(본체 중심 위 1 mm)"
규약과 달리 오프셋이 없다 — 새 본체가 정육면체라 그 quirk가 사라졌다.

| 마커 | 면 | 크기 | 중심 (x, y, z) mm |
|---|---|---|---|
| 0 | +Z | 25 | (0, −28, +64.5) |
| 1 | +Z | 25 | (0, +28, +64.5) |
| 2 | +X | 51 | (+29.5, 0, 0) |
| 3 | +Y | 51 | (0, +29.5, 0) |
| 4 | −X | 51 | (−29.5, 0, 0) |
| 5 | −Y | 51 | (0, −29.5, 0) |

- 마커 2~5는 본체 면 중앙 → z = 0. 메인 큐브(z = −1)와 **1 mm 다르다.**
- 마커 0/1은 31×31 블록 정중앙(사방 3 mm 여백), 최상단면 위. 메인 큐브의 y = ∓14에서
  ∓28로 옮겨졌고, 두 태그 사이 간격은 3 mm → 31 mm로 벌어졌다.

## 도면 재생성

```bash
python tools/visualize_cube_model.py \
  --cube-config targets/gt_cube/cube_config.json \
  --solid targets/gt_cube/cube_solid.json \
  --title "GT validation cube  |  59(X) x 87(Y) x 94(Z) mm" \
  --output-dir ABLATION_TEST_result_0909/gt_cube
```

## 사용 전 확인이 필요한 것

1. **`validate_cube_config()`는 이 큐브에서 반드시 FAIL한다.** 검증 함수가 넘겨받은 cfg 대신
   `config.py`의 모듈 상수(`TOP_MARKER_PLANE_Z_M`, footprint)와 비교하기 때문이다
   (`calibration_pipeline/apriltag_cube.py`의 `validate_cube_config`). 메인 큐브 외의 어떤
   타깃도 통과할 수 없으므로, 이 큐브를 파이프라인에 물리려면 검증 기준을 cfg에서
   유도하도록 먼저 고쳐야 한다.
2. **`face_roll_deg`는 메인 큐브 값을 복사한 것이고 아직 검증되지 않았다.** 태그를 실제로
   어떻게 붙였는지에 대한 값이므로 이 큐브에서 face-roll self-calibration을 다시 돌려야 한다.
   그 전에 `ABLATION_TEST_result_0909/gt_cube/cube_model_net.png` 전개도로 실물과 눈으로 대조할 것.
3. **마커 ID가 메인 큐브와 완전히 같다.** 두 큐브가 한 장면에 동시에 들어오면 디텍터가
   구분하지 못한다. 동시 촬영 계획이 있으면 ID를 분리해야 한다.
