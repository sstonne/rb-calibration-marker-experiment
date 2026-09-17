# ChArUco 보드 정의

물리 ChArUco 보드마다 JSON 한 개. 코드에 칸 수나 ID를 박지 않고 여기서 이름으로 읽는다.

| 이름 | 칸 (가로x세로) | 한 칸 / 마커 | 마커 ID | 배치 | 내부 코너 | 용도 |
|---|---|---|---|---|---|---|
| [`11x7_id5`](board_11x7_id5.json) | 11 x 7 | 25 / 18 mm | 5..42 (38장) | modern | 60 | 메인 보드. session09/10/11 hand-eye, `config.py` 기본값 |
| [`9x6_id90`](board_9x6_id90.json) | 9 x 6 | 25 / 18 mm | 90..116 (27장) | **legacy** | 40 | 2026-09-15 intrinsic 촬영 |

둘 다 `DICT_4X4_250`.

## 사용

```python
from calibration_pipeline.board_config import resolve_charuco_config
from calibration_pipeline.charuco import CharucoTarget

cfg, source = resolve_charuco_config("9x6_id90")   # 이름 또는 JSON 경로
target = CharucoTarget(cfg)                         # legacy_pattern 까지 자동 적용
```

```bash
python3 02_calibrate_intrinsics.py --list_boards
python3 02_calibrate_intrinsics.py --intr_dir intrinsics_1280x720 --from_images --board 9x6_id90
```

| 함수 (`calibration_pipeline/board_config.py`) | 역할 |
|---|---|
| `list_charuco_boards()` | 등록된 보드 이름 목록 |
| `resolve_charuco_config(board, overrides)` | 이름/경로 → `CharucoBoardConfig`, 출처 문자열. `None`이면 `config.py` 기본 보드 |
| `load_charuco_config_from_json_file(path)` | JSON 한 개 로드. 파일이 없거나 깨졌으면 예외 |
| `charuco_topology(cfg)` / `describe_charuco_config(cfg)` | 마커 수, 최대 코너 수, legacy 필요 여부 등 요약 |

`source`는 `board_file:targets/charuco_boards/board_9x6_id90.json`처럼 저장소 기준
상대 경로로 기록된다. 개별 필드를 덮어쓰면 `+override(...)`가 붙는다.

## 새 보드 등록

1. `board_<이름>.json`을 만든다. 이름은 `<가로>x<세로>_id<시작ID>` 형식을 따른다.
2. 필수 키: `squares_x`, `squares_y`, `square_length_m`, `marker_length_m`,
   `dictionary_name`, `marker_id_start`. 선택 키: `legacy_pattern` (기본 `false`).
   `name`에는 파일 이름과 같은 값을, `_comment`에는 어떤 실물인지 적는다.
3. `python3 -m pytest tests/test_charuco_board_targets.py`로 확인한다.

### 값을 정할 때 틀리기 쉬운 것

- **`squares_x`는 가로 칸 수, `squares_y`는 세로 칸 수다.** 인쇄물 캡션(`6x9` 등)은
  행x열인 경우가 있으니 캡션을 옮기지 말고 사진에서 직접 센다. 내부 코너 수가 아니라
  칸 수다.
- **세로 칸 수가 짝수면 `legacy_pattern`을 반드시 확인한다.** OpenCV 4.6 이전에 만든
  보드(좌상단 칸이 검정)는 `true`다. 세로 칸 수가 홀수면 두 배치가 같아서 상관없다.
- 둘 중 하나만 틀려도 **마커는 전부 검출되고 ChArUco 코너만 0개**가 된다. 마커 수는
  `가로*세로/2`라 가로/세로를 뒤집어도 같기 때문에 개수만 봐서는 알 수 없다.
  `02`의 `[DIAG] layout ...` 로그가 뒤집은 배치와 legacy 배치의 코너 수를 함께 보여준다.

## 주의

- `board_11x7_id5.json`은 `config.py`의 `CharucoBoardConfig` 기본값과 같아야 하며
  테스트가 이를 강제한다. 이미 촬영된 세션의 `meta.json`이 이 값으로 굳어 있으니 고치지 않는다.
- `legacy_pattern`이 `false`인 보드는 직렬화할 때 이 키를 쓰지 않는다. 이 필드가 생기기
  전에 굳은 `meta.json`/manifest와 키 집합·SHA-256이 그대로 같아야 하기 때문이다.
