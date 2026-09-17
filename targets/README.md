# targets

물리 캘리브레이션 타깃 정의. 세션 데이터(`data/`)나 결과(`ABLATION_TEST_result_0909/`)가 아니라,
**어떤 물체를 촬영했는지**를 기술하는 파일만 둔다.

| 타깃 | 정의 위치 | 용도 |
|---|---|---|
| 메인 AprilTag 큐브 | [`calibration_pipeline/config.py`](../calibration_pipeline/config.py) 의 `CubeConfig` | 캘리브레이션 파이프라인 기본 타깃 |
| GT 검증용 큐브 | [`gt_cube/`](gt_cube/) | GT 검증 실험 전용, 별도 물체 |
| ChArUco 보드 (전부) | [`charuco_boards/`](charuco_boards/) | 보드별 JSON. `resolve_charuco_config(이름)` 으로 읽는다 |

메인 큐브는 `config.py`가 단일 소스이므로 여기에 사본을 두지 않는다.
`config.py`가 기술하지 않는 타깃만 이 디렉터리에 JSON으로 정의한다.

ChArUco 보드는 촬영마다 바뀔 수 있어서 메인 보드까지 포함해 **모든 보드를**
`charuco_boards/`에 JSON으로 둔다. 메인 보드(`11x7_id5`)는 `config.py` 기본값과
같다는 것을 테스트가 강제한다. 자세한 규칙은 [`charuco_boards/README.md`](charuco_boards/README.md).

큐브 타깃 디렉터리는 두 파일로 구성한다.

- `cube_config.json` — 마커 모델. `load_cube_config_from_json_file()` 이 그대로 읽는다.
- `cube_solid.json` — 마커 주변 형상(mm 박스 목록). 작도 전용이며 캘리브레이션에는 쓰이지 않는다.
