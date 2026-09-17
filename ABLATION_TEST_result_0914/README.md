# 0914 비교실험 결과

현재 확인할 결과: [0914 재촬영 요약 보고서](zeus_recapture_0914/ABLATION_TEST_table1/ABLATION_TEST_TABLE1_RESULTS.md) · [회의 피드백 반영 점검](zeus_recapture_0914/FEEDBACK_CHECK.md).

## 촬영 데이터와 결과 폴더

| 폴더 | 실제 촬영 | 평가 방식 |
| --- | --- | --- |
| [zeus_recapture_0914](zeus_recapture_0914/README.md) | `capture_replayed_0914` / `capture_placed_0914`, GT 큐브, 변경된 리그·intrinsics | 전체 placement 재fit + leave-one-placement-out; Cube/Cross-view 각각 ALL·Train·Held-out Test |
| [session11_zeus_handheld_floor_wrist_meta_0909](session11_zeus_handheld_floor_wrist_meta_0909/README.md) | 0909 촬영의 legacy meta 변환 | 기존 event split, 3 seed calibration 요약 |

## 기존 0914 폴더가 간결했던 이유

기존 session11 실행은 `05_calibrate.py` 후 `06_make_report.py`까지 실행했다.
당시 `06`은 `calibration_summary.csv`와 `calibration_matrices.json`만 생성했다.
0909 README가 설명하던 상세 Markdown·HTML은 별도 `tools/sync_table1_canonical_data.py`
워크플로의 산출물이며, 그 워크플로를 session11에 적용한 기록은 없다.
0909 폴더의 D1/FK 진단도 별도 실험이다. 이 로컬 checkout에는 0909 README가 가리키는
과거 session04 파일 중 일부가 없으므로 README의 링크 목록 자체를 실행 결과로 간주하지 않는다.

보고서 양뿐 아니라 데이터와 split도 달랐다. 기존 session11의 event split 결과를
LOPO 결과로 이름만 바꾸거나, 0909 결과를 0914 재촬영 결과로 복사할 수 없다.
새 보고서는 사용자가 지정한 **실제 0914 재촬영 데이터**를 다시 계산한 결과다.

현재 `06_make_report.py`는 v4 JSON을 입력받으면 같은 간략 MD·CSV 생성기를 호출한다.
새 데이터도 같은 LOPO 실행기로 계산한 v4 JSON을 사용해야 한다. 기존 event-split JSON의
점수를 보고서 단계에서 LOPO 점수로 바꾸지는 않는다.

## 결과 열

Cube 3개, Cross-view 3개, External GT 4개 지표를 Markdown과 CSV에 명시한다.
픽셀 지표는 모두 `sqrt(mean(dx² + dy²))`로 계산한다.
독립 External GT 미수집 항목과 비전 미사용 장착 변환이 확인되지 않은 A3는
숫자 대신 `Pending` 및 사유를 기록한다. 자세한 확인 근거는
[입력 감사](zeus_recapture_0914/INPUT_AUDIT.md)를 참고한다.
