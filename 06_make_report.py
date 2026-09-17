#!/usr/bin/env python3
"""06. 저장된 Table 1 JSON의 스키마에 맞춰 보고서를 재생성한다.

실행: python 06_make_report.py --table1 <결과 JSON> [--out_dir <출력 폴더>]
출력 폴더를 생략하면 입력 JSON과 같은 폴더에 저장한다. calibration 재추정,
재적합, held-out 성적에 따른 선택은 수행하지 않는다.

1. Zeus v4 (`table1_zeus_cube_and_cross_view_v4`)
   zeus_gello_calibration/report_table1.py의 공통 생성기를 사용한다.
   Cube/Cross-view의 ALL·Train·Held-out Test와 External GT를 동일한
   Markdown·요약 CSV·fold CSV 형식으로 기록한다. 미산출은 Pending이다.
   원본 JSON도 ABLATION_TEST_table1_methods.json에 바이트 그대로 보존한다.

2. 기존 protocol/rows + seed runs JSON (명시적 schema 없음)
   기존 calibration_summary.csv와 calibration_matrices.json을 생성한다.
   대표 seed는 --representative_seed(기본 0)로 고정한다. 행렬은 4×4이며
   이동 단위는 m: T_base_Ci와 T_gripper_cam은 카메라 보정값,
   T_base_board와 T_base_cube_by_set은 해당 세션의 표적 자세다.

제한: 보고서 변환만으로 기존 event split을 leave-one-placement-out으로
바꾸거나 누락된 지표를 만들 수 없다. 새 데이터도 동일 비교가 필요하면
같은 평가 프로토콜로 계산한 v4 JSON을 입력해야 한다.

진입점 구현: calibration_pipeline/report.py의 main()/write_report().
"""

# calibration_pipeline/report.py 의 main() 을 그대로 사용한다.
# (인자 파서와 write_report() 호출이 모두 그쪽에 있다)
from calibration_pipeline.report import main


if __name__ == "__main__":
    main()
