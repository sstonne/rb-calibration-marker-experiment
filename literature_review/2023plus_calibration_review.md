# 멀티카메라 및 로봇–카메라 캘리브레이션: 2023년 이후 실측 결과 비교

가장 직접적인 비교 논문은 **Allegro et al.의 Multi-Camera Hand-Eye Calibration(2024)**과 **Evangelista et al.의 Graph-based Multi-Camera Hand-Eye Calibration(2023)**이다. 로봇 FK 불확실성 처리를 주장하려면 **Ulrich–Hillemann(2024)**과 **Ha(2023)**를 함께 비교해야 한다. 카메라만 보정하는 분야에서는 **FusedBA(2024)**의 독립 3D 검증과 **Laven et al.(2026)**의 불완전한 3D 타깃 보정이 이 프로젝트에 특히 중요하다.[^R1][^R2][^R4][^R5][^C1][^C5]

조사 범위는 **2023년을 포함한 2023-01-01~2026-09-09**다. 정식 출판연도를 우선하되 최초 공개일과 최신 원고 버전을 구분했다. 핵심 문헌 25편의 문제 설정, 저자 주장, 평가 지표, 주요 정량 결과, SOTA 근거 및 프로젝트 관련성을 정리했다. 모든 논문의 모든 그래프 점을 수치화한 전수조사는 아니며, 주요 결과표와 결론을 바꾸는 비교·ablation을 중심으로 했다. 원문에서 확인하지 못한 수치는 추정해서 채우지 않았다.

**결과 선정 기준: 실제 장비·실촬영 데이터의 결과만 수록한다. 시뮬레이션 수치는 결과표와 SOTA 판단에서 제외한다.** 실험 종류를 확인하지 못한 초록 수치도 제외한다. 문헌 25편을 검토했으며, 실측 수치 확인이 부족한 C5·C7·R4는 정량 순위에 넣지 않는다.

**서로 다른 논문의 mm·px 수치만 나열해 전체 순위를 매길 수는 없다.** 실측 결과도 ① 독립 계측 기준, ② 영상·FK 기반 참조값 대비 오차, ③ 실제 targeting 오차, ④ 재투영·폐루프·반복 일관성으로 구분한다. ④의 작은 수치는 절대 물리 정확도를 입증하지 않는다. SOTA 표시는 실제 데이터의 같은 평가 조건에서 확인되는 우위만 뜻한다.

## 1. 프로젝트와 비교 범위

### 1.0 신규 물체 위치 오차 2~3 mm와의 비교

이 비교에서는 **로봇 FK로 산출한 물체 위치가 정확한 정답이며, 신규 물체 위치 추정 오차가 2~3 mm**라는 실험 조건을 채택한다. 아래는 실제 장비 결과만으로 고른 비교군이다. 서로 다른 지표도 포함하되, ‘논문의 보고 오차가 더 작다’와 ‘동일 신규 물체 문제에서 더 우수함이 입증됐다’를 구별한다.

| 논문 | 실제 장비에서 보고한 결과 | 2~3 mm 대비 | 처음 보는 물체 평가와의 관계 |
|---|---|---|---|
| **R15 GWM-View, 2023** | 복잡한 산업 부품의 평균 positioning error **1 mm 미만** | **확연히 작은 보고값; 물체 위치 작업 면에서 우선 확인** | 3카메라+ABB 로봇; 임의 초기 자세의 알려진 부품. 미지 물체 일반화는 미확인; 초록·공개 실험 절만 확인 |
| **R12 UAL-HED, 2026** | 별도 99개 측정에서 FK 대비 conversion **0.202~0.342 mm** | **확연히 작음; FK 기준 변환이라는 점에서 가까움** | 카메라 대신 laser tracker+T-Mac의 6DoF 측정. 신규 물체 vision 인식은 없음 |
| **R14 Enhanced robotic 3D scanning, 2025** | CMM 기준 구 사이 거리: 평균 **0.373 mm**, 최대 **0.421 mm** | **확연히 작음; 3D 형상 계측 기준** | 로봇+정밀 structured-light scanner. 구 간 거리이므로 물체 원점 위치와는 다름 |
| **C8 AIRLS multi-camera, 2024** | 검증 wand 길이 평균 오차 **0.42±0.09 / 0.46±0.13 mm** | **확연히 작음; 카메라만의 거리 측정 기준** | 4대 광학 카메라, 보정과 다른 검증 wand. 알려진 반사 마커 간 거리이며 robot-base 절대 위치 아님 |
| **C1 FusedBA, 2024** | 별도 기록의 mocap 기준 3D 점 평균 오차 **2.14310 mm** | **비슷함; 3D 점 위치 평가로 가장 가까운 카메라 비교군** | 마커 점의 위치, 궤적 간 강체 정합 후 평가. 일반 물체 중심·자세 인식은 없음 |
| C1의 MocapAssisted baseline | 같은 평가에서 **1.35763 mm** | 수치상 더 낮음 | 보정에도 mocap을 사용하는 추가 장비 조건 |
| C6 Ohshima, 2026 | 5카메라 3D 점 **2.113±0.662 mm**, 10카메라 **1.445±0.833 mm** | 5대는 비슷함, 10대는 낮음 | 영상 기반 참조 calibration과 비교. 알려진 고정점, 일반 물체 인식 아님 |
| **R8 EasyHeC++, 2024** | 실제 robot targeting **3.0 / 3.1 mm** | **비슷한 수준; 수치상 우리보다 작지는 않음** | 로봇이 목표점을 짚는 작업; 위치 추정 외 tool·동작 오차 포함 |

근거: 각 ID의 실물 결과표·원문 위치를 아래 개별 항목에 연결했다.[^R15][^R12][^R14][^C8][^C1][^C6][^R8] **넓은 의미의 실측 mm 성능에서 더 좋은 보고값은 존재한다. 그러나 이 목록만으로 ‘처음 보는 물체 + 우리의 센서 구성 + 같은 GT 평가’ 전체를 더 잘 푼 방법이 확인된 것은 아니다.** 가장 가까운 후속 검토 대상은 실제 부품 위치를 평가하는 GWM-View, 보정 후 별도 관측의 3D 점을 검증하는 FusedBA, 실제 로봇 목표점 접근을 평가하는 EasyHeC++다. UAL-HED·정밀 스캐닝은 FK와 센서 정확도의 상한을 논의할 때 중요하다.

### 1.1 현재 구현에서 확인한 구성

| 항목 | 프로젝트의 실제 구성 | 논문 비교에 주는 의미 |
|---|---|---|
| 센서 | 여러 고정 카메라 + 손목의 eye-in-hand 카메라; Zeus 문서는 고정 3대 + 손목 1대 | 고정 카메라만 다루는 방법과 혼합 구성을 구분 |
| 로봇 | UR 계열 세션과 Zeus 별도 실험 | 서로 다른 로봇·세션 결과를 한 표본처럼 합치지 않음 |
| 타깃 | ChArUco 보드 + 파지·배치 가능한 다면 AprilTag 큐브 | 다면 가시성, 타깃 강체 형상, 제작 오차가 핵심 |
| 최종 추정 | 원본 2D 코너의 pixel reprojection 최적화, 공유 target pose와 카메라 변환 추정 | PnP는 주로 초기화; pose-only AX=YB 방법과 목적함수가 다름 |
| intrinsic | 메인 비교에서 K,D 사전 보정 후 고정 | intrinsic도 함께 보정하는 외부 방법과 자유도 맞춤 필요 |
| 강건성 | soft_l1, f_scale=2 px, 학습 frame prune/refit | 외부 baseline에도 동일 관측 품질 조건 필요 |
| FK 처리 | 큐브 pose를 vision 추정 / raw-FK 고정 / 정렬 FK soft factor / 정렬 FK hard fixed로 변경 | 손목 카메라 운동을 위한 FK 사용과 큐브 앵커 사용을 분리 |
| 평가 | heldout cube px, cross-view px, camera-common mm/deg, 외부 GT 계획; Zeus에는 z/rz 소규모 실험 | 내부 일관성과 독립 물리 정확도를 분리 |

근거: [README](../README.md), [현재 연구 정의](../RESEARCH_STORYLINE.md), [pixel solver](../calibration_pipeline/reprojection.py), [Zeus 통합·독립 solver](../zeus_gello_calibration/fit_calibration_methods.py), [Zeus 결과](../zeus_gello_calibration/ABLATION_RESULTS.md).

**`no-FK`는 로봇 없이 카메라끼리만 보정한다는 뜻이 아니다.** 손목 카메라 pose는 여전히 `T_base_gripper(event) @ T_gripper_camera`로 계산된다. 이 이름은 주로 배치 큐브 pose를 FK로 고정하지 않는다는 뜻이다. 따라서 본 프로젝트의 주 분야는 아래 B이고, A는 카메라 네트워크 부분을 평가하는 보조 비교군이다.

| 분류 | 구하는 것 | 로봇 정보 | 우리 실험에서의 역할 |
|---|---|---|---|
| A. 카메라 간 캘리브레이션 | 카메라 상대 pose, 경우에 따라 intrinsic·시간 오프셋·타깃 형상 | 불필요 | 고정 카메라 그룹의 relative calibration baseline |
| B. 로봇 포함 캘리브레이션 | camera–base, camera–gripper, target–robot 등 | FK/robot poses 또는 대체 운동 측정 | 전체 시스템의 직접 비교군 |
| C. 벤치마크·평가 | 데이터와 GT·평가 정의 제공 | 데이터에 따라 사용 | 재현성 및 SOTA 주장 검증 |

### 1.2 저장소 문서의 차이

현재 [RESEARCH_STORYLINE](../RESEARCH_STORYLINE.md)의 A5는 **vision-aligned FK hard fixed**다. 과거 [SOTA_Claim_Protocol](../SOTA_Claim_Protocol.md)의 A5는 **후처리 residual correction**으로 적혀 있어 현재 정의와 다르다. 본 보고서는 현재 storyline와 구현을 따른다. [PAPER_BASELINES](../PAPER_BASELINES.md)의 “Tabb는 단일 카메라”, “Allegro만 카메라 결합”, “Kalib는 단일 설정만 지원” 등으로 읽힐 수 있는 서술도 그대로 채택하지 않았다. Tabb 제목 자체가 eye(s)를 포함하고, 2023년 graph 방법도 다중 카메라를 결합하며, Kalib는 eye-in-hand와 eye-on-base를 모두 설명한다.[^R1][^R6][^OLD]

Zeus 문서의 FK 기준 검증에는 `session1` 비전으로 구한 `T_gripper_cube`가 들어간다. 따라서 **완전히 비전과 독립된 외부 GT로 분류하지 않았다.** 또한 Zeus의 `raw-fk` 변형은 메인 A3의 순수 기계적 raw-FK 고정과 동일하지 않다. 이 구분은 논문 수치와 우리 수치를 비교할 때 중요하다.

## 2. SOTA·관련성 표시 읽는 법

| 표시 | 의미 |
|---|---|
| **S1: 제한된 실측 SOTA 근거** | 논문이 비교한 동일 실측 데이터·지표에서 우위가 확인됨. 현재 전 분야 최고라는 뜻은 아님 |
| **S2: SOTA 후보** | 최신 방법이고 유망하지만 preprint, 제한된 비교, 또는 재현·GT 한계가 있음 |
| **B: 강한 baseline / 선행연구** | 비교·인용 가치가 높지만 현재 최고라고 부를 근거는 부족 |
| **E: 평가·도구** | 데이터셋·실용 도구·평가 설계 논문. 정확도 SOTA와 구별 |
| 관련성 5/5 | 핵심 방법·기여 또는 직접적인 평가 문제와 겹침 |
| 관련성 4/5 | 중요 비교군이지만 센서 구성·목적함수·타깃 조건이 다름 |
| 관련성 3/5 | 보조 비교·구현 참고 |
| 관련성 2/5 | 주변 분야·추가 조건이 많음 |

관련성은 본 보고서의 분석 판단이다. 논문의 우수성이나 학술지 등급을 뜻하지 않는다. “지표별 부분 우위”, “후속 논문 존재”, “독립 GT 없음”을 함께 읽어야 한다.

## 3. 전체 논문 목록

실측 근거를 아래처럼 나눠 읽는다. **직접 계측 기준의 절대 정확도만 비교하려면 첫 행을 우선하고, 나머지 행의 수치를 같은 정확도 순위에 섞지 않는다.**

| 실측 근거 | 해당 논문 | 실제로 검증하는 것 |
|---|---|---|
| 별도 계측 장비 기준 | C1 mocap, C9 CMM, R10의 OptiTrack 관련 항목 | 3D 점·타깃 변환 오차; C1은 궤적 정합 후 평가 |
| 실제 물체의 알려진 거리 | C2, C8, R14(CMM 기준 거리) | 두 구 사이 거리 재구성; camera pose의 독립 6DoF GT 아님 |
| 영상·FK 기반 참조 calibration | C3, C6, R2, R10의 Kalibr 항목, D1 | 기준 calibration 대비 차이; 기준 자체의 오차 존재 |
| 실측 pose 결과지만 GT 구축 상세 미확인 | R9 | 실촬영 데이터의 pose 오차; 독립 계측 SOTA 판단 보류 |
| 실제 로봇 targeting·부품 positioning | R7, R8, R15(상세 GT 미확인) | 목표점 접근 오차; robot·tool·vision 오차가 함께 포함 |
| 영상 잔차·운동학 일관성·반복성 | C4, R1, R3, R5, R6, R11, R12, R13 | 실측 데이터 적합도·정밀도; 독립 절대 정확도와 구분 |
| 실측 정량 검증 부족 | C5, C7, R4 | 관련 연구로 유지하되 정량 SOTA 비교 제외 |

| ID | 논문 / 방법 | 출판·버전 | 분류 | 관련성 | SOTA 판단 | 가장 중요한 결과 또는 한계 |
|---|---|---|---|---|---|---|
| C1 | Multi-Camera Calibration Using Far-Range Dual-LED Wand and Near-Range Chessboard Fused in Bundle Adjustment | Sensors 2024 | 카메라 | **5/5: 평가 설계** | **S1: 해당 stand-alone 비교** | FusedBA 독립 기록 3D 오차 2.14310 mm; mocap-assisted는 더 낮음 |
| C2 | Wand-Based Calibration of Unsynchronized Multiple Cameras for 3D Localization | Sensors 2024 | 카메라 | 2/5 | B | 비동기 3카메라; 거리 재구성 오차 1.3425~18.5684 mm |
| C3 | Spatiotemporal Multi-Camera Calibration using Freely Moving People | RA-L 2025, arXiv v3 | 카메라 | 2/5 | S1: 실영상 pose·대응 추정 | 동기화 실영상 Table I만 사용; translation은 정규화 값, mm 아님 |
| C4 | Incremental Multi-Camera Extrinsic Calibration Method Based on PnP Integrating Weighted AprilTag Detections and Multi-View Triangulation | Algorithms 2026 | 카메라 | 4/5 | B: 초기화 비교 | 평균 reprojection 7.451→1.006 px; 전역 BA 이전 초기화 성능 |
| C5 | Multi-Camera Calibration With Imperfect 3-D Targets for Imaging Biological Systems | IEEE TIM 2026 | 카메라 | **5/5: 큐브 형상** | B: 실측 정량 판단 보류 | 초록만 확인; 실측 조건과 수치 대응을 확인할 수 없어 정량 비교 제외 |
| C6 | Wand-Based Calibration Accuracy for Unsynchronized Multicamera Systems Without Timestamps | Sensors 2026 | 카메라 | 3/5 | B | 10카메라 3D 복원 1.445±0.833 mm; 3카메라는 개선 안 됨 |
| C7 | Caliscope: GUI Based Multicamera Calibration and Motion Tracking | JOSS 2024 | 카메라 | 3/5 | E | 공개 실용 도구; 논문에 정량 정확도 비교표 없음 |
| C8 | Enhanced Three-Axis Frame and Wand-Based Multi-Camera Calibration Method Using Adaptive Iteratively Reweighted Least Squares and Comprehensive Error Integration | Photonics 2024 | 카메라 | **5/5: 별도 물체 검증** | S1: 검증 wand 거리 | 실물 4카메라, 검증 wand 거리 0.42±0.09 / 0.46±0.13 mm |
| C9 | Calibration Procedure of a Multi-Camera System: Process Uncertainty Budget | Sensors 2023 | 카메라·계측 | 4/5 | E: CMM 기반 평가 | CMM 기준 LME, 선택 조건 최대 0.030 mm; 정밀 측정 환경 |
| R1 | A Graph-Based Optimization Framework for Hand-Eye Calibration for Multi-Camera Setups | ICRA 2023 | 로봇 | **5/5: 직접 비교** | S1: 발표 당시 비교 | 실물 3카메라 평균 reprojection 1.6818 px |
| R2 | Multi-Camera Hand-Eye Calibration for Human-Robot Collaboration in Industrial Robotic Workcells | RA-L 2024 | 로봇 | **5/5: 최우선 비교** | **S1: METRIC 실물 위치 오차** | 실물 13.21~51.18 mm; GT는 AprilTag+CAD+FK, 지표별 예외 있음 |
| R3 | Simultaneously Calibration of Multi Hand-Eye Robot System Based on Graph | 2023 공개 / TIE 71(5), 2024 | 로봇 | 4/5 | B | UR3+UR5×2, 카메라 5대; 0.6274~0.6938 mm는 폐루프 오차 |
| R4 | Probabilistic Framework for Hand–Eye and Robot–World Calibration AX=YB | T-RO 2023, online 2022 | 로봇 | **5/5: 불확실성** | B | 측정별 covariance와 MLE; 정량 결과 원문표 미확인 |
| R5 | Uncertainty-Aware Hand–Eye Calibration | T-RO 2024, online 2023 | 로봇 | **5/5: 불확실성** | **S1: 실측 잔차·재구성** | UR3e 0.193 px, RAE 0.00073 mm²; robot poses 보정 포함 |
| R6 | Kalib: Easy Hand-Eye Calibration with Reference Point Tracking | IROS 2025 / arXiv 2024→2025 | 로봇 | 4/5 | B | CAD 없는 기준점 추적; DROID IoU 0.80, 전체 GT pose 오차 아님 |
| R7 | EasyHeC: Accurate and Automatic Hand-eye Calibration via Differentiable Rendering and Space Exploration | RA-L 2023 | 로봇 | 3/5 | B: 후속 EasyHeC++ 존재 | 실물 targeting 약 4 mm; extrinsic GT 오차 아님 |
| R8 | EasyHeC++: Fully Automatic Hand-Eye Calibration with Pretrained Image Models | IROS 2024 | 로봇 | 4/5 | S1: 실측 targeting 비교 | 실제 targeting 3.0/3.1 mm; PCK에서는 일부 baseline보다 낮음 |
| R9 | Calib3R: Hand-Eye Calibration and 3D Metric-Scaled Scene Reconstruction with 3D Foundation Models | arXiv 2025, **v2 2026-08-08** | 로봇 | 4/5 | **S2: markerless joint** | UR Object 6.36 mm / 0.017 rad; 모든 marker-based보다 좋지는 않음 |
| R10 | A Certifiably Correct Algorithm for Generalized Robot-World and Hand-Eye Calibration | arXiv 2025 / IJRR 2026 | 로봇·일반 운동계 | 4/5 | B: 실측 지표별 부분 우위 | 실물 카메라 상대 오차 3.20 cm / 0.99°; 목적함수 최적성은 별도 기여 |
| R11 | PlaneHEC: Efficient Hand-Eye Calibration for Multi-view Robotic Arm via Any Point Cloud Plane Detection | arXiv 2025 v1 | 로봇 | 3/5 | S2: depth·plane | 3.35 mm / 0.13°는 반복 추정 분산 기반; 독립 GT 아님 |
| R12 | Optimal Uncertainty-Aware Calibration for the AX=YB Problem | arXiv 2026, IJRR 심사 중 | 로봇·레이저 트래커 | 4/5 | S2: 불확실성 적응 | 실측 conversion 0.342 mm; 카메라 실험 아닌 tracker–FK 비교 |
| R13 | Multi-Camera Robot-World Hand-Eye Calibration by Solving Multi-Unit Dual Quaternion Equations | CSIAM Trans. Applied Math. 2026 | 로봇 | 4/5 | B: 빠른 대수 해법 | 실물 dataset 7: 2.48/2.54/1.67 px, 0.30 s; pixel 정확도 최고 아님 |
| R14 | Enhanced Calibration Method for Robotic Flexible 3D Scanning System | Sensors 2025 | 로봇·3D 스캐너 | 4/5 | S1: 해당 구 간격 비교 | 실물 평균 구 간 거리 오차 0.373 mm, 최대 0.421 mm |
| R15 | GWM-view: Gradient-weighted multi-view calibration method for machining robot positioning | RCIM 2023 | 로봇·멀티카메라 | **5/5: 실제 부품 위치** | B: 상세 SOTA 판단 보류 | 실제 부품 평균 positioning 1 mm 미만; 실험표·GT 상세 미확인 |
| D1 | METRIC—Multi-Eye to Robot Indoor Calibration Dataset | Information 2023 | 벤치마크 | **5/5: 공개 평가** | E | 실제 GT는 AprilTag+CAD+FK로 구성; laser tracker GT 아님 |

각 행의 원문 링크와 출처는 아래 개별 항목 및 참고문헌에 있다. EoB는 eye-on-base(eye-to-hand), EiH는 eye-in-hand다.

## 4. 카메라끼리만 하는 캘리브레이션

### C1. FusedBA — Jatesiktat, Lim, Ang, 2024

7대 동기화 RGB 카메라의 먼 거리 dual-LED wand 관측과 근거리 checkerboard 관측을 함께 BA한다. 주장하는 핵심은 wand-only 최적화의 과적합을 줄이고, 보정에 쓰지 않은 기록의 3D 정확도를 개선한다는 것이다. 33회 calibration 후 별도 2분 mocap 기록에서 삼각측량 점의 평균 Euclidean 오차를 평가한다. 좌표계는 검증 궤적 사이의 강체 정합으로 맞춘다.[^C1]

| 방법 | 독립 기록 3D 평균 오차 RDB, mm | DLT, mm | 학습 wand RMS 재투영, px |
|---|---:|---:|---:|
| Anipose | 10.05912 ± 2.07237 | 10.67667 ± 2.77518 | N/A |
| MocapAssisted | 1.35763 ± 0.00705 | 1.42898 ± 0.00808 | N/A |
| BAp15 | 2.32816 ± 0.02848 | 2.39151 ± 0.02804 | **0.23108 ± 0.00419** |
| **FusedBA** | **2.14310 ± 0.01418** | **2.22776 ± 0.01427** | 0.26048 ± 0.00365 |
| FusedBA+W | 2.14911 ± 0.01470 | 2.22979 ± 0.01464 | 0.26230 ± 0.00367 |

출처: Table 1, §4.6, pp.17–19. ±는 calibration 반복 간 변동이다. **낮은 학습 px가 좋은 독립 3D 성능을 보장하지 않는 실제 반례**다. 같은 비교의 독립 보정 방법 중 FusedBA가 우수하지만, mocap-assisted가 더 정확하다. 검증 궤적으로 강체 정합하므로 robot-base 절대 정렬 평가와 같지 않다. 우리에게는 새 solver보다 **평가 설계**가 가장 중요하다. [데이터·프로젝트](https://koonyook.github.io/MCalib/).

### C2. 비동기 wand — Zhang, Fu, 2024

Timestamp matching과 pixel fitting으로 프레임레이트·전송 지연이 다른 카메라를 보정하고 EKF 기반 3D localization에 사용한다. 지표는 카메라별 RMS reprojection과 알려진 구 사이 거리의 재구성 오차다.[^C2]

| 평가 | 조건 | 보고 결과 |
|---|---|---|
| 3카메라 RMS reprojection | 60/60/110 Hz, 3회×3카메라 | 2.405~7.780 px |
| 3카메라 RMS reprojection | 90/90/110 Hz, 3회×3카메라 | 1.066~5.314 px |
| 거리 재구성 오차 | 60/60/110 Hz | 7.9496 / 1.3425 / 4.3363 mm |
| 거리 재구성 오차 | 75/75/110 Hz | 12.1173 / 18.5684 / 5.6703 mm |
| 거리 재구성 오차 | 90/90/110 Hz | 8.2264 / 1.6776 / 16.1708 mm |

출처: Tables 4–6, pp.12–13. 범위는 해당 표 원소의 최솟값~최댓값이며 평균이 아니다. 세 거리 오차는 카메라 위치 오차와 다르다. 정지 촬영 중심인 우리 세팅의 직접 경쟁자는 아니지만, 이동 중 촬영·비동기 문제의 참고문헌이다.

### C3. 사람 움직임 기반 시공간 보정 — Lee, Nishino, Nobuhara, 2025

사전학습 3D 인체 pose를 이용해 camera pose, 시간 오프셋, 사람 대응을 결합 추정한다. rotation은 rad, translation RMSE는 **첫 두 카메라 GT 거리로 정규화한 무차원 값**, 재투영은 px, 시간 오차는 frame, 대응 품질은 precision이다.[^C3]

| 동기화 실촬영 데이터 / IBA | 회전 rad | 정규화 translation | reprojection px | 대응 precision |
|---|---:|---:|---:|---:|
| Panoptic 160906_pizza1 | 0.027 | 0.023 | 3.875 | 0.706 |
| ZJU soccer1_6 | 0.006 | 0.004 | 3.778 | 0.722 |
| MMPTRACK industry_safety_2 | 0.024 | 0.033 | 2.415 | 0.896 |

출처: Table I, §§V-A/B. 실제 촬영한 각 장면에서 카메라 4대를 사용한다. 비동기 Table II는 실영상에 인위적 시간 offset을 주었으므로 엄격한 실측 비교에서는 제외했다. 같은 세 장면에서 ReID-Calib IBA의 회전은 1.448/1.465/1.051 rad로 제안보다 크다. **사람 장면의 pose·대응 추정에서 제한된 실측 SOTA 근거**이며, 절대 mm 스케일이나 robot-base alignment와는 다른 문제다.

### C4. Weighted AprilTag PnP + multi-view triangulation — Demidova, Zhuravlev, 2026

20카메라 object-centric rig에서 카메라를 순차 등록할 때, 가중 PnP와 다중시점 marker triangulation을 사용해 누적 drift를 줄인다. 실물 reprojection·angular·epipolar consistency와 후속 BA 수렴을 평가한다.[^C4]

| 초기화 방법 | median reprojection px | mean reprojection px |
|---|---:|---:|
| Baseline | 6.417 | 7.451 |
| Weighted PnP만 | 5.828 | 7.244 |
| Triangulation만 | 0.894 | 1.180 |
| **둘 다** | **0.757** | **1.006** |

출처: Table 2, pp.22–23. 이 표는 **후속 global BA 성능과 분리된 초기화 비교**다. 주요 이득은 triangulation에서 나온다. 우리의 AprilTag 관측·초기값 생성에 관련되지만, 로봇 포함 최종 pixel BA보다 정확하다는 증거는 아니다.

### C5. 불완전한 3D 타깃 — Laven et al., 2026

정확하다고 가정한 3D calibration target 자체에 제작 오차가 있을 때 target geometry와 calibration을 개선한다. 고해상도 인체·피부·장기 영상의 세 카메라 시스템에서 평가했다.[^C5]

출처: 출판사 abstract. **실험표 전체와 각 수치의 실측 조건을 확인하지 못해, 이 보고서에서는 정량 결과를 제외했다.** 주장 지표는 Euclidean reprojection, target geometry 복원 오차, 대상 복원량이다. 타깃 형상 복원 정밀도를 camera extrinsic/TRE 정확도로 해석하면 안 된다. 우리 다면 큐브의 면 간 위치·각도·출력 치수 오차와 매우 관련 높지만 실측 SOTA 판단은 보류한다.

### C6. Timestamp 없는 비동기 wand — Ohshima, 2026

시간 offset과 프레임 간격 비율을 최적화 변수로 두고 보간한다. 20회 반복, 45개 고정점 재구성을 gold-standard calibration과 비교했다. 원문의 gold standard도 영상 기반 보정 결과이므로 독립 계측 장비의 절대 GT와 구분한다.[^C6]

| 카메라 수 | 3D 복원 Interp, mm | No-interp, mm | ray distance Interp, mm | No-interp, mm |
|---|---:|---:|---:|---:|
| 10 | 1.445 ± 0.833 | 1.747 ± 0.908 | 1.914 ± 0.222 | 2.024 ± 0.186 |
| 5 | 2.113 ± 0.662 | 2.324 ± 1.019 | 1.777 ± 0.185 | 2.030 ± 0.256 |
| 3 | 5.121 ± 2.753 | 4.745 ± 2.304 | 1.381 ± 0.367 | 1.493 ± 0.448 |

출처: Tables 1–2, pp.11–12. 3D 오차의 유의한 개선은 10카메라에서만 확인된다. 3카메라에서는 보간이 수치상 더 나쁘므로 ‘항상 개선’이라고 할 수 없다. 초록·본문에 No-interp 1.746/1.747/1.757 표기 차이가 있어 **Table 1의 1.747**을 채택했다.

### C7. Caliscope — Prible, 2024

비디오 기반 multicamera calibration·triangulation GUI 도구다. OpenCV 계열 intrinsic 4개와 distortion 5개를 추정하며 카메라 extrinsic 보정과 3D tracking을 연결한다. 3쪽 JOSS 논문에는 독립 GT pose error나 비교 정확도 결과표가 없다. 따라서 **정량 성능은 ‘미보고’, SOTA는 해당 없음**으로 둔다. 우리 고정 카메라 그룹의 실용적인 구현 비교 대상으로는 적합하다.[^C7]

### C8. AIRLS multi-camera — Yuhai et al., Photonics 2024

4대 OptiTrack FLEX13 카메라로 three-axis frame과 wand 기반 intrinsic·extrinsic·distortion을 보정한다. 약 2 m 거리에서 촬영하며, calibration과 다른 상용 검증 wand 두 종류를 5회씩 측정한다. **보정 후 별도 물체를 측정한다는 점에서 직접 참고할 가치가 높다.**[^C8]

| 검증 wand의 거리 오차 mm | Normalized DLT | Related method 1 | Related method 2 | AIRLS |
|---|---:|---:|---:|---:|
| 390 mm wand, 평균 | 5.50 ± 1.80 | 3.45 ± 0.15 | 1.21 ± 0.17 | **0.42 ± 0.09** |
| 500 mm wand, 평균 | 6.73 ± 1.83 | 3.77 ± 0.44 | 1.74 ± 0.07 | **0.46 ± 0.13** |
| 390 mm wand, 최대의 반복 통계 | 11.89 ± 2.67 | 7.65 ± 0.51 | 3.88 ± 1.05 | **1.48 ± 0.64** |
| 500 mm wand, 최대의 반복 통계 | 14.22 ± 3.27 | 9.29 ± 1.14 | 5.55 ± 0.48 | **1.82 ± 0.29** |

출처: Table 3, §§2.4/3. ±는 5개 trial의 통계다. **두 반사 마커 사이의 재구성 거리**이므로 두 점이 함께 이동하는 공통 위치 편향은 검출하지 못한다. 우리 물체 원점의 FK 대비 오차와 동일 지표는 아니지만 sub-mm camera-only 계측을 입증하는 강한 비교군이다. 관련 비교에서 제한된 S1로 분류한다.

### C9. Process uncertainty budget — Leizea et al., Sensors 2023

CMM으로 이동시킨 마커가 만드는 virtual grid와 photogrammetry를 비교하고 calibration chain의 uncertainty를 분석한다. 여기서 virtual grid는 렌더링 영상이 아니라 **실물 CMM 이동으로 생성한 관측**이다. 두 카메라 계측계에서 실제 기준 좌표와의 LME를 검증한다.[^C9]

| Table 4 선택 조건 | LME 최대 mm | 평균 mm | 표준편차 mm |
|---|---:|---:|---:|
| CMM cube intrinsic + CMM extrinsic | 0.030 | 0.013 | 0.008 |
| Test-field intrinsic + photogrammetry extrinsic | 0.160 | 0.036 | 0.0244 |

출처: §3.3, Table 4. CMM 기준의 정밀 측정 조건에서는 우리보다 훨씬 작은 길이·좌표 측정 잔차가 가능함을 보여준다. **일반 물체의 신규 자세 인식 또는 로봇 포함 정확도 결과가 아니다.** 원문이 LME 평균·표준편차를 보고하므로 이를 곧바로 신규 물체 3D norm 평균으로 바꾸지 않는다. 정확도 SOTA보다 계측·불확실성 평가의 선행연구로 둔다.

## 5. 로봇을 포함한 캘리브레이션

### R1. Graph-based multi-camera hand–eye — Evangelista et al., ICRA 2023

Robot/world·target·camera 관계를 graph로 만들고 기하 및 pixel 제약을 최적화한다. 카메라 사이 cross-observation을 활용하며 PnP는 초기화에 사용한다. 우리 **순차 대 통합**, **pose 입력 대 원본 코너 최적화** 논리와 직접 연결된다.[^R1]

| 실물 reprojection, px | Camera 1 | Camera 2 | Camera 3 | 카메라 평균 |
|---|---:|---:|---:|---:|
| Park | 0.520446 | 0.346938 | 26.2883 | 9.0519 |
| Horaud | 0.432767 | 0.350170 | 25.0635 | 8.61548 |
| **Graph Multi** | **0.337803** | **0.284408** | **4.4232** | **1.6818** |

출처: Table I, PDF p.5. **실물 1.6818 px는 독립 mm 정확도가 아니고, per-camera 값의 평균**이다. 2023년 논문 내 실측 비교에서 우수하나 후속 2024 Allegro와 함께 평가해야 한다. [공개 코드](https://bitbucket.org/freelist/gm_handeye).

### R2. Multi-Camera Hand-Eye Calibration — Allegro, Terreran, Ghidoni, RA-L 2024

여러 카메라가 공유하는 board–end-effector 변환과 camera–camera 관계를 함께 최적화한다. Board 원본 corner 재투영과 cross-camera 재투영을 사용한다. **이 프로젝트의 외부 baseline 최우선 후보**다.[^R2]

| 데이터 / 센서 | GT translation mm | GT rotation deg | AX=ZB translation mm | AX=ZB rotation deg | 시간 s |
|---|---:|---:|---:|---:|---:|
| 실물 small / Kinect V2 | 22.01 | 0.09 | 6.54 | 0.55 | 16.79 |
| 실물 small / D455 | 13.21 | 0.07 | 9.21 | 0.66 | 20.11 |
| 실물 small / L515 | 13.68 | 0.02 | 24.95 | 1.32 | 42.21 |
| 실물 large / Kinect V2 | 51.18 | 0.30 | 12.17 | 0.65 | 77.99 |
| 실물 large / D455 | 45.18 | 0.16 | 35.98 | 2.12 | 35.43 |
| 실물 large / L515 | 19.34 | 0.09 | 12.45 | 0.85 | 17.33 |
| 산업 ABB | 미제공 | 미제공 | 6.31 | 0.98 | 3.12 |
| 산업 KUKA | 미제공 | 미제공 | 14.12 | 0.67 | 7.15 |

출처: 실물 Tables II–III. **실측 camera–robot 위치 오차는 13~51 mm**다. METRIC 실제 GT는 AprilTag+CAD+FK 기반이다. 논문 내 GT translation은 강하지만 모든 지표에서 1위는 아니다. 예: large Kinect rotation은 Evangelista 0.24°가 제안 0.30°보다 낮고, large L515 AX translation은 Evangelista 11.63 mm가 제안 12.45 mm보다 낮다. 따라서 **METRIC 실물 위치 오차에서 강한 baseline**이라는 표현이 정확하다. [공개 코드](https://github.com/davidea97/Multi-Camera-Hand-Eye-Calibration).

### R3. Multi-hand/eye graph — Zhou et al., 2023 공개 / TIE 2024

최소신장트리로 초기 calibration 순서를 정하고 closed-loop graph optimization으로 누적 오차를 줄인다. 실물은 UR5 2대+UR3 1대, 카메라 5대(손목 3+고정 2)로, mixed 구성 면에서 우리와 가깝다.[^R3]

| 실물 폐루프 | 제안 mm | Wu baseline mm |
|---|---:|---:|
| C–F–I–J–G–D | 0.6274 | **0.4213** |
| D–E–H–K–J–G | 0.6345 | **0.5812** |
| C–E–H–K–J–I–F | **0.6938** | 1.5923 |

출처: Table IV, PDF p.8. **실물은 참값을 측정하지 못해 폐루프 평가**를 사용했다. 일부 loop만 개선되므로 포괄적 SOTA로 표시하지 않는다.

### R4. Probabilistic AX=YB — Junhyoung Ha, T-RO 2023

측정 A와 B의 noise configuration, 개별 covariance, 상대 신뢰도를 명시해 MLE와 추정 uncertainty를 계산한다. ‘FK를 불확실하게 취급한다’는 아이디어의 직접 선행연구다. 저자 공개 MATLAB 코드에는 측정별 translation/rotation inverse covariance 입력과 pose uncertainty 계산이 존재한다.[^R4]

**확인 범위:** 출판사 초록·저자 코드에서 이론, 수치·하드웨어 검증의 존재와 covariance 출력은 확인했다. 실험 결과표 원문은 확보하지 못했으므로 **mm/deg 성능 수치는 미확인**이다. 다른 논문이 이 방법을 평가한 수치를 이 논문의 자체 결과로 바꾸어 쓰지 않았다. 정식 권호는 2023년, online publication은 2022-10-28이므로 경계 사례다. [공개 코드](https://github.com/hjhdog1/probabilisticAXYB).

### R5. Uncertainty-Aware Hand–Eye Calibration — Ulrich, Hillemann, T-RO 2024

Robot pose와 image noise를 확률적으로 모델링하는 GMF를 제시하고 robot pose correction 및 uncertainty estimation을 수행한다. **우리 A4의 target-pose soft factor와 적용 위치는 다르지만, FK 불확실성을 반영하는 최신 필수 baseline**이다.[^R5]

| UR3e 실물, Table II | RRMSE px | RAE **mm²** | 시간 s |
|---|---:|---:|---:|
| Daniilidis | 0.708 | 0.00462 | 0.2 |
| Tabb rp1 | 0.545 | 0.00212 | 0.3 |
| GMF, original robot poses로 평가 | 0.771 | 0.02103 | 3.4 |
| GMF, intrinsic 고정 | 0.196 | 0.00073 | 3.4 |
| **GMF** | **0.193** | **0.00073** | 3.4 |
| GMF self-calibration | 0.178 | N/A | 14.2 |

| Tabb 공개 데이터, Table VI | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
|---|---:|---:|---:|---:|
| GMF RRMSE px | 0.076 | 0.176 | 0.479 | 0.389 |
| GMF RAE mm² | 0.113 | 0.031 | 1.188 | 1.380 |

출처: Tables II/VI, PDF pp.11–12. RAE는 **평균 제곱 Euclidean 재구성 오차**다. 0.00073 mm²를 0.00073 mm 정확도로 읽으면 안 된다. 작은 pixel 잔차는 **보정된 robot pose를 사용한 결과**다. 실측 잔차·재구성에서 강하지만, 우리처럼 raw FK를 고정한 평가와 직접 동일하지 않다.

### R6. Kalib — Tang et al., IROS 2025

로봇 운동학으로 3D 위치를 알 수 있는 기준점 하나를 foundation tracker로 추적하고 PnP로 변환을 추정한다. CAD mesh·새 로봇별 재학습 없이 EiH와 eye-on-base를 모두 설명한다. **멀티카메라 joint solver는 아니므로 카메라별 적용이 필요**하다.[^R6]

| DROID 실영상 평가 | mean IoU | 조건 |
|---|---:|---|
| 기존 dataset calibration | 0.87 | 논문의 비교 기준 |
| EasyHeC | 0.77 | 동일 실영상 비교 |
| Kalib | 0.80 | 전체 60 sequence |
| Kalib, 실패성 sequence 제외 | 0.85 | 9/60 제외; 전체 결과와 분리 |

출처: §IV-C의 real-world validation. 약 2분/calibration, tracking 약 5 FPS(RTX3090). **실물 extrinsic의 mm/deg GT 수치는 확인되지 않았으며, IoU를 대신 변환해서 제시하지 않는다.** 기존 dataset calibration보다 IoU가 낮으므로 실측 정확도 SOTA로 표시하지 않는다. 손목 적용은 관측 가능한 기준점과 해당 기구학을 확보해야 한다. [코드](https://github.com/robotflow-initiative/Kalib).

### R7. EasyHeC — Chen et al., RA-L 2023

Robot segmentation과 differentiable rendering으로 pose를 보정하고, informative joint pose를 선택한다. 정확한 robot geometry와 visibility 조건이 필요하다. 실제 xArm에서 pointer로 ArUco board corner를 짚은 평균 거리 오차는 **약 0.4 cm=4 mm**다(§IV-C). Baxter 실영상에서는 2D/3D PCK를 평가하며 3 views PCK3D@2 cm는 **0.15**, @5 cm는 **0.80**다.[^R7]

‘실제 4 mm’는 camera extrinsic translation GT가 아니라 **로봇·검출·tool offset이 합쳐진 targeting 오차**다. EasyHeC++가 후속으로 존재하므로 현재 학습 기반 SOTA를 대표할 때는 두 논문을 구분한다. [코드·프로젝트](https://ootts.github.io/easyhec/).

### R8. EasyHeC++ — Hong, Zheng, Chen, IROS 2024

사전학습 영상 모델을 사용해 초기화와 segmentation을 개선하고 EiH로 확장한다. 로봇별 학습 부담은 줄지만 differentiable rendering용 robot geometry는 필요하다.[^R8]

| 설정 / 지표 | 제안 결과 | 비교 결과 |
|---|---|---|
| 실물 eye-to-hand targeting | 0.30 cm=**3.0 mm** | EasyHeC 4.0 mm; EasyHeC(SAM) 3.0 mm |
| 실물 EiH targeting, 10 trials | 0.31 cm=**3.1 mm** | Zhang 13.5 mm; Valassakis 43.0 mm |
| Baxter PCK3D, 3 views | @2 cm **0.30**, @5 cm **0.65** | EasyHeC @2 cm 0.15, @5 cm **0.80** |

출처: 실물 Tables VI–VIII. 실측 targeting에서 강한 결과지만 PCK threshold에 따라 승자가 달라진다. ‘Fully Automatic’ 제목에도 원문 §III-F는 초기 EiH calibration의 수동 pose 초기화를 명시한다. 우리 Zeus에 CAD가 없으면 Kalib보다 적용 비용이 높다. [프로젝트](https://ootts.github.io/easyhec_plus/).

### R9. Calib3R — Allegro et al., arXiv 2025 / v2 2026

Foundation-model pointmap, robot motion, camera-rig rigidity를 한 최적화에 넣어 hand–eye·scale·3D scene을 추정한다. **v2는 2026-08-08 원고**이며 v1과 제목·실험 범위가 달라 v2 표만 사용했다. 실험의 UR5는 손목에 D435 두 대를 단 구성으로, 우리 고정+손목 혼합 구성과 동일하지 않다.[^R9]

| Calib3R Table I | translation 원문 cm | 환산 mm | rotation 원문 rad | 환산 deg |
|---|---:|---:|---:|---:|
| Franka Pattern | 1.127 | 11.27 | 0.014 | 0.802 |
| Franka Object | 0.415 | 4.15 | 0.011 | 0.630 |
| UR Pattern | 1.512 | 15.12 | 0.014 | 0.802 |
| UR Object | 0.636 | 6.36 | 0.017 | 0.974 |
| GraspNet scene_0100 | 1.744 | 17.44 | 0.023 | 1.318 |

Franka Pattern scale error는 **0.11 cm=1.1 mm / 3.67%**, baseline JCR+VGGT는 **5.67%**다(Table II). 하지만 pattern-based Evangelista의 translation은 Franka Pattern **7.81 mm**, UR Pattern **6.91 mm**로 Calib3R보다 낮다. 따라서 **markerless joint 방식의 SOTA 후보**이지 모든 marker 방식보다 정확한 확정 SOTA는 아니다. 원고는 code를 acceptance 이후 공개한다고 명시한다. GT 구축·독립성 세부 설명은 이 평가 절에서 충분히 확인되지 않아 주의가 필요하다.

### R10. Certifiable generalized RWHEC — Wise et al., IJRR 2026

여러 센서·타깃·unknown scale을 최대우도 문제로 묶고 전역 최적성 certificate를 제공한다. 이는 **정의된 목적함수의 최적성**이며 실제 calibration 오차가 모든 방법보다 작다는 보장이 아니다.[^R10]

| 실물 Table 6 | 카메라 상대 translation cm | rotation deg | AprilTag20–OptiTrack translation cm | rotation deg |
|---|---:|---:|---:|---:|
| Wang2022 | 3.46 ± 1.69 | 1.38 ± 0.81 | 11.7 | 5.30 |
| LOM(Wang) | 3.34 ± 2.44 | **0.88 ± 0.49** | **11.4** | **4.50** |
| 제안 known scale | **3.20 ± 1.88** | 0.99 ± 0.52 | 11.5 | 4.73 |
| 제안 unknown scale | 2.87 ± 1.94 | 0.99 ± 0.52 | 7.94 | 4.72 |

출처: Table 6, §§9–10. 카메라 상대 GT는 Kalibr 기준, target–world는 OptiTrack 기준이다. 잘 초기화한 local solver와 정확도는 거의 같다. 실물 unknown-scale global solver는 최대 약 5분, local solver는 최대 1초다. 실측 모든 지표의 최고 방법은 아니며 전역 최적성 certificate를 정확도 SOTA와 구분한다. [공개 Julia 코드](https://github.com/utiasSTARS/certifiable-rwhe-calibration).

### R11. PlaneHEC — Wang et al., 2025 preprint

Depth point cloud의 벽·테이블 평면만으로 closed-form 초기화와 iterative hand–eye optimization을 수행한다. RealSense D435+Flexiv-Rizon4 실험이며, 논문의 multi-view는 여러 로봇 자세 관측을 포함하므로 다중 고정 카메라 network와 같지 않다.[^R11]

| 표본 수 | iterative e_t mm | e_r deg | e_xy mm | e_z mm |
|---|---:|---:|---:|---:|
| 5 | 26.22 | 0.48 | 8.56 | 24.78 |
| 10 | 7.54 | 0.27 | 2.67 | 7.05 |
| 20 | 4.67 | 0.17 | 1.63 | 4.38 |
| 30 | 3.35 | 0.13 | 1.20 | 3.13 |

출처: Tables I–II, §IV-B1. **300개 관측에서 subset을 50회 추출해 얻은 변환의 표준편차 기반 평가**다. 표 안에 mean error라는 표현도 섞여 있지만, 본문 절차상 독립 GT 절대오차로 읽어서는 안 된다. 0.28 s runtime은 데이터 수집을 제외한다. 기존 논문별 센서 정밀도가 다른 결과를 `e_t/max(E_c,E_R)`로 비교하므로 전체 accuracy SOTA 근거는 제한적이다.

### R12. UAL-HED — Chen et al., 2026 preprint

AX=YB의 Lie-algebra iterative solution에 source-data uncertainty 지표를 적용해 업데이트를 조정한다. 실물 장비는 ABB IRB 6700과 Leica AT960 laser tracker·T-Mac이며, RGB 카메라 실험은 아니다. Tracker의 6DoF 측정과 로봇 교시기/FK의 변환 차이를 평가한다. 원고 상태는 **IJRR under review**다.[^R12]

| 방법 | 높은 불확실성 공간 conversion error mm | 낮은 불확실성 공간 mm |
|---|---:|---:|
| L-HED | 0.346 | **0.193** |
| UAL-HED | **0.342** | 0.202 |
| Dual-Quaternion | 0.671 | 0.256 |
| Kronecker-Product | 1.170 | 0.557 |
| LMI | 1.137 | 0.843 |
| Point-Cloud Matching | 1.160 | 0.607 |
| SI-AH | 0.501 | 0.267 |

출처: Tables 8–9, §5.2. Calibration에 쓰지 않은 추가 99 data pairs로 6D conversion consistency를 검증한다. **변환 일관성 오차이지 독립 external TRE라고 확인된 수치는 아니다.** 높은 불확실성에서는 최소지만 낮은 불확실성에서는 L-HED보다 나쁘다. 우리 FK 처리의 보조 비교·추가 인용 후보이며 확정 SOTA는 아니다.

### R13. Multi-unit dual quaternion — Zhu, Shen, Ng, 2026

다중 카메라 RWHE를 부호 모호성을 다루는 subspace-constrained least-squares로 풀고 real-data correction을 제안한다. 대수적 잔차, camera별 RRMSE, runtime을 평가한다. 저자 주장도 ‘competitive’이며 모든 지표 우위를 주장하는 논문은 아니다.[^R13]

| Tabb 실물 데이터 | 제안 camera별 RRMSE px | 제안 시간 s | 비교: rp1_axis RRMSE px / 시간 s |
|---|---|---:|---|
| Dataset 6, 2 cameras | 18.0 / 17.7 | 0.27 | 0.683 / 0.695 / 1233.82 s |
| Dataset 7, 3 cameras | 2.48 / 2.54 / 1.67 | 0.30 | 1.06 / 1.09 / 0.767 / 3593.28 s |
| Dataset 8, 3 cameras | 2.72 / 2.87 / 1.38 | 0.31 | 1.34 / 1.52 / 0.744 / 3345.76 s |

출처: Tables 2–4, §4.2. 예를 들어 dataset 7의 Wang2022는 2.50/2.56/1.67 px, 0.04 s다. 제안은 비선형 반복군보다 매우 빠르지만 Wang보다 빠르지는 않고, pixel refinement보다 정확하지도 않다. **빠른 pose-level baseline**으로 가치가 있으며 pixel-level SOTA로 부르지 않는다. 극소 대수 잔차를 mm pose 정확도로 해석해서는 안 된다.

### R14. Enhanced robotic flexible 3D scanning — Zhou et al., Sensors 2025

KUKA 로봇과 LMI Gocator structured-light scanner의 hand–eye 및 robot kinematic parameters를 함께 보정한다. Sphere target의 기하 제약과 scanner 측정 오차를 모델링한다. 검증은 별도 standard scale의 두 구 중심 간 거리를 CMM 기준으로 측정하는 방식이다.[^R14]

| 실제 standard scale, 10개 위치 | 평균 거리 오차 mm | 최대 오차 mm |
|---|---:|---:|
| 보정 전 | 0.814 | 1.053 |
| 보정 후 | **0.373** | **0.421** |

출처: Table 4, §4.2. CMM 기준 구 간격은 299.546 mm다. 다른 실측 3개 위치에서도 제안 평균 0.385/0.322/0.307 mm가 Ren의 0.627/0.598/0.601 mm 및 Mu의 0.536/0.501/0.457 mm보다 낮다(Fig.8 본문). **실측 sub-mm 근거는 강하지만 정밀 scanner와 알려진 구를 사용한 거리 평가**이며 물체 원점의 base 좌표 오차와 다르다. 해당 비교에 한정한 S1이다.

### R15. GWM-View — Liu et al., RCIM 2023

3대 산업용 카메라와 ABB 6700을 사용해 자동차 부품의 임의 초기 자세를 추정하고 로봇 가공 위치를 정한다. Main/auxiliary camera 관계와 gradient-weighted suppression으로 시야 부족·가림·큰 관측각의 매칭 오차를 줄인다. Multi-view 좌표를 hand–eye 보정으로 robot base에 연결한다.[^R15]

| 실측 작업 | 논문의 보고 결과 | 확인 범위 |
|---|---|---|
| 임의 초기 자세의 복잡 부품 positioning | **평균 1 mm 미만** | 출판사 Highlights·abstract 및 공개 Experimental setup |

**새 물체 위치 추정 2~3 mm보다 낮은 작업 오차를 보고한 관련성 높은 후보**다. 다만 확인한 공개 범위에는 세부 trial 수, 독립 GT 구축, 물체별 정확한 평균 표가 없어 그 이상 정밀한 값이나 전체 SOTA 우위를 주장하지 않는다. 알려진 산업 부품의 임의 자세와 학습·모델에 없는 신규 물체 일반화도 구분한다. 원문 상세 실험표 확보 우선순위가 높다.

## 6. 데이터셋과 GT

### D1. METRIC — Allegro et al., Information 2023

카메라 네트워크와 robot-world hand–eye를 함께 평가하는 공개 데이터셋이다. 여기서는 실제 2개 작업셀과 Kinect V2·D455·L515 네트워크만 다룬다. 지표는 camera-to-camera와 camera-to-robot의 translation mm·rotation deg다.[^D1]

| 실물 RWHE / 평균 translation mm | Small Kinect | Small D455 | Small L515 | Large Kinect | Large D455 | Large L515 |
|---|---:|---:|---:|---:|---:|---:|
| Evangelista | 42.79 ± 16.70 | 45.14 ± 16.06 | 26.20 ± 15.50 | 77.26 ± 8.27 | 136.39 ± 57.89 | 60.59 ± 8.40 |
| Tabb | 51.57 ± 17.48 | 63.98 ± 17.53 | 34.59 ± 15.25 | 105.01 ± 9.11 | 153.2 ± 37.05 | 75.33 ± 9.01 |
| Li | 66.67 ± 17.07 | 51.03 ± 9.83 | 23.70 ± 4.65 | 129.39 ± 25.77 | 수렴 실패 | 26.39 ± 6.56 |
| Shah | 27.22 ± 6.12 | 23.90 ± 6.99 | 18.51 ± 4.22 | 54.92 ± 9.13 | 수렴 실패 | 26.15 ± 5.30 |

| 실물 RWHE / 평균 rotation deg | Small Kinect | Small D455 | Small L515 | Large Kinect | Large D455 | Large L515 |
|---|---:|---:|---:|---:|---:|---:|
| Evangelista | 0.57 ± 0.14 | 0.43 ± 0.23 | 0.39 ± 0.10 | 0.77 ± 0.28 | 1.33 ± 0.64 | 0.43 ± 0.06 |
| Tabb | 0.73 ± 0.14 | 1.36 ± 0.24 | 0.94 ± 0.16 | 0.75 ± 0.12 | 1.74 ± 0.24 | 0.77 ± 0.07 |
| Li | 0.73 ± 0.16 | 0.72 ± 0.43 | 0.31 ± 0.07 | 0.72 ± 0.28 | 수렴 실패 | 0.30 ± 0.07 |
| Shah | 0.75 ± 0.06 | 0.54 ± 0.23 | 0.34 ± 0.10 | 0.73 ± 0.19 | 수렴 실패 | 0.31 ± 0.08 |

출처: Tables 9–10, §5. 표 caption에는 camera network라고 적혀 있으나 §5 본문 및 Appendix A9/A10은 camera–robot RWHE 평가임을 명시한다. ±는 카메라 간 분산이며 반복 calibration의 신뢰구간이 아니다. R2 후속 논문의 재평가 결과와 섞지 않는다.

실제 GT는 **15 cm AprilTag + 3D-printed mount의 CAD + robot FK + AprilTag pose 추정**을 조합한다(§3.2). 따라서 외부 laser tracker나 mocap이 직접 측정한 카메라 pose와 동일하지 않다. R2의 실물 GT 오차 해석에도 이 제한이 이어진다. 공개 benchmark 재현에는 최우선이지만, 우리 논문의 독립 6DoF GT 실험을 대체하지는 못한다.

## 7. 지금 무엇을 SOTA라고 표시할 수 있나

| 세부 문제 | 표시할 방법 | 정확한 판단 |
|---|---|---|
| 마커 기반 fixed multi-camera + robot | **Allegro 2024** | METRIC 내 translation에서 강한 비교 근거; 우리 주 baseline. 모든 rotation·runtime 1위는 아님 |
| Robot uncertainty를 모델링한 hand–eye | **Ulrich–Hillemann 2024** | 실물 재투영·재구성에서 강한 근거. robot pose correction 포함; 독립 extrinsic GT 우위로 확대 불가 |
| Markerless reconstruction + multi-camera hand–eye joint | **Calib3R v2** | 최신 SOTA 후보; preprint·코드 공개 예정·구성 차이 명시 |
| 다중 센서·타깃·unknown-scale | **Wise et al. 2026: baseline** | 실물 translation에서 부분 우위; rotation 등에서 열세. certificate는 이론 기여로 별도 표시 |
| Camera-only, stand-alone calibration의 독립 3D 검증 | **FusedBA 2024** | 자체 MCalib 조건에서 강함; mocap-assisted보다 우수하지 않음 |
| Camera-only, 별도 wand의 거리 측정 | **AIRLS 2024** | 4카메라·두 종류 검증 wand의 Table 3에서 비교군보다 낮은 평균 오차; 절대 point 위치 SOTA와 구분 |
| Robot+scanner의 구 간격 측정 | **Zhou et al. 2025** | CMM 기준 실측 및 논문 내 Ren·Mu 비교에서 강함; 센서와 평가량이 다름 |
| Multi-camera+robot의 실제 산업 부품 위치 | **GWM-View 2023: 우선 확인 후보** | 평균 1 mm 미만 보고; 상세 결과표·GT 조건 미확인으로 SOTA 확정 보류 |
| Camera-only, 인체 기반 보정 | **Lee et al. 2025** | 동기화 실영상의 pose·대응에서 강함; 인위적 비동기 평가는 제외; mm 순위와 분리 |
| Robot segmentation/rendering 기반 calibration | **EasyHeC++ 2024** | 실물 targeting에서 강함. PCK threshold에 따라 승자는 다름 |
| 우리 mixed fixed+wrist + board+cube + FK target handling 전체 | **확정된 단일 SOTA 없음** | 조사한 논문과 우리 구현을 동일 세팅·독립 GT에서 직접 비교해야 판정 가능 |

위 판단은 R2/R5/R9/R10/C1/C3/R8의 결과표를 종합한 분석이다. 최신이라는 이유만으로 SOTA로 승격하지 않았으며, 본 조사가 모든 관련 논문을 망라했다는 의미도 아니다.

## 8. 우리 결과와 논문을 연결하는 평가 지표

| 지표 | 정의 / 단위 | 검증 대상 | 대표 문헌 | 해석상 주의 |
|---|---|---|---|---|
| Extrinsic translation error | norm(t_est−t_GT), mm | 특정 camera transform의 위치 정확도 | R2, R9 | transform 방향·원점·GT 측정법을 통일 |
| Geodesic rotation error | acos((tr(R_GTᵀR_est)−1)/2), deg | orientation 정확도 | R2 | rad와 deg 구분; matrix Frobenius error와 구별 |
| Pixel reprojection RMSE | sqrt(mean(norm(u_pred−u_obs)²)), px | 이미지 적합도 | C4, R1, R5 | mean Euclidean·per-axis RMS·per-camera 평균과 다름 |
| Heldout reprojection | 학습에 안 쓴 관측에 frozen calibration 적용 | 관측 일반화 | 우리 main pipeline | test target pose를 test pixel로 재적합하면 별도 조건 |
| Camera-common consistency | 각 카메라 target pose 추정 간 거리·회전 | 상대 일관성 | 우리 Zeus, R3의 관련 loop 지표 | 공통 systematic bias는 검출 못함 |
| AX=YB / AX=ZB closure | 운동학 변환 양변의 translation·rotation 차이 | 모델·운동학 일관성 | R2, R12 | robot/image 측정이 공통으로 틀려도 작아질 수 있음 |
| Independent target TRE | robot frame의 예측 target point와 독립 GT point 거리 | 실제 작업공간 위치 정확도 | 우리 주 평가 제안 | camera-origin translation error와 동일하지 않음 |
| Reconstruction error | 복원한 3D 점·거리와 기준의 차이, mm | 3D geometry | C1, C2, C6 | pointwise/거리/정합 후 평가를 구분 |
| RAE squared | 평균 squared 3D 거리, mm² | 타깃 재구성 | R5 | mm로 쓰려면 제곱근이며 원문 지표와 구분 표시 |
| Repeatability / dispersion | 반복 추정 변환의 표준편차 | 정밀도·안정성 | R11 | 절대 정확도 증거 아님 |
| IoU / PCK | mask overlap / threshold 이내 point 비율 | 투영·keypoint·응용 품질 | R6–R8 | GT extrinsic mm 오차로 대체 불가 |
| Runtime / data efficiency | 실행시간, 입력 수, 성공률 | 실용성 | R2, R6, R9, R10, R13 | CPU/GPU·검출·수집 포함 여부와 실패 제외 조건 명시 |

### 8.1 저장소의 기존 calibration 결과와 신규 물체 결과

| 내부 실험 | 방법 | heldout cube px | camera-common mm / deg | 외부 검증 상태 |
|---|---|---:|---|---|
| Session04 canonical | A1 | 4.1402 | 8.8294 / 1.0503 | External cube GT pending |
| Session04 canonical | A2 | 3.5958 | 7.2868 / 1.0280 | pending |
| Session04 canonical | A4 | 3.5805 | 7.3077 / 1.0151 | pending; FK covariance 측정 대기 |
| Session04 canonical | A5 | 3.2274 | 6.6745 / 0.8862 | pending; GT 평가 전 방법 고정 필요 |
| Zeus 2026-09-09 | 통합_no-fk_px | 2.9557 | 4.2250 / 1.0534 | z 1.24 mm / rz 0.69°, n=3 |
| Zeus 2026-09-09 | 통합_raw-fk_px | 2.2087 | 4.6416 / 1.1567 | z 0.94 mm / rz 0.65°, n=3 |
| Zeus 2026-09-09 | 독립_no-fk_px | 3.8053 | 3.9182 / 1.1376 | z 1.65 mm / rz 0.63°, n=3 |

신규 물체의 FK 기준 2~3 mm는 본 비교에서 유효한 위치 GT 오차로 채택한다. 위 표는 저장소의 별도 calibration·큐브 평가 기록으로, 신규 물체 결과와 자동으로 동일시하지 않는다.

출처: [Session04 Table 1](../CP_result/session04/late_table1/TABLE1_RESULTS.md), [Zeus 현재 결과](../zeus_gello_calibration/ABLATION_RESULTS.md). 두 평가의 split·GT 정의가 다르므로 세션끼리나 논문 수치와 직접 순위를 매기지 않는다. Zeus 결과는 일부 축과 세 trial에 한정되고, FK·비전으로 산출한 grasp offset 및 수동 정렬 영향을 받으므로 **full 6DoF independent TRE 달성으로 해석하지 않는다.**

### 8.2 우선 실행할 baseline

| 우선순위 | 비교군 | 필요한 입력·변경 | 답할 질문 |
|---|---|---|---|
| **1** | R2 Allegro 2024 | 고정 카메라의 board-on-gripper 영상+FK; 손목 구성은 별도 지원 확인 | 최근 멀티카메라 joint hand–eye보다 이득이 있는가 |
| **2** | R1 Evangelista 2023 | 같은 corner·intrinsic·FK, 포맷 adapter | 기존 pixel graph에 비해 mixed target 연결이 유효한가 |
| **3** | R5 Ulrich–Hillemann, R4 Ha | covariance 모델·robot pose 처리 차이 기록; Ha는 PnP pose covariance 입력 | FK-aware 처리의 이득이 기존 uncertainty 방법 이상인가 |
| **4** | C1 평가 프로토콜 + C8 AIRLS + camera-only BA baseline | 고정 카메라 relative transform만 평가, 별도 blind target 사용 | 로봇 앵커의 영향과 순수 camera-network 품질을 분리할 수 있는가 |
| **5** | R6 Kalib 또는 R8 EasyHeC++ | 연속 video+FK+reference point; EasyHeC++는 robot geometry | 마커 사용에 따른 정확도·준비시간 trade-off |
| **6** | R9 Calib3R / R10 Wise / R13 Zhu | 코드 가용성·rig 구성·pose 입력을 맞춤 | markerless joint, global solver, 빠른 algebraic solver와의 위치 |

신규 물체의 위치 정확도를 주요 기여로 삼는다면 **R15 GWM-View의 실험표 확보와 C8의 별도 검증 물체 평가를 추가 우선순위**로 둔다. 이 우선순위는 프로젝트 구조와 재현 가능성에 대한 분석 제안이다. **모든 외부 방법을 지금 실행했다는 의미는 아니다.** 아래 결과는 문헌 보고값이며 이 프로젝트에서 재현 실행한 외부 baseline 결과가 아니다.

### 8.3 논문 주장과 필요한 추가 검증

통합 pixel 최적화, multi-camera graph, uncertainty-aware hand–eye, markerless calibration은 각각 이미 선행연구가 있다. 따라서 ‘통합 최적화를 최초 제안’, ‘FK를 soft하게 믿는 최초 방법’, ‘다면 타깃 자체가 최초’처럼 넓게 주장하기 어렵다.[^R1][^R2][^R4][^R5][^C5][^OLD]

우리 기여는 **고정·손목 혼합 카메라에서 파지/배치 가능한 다면 큐브와 보드를 공유 latent target pose로 연결하고, target FK 처리 전략을 같은 관측·같은 목적함수·같은 독립 평가로 비교하는 것**으로 좁히는 편이 현재 증거에 맞다. Cube geometry를 고정한 실험과 형상 오차에 민감한 실험, board-only와 cube-only, 독립과 순차와 통합을 구분하면 각 효과를 설명할 수 있다.

최종 비교는 같은 설치 세션에서 mean/median target TRE, rotation, P95, 실패율·카메라 coverage, runtime·촬영량을 기록하는 것이 적합하다. 한 카메라 원점의 t error와 작업영역 target TRE를 함께 기록하면 lever-arm에 따른 회전오차 영향을 드러낼 수 있다. 외부 GT를 본 뒤 parameter를 바꾸지 말고, train/heldout split과 타깃 pose 추정 규칙을 모든 방법에 동일하게 적용해야 한다.

## 9. 추가 후보·연도 제외

| 항목 | 처리 |
|---|---|
| A method for calibrating multi-camera systems based on sparse reconstruction of a 3D object, Measurement 240, 2025, 115561 | 3D 타깃·graph 기반으로 관련성 높음. 출판사 초록까지 확인했으나 상세 정량 결과는 미확인; 핵심 25편의 수치 비교에는 미포함. [출판사](https://www.sciencedirect.com/science/article/pii/S0263224124014465) |
| CALICO, 2024 개정 | 최초 공개는 **2019**, 2024는 v3. 신규 2023+ 논문으로 세지 않음. Pattern rig·non-overlap 특성은 큐브 관련 선행연구로 중요. [버전 이력](https://arxiv.org/abs/1903.06811) |
| MC-Calib | 저자 저장소 citation의 논문은 **2022**. 최신 코드 업데이트를 새 논문 연도로 취급하지 않음. [저자 코드](https://github.com/rameau-fr/MC-Calib) |
| Tsai 1989, Daniilidis 1999, Shah 2013, Tabb 2017 | 연도 범위 밖이므로 독립 상세 항목 제외. 2023+ 논문의 비교 baseline 또는 우리 재현 실험 기준선으로는 유지 |
| Simultaneous Robot-World and Hand-Eye Calibration, arXiv:2311.11818 | arXiv 업로드 연도만 보고 신작으로 포함하지 않음. 원출판연도 검증 없이 2023 논문으로 세지 않음 |

## 10. 참고문헌·원문 위치

아래 링크는 논문 원문·출판사·저자 공개본이다. 수치에는 해당 Table/section을 함께 기록했다. `미보고`는 확인한 논문에 해당 결과가 없음을, `미확인`은 접근 또는 원문 확인 범위가 부족함을 뜻한다.

[^C1]: Prayook Jatesiktat, Guan Ming Lim, Wei Tech Ang. “Multi-Camera Calibration Using Far-Range Dual-LED Wand and Near-Range Chessboard Fused in Bundle Adjustment.” Sensors 24(23), 7416, 2024-11-21. [논문](https://www.mdpi.com/1424-8220/24/23/7416), [공개 PDF](https://mdpi-res.com/d_attachment/sensors/sensors-24-07416/article_deploy/sensors-24-07416.pdf). Table 1, §4.6; DOI 10.3390/s24237416.
[^C2]: Sujie Zhang, Qiang Fu. “Wand-Based Calibration of Unsynchronized Multiple Cameras for 3D Localization.” Sensors 24(1), 284, 2024-01-03. [논문](https://www.mdpi.com/1424-8220/24/1/284). Tables 4–6; DOI 10.3390/s24010284.
[^C3]: Sang-Eun Lee, Ko Nishino, Shohei Nobuhara. “Spatiotemporal Multi-Camera Calibration using Freely Moving People.” IEEE Robotics and Automation Letters, 2025. [원문 v3](https://arxiv.org/html/2502.12546v3), [서지](https://arxiv.org/abs/2502.12546). Table I, §§V-A/B.
[^C4]: Liliya A. Demidova, Vladimir E. Zhuravlev. “Incremental Multi-Camera Extrinsic Calibration Method Based on PnP Integrating Weighted AprilTag Detections and Multi-View Triangulation.” Algorithms 19(5), 371, 2026-05-08. [논문](https://www.mdpi.com/1999-4893/19/5/371), [PDF](https://mdpi-res.com/d_attachment/algorithms/algorithms-19-00371/article_deploy/algorithms-19-00371.pdf). Table 2; DOI 10.3390/a19050371.
[^C5]: Robin Laven, Thiranja Prasad Babarenda Gamage, Alexander Dixon, Gonzalo D. Maso Talou, Merryn H. Tawhai, Andrew J. Taberner, Martyn P. Nash, Poul M. F. Nielsen. “Multi-Camera Calibration With Imperfect 3-D Targets for Imaging Biological Systems.” IEEE Transactions on Instrumentation and Measurement 75, 5005208, 2026-02-23. [출판사 초록](https://ieeexplore.ieee.org/document/11408373/). DOI 10.1109/TIM.2026.3667321. 초록 확인; 실측 조건별 수치는 미확인으로 정량 비교 제외.
[^C6]: Yuji Ohshima. “Wand-Based Calibration Accuracy for Unsynchronized Multicamera Systems Without Timestamps.” Sensors 26(3), 777, 2026-01-23. [논문](https://www.mdpi.com/1424-8220/26/3/777), [PDF](https://mdpi-res.com/d_attachment/sensors/sensors-26-00777/article_deploy/sensors-26-00777.pdf). Tables 1–2; DOI 10.3390/s26030777.
[^C7]: Donald Prible. “Caliscope: GUI Based Multicamera Calibration and Motion Tracking.” JOSS 9(102), 7155, 2024-10-15. [논문](https://joss.theoj.org/papers/10.21105/joss.07155), [PDF](https://www.theoj.org/joss-papers/joss.07155/10.21105.joss.07155.pdf). pp.1–3; DOI 10.21105/joss.07155.
[^R1]: Daniele Evangelista, Emilio Olivastri, Davide Allegro, Emanuele Menegatti, Alberto Pretto. “A Graph-Based Optimization Framework for Hand-Eye Calibration for Multi-Camera Setups.” ICRA 2023, pp.11474–11480. [PDF](https://arxiv.org/pdf/2303.04747), [코드](https://bitbucket.org/freelist/gm_handeye). Table I; DOI 10.1109/ICRA48891.2023.10160758.
[^R2]: Davide Allegro, Matteo Terreran, Stefano Ghidoni. “Multi-Camera Hand-Eye Calibration for Human-Robot Collaboration in Industrial Robotic Workcells.” IEEE Robotics and Automation Letters, 2024. [원문](https://arxiv.org/html/2406.11392), [카메라 레디](https://www.research.unipd.it/retrieve/059b99bf-2f5a-4deb-b5f3-4cbc80ca1e99/RAL_MultiCameraHandEye_CameraReady.pdf), [코드](https://github.com/davidea97/Multi-Camera-Hand-Eye-Calibration). Tables II–III.
[^R3]: Zishun Zhou, Liping Ma, Xilong Liu, Zhiqiang Cao, Junzhi Yu. “Simultaneously Calibration of Multi Hand-Eye Robot System Based on Graph.” arXiv 2023; IEEE Transactions on Industrial Electronics 71(5), 5010–5020, 2024. [PDF](https://arxiv.org/pdf/2305.02518). Table IV; DOI 10.1109/TIE.2023.3283693.
[^R4]: Junhyoung Ha. “Probabilistic Framework for Hand–Eye and Robot–World Calibration AX=YB.” IEEE Transactions on Robotics 39(2), 1196–1211, 2023; online 2022-10-28. [출판사](https://doi.org/10.1109/TRO.2022.3214350), [저자 코드](https://github.com/hjhdog1/probabilisticAXYB), [저자 서지](https://intro.unist.ac.kr/publications/). 실험표 미확인.
[^R5]: Markus Ulrich, Markus Hillemann. “Uncertainty-Aware Hand–Eye Calibration.” IEEE Transactions on Robotics 40, 573–591, 2024; online 2023-11-06. [저자기관 서지](https://publikationen.bibliothek.kit.edu/1000167305), [공개 PDF](https://publikationen.bibliothek.kit.edu/1000167305/152029292). Tables II/VI, Fig.9; DOI 10.1109/TRO.2023.3330609.
[^R6]: Tutian Tang, Minghao Liu, Wenqiang Xu, Cewu Lu. “Kalib: Easy Hand-Eye Calibration with Reference Point Tracking.” IROS 2025; arXiv first 2024, v2 2025-03-24. [원문](https://arxiv.org/html/2408.10562), [서지·DOI](https://arxiv.org/abs/2408.10562), [코드](https://github.com/robotflow-initiative/Kalib). §IV-C; DOI 10.1109/IROS60139.2025.11247188.
[^R7]: Linghao Chen, Yuzhe Qin, Xiaowei Zhou, Hao Su. “EasyHeC: Accurate and Automatic Hand-eye Calibration via Differentiable Rendering and Space Exploration.” IEEE Robotics and Automation Letters, 2023. [저자 PDF](https://ootts.github.io/easyhec/files/EasyHeC.pdf), [프로젝트](https://ootts.github.io/easyhec/). 실물 PCK 표 및 §IV-C.
[^R8]: Zhengdong Hong, Kangfu Zheng, Linghao Chen. “EasyHeC++: Fully Automatic Hand-Eye Calibration with Pretrained Image Models.” IROS 2024. [원문](https://arxiv.org/html/2410.09293), [서지](https://arxiv.org/abs/2410.09293). Tables VI–VIII, §III-F.
[^R9]: Davide Allegro, Matteo Terreran, Stefano Ghidoni. “Calib3R: Hand-Eye Calibration and 3D Metric-Scaled Scene Reconstruction with 3D Foundation Models.” arXiv:2509.08813 **v2**, 2026-08-08; first 2025-09-10. [사용 원문 v2](https://arxiv.org/html/2509.08813v2). Tables I–II. v1 제목은 “Calib3R: A 3D Foundation Model for Multi-Camera to Robot Calibration and 3D Metric-Scaled Scene Reconstruction”.
[^R10]: Emmett Wise, Pushyami Kaveti, Qilong Cheng, Wenhao Wang, Hanumant Singh, Jonathan Kelly, David M. Rosen, Matthew Giamou. “A Certifiably Correct Algorithm for Generalized Robot-World and Hand-Eye Calibration.” The International Journal of Robotics Research, online 2026-03-09; preprint 2025. [출판사](https://journals.sagepub.com/doi/10.1177/02783649261420308), [공개 원문](https://arxiv.org/html/2507.23045), [코드](https://github.com/utiasSTARS/certifiable-rwhe-calibration). Table 6, §§9–10. 출판본 저자 표기를 사용; preprint metadata에는 Chen 표기도 있음.
[^R11]: Ye Wang, Haodong Jing, Yang Liao, Yongqiang Ma, Nanning Zheng. “PlaneHEC: Efficient Hand-Eye Calibration for Multi-view Robotic Arm via Any Point Cloud Plane Detection.” arXiv:2507.19851v1, 2025-07-26. [원문](https://arxiv.org/html/2507.19851), [PDF](https://arxiv.org/pdf/2507.19851). Tables I–II, §IV-B1.
[^R12]: Yanjia Chen, Xiangfei Li, Huan Zhao, Yiyuan Hong, Guanxiao Xia, Jiexin Zhang, Han Ding. “Optimal Uncertainty-Aware Calibration for the AX=YB Problem.” arXiv:2605.04809v1, 2026-05-06, under review in IJRR. [원문](https://arxiv.org/html/2605.04809), [서지·상태](https://arxiv.org/abs/2605.04809). Tables 8–9.
[^R13]: Hong Zhu, Yuqing Shen, Michael K. Ng. “Multi-Camera Robot-World Hand-Eye Calibration by Solving Multi-Unit Dual Quaternion Equations.” CSIAM Transactions on Applied Mathematics 7(6), 1047–1079, 2026; online 2026-04-14. [출판사 원문](https://www.global-sci.com/csiam-am/article/view/24054), [저자기관 서지](https://scholars.hkbu.edu.hk/en/publications/multi-camera-robot-world-hand-eye-calibration-by-solving-multi-un/). Tables 2–4; DOI 10.4208/csiam-am.SO-2025-0052.
[^D1]: Davide Allegro, Matteo Terreran, Stefano Ghidoni. “METRIC—Multi-Eye to Robot Indoor Calibration Dataset.” Information 14(6), 314, 2023. [논문](https://www.mdpi.com/2078-2489/14/6/314), [PDF](https://mdpi-res.com/d_attachment/information/information-14-00314/article_deploy/information-14-00314-v2.pdf). §3.2, Tables 9–10, Appendix A9/A10; DOI 10.3390/info14060314.
[^OLD]: Amy Tabb, Khalil M. Ahmad Yousef. “Solving the Robot-World Hand-Eye(s) Calibration Problem with Iterative Methods.” Machine Vision and Applications, 2017. [저자 공개본](https://arxiv.org/abs/1907.12425). Amy Tabb et al., “Multi-camera calibration with pattern rigs, including for non-overlapping cameras: CALICO,” first 2019, revision 2024. [버전 이력](https://arxiv.org/abs/1903.06811). 연도 제외 및 multi-camera 선행연구 확인용.

[^C8]: Oleksandr Yuhai, Yubin Cho, Ahnryul Choi, Joung Hwan Mun. “Enhanced Three-Axis Frame and Wand-Based Multi-Camera Calibration Method Using Adaptive Iteratively Reweighted Least Squares and Comprehensive Error Integration.” Photonics 11(9), 867, 2024. [논문](https://www.mdpi.com/2304-6732/11/9/867), [PDF](https://mdpi-res.com/d_attachment/photonics/photonics-11-00867/article_deploy/photonics-11-00867.pdf). Table 3, §2.4; DOI 10.3390/photonics11090867.
[^C9]: Ibai Leizea, Imanol Herrera, Pablo Puerto. “Calibration Procedure of a Multi-Camera System: Process Uncertainty Budget.” Sensors 23(2), 589, 2023-01-04. [논문](https://www.mdpi.com/1424-8220/23/2/589), [공개 원문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9864742/), [PDF](https://mdpi-res.com/d_attachment/sensors/sensors-23-00589/article_deploy/sensors-23-00589.pdf). §3.3, Table 4; DOI 10.3390/s23020589.
[^R14]: Zhilong Zhou, Jinyong Shangguan, Xuemei Sun, Yunlong Liu, Xu Zhang, Dengbo Zhang, Haoran Liu. “Enhanced Calibration Method for Robotic Flexible 3D Scanning System.” Sensors 25(15), 4661, 2025-07-27. [논문](https://www.mdpi.com/1424-8220/25/15/4661), [PDF](https://mdpi-res.com/d_attachment/sensors/sensors-25-04661/article_deploy/sensors-25-04661.pdf). Table 4, §4.2; DOI 10.3390/s25154661.
[^R15]: Hongdi Liu, Jiahao Fu, Minqi He, Lin Hua, Dahu Zhu. “GWM-view: Gradient-weighted multi-view calibration method for machining robot positioning.” Robotics and Computer-Integrated Manufacturing 83, 102560, 2023. [출판사 Highlights·abstract](https://www.sciencedirect.com/science/article/abs/pii/S0736584523000364), [공개 실험 절](https://www.sciencedirect.com/science/article/pii/S0736584523000364). DOI 10.1016/j.rcim.2023.102560. 실제 부품 평균 위치 오차 1 mm 미만은 Highlights에 명시; 상세 결과표·GT 절차 미확인.
