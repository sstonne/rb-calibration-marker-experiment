# Review A — 기여 입증에 필요한 비교와 실험

> 역할: IEEE T-RO / RA-L 급 **리뷰어 A**.
> 작성일: 2026-09-16. 대상: `/home/sstone/rb-calibration-marker-experiment` 현재 상태.
> 목적: "이 논문이 기여를 입증하려면 무엇이 더 필요한가"를 뽑는다. 장점 서술은 하지 않는다.
> 저장소에서 확인하지 못한 값은 전부 `미확인`으로 적었다. 추정으로 채우지 않았다.

## 0. 용어 (처음 나올 때 한 줄 정의)

| 용어 | 뜻 |
|---|---|
| **EIH (eye-in-hand, 손목 카메라)** | 로봇 팔 끝에 붙어 같이 움직이는 카메라. 여기서는 D435 1대 |
| **EoB (eye-on-base, 고정 카메라)** | 작업공간에 고정된 카메라. 여기서는 D415 3대 |
| **Hand–Eye 변환 $T^G_C$** | 그리퍼 좌표와 손목 카메라 좌표 사이의 6자유도 변환. EIH 보정의 미지수 |
| **게이지(gauge)** | 최적화에서 "전체를 통째로 옮겨도 잔차가 안 변하는" 자유도. 이걸 고정해 주는 관측이 없으면 절대 좌표가 정해지지 않는다 |
| **관측가능성(observability)** | 그 미지수가 데이터로 실제로 구속되는가. 구속이 약하면 값은 나와도 의미가 없다 |
| **여기(excitation)** | 미지수를 구속하려고 로봇 자세를 얼마나 다양하게 움직였는가. 특히 회전축의 다양성 |
| **순환(circularity)** | A를 구하는 데 B를 쓰고, B를 구하는 데 A를 쓴 구조. "안다"고 말할 수 없는 상태 |

---

## ① 기여 주장 판정

### 1.1 한 줄 판정

> **주장 4번("마커 절대 위치를 알고 EIH를 푼다")은 현재 서술 그대로는 성립하지 않는다.
> 부분적으로는 실재하는 정보 이득이지만, 저장소의 현재 문서·수식·데이터로는 그 이득이
> (a) 순환이 아님을 보이지 못했고, (b) 크기를 잰 적이 없으며, (c) 분리해서 잰 지표가 아예 없다.**

주장을 **버리라는 것이 아니다.** 지금 형태(“절대 위치를 안다”)로는 리뷰에서 깨진다. 살아남는 형태로 바꾸고, 그 형태를 증명하는 실험을 붙여야 한다.

### 1.2 주장 1~5에 대한 개별 판정

| # | 저자 주장 | 판정 | 근거 |
|---|---|---|---|
| 1 | 큐브를 여러 카메라가 동시 관측 | **기여 아님** | 셋업 서술이다. R1·R2·R3·R15·D1·P3이 전부 동시 관측한다 ([SENSOR_TOPOLOGY_MATRIX §2.1](SENSOR_TOPOLOGY_MATRIX.md)) |
| 2 | 큐브 위치를 로봇 FK로도 안다 | **기여 아님** | 저자 본인이 3번에서 인정했다. 보드를 EE에 붙이는 R2(Allegro)·D1(METRIC)·P3(Miseikis)가 같은 성질을 가진다 |
| 3 | 고정카메라만 보면 특별하지 않다 (자기 반론) | **옳다. 유지하라** | 이 문장은 논문에 그대로 써야 한다. 리뷰어가 먼저 쓰기 전에 저자가 쓰는 편이 낫다 |
| 4 | **EIH를 "마커 절대 위치를 알고" 푸는 것이 차별점** | **조건부 성립. 현재 서술은 부정확** | §1.3 참조 |
| 5 | 덤으로 멀티카메라도 한 번에 | **기여 아님 (편의성)** | R2·R1이 이미 한다 |

### 1.3 주장 4를 냉정하게 뜯는다

**(a) "안다"는 표현은 틀렸다 — 큐브 절대 위치는 추정량이지 관측량이 아니다.**

[README §2](../README.md)의 변수 표를 보면 $T^B_{\mathrm{cube}}(s)$는 A2·A4에서 **자유변수**이고, 고정카메라 외부파라미터 $T^B_{C_i}$도 **자유변수**다. 같은 최적화 안에서 동시에 추정되는 두 양을 두고 한쪽을 "안다"고 부를 수 없다. 정확한 서술은 이것이다.

> 큐브의 base-frame pose는 **여러 카메라가 공유하는 잠재변수(shared latent variable)**이고,
> 고정 카메라 3대가 그 잠재변수를 **과결정(over-determine)**하므로 사후 분산이 줄어든다.

이건 "안다"가 아니라 **"덜 틀린다"**다. 논문 문장은 반드시 이렇게 바뀌어야 한다. "we know the absolute marker pose"라고 쓰면 리뷰어 세 명이 전부 같은 곳을 찌른다.

**(b) 그런데 순환은 실제로 존재한다 — 그리고 그 위치가 저장소에 명시돼 있지 않다.**

게이지 분석을 직접 해 보면 이렇다. 고정카메라 잔차는
$$\hat{\mathbf u} = \pi\!\left(K_i, D_i,\ (T^B_{C_i})^{-1} T^B_O \mathbf X_n\right)$$
이다 ([README §6.1, §6.3](../README.md)). 여기서 임의의 강체변환 $G$를 **모든** $T^B_{C_i}$와 **모든** $T^B_O$에 동시에 곱하면 잔차는 정확히 불변이다. 즉 **고정카메라 관측만으로는 robot base 좌표계가 전혀 정해지지 않는다.** 게이지를 고정하는 건 EIH 잔차뿐이다.
$$\hat{\mathbf u} = \pi\!\left(K, D,\ (T^B_G(e)\,T^G_C)^{-1} T^B_O \mathbf X_n\right)$$
$T^B_G(e)$가 event마다 다르므로 $G$가 흡수되지 않는다.

**결론: 현재 legacy Session04 구성에서는 "고정 카메라가 큐브의 절대 위치를 알려준다"가 문자 그대로 순환이다.** 고정 카메라가 base 좌표계와 연결되는 유일한 경로가 $T^G_C$ 자신이기 때문이다. $T^G_C$로 앵커된 좌표계에서 잰 큐브 위치로 다시 $T^G_C$를 개선한다 — 이 구조를 그대로 두면 reject 사유가 된다.

**(c) 순환을 깨는 경로는 저장소에 이미 설계돼 있다. 그런데 논문 서술이 그걸 안 쓰고 있다.**

[CAPTURE_PROTOCOL.md §3.1](../CAPTURE_PROTOCOL.md)의 **P1 Moving Rig 15 event**는 composite rig를 **그리퍼에 쥔 채** 고정 카메라로만 촬영한다. 여기서 $T_{\mathrm{flange\_rig}}$가 추정되면 고정 카메라 $T^B_{C_i}$는 **$T^G_C$를 거치지 않고 FK로 직접 base에 앵커된다.** 그 다음 P2에서 내려놓은 큐브를 손목 카메라가 보면, 그때는 큐브 pose가 **$T^G_C$와 무관한 경로로** 결정돼 있다. **이때에만 주장 4가 순환이 아니다.**

즉 저자의 진짜 기여 문장은 이렇게 써야 한다.

> **살아남는 형태:** "그리퍼에 쥔 타깃(P1)이 고정 카메라 네트워크를 robot base에 FK로 앵커하고,
> 그 앵커된 네트워크가 내려놓은 큐브(P2)의 base-frame pose를 과결정한다.
> 손목 카메라의 hand–eye는 그 **외부에서 결정된** 타깃 pose에 대해 풀리므로,
> 타깃 pose와 hand–eye를 같은 단일 카메라 데이터에서 함께 풀어야 하는 기존 EIH 대비
> 두 미지수의 상관이 끊긴다."

이건 **정보 이득의 정체를 "절대 지식"이 아니라 "상관 해제(decorrelation)"로 재정의**한 것이다. 이 형태면 방어 가능하고, ②의 실험으로 직접 잴 수 있다.

**(d) 그런데 이 재정의도 무조건 이득이라는 보장이 없다 — 방향이 반대일 수 있다.**

[ADD_EXPERIMENTS_SUMMARY.md](../ADD_EXPERIMENTS_SUMMARY.md)의 external baseline package 행을 보면 Session04 관측은 **154 observations 중 fixed 50 / gripper 104**다. 즉 **손목 카메라 관측이 고정 카메라 3대를 합친 것의 2배**다. 고정 카메라 1대당 약 17개다.

그리고 손목 카메라(D435)는 큐브를 **근접해서** 보고, 고정 카메라(D415, 0.5~1 m)는 **멀리서 비스듬히** 본다. 정확한 관측 거리·시야각·코너당 픽셀 크기는 저장소에서 `미확인`이다.

**그렇다면 고정 카메라가 결정한 큐브 pose가 손목 카메라가 혼자 낼 수 있는 pose보다 나쁠 가능성이 충분하다.** 그 경우 이 구조는 이득이 아니라 **오염**이다. 저자는 이 방향을 한 번도 검사하지 않았다. 이게 이 논문의 가장 큰 미검증 가정이다.

**(e) 선행연구 위치도 저자 주장보다 좁다.**

[SENSOR_TOPOLOGY_MATRIX §4.2](SENSOR_TOPOLOGY_MATRIX.md)가 정직하게 적고 있듯 **R3 (Zhou, TIE'24)** 하나가 "손목 3 + 고정 2를 한 그래프"를 이미 한다. 저자 주장 4의 구조적 novelty는 R3가 이미 차지했다. 남는 건 (i) 소비자급 RGB-D, (ii) 독립 외부 GT 평가, (iii) **EIH 이득의 분리 측정** — 셋뿐이고, (iii)은 **아무도 안 했으므로 여기가 유일하게 큰 빈칸**이다. 그런데 저자도 아직 안 했다.

### 1.4 판정 요약

| 항목 | 상태 |
|---|---|
| "절대 위치를 안다" 문장 | **폐기**. 순환 구조가 실재한다 |
| "상관 해제 + 과결정" 재정의 | **성립 가능**. 단 P1 gripped-rig 앵커가 실제로 작동해야 함 |
| 그 이득의 크기 | **미측정**. 저장소에 EIH 단독 지표가 존재하지 않음 |
| 이득의 방향 | **미검증**. 고정캠이 손목캠보다 큐브를 못 볼 수 있음 |
| 구조적 novelty | **R3가 선점**. 남은 건 분리 측정과 독립 GT |

---

## ② 반드시 필요한 실험 (우선순위 순)

우선순위 표기:
- **R (Reject)** — 이게 없으면 나는 reject를 낸다.
- **M (Major)** — 이게 없으면 major revision. 있으면 기여가 산다.
- **N (Nice)** — 있으면 논문이 좋아진다. 없어도 reject 사유는 아니다.

### 표: 실험 목록 한눈에

| ID | 우선 | 한 줄 | 답하는 질문 |
|---|---|---|---|
| **E1** | **R** | 결합 끊기 대조군 (관측 수 동일) | 이득이 "정보량"인가 "정보의 종류"인가 |
| **E2** | **R** | 여기(excitation) 감소 sweep | 절대 타깃 지식이 EIH의 회전 여기 요구를 없애는가 |
| **E3** | **R** | 외부 GT의 독립성 확보 | GT가 FK로 만들어졌으면 FK 계열이 자동 승리 |
| **E4** | **R** | 같은 데이터로 EIH 고전 baseline | 우리 EIH가 Tsai/Daniilidis/Tabb/Koide보다 나은가 |
| **E5** | **M** | 고정 카메라 대수 sweep 0→3 | 이득이 단조인가, 3대가 우연인가 |
| **E6** | **M** | 관측 품질 비대칭 진단 | 고정캠이 손목캠보다 큐브를 잘 보는가 |
| **E7** | **M** | P1 gripped-rig 앵커 제거 ablation | 순환을 실제로 깼는가 |
| **E8** | **M** | 파지 반복성 실측 | FK prior(A3/A4/A5)의 전제가 존재하는가 |
| **E9** | **M** | 큐브 면간 기하 오차 보정 변수화 | 21.9 mm board–cube 충돌이 EIH 이득보다 큰가 |
| **E10** | **M** | $T^G_C$ 사후 공분산 / 세션 재설치 반복 | 이득이 우연 1회인가 |
| **E11** | **N** | 고정캠 계통오차 주입 sweep (시뮬) | 오염 경로의 크기 |
| **E12** | **N** | 촬영 수 sweep을 EIH 지표로 | 데이터 효율 주장 |

---

### E1 — 결합 끊기 대조군 (우선순위 **R**)

**목적.** "고정 카메라를 더 썼으니 당연히 좋다"와 "공유 잠재 큐브 pose로 묶었기 때문에 좋다"를 분리한다. 이게 안 되면 기여가 "데이터를 더 썼다"로 축소된다.

**조건.** 동일한 45 event, 동일한 검출 코너, 동일한 split·seed·solver·$(K,D)$. 세 팔(arm)만 다르다.

| arm | 고정캠 관측 | 큐브 pose 변수 | 의미 |
|---|---|---|---|
| **A-solo** | 전부 masking | 손목캠 데이터에서만 추정 | 기존 EIH (우리 solver로 구현한 공정판) |
| **A-decoupled** | **전부 사용** | **카메라별로 따로** ($T^{B,(i)}_{\mathrm{cube}}(s)$를 카메라마다 독립 변수로) | **관측 수는 같고 결합만 없앤 핵심 대조군** |
| **A-ours** | 전부 사용 | 모든 카메라가 **공유** | 제안 방법 |

**지표.** ③의 M1(고정캠-only 앵커 기반 EIH held-out 재투영 px), M2($T^G_C$ 산포 mm/deg), M4(부트스트랩 공분산). **px와 mm는 같은 표에 넣지 않는다.**

**기대 결과.** A-ours < A-decoupled < A-solo. 특히 **A-decoupled와 A-ours의 차이**가 기여의 크기다.

**없으면 나올 반론.** "고정 카메라 3대 분량의 관측을 더 넣었으니 좋아지는 건 당연하다. 저자가 주장하는 '절대 위치를 안다'는 메커니즘의 증거가 아니다." — 이 한 줄로 기여가 사라진다.

---

### E2 — 여기(excitation) 감소 sweep (우선순위 **R**) ⭐ 가장 중요

**목적.** 저자 주장 4를 **직접** 시험하는 유일한 실험이다. 고전 EIH($AX=XB$)는 비평행한 회전축이 최소 2개 필요하다. 타깃의 base-frame pose가 외부에서 결정돼 있으면 **자세 하나하나가 독립적으로 $T^G_C$를 구속하므로 회전 여기 요구가 사라진다.** 저자 주장이 참이면 **여기가 줄어들수록 격차가 커져야 한다.**

**조건.** 손목 카메라 event를 여기 수준별로 부분집합 추출.

| 수준 | 자세 구성 |
|---|---|
| L0 | 전체 (회전축 다양, 큰 각도) |
| L1 | 회전 각도만 절반으로 |
| L2 | 회전축을 1개로 제한 (예: tool z축만) |
| L3 | 회전 거의 없음, 병진만 |

각 수준에서 A-solo(기존 EIH)와 A-ours를 푼다. 부분집합은 **seed 10개 이상**으로 무작위 추출해 산포를 낸다.

**지표.** 수준별 M2($T^G_C$ 오차 mm/deg, 독립 GT 대비) 곡선. 보조로 M4(공분산 조건수).

**기대 결과.** A-solo는 L2·L3에서 급격히 발산하거나 조건수가 폭증한다. A-ours는 평평하다. **이 그림 한 장이 논문의 Figure 1이 되어야 한다.**

**없으면 나올 반론.** "저자는 정보 이득을 주장하지만 그 이득이 나타나야 할 조건(약한 여기)에서 실험하지 않았다. 여기가 충분한 조건에서는 기존 EIH도 잘 풀리므로, 보고된 차이는 방법이 아니라 데이터 양의 효과일 수 있다."

**추가 지적.** 저장소에는 여기 수준을 정량화한 값이 없다. `미확인`. P2·P3의 로봇 자세 다양성(회전축 분포, 각도 범위)을 **먼저 보고**해야 이 sweep이 해석된다.

---

### E3 — 외부 GT의 독립성 확보 (우선순위 **R**)

**목적.** 최종 순위를 정하는 지표의 정당성.

**현재 상태의 심각한 문제.** [`zeus 외부gt.txt`](../zeus%20외부gt.txt)에 적힌 절차는 이렇다.

```
Pos, -242.01, 430.02, 178.71, ... 에다가 큐브 놔두고,
z + 300 올라가서 촬영
다시 아래로 내려가서 큐브 잡고 z 축으로 10 올린 뒤에 ...
```

**즉 "외부 GT"를 로봇이 직접 큐브를 놓아서 만든다.** 그러면 GT = `로봇 FK` + `파지 반복성` + `해제 슬립`이다. **이건 FK와 독립이 아니다.** 이 GT로 A3/A4/A5(FK 계열)와 A2(VISION)를 비교하면 **FK 계열이 구조적으로 유리하다.** [RESEARCH.md §3](../RESEARCH.md)이 "FK-reference는 FK 계열에 구조적으로 유리하다"고 내부 지표에 대해 이미 인정해 놓고, 외부 GT에서 같은 함정을 반복하고 있다.

[README §17](../README.md)은 "tracker/CMM/6-DoF kinematic jig로 독립 측정"이라고 적었지만, 실제 실행 계획 문서는 로봇 배치식이다. **둘 중 무엇이 최종인지 `미확인`이며, 이 불일치 자체가 지적 대상이다.**

**조건.** 다음 중 하나 이상.

| 안 | 내용 | 독립성 |
|---|---|---|
| GT-A | 광학 트래커(OptiTrack 등)로 큐브와 robot base를 직접 측정 | 완전 독립 |
| GT-B | 큐브를 **사람이** 기계 지그의 각인된 위치에 놓고, 지그를 CMM/캘리퍼로 실측 | 로봇과 독립 |
| GT-C | 로봇이 놓되, **놓은 뒤 로봇 전원을 끄고** 독립 계측기로 재측정 | 부분 독립 |

**필수 부수 작업.** GT 자체의 불확도 floor를 **먼저** 보고한다. 반복 배치 20회 이상의 산포로. [SENSOR_TOPOLOGY_MATRIX §8.5](SENSOR_TOPOLOGY_MATRIX.md)가 C9(uncertainty budget)를 참고처로 이미 지목해 놓았다.

**지표.** TRE mm, rotation deg, P95, failure rate — [SOTA_Claim_Protocol.md §4](../SOTA_Claim_Protocol.md)의 사전 합격 조건 그대로. 단 **GT floor보다 작은 차이는 유의미하다고 쓰지 않는다.**

**없으면 나올 반론.** "제안 방법의 우위 판정에 쓰인 ground truth가 제안 방법이 사용하는 것과 동일한 forward kinematics로 생성되었다. 이 비교는 순환이다." — **단독 reject 사유다.**

---

### E4 — 같은 데이터로 EIH 고전 baseline (우선순위 **R**)

**목적.** EIH가 차별점이라고 주장하면, EIH를 **같은 데이터에서** 고전 방법과 비교해야 한다.

**현재 상태의 문제.** [PAPER_BASELINES.md §10](../PAPER_BASELINES.md)은 그리퍼캠 baseline을 **"기존 바닥 보드 데이터로 gTc 계산"**이라고 적었다. 즉 **baseline은 다른 데이터로 푼다.** 이건 불공정 비교이고, 리뷰어가 바로 잡아낸다.

**조건.** 손목 카메라가 큐브/보드를 본 **동일한 event 집합**을 모든 방법에 준다.

| baseline | 형태 | 비고 |
|---|---|---|
| Tsai–Lenz (1989) | $AX=XB$ | OpenCV `calibrateHandEye(TSAI)` |
| Daniilidis (1999) | $AX=XB$ dual quaternion | closed-form 최강 |
| Park–Martin, Horaud–Dornaika | $AX=XB$ | [SOTA_Claim_Protocol §6](../SOTA_Claim_Protocol.md) C2/C3 행에 이미 자리가 있다 |
| Shah (2013) | $AX=ZB$ | 타깃 pose $Z$도 같이 푼다 — **주장 4의 가장 가까운 경쟁자** |
| Tabb & Ahmad Yousef (2017) = P1 | 재투영 반복, 다중 EIH | **우리 목적함수의 직계 선행연구** |
| Koide & Menegatti (RA-L 2019) = P2 | 일반 hand–eye BA | 18 images로 푼다. EIH 정식화가 우리와 가장 가깝다 |
| Ulrich & Hillemann (T-RO 2024) = R5 | 불확도 인지 EIH | soft-FK 주장의 선행연구 |

**특히 Shah를 소홀히 하면 안 된다.** Shah($AX=ZB$)도 **타깃 pose $Z$를 같이 추정**한다. 즉 "타깃 위치를 모른 채 푼다"는 저자의 기존-EIH 묘사가 Shah에는 해당하지 않는다. 저자가 §3에서 스스로 한 반론(주장 3)은 **EIH에도 적용된다**는 사실을 논문이 다뤄야 한다.

**지표.** ③의 M1·M2. baseline이 지원 못 하는 카메라는 [SOTA_Claim_Protocol §6](../SOTA_Claim_Protocol.md)의 `Cameras supported/evaluated` 열에 반드시 명시.

**없으면 나올 반론.** "제안 방법의 eye-in-hand 정확도가 고전 hand–eye 방법과 동일한 입력에서 비교되지 않았다. 보고된 우위는 데이터 차이일 수 있다."

---

### E5 — 고정 카메라 대수 sweep (우선순위 **M**)

**목적.** "고정 카메라가 큐브를 과결정해서 EIH가 좋아진다"가 참이면 **대수에 대해 단조**여야 한다.

**조건.** 고정 카메라 $n \in \{0, 1, 2, 3\}$. $n=0$은 A-solo와 같다. $n=1,2$는 3개 카메라에서 가능한 모든 조합(각각 3개, 3개)을 전부 돌리고 평균·산포를 낸다.

**지표.** M1, M2. 카메라 대수 대비 곡선.

**기대 결과.** 단조 감소. 그리고 **어느 대수에서 포화되는지**가 실용적 결론이 된다.

**즉시 실행 가능하다는 점을 지적한다.** `Simulation/core/scene.py`의 `SimScene`은 이미 **`n_fixed_cams` 파라미터를 받는다.** 그런데 `run_*.py` 전부가 `n_fixed_cams=3`으로 하드코딩돼 있고, 대수 sweep 스크립트가 **하나도 없다.** [SIM_RESULTS.md 표 1 주석](../Simulation/SIM_RESULTS.md)도 "고정 카메라가 적거나 없으면 FK의 가치가 커질 수 있다 — **미검증(다음 실험 후보: 카메라 수 sweep)**"이라고 스스로 적어 놓았다. **저자가 필요성을 인지하고도 1년 가까이 안 돌렸다.** 코드가 이미 지원하므로 며칠 작업이다.

**없으면 나올 반론.** "3대라는 특정 구성에서만 보인 효과다. 일반화 근거가 없다."

---

### E6 — 관측 품질 비대칭 진단 (우선순위 **M**)

**목적.** §1.3(d)의 미검증 가정을 직접 검사한다. **고정 카메라가 손목 카메라보다 큐브를 잘 보는가?**

**조건.** 추정 없이 관측만 집계. 각 (event, camera, cube face)에 대해:

| 측정 | 왜 |
|---|---|
| 카메라–큐브 거리 mm | 삼각측량 오차는 거리에 대체로 비례 |
| 큐브 면 법선과 시선 사이 각 deg | grazing angle이면 PnP가 뒤집힌다 |
| 검출된 코너 수, 코너 간 픽셀 거리 | 유효 해상도 |
| 단일 이미지 PnP 재투영 잔차 px | 관측 품질 직접 대리값 |
| 단일 이미지 PnP pose의 수치적 공분산 | **핵심.** 이게 고정캠 > 손목캠이면 이 논문의 전제가 뒤집힌다 |

**전부 `미확인`이다.** 저장소에 카메라–타깃 거리 통계가 없다.

**지표.** 카메라별 분포(중앙값·IQR). **px와 mm를 한 표에 넣지 않는다 — 별도 표 두 개.**

**기대 결과.** 손목캠 PnP 공분산이 고정캠보다 작으면(=더 정확하면) 저자는 **"그럼에도 왜 이득인가"**를 설명해야 한다. 답은 있다 — 손목캠의 정확한 상대 pose는 $T^G_C$와 **상관**되어 있고 고정캠 정보는 **비상관**이라 상관 해제에 기여한다. 하지만 이 논증은 **수치로 뒷받침되기 전엔 말잔치**다.

**없으면 나올 반론.** "wrist camera가 target에 훨씬 가깝다. fixed camera가 결정한 target pose는 wrist camera 자신의 추정보다 부정확할 수 있고, 그렇다면 제안 구조는 hand–eye를 개선하는 것이 아니라 오염시킨다. 저자는 이 가능성을 검토하지 않았다."

---

### E7 — P1 gripped-rig 앵커 제거 ablation (우선순위 **M**)

**목적.** §1.3(c)에서 밝힌 **순환 탈출 경로가 실제로 작동하는지** 증명한다.

**조건.**

| arm | P1 (rig를 쥔 채 고정캠 촬영) | base 앵커 경로 |
|---|---|---|
| **No-P1** | masking | $T^G_C$ 한 경로뿐 (**순환**) |
| **With-P1** | 사용 | FK 직접 앵커 + $T^G_C$ (**순환 없음**) |

**지표.** M1, M2, 그리고 **$T^G_C$와 $T^B_{C_i}$ 사이의 추정 상관계수 행렬**. With-P1에서 상관이 줄어야 한다.

**기대 결과.** No-P1에서 $T^G_C$ 산포가 크고 고정캠 외부파라미터와 강하게 상관된다. With-P1에서 둘 다 줄어든다.

**없으면 나올 반론.** "고정 카메라의 base-frame 외부파라미터 자체가 hand–eye를 통해서만 결정된다면, '고정 카메라가 절대 위치를 제공한다'는 서술은 순환이다." — **이게 주장 4에 대한 가장 치명적인 반론이며, E7 없이는 반박 불가능하다.**

---

### E8 — 파지 반복성 실측 (우선순위 **M**)

**목적.** A3(FK hard fixed) / A4 / A5의 전제 검증. [README §17-1](../README.md)이 "눈금 cube jig의 반복 파지 실측 없이 FK를 정답으로 주장할 수 없다"고 **스스로 적어 놓고 실측이 없다.**

**조건.** 같은 명령 pose로 큐브를 쥐고 → 놓고 → 다시 쥐기를 30회 이상 반복. 매번 전 카메라로 촬영.

**지표.** 큐브가 그리퍼 안에서 얼마나 움직였는지 mm/deg 산포. 그리고 놓는 순간의 슬립. (git log에 `analyze_release_slip`이 보이지만 결과 문서를 찾지 못했다 — `미확인`)

**기대 결과.** 이 산포가 A3/A4/A5의 FK prior 정확도 **상한**이다. 산포가 외부 GT 목표 오차(2~3 mm 대)와 같은 크기면 FK 계열 세 행의 차이는 전부 노이즈다.

**없으면 나올 반론.** "FK를 target pose prior로 사용하는 세 변형(A3/A4/A5)의 전제인 파지 반복성이 측정되지 않았다. 세 방법의 차이가 유의미하다는 근거가 없다."

---

### E9 — 큐브 기하 오차 변수화 (우선순위 **M**)

**목적.** [README §17-9](../README.md)가 인정한 미해결 문제: **camera 1에서 21.9 mm, camera 3에서 19.5 mm의 board 기반 vs cube 기반 상대자세 불일치.**

**이 크기는 논문이 주장하려는 EIH 이득보다 거의 확실히 크다.** 21.9 mm짜리 계통 불일치가 남아 있는 상태에서 "EIH가 더 정확해졌다"를 mm 단위로 주장하면 리뷰어가 받아들이지 않는다.

**조건.** 큐브의 면간 변환 $T^{\mathrm{cube}}_{\mathrm{face}_k}$를 **최적화 자유변수로 승격**(제작 오차 보정). [SENSOR_TOPOLOGY_MATRIX §8.3](SENSOR_TOPOLOGY_MATRIX.md)이 지목한 **C5 (Imperfect 3-D targets, TIM 2026)**가 정확히 이 문제를 다룬 선행연구다. 인용하고 따라 하라.

- arm-fixed: 면간 기하를 CAD/캘리퍼 값으로 고정 (현재)
- arm-free: 면간 기하를 자유변수로 (게이지 고정을 위해 1개 면은 고정)

**지표.** board–cube 충돌 mm(별도 표), 그리고 M1(px, 별도 표).

**기대 결과.** 충돌이 크게 줄면 "이건 타깃 제작 오차였다"가 확정되고, 그 뒤에야 EIH 이득을 잴 수 있다. 안 줄면 원인은 intrinsic 계통오차이고 그것도 별도로 다뤄야 한다.

**없으면 나올 반론.** "저자 스스로 20 mm 규모의 미해명 target-dependent 불일치를 보고했다. 그 상태에서 mm 단위 정확도 우위를 주장할 수 없다."

---

### E10 — $T^G_C$ 사후 공분산 / 세션 재설치 반복 (우선순위 **M**)

**목적.** 우연 1회가 아님을 보인다. [SOTA_Claim_Protocol.md §3](../SOTA_Claim_Protocol.md)이 "추론 단위는 camera-installation session"이라고 이미 규정했다.

**조건.** 카메라를 **물리적으로 떼었다 다시 설치**한 독립 세션 5회 이상. 각 세션에서 전 방법 실행.

**지표.** 세션 간 $T^G_C$ 산포(mm/deg), 부트스트랩 95% CI, paired difference.

**주의.** 같은 세션 안의 여러 blind pose를 독립 표본처럼 세면 안 된다 — 저장소 규약에 이미 적혀 있다. 현재 세션 수는 1이다(`session02_NOUSE_session04_0814`). **n=1로 통계 주장을 하면 즉시 지적된다.**

**없으면 나올 반론.** "단일 camera installation session의 결과다. 보고된 차이가 설치 간 변동보다 크다는 증거가 없다."

---

### E11 — 고정캠 계통오차 주입 sweep (우선순위 **N**, 시뮬)

**목적.** §1.3(d) 오염 경로의 크기를 안전하게 잰다.

**조건.** 시뮬에서 고정 카메라에만 intrinsic/왜곡 계통편향을 0→3% 주입하고 손목 카메라는 깨끗하게 둔다(그리고 반대 조건도). [WHICH_WINS_WHERE.md §3(2)](../Simulation/WHICH_WINS_WHERE.md)의 계통노이즈 sweep이 이미 있지만 **전 카메라에 동일 적용**이라 비대칭 오염을 못 본다.

**지표.** 시뮬은 이미 `gTc_mm`을 계산한다 (`Simulation/core/metrics.py:171`, `core/experiment.py:186`의 출력 열에 `gTc_mm` 존재). **그런데 [SIM_RESULTS.md](../Simulation/SIM_RESULTS.md)와 [WHICH_WINS_WHERE.md](../Simulation/WHICH_WINS_WHERE.md) 어느 표에도 `gTc_mm` 숫자가 없다.** WHICH_WINS는 "전체 지표(gTc/e_X/reproj)는 fig_ww_metrics.png 참고"라고 그림으로 미뤘다. **EIH가 핵심 기여인 논문에서 EIH 지표를 그림 각주로 미룬 것 자체가 지적 대상이다.** 기존 시뮬 결과에서 `gTc_mm`을 표로 꺼내는 건 오늘 할 수 있는 일이다.

---

### E12 — 촬영 수 sweep을 EIH 지표로 (우선순위 **N**)

[SIM_RESULTS.md](../Simulation/SIM_RESULTS.md)에 `e_task` 기준 set 수 sweep이 있다. 같은 sweep을 `gTc_mm` 기준으로 다시 그린다. "EIH가 데이터에 어떻게 반응하는가"는 EIH 기여 주장의 보조 증거다.

---

## ③ 단순히 "멀티카메라–로봇 캘리브레이션 비교"로 충분한가 — **아니다**

부족하다. 빠진 **비교 축이 5개** 있다.

| 빠진 축 | 무엇이 없나 | 왜 필요한가 |
|---|---|---|
| **축 1. EIH 단독 축** | EIH만 떼어 재는 지표·baseline·ablation이 전무 | 저자 주장 4가 여기 걸려 있다. 현재 저장소의 모든 지표는 EoB+EIH 혼합이다 |
| **축 2. 여기(excitation) 축** | 자세 다양성을 조절한 실험이 없음 | 정보 이득이 가장 크게 나타나야 할 축인데 안 건드렸다 |
| **축 3. 기하(geometry) 축** | 고정캠 대수·배치·거리·각도 sweep 없음 | 일반화 주장의 근거. 코드는 이미 지원(`n_fixed_cams`) |
| **축 4. 타깃 축** | 큐브 면 수·크기·면간 제작오차 sweep 없음. board vs cube vs board+cube는 있으나 **큐브 내부 설계**는 고정 | 다면 큐브가 기여면 "몇 면이 필요한가"를 답해야 한다 |
| **축 5. 실패/열화 축** | 가림, 카메라 드리프트, 부분 관측, 오검출 — 실데이터에서 전무 (시뮬에만 오검출 있음) | 로봇 셀에서 고정캠이 큐브를 못 보는 건 예외가 아니라 상시다 |

그리고 **비교 대상 축**도 하나 더 필요하다.

| 빠진 비교군 | 이유 |
|---|---|
| **R3 (Zhou, TIE'24)** | [SENSOR_TOPOLOGY_MATRIX §5.1](SENSOR_TOPOLOGY_MATRIX.md)이 "둘을 합친 계보는 R3 하나뿐"이라고 스스로 적었다. **유일한 직접 경쟁자를 실행 비교하지 않으면 안 된다.** [PAPER_BASELINES.md](../PAPER_BASELINES.md)의 baseline 6개에 **R3가 없다.** 코드 공개 여부 `미확인` — 없으면 재구현하거나, 최소한 R3의 폐루프 지표를 우리 데이터에서 계산해 같은 축으로 보고 |
| **Shah / Koide** | §E4 참조. "타깃 pose도 같이 푼다"는 성질을 이미 가진 방법들 |

---

## ④ EIH 정확도를 단독으로 재는 지표 — 현재 지표는 전부 실격

### 4.1 왜 현재 지표로는 EIH 기여가 분리되지 않는가

| 현재 지표 | EIH 분리 실패 이유 |
|---|---|
| **Heldout cube RMSE px** | 고정캠·손목캠 관측이 **한 숫자로 pooling**돼 있다. 카메라별 분해가 최종 표에 없다. $T^G_C$가 좋아졌는지 고정캠 외부파라미터가 좋아졌는지 알 수 없다 |
| **Cross-view pixel transfer px** | [README §11.5](../README.md)가 **직접 인정**한다: "Fixed-gripper 27 pair 중 **18 pair는 train fixed-anchor와 heldout gripper event를 연결하므로 mixed-anchor 내부 closure**". 즉 절반 이상이 train 정보로 오염됐다. **EIH 평가 지표로 쓸 수 없다** |
| **Cam-common Obj-Cam mm/deg** | [README §11.6](../README.md)이 인정: "Cross-view px와 같은 pair discrepancy를 다른 단위로 본 값이므로 **독립된 두 번째 증거가 아니며, 공통 systematic error도 검출하지 못한다**". 두 카메라가 **같은 방향으로 틀리면 0이 나온다** — EIH와 고정캠이 함께 틀린 경우를 놓친다 |
| **External cube GT TRE** | EoB+EIH 시스템 전체의 출력. EIH 기여가 섞여 있다. 게다가 §E3의 독립성 문제 |
| **ALL / Train cube RMSE** | in-sample. 저자도 순위 지표 아니라고 명시 |

**결론: 저장소에 EIH를 단독으로 재는 지표가 하나도 없다.** EIH가 핵심 기여라는 논문에서 이건 치명적이다.

### 4.2 제안하는 EIH 단독 지표 5개

각 지표는 **단위가 다르므로 절대 한 표에 섞지 않는다.** 표를 M1(px), M2·M3(mm/deg), M4(무차원), M5(mm/deg)로 나눈다.

---

#### **M1. Leave-gripper-out EIH 예측 재투영** — 단위 px

**정의.** 손목 카메라 관측을 **캘리브레이션에서 완전히 제외**하고 고정 카메라만으로 큐브 pose $T^B_{\mathrm{cube}}(s)$를 확정한다. 그 pose를 frozen한 뒤, 추정된 $T^G_C$와 event FK로 손목 카메라 이미지에 재투영해 검출 코너와 비교한다.

$$\mathrm{RMSE}^{\mathrm{EIH}}_{px} = \sqrt{\frac{1}{2N}\sum_n \left\| \pi\!\left(K_g, D_g, (T^B_G(e)T^G_C)^{-1} T^B_{\mathrm{cube}}(s)\mathbf X_n\right) - \mathbf u_n \right\|^2}$$

**왜 이게 낫나.** 손목 관측이 $T^B_{\mathrm{cube}}$ 추정에 **한 방울도 안 들어가므로** cross-view가 가진 mixed-anchor 오염이 원천적으로 없다. 순수하게 "$T^G_C$가 맞나"만 잰다.

**주의.** $T^G_C$ 자체는 손목 관측으로 추정되므로, **$T^G_C$ 추정에 쓴 event와 평가 event를 반드시 분리**한다(leave-one-event-out 또는 leave-one-placement-out).

---

#### **M2. $T^G_C$ 직접 오차** — 단위 mm / deg

**정의.** 독립 GT 대비 hand–eye 변환 자체의 오차.

**문제.** 손목 카메라의 광학중심을 트래커로 직접 재기는 어렵다. **두 가지 실용 대안.**

- **M2a (권장). 재설치 없는 반복 산포.** 동일 세션에서 손목 카메라로 **같은 고정 기준 타깃**을 서로 다른 로봇 자세 $K$개에서 본다. 각 자세에서 $T^B_{\mathrm{target}}(e) = T^B_G(e)\,T^G_C\,T^{C}_{\mathrm{target}}(e)$를 계산한다. 타깃이 안 움직였으므로 **$K$개 결과의 산포가 곧 $T^G_C$ 오차가 만드는 산포**다. GT가 필요 없다. 로봇 자세 다양성이 클수록 민감하다.
- **M2b. 트래커 기준.** 손목에 트래커 마커를 붙이고 $T^{G}_{\mathrm{tracker}}$를 별도 보정한 뒤 체인으로 비교. 장비 필요.

**M2a는 장비 없이 오늘 가능하다.** 이게 EIH 단독 지표 중 가장 실행 쉬운 것이다.

---

#### **M3. EIH 단독 물리 타기팅** — 단위 mm

**정의.** **고정 카메라를 전부 끄고**, 손목 카메라만으로 물체를 인식해 집거나 핀을 꽂는다. 접촉 위치 오차 mm와 성공률.

**왜.** [README §17](../README.md)의 후속 실험 3번(peg-in-hole / grasp)이 이미 계획돼 있지만 **"고정캠 끄고 손목캠만"이라는 조건이 없다.** 그 조건을 붙여야 EIH 단독 지표가 된다. 그리고 이건 [SENSOR_TOPOLOGY_MATRIX §8.4](SENSOR_TOPOLOGY_MATRIX.md)의 R8(EasyHeC++ 3.0 mm), R7(약 4 mm)과 **종류가 같은 유일한 지표**라 외부 비교가 가능하다.

---

#### **M4. $T^G_C$ 사후 불확도** — 무차원 + mm/deg

**정의.** 수렴점에서 $T^G_C$에 해당하는 Jacobian 블록의 **Schur complement 공분산**, 그 **조건수**, 그리고 $T^G_C$ ↔ $T^B_{C_i}$ ↔ $T^B_{\mathrm{cube}}(s)$ 사이의 **상관계수 행렬**.

**왜 이게 핵심인가.** 저자 주장 4는 정확히 **"$T^G_C$와 $T^B_{\mathrm{cube}}$의 상관을 끊는다"**는 주장이다(§1.3c의 재정의). 그러면 **그 상관계수를 직접 보고하면 된다.** A-solo에서 높고 A-ours에서 낮으면 주장이 수치로 증명된다. 이게 이 논문에서 가장 직접적이고 가장 값싼 증거인데 아무도 안 냈다.

솔버가 SciPy `least_squares`이므로 `result.jac`에서 바로 계산된다. **구현 반나절.**

---

#### **M5. 여기 저항성 곡선** — 단위 mm/deg (E2의 출력)

**정의.** 여기 수준 L0→L3에 대한 M2a 산포 곡선. 두 방법의 곡선 기울기 차이가 기여의 크기다.

---

### 4.3 지표 사용 규칙 (저장소 규율 유지)

1. M1(px)과 M2/M3/M5(mm/deg)는 **절대 같은 표에 넣지 않는다.**
2. M4(상관·조건수)는 세 번째 별도 표.
3. 각 표 캡션에 **"무엇을 재는가"** 한 줄을 반드시 붙인다.
4. 외부 논문 수치를 인용할 때는 **센서 열과 "무엇을 재나" 열을 같이 넣는다** ([SENSOR_TOPOLOGY_MATRIX §7-2](SENSOR_TOPOLOGY_MATRIX.md) 규칙).

---

## ⑤ 저자가 놓친 반례 시나리오와 검증 실험

### 반례 1 — 고정 카메라가 손목 카메라보다 큐브를 못 본다 (오염 역전)

**언제.** 고정캠이 멀리(1 m+) 비스듬히 보고, 손목캠이 가까이(0.2 m) 정면으로 본다. 현재 셋업이 정확히 이럴 가능성이 높다 — 관측 수가 fixed 50 / gripper 104이고, 손목캠은 큐브에 접근하는 위치에 있다.

**무슨 일이 생기나.** 고정캠이 결정한 $T^B_{\mathrm{cube}}$가 손목캠 자신의 추정보다 나쁘다. 공유 잠재변수로 묶으면 그 오차가 $T^G_C$로 **전이**된다. 특히 A5처럼 **hard fixed**하면 전이가 100%다.

**검증 실험 = E6 + 다음.**
- 조건: 고정캠 관측에 인위적 픽셀 편향(0.5/1/2 px 계통)을 주입하고 손목캠은 그대로 둔다. 실데이터에서 후처리로 가능.
- 지표: M2a 산포, M4 상관.
- 기대: 고정캠 편향이 커질수록 A-ours의 $T^G_C$가 A-solo보다 **나빠지는 교차점**이 나온다. **그 교차점이 논문에서 가장 정직한 그림이다.**
- 없으면: "제안 구조는 fixed camera 오차를 hand–eye로 전파하는 경로를 새로 만든다. 저자는 이 경로를 분석하지 않았다."

### 반례 2 — 고정 카메라가 큐브를 아예 못 본다 (가림·좁은 셀)

**언제.** 로봇 몸통이 시선을 가릴 때, 큐브가 작업공간 구석에 있을 때, 산업 셀처럼 카메라를 멀리 못 둘 때. [SENSOR_TOPOLOGY_MATRIX §2.1](SENSOR_TOPOLOGY_MATRIX.md)의 R2 산업 셀은 이미지가 **10장 미만**이다 — 현장에서는 관측이 귀하다.

**무슨 일이 생기나.** 일부 placement에서 고정캠 관측이 0 또는 1대뿐이면 과결정이 사라진다. 그때 A-ours는 A-solo로 **퇴화**하거나, 관측 1대일 때는 **오히려 나쁜 prior**를 받는다.

**검증 실험.**
- 조건: placement별로 큐브를 본 고정캠 대수로 **층화(stratify)**한다 (0대 / 1대 / 2대 / 3대). 실데이터에서 자연 발생하는 층을 그대로 쓴다.
- 지표: 층별 M1, M2a. 그리고 [SOTA_Claim_Protocol §4-4](../SOTA_Claim_Protocol.md)가 요구하는 **"worst workspace stratum에서 큰 열화가 없다"** 조건을 이 층화로 검사한다.
- 기대: 0~1대 층에서 이득이 사라지거나 음수. **그러면 "고정캠 2대 이상이 보이는 영역에서만 유효"라는 적용 범위를 논문에 명시해야 한다.**
- 없으면: "제안 방법은 fixed camera가 target을 보는 workspace 영역에서만 유효하다. 그 영역이 얼마나 되는지, 밖에서는 어떻게 되는지 보고되지 않았다."

**추가 지적.** 저장소에 **carving/가시성 통계가 없다.** placement별로 큐브를 본 카메라 수 분포 = `미확인`.

### 반례 3 — 파지·해제가 반복적이지 않다

**언제.** 그리퍼가 큐브를 매번 조금 다르게 쥐거나, 놓는 순간 미끄러진다. git log에 `analyze_release_slip`과 `session2_pick_and_place: double gripper-settle time`가 있는 걸 보면 **저자도 이미 슬립 문제를 겪고 있다.** 그런데 정량 결과 문서를 못 찾았다 — `미확인`.

**무슨 일이 생기나.** (i) A3/A4/A5의 FK prior가 무의미해진다. (ii) 더 중요하게, **E3의 로봇 배치식 외부 GT가 통째로 무너진다** — GT pose 자체가 슬립만큼 틀린다.

**검증 실험 = E8 + 다음.**
- 조건: 같은 명령 pose로 pick→place 30회. 매번 놓은 뒤 전 카메라 촬영. 그리고 **놓기 직전(gripped)과 놓은 직후(released)를 같은 카메라로 비교**해 슬립만 분리.
- 지표: 배치 산포 mm/deg, 슬립 mm/deg. 별도 표.
- 기대: 이 산포가 GT floor다. **주장하려는 개선폭이 이 floor보다 작으면 논문의 주 결론이 성립하지 않는다.**
- 없으면: "ground truth가 로봇 파지 반복성에 의존하는데 그 반복성이 측정되지 않았다."

### 반례 4 — 카메라가 움직인다 (드리프트)

**언제.** 열팽창, 진동, 삼각대 크리프. 캘리브레이션 후 몇 시간~며칠.

**무슨 일이 생기나.** 손목 카메라의 $T^G_C$는 **기계적으로 볼트 고정**이라 안정적이다. 고정 카메라는 **삼각대**라 훨씬 잘 움직인다. 제안 구조는 안정적인 것(EIH)을 불안정한 것(EoB)에 **묶는다.** 즉 시간이 지날수록 제안 방법이 기존 EIH보다 **나빠질 수 있다.**

**이건 저자가 전혀 언급하지 않은 축이다.**

**검증 실험.**
- 조건: 캘리브 직후 / 6시간 후 / 24시간 후 / 48시간 후 동일 blind pose 재촬영. 카메라는 손대지 않는다. 온도 로깅.
- 지표: 시간별 M2a, M1, 그리고 외부 GT TRE.
- 기대: 고정캠 외부파라미터가 드리프트하고 그게 $T^G_C$ 사용 경로로 전이되는지 확인. 전이되면 **"재보정 주기"**를 논문에 명시해야 한다.
- 없으면: "제안 방법은 wrist camera calibration을 tripod-mounted camera extrinsics에 결합시킨다. 시간 안정성이 평가되지 않았다."

### 반례 5 — 큐브 면간 제작 오차 (다면성이 독이 되는 경우)

**언제.** 면 A와 면 B의 상대 위치가 설계값과 다르다. 카메라마다 **다른 면**을 보면, 같은 큐브 pose에 대해 카메라별로 다른 답이 나온다.

**무슨 일이 생기나.** "여러 카메라가 같은 큐브를 본다"의 이점이 **계통 편향으로 바뀐다.** 다면이라서 좋은 게 아니라 다면이라서 나쁘다. [README §17-9](../README.md)의 **21.9 mm 불일치**가 바로 이 증상일 수 있다.

**검증 실험 = E9 + 다음.**
- 조건: 카메라가 본 **면 조합**으로 층화. "카메라들이 같은 면을 봤을 때" vs "서로 다른 면을 봤을 때"의 불일치를 분리.
- 지표: 층별 cross-view 불일치 mm (별도 표), 면간 보정량 mm/deg.
- 기대: 서로 다른 면 층에서 불일치가 크면 원인이 큐브 기하로 확정된다.
- 없으면: "multi-face cube의 face-to-face manufacturing error가 보정 변수로 다뤄지지 않았다. 여러 카메라가 서로 다른 면을 관측하면 이 오차가 systematic bias로 들어간다."

### 반례 6 — 손목 카메라 이미지 품질 (모션 블러·롤링셔터·노출)

**언제.** 손목 카메라는 움직인다. RealSense color는 **롤링셔터**다 ([SENSOR_TOPOLOGY_MATRIX §3](SENSOR_TOPOLOGY_MATRIX.md)이 직접 지적).

**무슨 일이 생기나.** 정지 후 촬영이면 문제 없지만, 정지 시간(settle time)이 부족하면 코너가 흐르고 EIH만 선택적으로 열화된다. git log의 "double gripper-settle time"은 저자가 이미 이 문제를 겪었다는 신호다.

**검증 실험.**
- 조건: settle time 0.2 / 0.5 / 1.0 / 2.0 s로 같은 pose 촬영.
- 지표: 카메라별 코너 검출 잔차 px (별도 표), M1.
- 기대: 손목캠만 settle time에 민감하면 프로토콜에 최소 settle time을 명시해야 한다.
- 없으면: 큰 반론은 아니지만 재현성 질문을 받는다.

---

## ⑥ 지금 상태로 제출하면 받을 코멘트 (예상)

우선순위 순. 내가 리뷰어라면 이 순서로 쓴다.

### R-1 (reject 사유) — Ground truth가 평가 대상과 독립이 아니다
> The external ground truth appears to be generated by the robot itself placing the cube at commanded poses (`zeus 외부gt.txt`). The GT therefore contains the same forward-kinematics errors that the FK-based variants (A3/A4/A5) exploit. Comparing FK-based and vision-only methods against this GT is circular. An independent measurement (tracker, CMM, or a robot-independent jig) and a reported GT uncertainty floor are mandatory.

### R-2 (reject 사유) — 핵심 기여를 재는 지표가 존재하지 않는다
> The paper claims that solving eye-in-hand calibration with an externally determined target pose is the main contribution. However, every reported metric (held-out cube px, cross-view px, cam-common mm/deg, external TRE) mixes eye-on-base and eye-in-hand contributions. The authors themselves state that 18 of 27 fixed-gripper cross-view pairs are "mixed-anchor internal closure" and that cam-common consistency "cannot detect common systematic error". No metric isolates $T^G_C$. The central claim is therefore unmeasured.

### R-3 (reject 사유) — "절대 위치를 안다"는 순환이다
> The fixed-camera reprojection residual is invariant to a global rigid transform applied jointly to all $T^B_{C_i}$ and all target poses. The robot base gauge is fixed exclusively through the eye-in-hand chain $T^B_G(e)T^G_C$. Claiming that fixed cameras supply "absolute" target knowledge to the hand–eye problem is circular unless a gripper-mounted, FK-anchored target phase is demonstrated to break it. This is not shown.

### R-4 (reject 사유) — 대조군이 잘못 설계됐다
> The comparison "with fixed cameras" vs "without" confounds two things: the amount of observations and the coupling through a shared latent target pose. A decoupled control (same observations, per-camera independent target poses) is required. Furthermore, the eye-in-hand baselines are reported as being computed on a *different* dataset (`PAPER_BASELINES.md` §10: "그리퍼캠: 기존 바닥 보드 데이터로"), which is not a fair comparison.

### M-1 — 유일한 직접 경쟁자를 비교하지 않았다
> The authors' own literature matrix identifies Zhou et al. (TIE 2024) as the only prior work solving mixed fixed + wrist cameras in one graph, yet this method is absent from the baseline set (`PAPER_BASELINES.md` lists six baselines, none of which is Zhou). Comparison against the nearest competitor is required.

### M-2 — 정보 이득이 나타나야 할 조건에서 실험하지 않았다
> If knowing the target pose decouples the hand–eye estimate, the benefit should grow as rotational excitation decreases, since classical $AX=XB$ requires two non-parallel rotation axes. No excitation ablation is reported. Likewise, no sweep over the number of fixed cameras is reported, although the simulation code already exposes `n_fixed_cams` and the authors' own `SIM_RESULTS.md` lists this sweep as "미검증 / 다음 실험 후보".

### M-3 — 미해명 20 mm 불일치 위에서 mm 단위 주장을 한다
> `README.md` §17.9 reports a 21.9 mm / 19.5 mm discrepancy between board-derived and cube-derived relative poses that remains unexplained. Accuracy claims at the millimetre level cannot be made while an unmodelled systematic error of this magnitude is present. Face-to-face cube manufacturing error should be included as a calibration variable (cf. Imperfect 3-D targets, TIM 2026).

### M-4 — 통계 단위가 n=1이다
> The authors' own protocol states that the inference unit is a camera-installation session, yet all results come from a single session. Bootstrap CIs over blind poses within one session do not support the stated superiority contract.

### M-5 — FK 기반 변형들의 전제가 측정되지 않았다
> A3/A4/A5 differ in how they use the FK-derived target pose prior, yet grasp-and-release repeatability — the upper bound on that prior's accuracy — is not measured. The authors acknowledge this in §17.1 but provide no data.

### m-1 (minor) — 지표 혼합
> Table in `FK_use_A2-A5.md` places "Train overall px", "Heldout Cube px", "Cross-view Cube px" and "Cam-common mm/deg" in a single table. These quantify different things in different units on different populations (in-sample vs held-out vs pairwise). Table 1 in `Simulation/SIM_RESULTS.md` similarly mixes e_task mm, e_X mm, bTf mm, reproj px and cross mm. Please separate. **이건 저장소가 스스로 세운 규율을 스스로 어긴 사례다.**

### m-2 (minor) — 인용 불가 상태의 근거 문서
> `Simulation/SIM_RESULTS.md` and `Simulation/WHICH_WINS_WHERE.md` both carry a banner saying the numbers are "구버전(불공정) 코드로 산출된 것이라 폐기 대상 — 재산출 전까지 인용하지 마세요." Yet these are listed as supporting evidence. Either regenerate or remove from the evidence chain.

### m-3 (minor) — 근거 문서가 없다
> `RESEARCH.md`, `FK_use_A2-A5.md`, `ADD_EXPERIMENTS_SUMMARY.md` and `README.md` all declare `CALIBRATION_EXPERIMENT_VALIDATION.md` to be the "단일 기준" document. **이 파일이 저장소에 존재하지 않는다.** 네 문서가 존재하지 않는 기준 문서를 가리키고 있다.

### m-4 (minor) — 근거 없는 수치 인용
> `SENSOR_TOPOLOGY_MATRIX.md` §8.6 proposes the sentence "우리는 로봇 좌표계에서 새 물체 위치를 2~3 mm로 맞힌다". No external-GT result exists in the repository (only templates and a test). 이 수치의 출처는 `미확인`이며, 현 상태로 논문에 쓰면 근거 없는 주장이다.

---

## ⑦ 최소 통과 조건 (저자가 무엇부터 해야 하나)

내가 accept 쪽으로 기울려면 **최소한** 이것들이 필요하다.

| 순서 | 항목 | 난이도 |
|---|---|---|
| 1 | **M4 상관/공분산 보고** — 주장 4를 수치로 만든 것. SciPy `result.jac`에서 바로 나온다 | 반나절 |
| 2 | **기존 시뮬 결과에서 `gTc_mm` 표 추출** — 이미 계산되고 있는데 표에 없다 | 반나절 |
| 3 | **E1 결합 끊기 대조군** — 기여 정의 자체 | 수일 |
| 4 | **M1 leave-gripper-out 지표** — EIH 단독 px | 수일 |
| 5 | **E5 고정캠 대수 sweep (시뮬)** — `n_fixed_cams` 이미 지원 | 수일 |
| 6 | **E2 여기 sweep** — 주장 4의 직접 증명. Figure 1 후보 | 1~2주 |
| 7 | **E8 파지 반복성 + E3 독립 GT** — 평가 정당성 | 2~4주, 장비 필요 |
| 8 | **E4 같은 데이터 EIH baseline** | 2~4주 |
| 9 | **R3(Zhou) 비교** | 재구현 필요 시 상당 |

1·2번은 **오늘 시작할 수 있고 기여 주장의 성패를 바로 가른다.** 여기서 상관이 안 줄어들면 주장 4는 폐기하고 논문 프레임을 다시 짜야 한다. 그 사실을 실험 6~9번에 몇 주 쓰기 **전에** 아는 게 낫다.

---

## ⑧ 이 리뷰에서 확인하지 못한 것 (`미확인` 목록)

정직하게 남긴다. 아래는 저장소에서 근거를 찾지 못한 항목이고, 추정으로 채우지 않았다.

| 항목 | 상태 |
|---|---|
| `CALIBRATION_EXPERIMENT_VALIDATION.md` | **파일 없음.** 4개 문서가 이걸 단일 기준으로 참조 |
| 외부 GT 실측 결과 | 없음. 템플릿·테스트·러너만 존재 |
| "우리 2~3 mm"의 출처 | 없음 |
| 카메라–큐브 관측 거리·시야각 분포 | 없음 |
| placement별 큐브를 본 고정캠 대수 분포 | 없음 |
| 파지/해제 반복성 정량 결과 | `analyze_release_slip`이 git log에 보이나 결과 문서 미발견 |
| 로봇 자세 다양성(여기) 정량화 | 없음 |
| 최종 외부 GT가 트래커식인지 로봇 배치식인지 | 문서 간 불일치 (README §17 vs `zeus 외부gt.txt`) |
| R3 (Zhou TIE'24) 코드 공개 여부 / 실촬영 장수 | 저자 문서도 `미확인`으로 표기 |
| 시뮬 결과 문서의 폐기 배너 해제 여부 | 배너는 남아 있고 MEMORY는 재판정했다고 함. 불일치 |

---

## 참고한 저장소 문서

- [`README.md`](../README.md) §2, §6, §7, §9~§13, §17
- [`RESEARCH.md`](../RESEARCH.md)
- [`FK_use_A2-A5.md`](../FK_use_A2-A5.md)
- [`PAPER_BASELINES.md`](../PAPER_BASELINES.md)
- [`SOTA_Claim_Protocol.md`](../SOTA_Claim_Protocol.md)
- [`CAPTURE_PROTOCOL.md`](../CAPTURE_PROTOCOL.md) §3
- [`ADD_EXPERIMENTS_SUMMARY.md`](../ADD_EXPERIMENTS_SUMMARY.md)
- [`ABLATION_TEST_result_0909/README.md`](../ABLATION_TEST_result_0909/README.md)
- [`SENSOR_TOPOLOGY_MATRIX.md`](SENSOR_TOPOLOGY_MATRIX.md) §2, §3, §4.2, §5, §7, §8
- [`2023plus_calibration_review.md`](2023plus_calibration_review.md)
- [`Simulation/SIM_RESULTS.md`](../Simulation/SIM_RESULTS.md), [`Simulation/WHICH_WINS_WHERE.md`](../Simulation/WHICH_WINS_WHERE.md)
- `Simulation/core/metrics.py`, `Simulation/core/scene.py`, `Simulation/core/experiment.py`
- `calibration_pipeline/external_gt.py`, `zeus 외부gt.txt`
