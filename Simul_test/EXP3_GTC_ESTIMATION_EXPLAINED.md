# Experiment 3 — Camera-based vs FK-based vs Camera+FK-correction (쉽게 읽는 설명서)

> **한 줄 요약:** 통합(Joint) 캘리브에서 gripper-target 변환(gTc)과 큐브 위치를 다루는
> 세 방식을 비교했다. 큐브를 **미지수로 추정(Camera-based)** vs **FK로 아는 값으로 고정
> (FK-based)** vs **추정 후 FK로 후보정(Camera+FK)**. → 노이즈가 커질수록 **FK-based 가
> 가장 취약**하고, **Camera+FK-correction 이 전 구간 최고**.

이 문서는 [fig_exp3_noise_sweep.png](figures/fig_exp3_noise_sweep.png) 를 읽는 데 필요한 것을
담는다: **① 세 방식이 뭔가 ② 지표 ③ 노이즈 sweep 결과.**

---

## 0. 목적

통합(Joint = bundle adjustment) 캘리브에서 **gripper-target 변환(gTc)을 미지수로 추정하는
방식**과, **로봇 FK·큐브 기하로 아는 값으로 고정/보정하는 방식**의 차이를 평가한다.
(gTc = 그리퍼와 그리퍼에 붙은 카메라 사이의 상대 위치, eye-in-hand 핸드아이의 핵심 미지수.)

---

## 1. 세 방식

세 방식 모두 **통합(Joint)** 을 베이스로 하고, 큐브(타깃) 위치를 어떻게 다루느냐만 다르다.

### ① Camera-based
- 큐브 위치를 **미지수로 두고** 카메라 관측으로 gTc·카메라와 **함께 추정**.
- 노이즈가 큐브·카메라·gTc 여러 파라미터로 분산됨 → gTc 가 덜 오염.

### ② FK-based (Robot FK-based target pose)
- 큐브 위치를 로봇 FK 로 아는 값으로 **고정**하고 카메라·gTc 만 최적화.
- 큐브가 상수라 관측 노이즈를 흡수할 곳이 gTc 밖에 없음 → 노이즈가 gTc 로 몰림.

### ③ Camera+FK-correction
- ① Camera-based 로 캘리브한 **뒤**, train 큐브의 FK 정답으로 학습한 **위치의존 잔차
  (Ridge 회귀)** 를 최종 예측에 후보정.
- = "camera-based estimation + robot FK-based target pose 를 이용한 후보정".
- gTc 자체는 안 바꾸고 **최종 큐브 예측만** 보정 → held-out 에서만 효과.

> **주의:** 이 시뮬은 FK 가 완벽하다(FK = GT 큐브 위치)고 가정한다. 그래서 노이즈가 작을
> 때는 ①②가 거의 같다. 차이는 **노이즈가 커질 때** 드러난다(아래 sweep).

---

## 2. 평가 지표

- **[핵심] Held-out 큐브 예측 오차 (mm):** train 으로 캘리브 후, 학습 안 한 새 큐브 위치를
  base 에서 예측한 오차. **실전(파지) 성능.**
- **[진단] gTc 복원 오차 (mm):** 추정한 gripper-target 변환이 GT 와 몇 mm 차이나나.
  세 방식의 핵심 차이가 드러나는 지표.
- (그 외 진단: Camera-to-base 오차, Prior/Target consistency. Reprojection error 는 시뮬에서
  주입 노이즈를 되비출 뿐 변별력이 없어 **제외**.)

---

## 3. 노이즈 sweep 결과

관측 노이즈를 2 → 15mm 로 키우며 세 방식의 오차 곡선을 그렸다 (systematic 노이즈, 8 sets).

### 왼쪽: Held-out 큐브 예측 (실전 성능)

| 노이즈(mm) | Camera-based | FK-based | Camera+FK-corr |
|---|---|---|---|
| 2  | 0.39 | 0.41 | **0.07** |
| 6  | 1.07 | 1.22 | **0.18** |
| 15 | 2.62 | 3.04 | **0.47** |

- **Camera+FK-correction 이 전 구간 압도적** (15mm 에서도 0.47mm, 다른 둘의 1/6).
- Camera-based 가 FK-based 보다 약간 나음 (노이즈가 분산되어서).

### 오른쪽: gTc(핸드아이) 복원

| 노이즈(mm) | Camera-based | FK-based | Camera+FK-corr |
|---|---|---|---|
| 2  | 0.12 | 0.13 | 0.12 |
| 6  | 0.15 | 0.39 | 0.15 |
| 15 | 0.25 | **0.98** | 0.25 |

- **FK-based 만 노이즈에 급격히 취약** (15mm 에서 0.98mm, 4배). 큐브를 FK 로 고정하면
  관측 노이즈를 gTc 가 다 떠안기 때문.
- Camera-based · Camera+FK-corr 는 완만 (겹침) — 후보정은 gTc 를 안 바꾸므로 파랑과 동일.

### 세 가지 메시지
1. **큐브를 FK 로 고정하면(FK-based) 노이즈에 취약하다.** 관측 노이즈가 gTc 로 몰려
   핸드아이 추정이 나빠진다 → gTc 0.98mm.
2. **큐브를 미지수로 추정하면(Camera-based) 더 강건하다.** 노이즈가 여러 파라미터로 분산.
3. **후보정(Camera+FK)이 실전 성능을 극적으로 높인다.** 노이즈가 커져도 held-out 이
   완만하게만 증가(0.47mm). FK 로 배운 위치의존 잔차를 최종 예측에서 제거하기 때문.

> 결론: gripper-target 변환은 **미지수로 추정**하는 편이 노이즈에 강건하고, 여기에
> **FK 기반 후보정**을 얹으면 실전 성능이 가장 좋다.

---

## 4. 재현

```bash
# 메인 3방식 비교 (표)
PYTHONPATH= python Simul_test/exp3_gtc_estimation.py --seeds 20

# 노이즈 sweep (곡선 데이터 저장)
PYTHONPATH= python Simul_test/exp3_gtc_estimation.py --sweep noise --seeds 12 \
    --dump Simul_test/figures/exp3_noise_sweep_data.json

# 곡선 figure 생성
PYTHONPATH= python Simul_test/viz_exp3_sweep.py
```

(선택) 데이터량 sweep: `--sweep set` — train set 수를 늘리며 세 방식 비교.
