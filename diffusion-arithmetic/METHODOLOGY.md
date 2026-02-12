# Experimental Methodology

Discrete Diffusion vs Autoregressive Models on Arithmetic Reasoning

---

## Shared Architecture

모든 실험에서 동일한 Transformer backbone을 사용한다. AR과 diffusion의 차이는 **attention mask**와 **학습 objective**뿐이며, 파라미터 수와 구조는 동일하다.

### Model

| 항목 | 값 | 비고 |
|------|-----|------|
| Layers | 6 | (모듈 3만 4) |
| Attention heads | 6 | (모듈 3만 4) |
| Embedding dim | 384 | (모듈 3만 192) |
| Dropout | 0.2 | (모듈 3만 0.1) |
| Weight tying | O | wte.weight = lm_head.weight |
| Params (모듈 1-2) | ~10.6M | |
| Params (모듈 3) | ~1.5M | 더 작은 분포 학습용 |

### Position Encoding 옵션

모듈 1-2에서는 두 가지를 모두 실험한다.

**Absolute PE**: 학습 가능한 position embedding `wpe(pos)`를 token embedding에 더한다. 학습 시 본 최대 위치까지만 의미 있는 표현을 갖는다.

**RoPE (Rotary Position Embedding)**: Query와 Key에 회전 행렬을 적용한다. 절대 위치가 아닌 상대 위치를 encoding하므로, 학습 시 보지 못한 길이에서도 위치 관계가 보존된다. 주파수 계산:

$$\theta_i = 10000^{-2i/d_{\text{head}}}$$

position $t$에서 $(q_{2i}, q_{2i+1})$을 $\theta_i \cdot t$만큼 회전시킨다. LLaDA와 동일하게, causal mask만 제거하고 순수 RoPE를 사용한다 (추가 position embedding 없음).

**RoPE + Diffusion 특성**: RoPE는 Q·K attention에만 위치 정보를 제공하며, token embedding 자체에는 위치를 부여하지 않는다. AR에서는 causal mask가 암묵적 위치 신호를 제공하지만, bidirectional diffusion에서는 이 신호가 없다. MASK 토큰들은 첫 번째 layer 입력에서 동일한 embedding을 가지며, attention weight의 상대 위치 정보만으로 구분되어야 한다. LLaDA (8B)에서는 이것이 충분히 작동하지만, 소규모 모델에서는 absolute PE 대비 성능 차이가 나타날 수 있으며, 이 자체가 분석 대상이다.

**왜 둘 다 테스트하는가**: Absolute PE를 사용하면 "length generalisation failure"가 position encoding 한계인지 모델의 알고리즘적 한계인지 구분할 수 없다. RoPE를 병렬 테스트함으로써 이 confound를 분리한다.

### Training Objective

**AR**: Teacher forcing. `=` 이후 answer 위치에 대해서만 cross-entropy loss를 계산한다. 매 step에서 모든 answer token이 supervision을 받는다.

**Diffusion (Masked)**: Answer 영역의 토큰을 random mask ratio $t \sim U(0,1)$로 마스킹한 후, 마스킹된 위치의 원래 토큰을 예측한다. 매 step에서 마스킹된 subset만 supervision을 받으므로, 동일 iteration 수에서 AR보다 effective supervision이 적다. 이를 보상하기 위해 convergence 기반 학습을 사용한다.

### Convergence 기반 학습

고정 iteration 대신, loss가 일정 기간(`patience`) 동안 개선되지 않으면 학습을 중단한다.

| 파라미터 | 모듈 1-2 | 모듈 3 |
|----------|---------|--------|
| Max iterations/epochs | 20,000 iter | 60 epoch |
| Patience | 3,000 iter | 8 epoch |
| Min delta | 1e-4 | 1e-4 |
| 중단 시 | best 시점의 weight로 복원 | best 시점의 weight로 복원 |

**왜 필요한가**: AR은 teacher forcing으로 매 step 전체 answer에 대한 gradient를 받지만, diffusion은 random subset만 받는다. 고정 iteration으로 비교하면 diffusion이 불리하다. Convergence 기준으로 맞추면 양쪽 모두 학습이 충분히 된 상태에서 비교할 수 있다.

### Decoding

| 항목 | AR | Diffusion |
|------|-----|-----------|
| 기본 방식 | Greedy (argmax) | Greedy (argmax) |
| Temperature | N/A (deterministic) | N/A (deterministic) |
| 생성 방향 | 항상 left-to-right | Policy에 따라 다름 |

**왜 둘 다 greedy인가**: AR은 관례적으로 greedy를 사용한다. Diffusion에서 sampling(temperature=1.0)을 사용하면 stochastic noise가 추가되어, 정확도 차이가 모델의 능력 차이인지 sampling noise인지 구분할 수 없다. 공정한 비교를 위해 양쪽 모두 deterministic argmax를 사용한다.

**예외**: 모듈 3 (toy distribution)에서는 sampling을 사용한다. 이유는 아래에서 설명한다.

---

## 모듈 1: Addition (exp_addition.py)

Multi-digit 덧셈을 통해 carry chain 추론 능력을 테스트한다. 3자리/5자리/7자리의 난이도 스케일링으로 AR과 diffusion의 차이가 드러나는 경계를 찾는다.

### 데이터 구성

각 sample은 문자열 형태이며, 세 가지 출력 format을 사용한다.

**Plain**: 답을 그대로 출력한다.
```
347+521=0868           (3d)
54321+12345=066666     (5d)
```
Operand는 nd자리로 zero-pad하고, 답은 (nd+1)자리로 zero-pad한다.

**Reverse**: 답을 뒤집어 출력한다.
```
347+521=8680
```
LSB가 먼저 나오므로 carry를 순서대로 처리할 수 있다. AR에게 유리한 format.

**Scratchpad**: 중간 과정을 명시적으로 출력한다.
```
347+521=C0S8C0S6C0S8>>0868
```
오른쪽 자릿수부터 처리하며, 각 step에서 carry(C)와 sum digit(S)를 기록한다. `>>` 뒤에 최종 답이 온다.

### Operand 생성 규칙

Operand는 0-999 범위에서 **정수 값 단위로** sampling한다. 모든 digit (0-9)이 학습 데이터에 등장한다.

**Number-level OOD (Lee et al. 2023과 동일)**: 1000개 operand 값 중 100개(10%)를 random으로 선택하여 학습에서 제외한다. 이는 1000×1000 덧셈 행렬에서 특정 row/column을 빈칸으로 두는 것과 동일하다. Low-rank matrix completion의 관점에서, 나머지 entry들이 빈칸을 제약하므로 모델이 일반화할 수 있다.

이 설계의 이점:
- **논문과 직접 비교 가능**: Lee et al.이 성공적으로 보인 일반화와 동일한 조건이다.
- **Digit embedding 문제 회피**: Digit-level exclusion (예: 5를 모든 위치에서 제거)은 해당 digit의 operand-position embedding이 학습되지 않아 일반화가 원천적으로 불가능하다. Number-level holdout은 모든 digit이 다양한 context에서 학습된다.
- **LRMC vs 알고리즘 학습 테스트**: LRMC는 row/column 전체가 빈 경우 복원 불가능하지만, 우리가 10%만 제외하면 각 operand의 다른 조합은 학습에 포함되므로 "진정한" 덧셈 알고리즘을 학습했는지 검증할 수 있다.

Seed를 고정(0)하여 held-out set의 재현성을 보장한다. 중복 operand 쌍 `(a, b)`은 제거한다.

### 데이터 분할

**Multi-digit scaling**: 2가지 자릿수에서 실험한다. 3자리에서 accuracy가 ~1.0에 포화되므로, 5자리로 난이도를 올려 AR과 diffusion의 차이가 드러나는 경계를 찾는다. (7자리 실험 결과, 5자리와 동일한 경향을 보여 제외.)

| nd | N_train | N_test | OOD 종류 | 비고 |
|----|---------|--------|---------|------|
| 3  | 10,000  | 2,000  | Number-level + Length | 논문 비교용 |
| 5  | 50,000  | 2,000  | Length only | 난이도 확장 |

**3자리 splits** (Lee et al. 2023 비교):

| Split | 자릿수 | Operand 조건 | N | 테스트 대상 |
|-------|--------|-------------|---|-----------|
| train | 3 | 양쪽 모두 TRAIN_OPS (900개) | 10,000 | — |
| test_id | 3 | 양쪽 모두 TRAIN_OPS | 2,000 | In-distribution 정확도 |
| test_ood_number | 3 | ≥1개가 HELD_OUT (100개) | 2,000 | Number-level 일반화 |
| test_ood_length | 4 | 제약 없음 (0-9999) | 2,000 | Length 일반화 |

**5자리 splits**: Operand space가 너무 커서 (10^5) number-level holdout이 의미 없으므로, uniform random sampling + length OOD만 테스트.

| Split | 자릿수 | N | 테스트 대상 |
|-------|--------|---|-----------|
| train | 5 | 50,000 | — |
| test_id | 5 | 2,000 | ID 정확도 |
| test_ood_length | 7 | 2,000 | Length 일반화 |

### 실험 조건

2(digit_config) × 2(objective) × 3(format) × 2(pos_enc) = **24 configurations**

{3d, 5d} × {ar, diffusion} × {plain, reverse, scratchpad} × {absolute, rope}

### 평가 지표

**Exact Match Accuracy**: 생성된 최종 답 문자열이 정답과 완전히 일치하는 비율. Scratchpad format에서는 `>>` 뒤의 최종 답만 비교한다.

$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

**Number OOD Breakdown**: `test_ood_number`의 결과를 두 그룹으로 세분한다.
- "one held-out": operand 중 정확히 하나만 HELD_OUT에 속하는 sample
- "both held-out": 양쪽 operand 모두 HELD_OUT에 속하는 sample

HELD_OUT 100개에서 양쪽 모두 held-out인 쌍의 비율은 약 (100/1000)² = 1%이므로 대부분은 "one held-out"이다. "Both held-out"은 덧셈 행렬에서 두 row/column 모두 빈 교차점에 해당하며, LRMC에서도 가장 어려운 경우이다.

**Scratchpad Decode Order Analysis** (diffusion + scratchpad에만 해당):

Diffusion의 scratchpad이 AR의 chain-of-thought과 기능적으로 동일한지 검증한다.

- `scratchpad_first_ratio`: 모든 scratchpad 위치가 모든 최종 답 위치보다 먼저 채워진 sample의 비율.
- `avg_scratchpad_rank`: scratchpad 위치들이 채워진 평균 순서 (0 = 가장 먼저).
- `avg_final_rank`: 최종 답 위치들이 채워진 평균 순서.

**Fixation Order Analysis** (diffusion의 계산 경로 분석):

"Diffusion이 AR과 다른 계산 경로를 학습하는가?"를 직접 검증하기 위한 핵심 분석이다.

답의 각 자릿수 위치(0=MSB, nd=LSB)에 대해, diffusion decoding 중 해당 위치가 **몇 번째 step에서 확정(fix)되었는지**를 추적한다.

- `mean_rank[i]`: 답의 i번째 위치가 확정된 평균 step. 낮을수록 먼저 결정됨.
- `carry_corr`: carry 발생 여부와 fixation rank 간의 Spearman 상관관계. 양수이면 carry 위치가 더 늦게 결정됨(carry가 불확실성을 높이므로).

**기대 패턴**:
- 만약 diffusion이 LSB부터 확정하면 → 내부적으로 reverse와 유사한 알고리즘을 학습한 것
- 만약 MSB부터 확정하면 → plain과 유사
- 만약 carry 위치를 늦게 확정하면 → carry 전파를 인식하고 있다는 증거
- 만약 format (plain/reverse/scratchpad)에 따라 fixation order가 달라지면 → format이 학습된 알고리즘에 영향을 줌

이 분석은 각 digit 설정 (3d/5d)에서 독립적으로 수행되며, 자릿수가 늘어남에 따라 패턴이 어떻게 변하는지도 관찰한다.

**Convergence Iteration**: 각 configuration이 수렴한 iteration. Training budget의 공정성을 확인한다.

**Difficulty Curve**: nd별 ID accuracy를 시각화하여, accuracy가 1.0 이하로 떨어지기 시작하는 digit 수와 그때의 AR/diffusion 차이를 분석한다.

### Secondary Analysis: Digit-Position Exclusion (Appendix B.2.1 재현)

메인 실험(number-level OOD)과 별도로, Lee et al.의 Appendix B.2.1을 재현하여 AR과 diffusion의 차이를 추가 분석한다.

**설계**: Digit 5를 **특정 자릿수 위치 하나에서만** 제외하고, 위치별로 별도 모델을 학습시킨다.

| 실험 | 제외 조건 | 학습 데이터 예시 |
|------|----------|---------------|
| excl_pos=0 (LSB, ones) | ones 자리에 5 없음 | `123+467=...` (O), `125+467=...` (X) |
| excl_pos=1 (tens) | tens 자리에 5 없음 | `123+467=...` (O), `153+467=...` (X) |
| excl_pos=2 (MSB, hundreds) | hundreds 자리에 5 없음 | `123+467=...` (O), `523+467=...` (X) |

총 3(position) × 2(ar, diffusion) = **6 models** 학습. Format은 plain만 사용, position encoding은 absolute만 사용.

**평가 지표**:
- **Overall accuracy**: 모든 digit, 모든 위치를 포함하는 10,000개 test set에서의 정확도.
- **Exclusion accuracy**: test set 중 제외된 위치에 digit 5가 등장하는 sample만 추출한 정확도. 이것이 해당 위치에서의 일반화 능력을 직접 측정한다.

**기대 패턴 (논문 기반)**:
- LSB exclusion이 가장 해롭다. LSB는 carry 없이 순수 덧셈을 배우는 위치이므로, 여기서 5의 패턴을 못 배우면 다른 위치의 carry 계산에도 전이가 안 된다.
- MSB exclusion이 가장 덜 해롭다. MSB는 carry에 의존하므로, 5가 아닌 다른 digit들의 패턴에서 5의 행동을 추론할 여지가 있다.

**우리의 추가 질문**: Diffusion 모델은 AR과 다른 패턴을 보이는가? Diffusion은 non-sequential decode order를 가지므로, LSB에 대한 의존도가 AR과 다를 수 있다.

### 신뢰성 분석

2,000 test samples에서 accuracy $p$의 standard error:

$$\text{SE} = \sqrt{\frac{p(1-p)}{N}} \approx 0.011 \quad (p=0.5, N=2000)$$

Accuracy 차이가 5% 이상이면 통계적으로 유의하다.

---

## 모듈 2: Tree Expression (exp_tree.py)

이진 트리 수식을 통해 병렬 계산 구조에서의 추론 능력을 테스트한다.

### 데이터 구성

**트리 구조**: 깊이 $d$의 perfect binary tree. 각 leaf는 단일 digit, 각 내부 node는 `+` 또는 `*` 연산자이다. 깊이 $d$에서 leaf 수는 $2^d$개.

예시 (depth 2):
```
((3+5)*(2+7))
= (8 * 9)
= 72
→ 출력: 072 (mod 1000, 3자리 zero-pad)
```

**Plain format**:
```
((3+5)*(2+7))=072
```

**Scratchpad format**: Level-wise intermediate 값을 명시한다.
```
((3+5)*(2+7))=[L1:8,9][L2:72]=>072
```
`[L1:8,9]`는 depth 1에서의 중간 결과(8=3+5, 9=2+7), `[L2:72]`는 depth 2에서의 최종 결과. `=>` 뒤에 최종 답이 온다.

**왜 tree인가**: 덧셈의 carry chain은 본질적으로 순차적이다(이전 자릿수의 carry를 알아야 다음 자릿수를 계산). 반면 tree의 하위 subtree들은 독립적으로 계산 가능하므로 병렬 구조를 가진다. Diffusion 모델이 이 병렬성을 활용할 수 있는지 테스트한다.

**mod 1000**: 답의 길이를 3자리로 고정하여 output length가 변수가 되지 않게 한다.

### 데이터 분할

| Split | 깊이 | Leaf digits | N | 테스트 대상 |
|-------|------|------------|---|-----------|
| train | {2, 3} | {0,...,9} | 10,000 | — |
| test_id | {2, 3} | {0,...,9} | 2,000 | In-distribution |
| test_ood_depth | {4, 5} | {0,...,9} | 2,000 | 순수 depth 일반화 |

Tree의 leaf는 단일 digit(0-9)이므로 number-level OOD가 적용되지 않는다. OOD는 depth generalization만 테스트한다.

### 실험 조건

2(objective) × 2(format) × 2(pos_enc) = **8 configurations**:

{ar, diffusion} × {plain, scratchpad} × {absolute, rope}

### 평가 지표

모듈 1과 동일한 Exact Match Accuracy를 사용한다. Scratchpad format에서는 `=>` 뒤의 최종 답(3자리)만 비교한다. Scratchpad decode order analysis도 동일하게 적용한다.

Number-level OOD breakdown은 없다 (leaf가 단일 digit이므로 적용 불가).

---

## 모듈 3: Toy Distribution (exp_toy_distribution.py)

Diffusion의 **sampling mechanism**을 controlled 환경에서 분석한다. Ground truth 분포 $p(x)$가 알려져 있으므로, 생성된 sample의 quality를 정확하게 측정할 수 있다.

### 모듈 1-2와의 관계

모듈 1-2는 conditional generation (question → answer) + exact match로 **reasoning 능력**을 측정한다. 모듈 3은 unconditional generation + distributional metrics로 **sampling mechanism 자체**를 분석한다. 결과는 "policy가 어떤 dependency 구조에서 어떻게 행동하는가"를 설명하지만, 이를 직접 arithmetic/tree 성능 예측에 사용해서는 안 된다 (dependency 구조가 다르기 때문).

### 왜 sampling인가 (greedy가 아닌)

모듈 1-2에서는 exact match 정확도를 측정하므로 deterministic greedy가 적절하다. 그러나 모듈 3에서는 distributional metric (TV distance, KL divergence)을 측정한다. Greedy는 deterministic이므로 N개 sample이 모두 동일한 시퀀스가 된다. 이 경우 empirical distribution은 단일 point mass가 되어 TV/KL 계산이 의미를 잃는다. 따라서 모듈 3에서는 sampling을 사용한다.

### Sequence 공간

| 항목 | 값 |
|------|-----|
| Vocabulary | {0, 1, 2, 3} (V=4) |
| Sequence length | 8 (L=8) |
| 가능한 시퀀스 수 | 4^8 = 65,536 |
| MASK token | id=4 (생성 시 -inf로 제외) |

V와 L을 작게 설정하여 **전체 분포를 enumerate**할 수 있다. Ground truth 분포 $p(x)$를 정확하게 계산할 수 있으므로, TV distance 등의 metric이 정밀하다.

### 분포 정의

네 가지 분포는 서로 다른 dependency 구조를 가진다.

**A. Near-Independent** (α=0.5)

$$p(x) = \prod_{i=1}^{L} p_i(x_i)$$

$p_i$는 Dirichlet(0.5, 0.5, 0.5, 0.5)에서 sampling한 position별 marginal이다. α=0.5는 sparse Dirichlet로, 각 위치에서 1-2개 token에 확률이 집중된다. 위치 간 dependency는 없다.

α의 효과: α=3.0(기존)일 때 marginal이 거의 uniform(~0.25 each)이어서 학습할 구조가 없었다. α=0.5로 낮추면 mode probability가 ~0.65까지 올라가 명확한 패턴이 생긴다.

**B. 2nd-order Markov Chain** (sparsity=0.1)

$$p(x) = p(x_1, x_2) \prod_{t=3}^{L} p(x_t | x_{t-2}, x_{t-1})$$

4^2 = 16개의 context에 대한 transition matrix를 Dirichlet로 생성한다. sparsity=0.1은 매우 peaked한 transition을 만든다 (각 context에서 dominant next token이 확률 ~0.9).

**C. Global Sum Constraint** (β=2.0, α=0.5)

$$p(x) \propto \exp\!\Bigl(-\beta \bigl(\sum_i x_i - \tau\bigr)^2\Bigr) \cdot \prod_i p_i(x_i)$$

τ = (V-1)·L/2 = 12. 위치별로는 독립적인 base distribution에 "전체 합이 12 근처여야 한다"는 global constraint가 추가된다. β=2.0은 강한 coupling으로, 한 위치의 값이 다른 위치들의 값을 강하게 제약한다.

**D. Markov + Global** (β=1.5, sparsity=0.1)

$$p(x) \propto p_{\text{Markov}}(x) \cdot \exp\!\Bigl(-\beta \bigl(\sum_i x_i - \tau\bigr)^2\Bigr)$$

Local dependency (Markov)와 global dependency (sum constraint)를 동시에 가진다. 가장 복잡한 dependency 구조로, decoding policy의 차이가 가장 크게 나타날 것으로 기대한다.

### 학습

| 항목 | 값 |
|------|-----|
| 학습 데이터 | 각 분포에서 200,000 sample |
| Batch size | 1,024 |
| Learning rate | 3e-4 (cosine annealing) |
| Position encoding | Absolute (길이가 고정이므로 RoPE 불필요) |

학습 objective는 모듈 1-2의 diffusion과 동일 (random masking → predict masked tokens). 다만 여기서는 conditional이 아니라 unconditional이므로 전체 시퀀스가 masking 대상이다.

### 이론적 최적 loss

학습이 제대로 되었는지 판단하기 위해 **Bayes-optimal masked loss**를 계산한다.

완벽한 모델이 masked token을 예측할 때의 cross-entropy:

$$\mathcal{L}^* = \mathbb{E}_{x, \text{mask}} \left[ -\sum_{i \in \text{masked}} \log p(x_i | x_{\text{unmasked}}) \right]$$

이를 Monte Carlo로 추정한다. 학습된 모델의 loss가 이 값에 가까우면 모델이 분포를 잘 학습한 것이다. Gap이 크면 모델 capacity 부족이다.

비교 기준:
- Uniform CE = ln(4) ≈ 1.386 (아무것도 학습하지 않은 상태)
- Optimal CE = 분포에 따라 다름 (구조가 많을수록 낮음)
- 학습된 loss = Optimal CE에 가까울수록 좋음

### Decoding Policy

시퀀스의 모든 위치가 MASK 상태에서 시작하여, 한 번에 하나씩(sequential) 또는 여러 개씩(parallel) 토큰을 결정한다.

**Sequential Policies** (한 step에 1 token, 총 L=8 forward pass):

| Policy | 위치 선택 기준 | 직관 |
|--------|-------------|------|
| confidence | max softmax probability가 가장 높은 위치 | "가장 확신하는 것부터" |
| low_entropy | 예측 entropy가 가장 낮은 위치 | confidence와 유사하나 전체 분포 고려 |
| high_entropy | 예측 entropy가 가장 높은 위치 | "가장 불확실한 것부터" (역직관적) |
| margin | top-1과 top-2 확률 차이가 가장 큰 위치 | "가장 명확한 선택지가 있는 곳부터" |
| random | masked 위치 중 uniform random | Baseline |
| l2r | 왼쪽에서 오른쪽 순서 | AR과 동일한 순서 |
| r2l | 오른쪽에서 왼쪽 순서 | 역순 |

**Adaptive Threshold** (variable NFE):

| Policy | τ | 동작 |
|--------|---|------|
| adaptive_τ0.5 | 0.5 | 공격적 병렬화 — 절반 이상 확신하면 동시 decode |
| adaptive_τ0.7 | 0.7 | 중간 — 70% 이상에서만 동시 decode |
| adaptive_τ0.9 | 0.9 | 보수적 — 거의 확실한 것만 동시 decode |

각 step에서 max softmax probability ≥ τ인 **모든** 위치를 동시에 decode한다. 아무 위치도 threshold를 넘지 못하면 가장 confident한 1개를 decode (fallback). 이 방식은 분포의 dependency 구조에 자연스럽게 적응한다: independent한 분포에서는 한번에 많은 위치가 확실하므로 NFE가 크게 줄고, strongly coupled한 분포에서는 한번에 적은 위치만 확실하므로 sequential에 가깝게 동작한다.

**Jacobi Iteration** (variable NFE):

| Policy | max_iter | 동작 |
|--------|----------|------|
| jacobi_i5 | 5 | 빠른 수렴 또는 조기 종료 |
| jacobi_i10 | 10 | 중간 |
| jacobi_i20 | 20 | 충분한 수렴 보장 |

모든 위치를 동시에 predict한 뒤, 그 결과를 context로 다시 predict하는 과정을 반복한다. Token이 더 이상 변하지 않으면 (fixed point) 수렴. Jacobi iteration의 diffusion 버전으로, 수렴 시 sequential confidence와 동일한 quality를 달성하면서도 수렴이 빠르면 NFE가 L보다 훨씬 적을 수 있다.

**기존 parallel_random/confidence를 제거한 이유**: 고정 k개를 무작위/confidence 순으로 동시 decode하는 방식은 dependency를 무시하기 때문에, sequential보다 항상 나쁜 결과를 보였다. Adaptive threshold와 Jacobi는 모델 자체의 confidence를 활용하여 "언제 병렬화할지"를 결정하므로, speed-quality tradeoff에서 의미있는 비교가 가능하다.

### 평가 지표

모든 metric은 N=500,000 sample에 대해 계산한다.

**TV Distance (Total Variation)**:

$$\text{TV}(p, q) = \frac{1}{2}\sum_{x \in \mathcal{X}} |p(x) - q(x)|$$

여기서 $q(x)$는 empirical distribution (= 생성된 sample의 빈도). 범위는 [0, 1]이며 0이면 완벽한 일치.

**TV Baseline (Irreducible Estimation Noise)**:

$p(x)$에서 직접 500K개를 sampling해도 finite sample effect로 TV > 0이다. 이를 5회 반복 측정하여 baseline을 구한다.

$$\text{TV}_{\text{baseline}} = \mathbb{E}\left[\text{TV}(p, \hat{p}_{\text{ideal}})\right]$$

이론적으로 $\text{TV}_{\text{noise}} \approx \sqrt{|\mathcal{X}| / 2\pi N}$. N=500K, |X|=65,536에서 약 0.14.

**TV Excess (Policy-Induced Error)**:

$$\text{TV}_{\text{excess}} = \text{TV}_{\text{observed}} - \text{TV}_{\text{baseline}}$$

이것이 실제로 policy의 quality를 반영하는 지표이다. Baseline 이하이면 noise 수준이고, 높을수록 policy가 분포를 왜곡한다.

**Bootstrap 95% CI**:

500K sample에서 bootstrap resampling (K=20)으로 TV의 confidence interval을 계산한다. 두 policy의 CI가 겹치지 않으면 차이가 통계적으로 유의하다.

**KL Divergence**:

$$D_{\text{KL}}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}$$

TV보다 tail behavior에 민감하다. 관측되지 않은 시퀀스에 $q(x)=0$이면 KL이 발산하므로, 이를 $10^{-10}$으로 clamp한다.

**Mode Coverage (Top-100)**:

True distribution의 확률 상위 100개 시퀀스 중, 생성된 sample에 1회 이상 등장한 시퀀스의 비율. Mode dropping을 감지한다.

**Spearman Correlation**:

각 생성된 sample의 path score (decoding 과정에서 누적된 log-probability)와 true log-probability $\log p(x)$의 순위 상관. Policy가 실제로 높은 확률의 시퀀스를 우선 생성하는지를 측정한다.

**Support Coverage**:

65,536개 가능한 시퀀스 중 1회 이상 생성된 unique 시퀀스의 비율. 500K sample (7.6× support size)이면 ~99.95%가 기대값이다.

**NFE (Number of Forward Evaluations)**:

Sequential policies는 항상 L=8. Parallel policies는 k에 따라 줄어든다. Quality가 동일하다면 NFE가 낮을수록 효율적이다.

**Decode Order Analysis** (새로 추가):

각 policy가 L개 위치를 어떤 순서로 decode하는지 추적한다. Sequential policy의 경우, 각 step에서 선택된 position을 기록하여:
- `mean_decode_order[j]`: position j가 평균 몇 번째 step에서 decode되었는지. 0이면 가장 먼저.

**MI-Order Alignment Score** (새로 추가):

Position별 total MI (= 해당 위치와 다른 모든 위치 간 MI의 합)와 decode order 간의 Spearman 상관.

$$\rho = \text{Spearman}\bigl(\sum_j I(X_i; X_j),\ \text{decode\_rank}(i)\bigr)$$

- ρ < 0 (음수): high-MI 위치를 먼저 decode → MI 구조를 활용하는 좋은 전략
- ρ ≈ 0: decode order가 MI와 무관
- ρ > 0: high-MI 위치를 나중에 decode → MI 구조와 반대

이 지표는 policy가 분포의 dependency 구조를 얼마나 잘 활용하는지 직접 측정한다.

**Policy Ranking Table** (새로 추가):

각 분포에서 sequential policy들을 excess TV 순으로 1~7위까지 ranking. 분포별로 어떤 policy가 일관되게 좋은지/나쁜지를 한눈에 파악한다.

### Mutual Information Matrix

각 분포의 dependency 구조를 시각화하기 위해 pairwise MI를 계산한다.

$$I(X_i; X_j) = \sum_{v_i, v_j} p(X_i=v_i, X_j=v_j) \log \frac{p(X_i=v_i, X_j=v_j)}{p(X_i=v_i) \cdot p(X_j=v_j)}$$

- Distribution A: MI ≈ 0 everywhere (독립)
- Distribution B: 인접 위치 pair에서 높은 MI (local dependency)
- Distribution C: 모든 pair에서 균일하게 낮지 않은 MI (global coupling)
- Distribution D: Local + global 패턴의 혼합

이 matrix는 decoding policy의 성능을 해석하는 데 사용된다. MI-Order Alignment Score와 함께 사용하면, 특정 policy가 어떤 dependency 구조에서 왜 잘/못 작동하는지 설명할 수 있다.

---

## Visualization 목록

### 모듈 1

- **Difficulty Curve**: nd별 (3d, 5d) ID accuracy 곡선. AR vs diffusion이 갈라지는 지점.
- **Accuracy by Split (per nd)**: 각 자릿수별 bar chart. AR(빨강) vs diffusion(파랑), hatched=RoPE.
- **Fixation Order**: Diffusion이 답의 각 자릿수를 확정하는 순서. MSB부터인지 LSB부터인지.
- **Fixation × Format 비교**: plain/reverse/scratchpad에서 fixation order가 어떻게 달라지는지.
- **Digit-Position Exclusion**: LSB/tens/MSB 위치별로 digit 5를 제외했을 때 AR vs diffusion 비교.

### 모듈 2

- **Training Curves**: Position encoding별 loss 곡선.
- **Accuracy by Split**: ID + depth OOD bar chart.
- **RoPE vs Absolute Scatter**: Depth OOD에서 absolute vs RoPE 비교.

### 모듈 3

- **MI Matrix Heatmap**: 4개 분포의 pairwise MI. Dependency 구조 시각화.
- **TV Excess Heatmap**: 전체 13 policies (seq 7 + adaptive 3 + jacobi 3) × 4 distributions.
- **NFE vs TV Scatter**: 핵심 시각화. Sequential은 모두 NFE=8에 모이고, adaptive/jacobi는 NFE가 달라짐. Quality 손실 없이 NFE를 줄이는 정도를 직접 비교.
- **Decode Order Heatmap**: Adaptive policy별 position decode 순서 (10K 샘플 기반).
- **MI-Order Alignment**: Sequential policy의 decode order가 MI 구조와 얼마나 일치하는지.
- **Policy Ranking Table**: 전체 13 policies의 분포별 ranking.
- **Spearman Correlation**: Path score와 true probability의 순위 상관.
