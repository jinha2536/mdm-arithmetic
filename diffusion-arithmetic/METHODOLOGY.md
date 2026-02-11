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

position $t$에서 $(q_{2i}, q_{2i+1})$을 $\theta_i \cdot t$만큼 회전시킨다.

**왜 둘 다 테스트하는가**: Absolute PE를 사용하면 "length generalisation failure"가 position encoding 한계인지 모델의 알고리즘적 한계인지 구분할 수 없다. RoPE를 병렬 테스트함으로써 이 confound를 분리한다.

### Training Objective

**AR**: Teacher forcing. `=` 이후 answer 위치에 대해서만 cross-entropy loss를 계산한다. 매 step에서 모든 answer token이 supervision을 받는다.

**Diffusion (Masked)**: Answer 영역의 토큰을 random mask ratio $t \sim U(0,1)$로 마스킹한 후, 마스킹된 위치의 원래 토큰을 예측한다. 매 step에서 마스킹된 subset만 supervision을 받으므로, 동일 iteration 수에서 AR보다 effective supervision이 적다. 이를 보상하기 위해 convergence 기반 학습을 사용한다.

### Convergence 기반 학습

고정 iteration 대신, loss가 일정 기간(`patience`) 동안 개선되지 않으면 학습을 중단한다.

| 파라미터 | 모듈 1-2 | 모듈 3 |
|----------|---------|--------|
| Max iterations/epochs | 15,000 iter | 60 epoch |
| Patience | 2,000 iter | 8 epoch |
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

3자리 덧셈을 통해 carry chain 추론 능력을 테스트한다.

### 데이터 구성

각 sample은 문자열 형태이며, 세 가지 출력 format을 사용한다.

**Plain**: 답을 그대로 출력한다.
```
347+521=0868
```
Operand는 3자리로 zero-pad하고, 답은 4자리(3+1)로 zero-pad한다. 모델은 `=` 이후를 생성한다.

**Reverse**: 답을 뒤집어 출력한다.
```
347+521=8680
```
Least significant digit이 먼저 나오므로, carry를 순서대로 처리할 수 있다. AR에게 유리할 수 있는 format이다.

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

| Split | 자릿수 | Operand 조건 | N | 테스트 대상 |
|-------|--------|-------------|---|-----------|
| train | 3 | 양쪽 모두 TRAIN_OPS (900개) | 10,000 | — |
| test_id | 3 | 양쪽 모두 TRAIN_OPS | 2,000 | In-distribution 정확도 |
| test_ood_number | 3 | ≥1개가 HELD_OUT (100개) | 2,000 | Number-level 일반화 |
| test_ood_length | 4 | 제약 없음 (0-9999) | 2,000 | Length 일반화 |

**왜 이렇게 분리하는가**: Number OOD는 학습 시 보지 못한 operand 값에 대한 일반화를 측정한다. Length OOD는 3자리에서 4자리로의 positional generalization을 측정한다. 두 축은 독립적이다.

### 실험 조건

2(objective) × 3(format) × 2(pos_enc) = **12 configurations**:

{ar, diffusion} × {plain, reverse, scratchpad} × {absolute, rope}

### 평가 지표

**Exact Match Accuracy**: 생성된 최종 답 문자열이 정답과 완전히 일치하는 비율. Scratchpad format에서는 `>>` 뒤의 최종 답만 비교한다.

$$\text{Acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$$

**Number OOD Breakdown**: `test_ood_number`의 결과를 두 그룹으로 세분한다.
- "one held-out": operand 중 정확히 하나만 HELD_OUT에 속하는 sample
- "both held-out": 양쪽 operand 모두 HELD_OUT에 속하는 sample

HELD_OUT 100개에서 양쪽 모두 held-out인 쌍의 비율은 약 (100/1000)² = 1%이므로 대부분은 "one held-out"이다. "Both held-out"은 덧셈 행렬에서 두 row/column 모두 빈 교차점에 해당하며, LRMC에서도 가장 어려운 경우이다.

**Scratchpad Decode Order Analysis** (diffusion + scratchpad에만 해당):

Diffusion의 scratchpad이 AR의 chain-of-thought과 기능적으로 동일한지 검증한다.

- `scratchpad_first_ratio`: 모든 scratchpad 위치가 모든 최종 답 위치보다 먼저 채워진 sample의 비율. 1.0이면 "항상 scratchpad를 먼저 완성한 후 답을 생성"한다는 뜻.
- `avg_scratchpad_rank`: scratchpad 위치들이 채워진 평균 순서 (0 = 가장 먼저).
- `avg_final_rank`: 최종 답 위치들이 채워진 평균 순서.

**Convergence Iteration**: 각 configuration이 수렴한 iteration. Training budget의 공정성을 확인한다.

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

**Parallel Policies** (한 step에 k tokens):

| Policy | 위치 선택 기준 | k |
|--------|-------------|---|
| parallel_random | Random k개 | 2, 4 |
| parallel_confidence | Confidence top-k | 2, 4 |
| parallel_low_dep | Confidence 기반 + logit 유사도가 낮은 조합 | 2, 4 |

`parallel_low_dep`은 서로 독립적인 위치를 골라서 동시에 decode하는 전략이다. Logit vector의 cosine similarity가 낮은 위치 쌍을 선택한다.

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

### Mutual Information Matrix

각 분포의 dependency 구조를 시각화하기 위해 pairwise MI를 계산한다.

$$I(X_i; X_j) = \sum_{v_i, v_j} p(X_i=v_i, X_j=v_j) \log \frac{p(X_i=v_i, X_j=v_j)}{p(X_i=v_i) \cdot p(X_j=v_j)}$$

- Distribution A: MI ≈ 0 everywhere (독립)
- Distribution B: 인접 위치 pair에서 높은 MI (local dependency)
- Distribution C: 모든 pair에서 균일하게 낮지 않은 MI (global coupling)
- Distribution D: Local + global 패턴의 혼합

이 matrix는 decoding policy의 성능을 해석하는 데 사용된다. 예를 들어 confidence policy가 Distribution B에서 잘 작동하면, local dependency가 있는 구조에서 "확신이 높은 위치부터"가 효과적이라는 해석이 가능하다.

---

## Visualization 목록

### 모듈 1-2 공통

- **Training Curves**: Position encoding별 loss 곡선. Convergence 확인용.
- **Convergence Bar Chart**: 각 configuration의 수렴 iteration. Budget 공정성 확인.
- **Accuracy by Split**: Split별 bar chart. AR(빨강) vs diffusion(파랑), hatched=RoPE.
- **RoPE vs Absolute Scatter**: OOD split에서 absolute(x축) vs RoPE(y축). 대각선 위이면 RoPE가 우세.

### 모듈 3

- **MI Matrix Heatmap**: 4개 분포의 pairwise MI. Dependency 구조 시각화.
- **TV Heatmap (Raw + Excess)**: Policy × Distribution. Raw TV와 baseline 제거 후 excess TV 병렬 비교.
- **Spearman Correlation Heatmap**: Policy × Distribution. Path score와 true probability의 순위 상관.
- **Pareto Chart (NFE vs TV)**: Speed-quality tradeoff. Parallel policies의 k에 따른 변화 추적.
