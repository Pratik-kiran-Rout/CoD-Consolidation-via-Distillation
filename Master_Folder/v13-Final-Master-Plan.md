# MASTER PLAN V13 (FINAL SPECIFICATION): Calibrated Intra-Layer State-Survival KV Eviction & Distilled State Consolidation for Attention–SSM Hybrids

**Subtitle:** *Token-Level Realized State-Survival Calibration, Risk-Adaptive KV Budget Allocation, and Distilled Ridge State Consolidation on Pretrained & Controlled Hybrid Language Models*

---

## 0. Executive Summary: What Changed from V12 / Audit V2 to Final V13

This document is the **definitive, final Master Plan V13**. It resolves every mathematical, architectural, and experimental flaw identified across V12 and the three technical review rounds, strips away stacked complexity, and focuses 100% of our effort on a clean, high-payoff architecture and experimental plan executable on **Kaggle Free Tier ($2 \times \text{NVIDIA T4 16GB GPUs}$)**.

### 0.1 Realistic Novelty & Reviewer Calibration (Zero Sugarcoating)

* **No Self-Awarded Score Inflation:** A paper's final score at ICLR/NeurIPS/ICML/TMLR is determined by empirical outcomes on Gate 0–2 and pretrained benchmarks, not by design promises.
* **Design Baseline vs. Empirical Ceiling:**
  * **Pre-Experiment Design Rating:** **7.0 – 7.5 / 10** (mathematically sound, training-free + 1,728-parameter distilled consolidation on real 0.5B–1.5B hybrids, with a built-in diagnostic fallback paper if consolidation or survival signals saturate).
  * **Path to 8.0 – 8.5 / 10:** Achieved **if and only if** (1) the calibrated survival score $\hat{p}_{\text{surv}}$ beats `DeltaS` (`2609.27470`), `DASC` (`2608.30386`), and write-time surprise (`HOLA`, `2607.02303`) on Gate 0 while controlling for token age $P - t$, and (2) token-level state-aware eviction + distilled state consolidation outperforms `SnapKV`, `KVzip` (`2505.23416`), `DeltaS`, and `LORE-KV` (`2610.07643`) at matched memory budgets on **`Falcon-H1-1.5B-Deep-Instruct`**.

### 0.2 Eight Final Fixes Locked into V13

| Area | Flaw Caught in Final Review | Exact Resolution Locked into Master Plan V13 |
| :--- | :--- | :--- |
| **1. Granularity & Scope** | Block granularity ($B=32$) penalizes baselines on single-token needles and requires complex block pooling; learned `SR-TransMamba` adds bloat before gates pass. | **Primary Path is Token-Level ($B=1$) on Pretrained Hybrids (`Falcon-H1-0.5B` & `1.5B-Deep`).** Custom from-scratch `SR-TransMamba` is cut from the critical path. |
| **2. Memory Accounting & Scale** | In `Falcon-H1-0.5B`, KV/token/layer $= 256$ floats while $S_P = 196,608$ floats ($768$ token-equivalents, not $32$), and 8K KV cache is only $151\text{ MB}$ with weak 16K 0.5B accuracy. | **Primary benchmark upgraded to `Falcon-H1-1.5B-Deep-Instruct` ($L=66$)** alongside `0.5B`; budgets reported as **% of Retained KV Cache ($10\% \dots 50\%$)** on tasks where Full-KV accuracy is high, plus a **Streaming Chunked-Prefill Variant**. |
| **3. Exact Tensor Hook in `FalconH1Mixer`** | Hooking `in_proj` misses $\mu$P multipliers (`ssm_in_multiplier`, 5-vector `ssm_multipliers`) and `conv1d + SiLU` on `xBC`. | Extract $(x_t, B_t, C_t, \Delta_t)$ **strictly after** $\mu$P scaling, `conv1d`, and `SiLU` inside `FalconH1Mixer`, and read $S_P$ directly from `cache_params.ssm_states[layer_idx]`. |
| **4. $\rho_{\text{surv}}$ Denominator & Calibration** | Audit V2 mixed undecayed $\|u_{t,m}\|_2^2$ with decayed $\|\nu_{t,m}(P)\|_2^2$ and relied on Spearman correlation alone. | Use time-consistent **Decayed Signal-to-Interference Ratio** $\text{SIR}_{t,m}(P) = \frac{\alpha_{t \to P,m}^2 \|u_{t,m}\|_2^2}{\|\nu_{t,m}(P)\|_2^2 + \epsilon}$ + **Directional Cosine** $c_{t,m}(P)$ inside a **Calibrated Survival Predictor $\hat{p}_{\text{surv}}(t,m \mid P)$**, evaluated via **AUROC, ECE Calibration Curves, and Age-Controlled Partial Correlation**. |
| **5. Proposition 2 Spectral Bound** | Claimed operator norm $1/(1+\lambda)$ instead of $\max_i \sigma_i/(\sigma_i^2+\lambda)$, used mean $\eta$ for non-symmetric $\Pi$, and averaged over isotropic queries. | Corrected to the **Exact Singular-Value Perturbation Bound** using $\max_i \frac{\sigma_i}{\sigma_i^2 + \lambda}$ and $\gamma_{\max, m}$, paired with **direct empirical measurement of collateral damage on stored pairs** in Gate 2. |
| **6. Fair Consolidation Comparison & Learned Gate** | Consolidating 16 tokens saves only 16 KV token-equivalents, and hand-sweeping $(\gamma, \lambda)$ risks corrupting long-horizon heads. | Compare consolidating $k \in \{32, 64, 128\}$ evicted tokens against **keeping $k_{\text{eq}}$ extra KV tokens**, and train **1,728–3,168 scalar gates $\{\gamma_{l,m}, \lambda_{l,m}\}$ via 300-step Full-KV Teacher KL Distillation**. |
| **7. De-Circularizing Gate 0** | R5 is synthetically constructed for survival to win, and $\alpha_{t \to P}$ correlates strongly with token age $(P - t)$. | Add **Natural-Text Prefix Recall + Partial Correlation controlling for Token Age $(P - t)$** in Gate 0 so $\hat{p}_{\text{surv}}$ must prove value beyond simple recency. |
| **8. Built-In Fallback Paper** | If state consolidation yields small gains on pretrained models, the project still needs a guaranteed publication path. | Structure Phase 1–2 so that **Calibrated State Survival + Risk-Adaptive Token-Level `SA-SnapKV`** forms a complete, standalone paper even before Phase 3 consolidation is added. |

---

## 1. Core Thesis & The Three Publishable Pillars

### 1.1 The Architectural Thesis

In parallel and interleaved Attention–SSM hybrid language models (`Falcon-H1`, `Hymba`, `Zamba2`), every layer maintains two memory systems across a context prefix $0 \dots P-1$:
1. **The Parametric Recurrent State ($S_{P, l, m} \in \mathbb{R}^{d_{\text{state}} \times d_{\text{head}}}$):** Fixed size regardless of sequence length, but subject to **heterogeneous per-head exponential decay** ($\alpha_{t \to P, l, m}$) and **key-overlap superposition interference** ($\nu_{t, l, m}(P)$).
2. **The Nonparametric Attention KV Cache ($\mathcal{K}_{P,l}, \mathcal{V}_{P,l} \in \mathbb{R}^{P \times H_{kv} \times d_h}$):** Exact lookup, but grows linearly with context length $P$.

Existing KV eviction heuristics (`SnapKV`, `H2O`, `PyramidKV`, `KVzip`) ignore the recurrent state $S_P$ and waste KV budget retaining tokens that $S_P$ already remembers. Conversely, existing state-aware methods (`HOLA`, `HAM`, `DeltaS`) score tokens using **write-time surprise** $\|v_t - k_t S_{t-1}\|$ or **write-time chunk drift** $\|S_c - S_{c-1}\|_F$, which fails when early writes decay between $t$ and $P$ or when recent high-drift writes near $P$ remain perfectly intact inside $S_P$.

### 1.2 Three Clean, Modular Contributions

1. **Pillar 1 — Calibrated Realized State-Survival Predictor ($\hat{p}_{\text{surv}}(t, l, m \mid P)$):**
   Using the exact additive state-readout decomposition $\hat{b}_{t,m} S_{P,m} = \alpha_{t \to P, m} u_{t,m} + \nu_{t,m}(P)$, we extract three time-consistent, dimensionless survival signals at position $P$—**Decayed Signal-to-Interference Ratio** $\text{SIR}_{t,m}(P)$, **Readout Cosine Alignment** $c_{t,m}(P)$, and **Cumulative Log-Decay** $\log \alpha_{t \to P, m}$—and calibrate them to predict per-token, per-head state-only recall probability $\hat{p}_{\text{surv}}(t,l,m \mid P) \in (0, 1)$, validated via AUROC, Expected Calibration Error (ECE), and partial correlation controlling for token age $P - t$.
2. **Pillar 2 — Risk-Adaptive Token-Level State-Aware KV Eviction (`SA-SnapKV`):**
   A training-free, token-level ($B=1$) KV cache eviction and budget allocation rule that retains tokens maximizing **Prospective Utility $\times$ State-Survival Deficit**:
   $$s_{t,l} = D_{t,l} \cdot \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$
   and allocates per-layer KV budgets $K_l$ proportionally to each layer's **Expected State Forgetting Risk** $\mathcal{R}_l(P) = \sum_{t=0}^{P-1} D_{t,l}\big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$.
3. **Pillar 3 — Teacher-Distilled Ridge State Consolidation (`Distilled-Consolidate`):**
   Before evicting runner-up high-deficit tokens $\mathcal{T}_{\text{cons}, l}$ from the KV cache at position $P$, we consolidate their target values into the fast-decaying heads of $S_{P,l,m}$ via a batched ridge-regularized delta-rule update controlled by **per-layer, per-head scalar gates $\{\gamma_{l,m}, \lambda_{l,m}\}$** (only $1,728$ parameters on `0.5B` / $3,168$ parameters on `1.5B-Deep`), trained in 15 minutes via Full-KV teacher KL self-distillation.

---

## 2. Exact Hardware & Memory Physics on Kaggle $2 \times \text{T4}$ GPUs

### 2.1 Verified Architecture & Memory Table (`Falcon-H1-0.5B` & `1.5B-Deep`)

From the official `tiiuae/Falcon-H1` model configurations, both models run parallel Attention + Mamba-2 (`FalconH1Mixer`) in every layer:

| Specification | `Falcon-H1-0.5B-Instruct` (Fast Dev / Gate 1–2) | `Falcon-H1-1.5B-Deep-Instruct` (Primary Paper Benchmark) |
| :--- | :--- | :--- |
| **Layers ($L$)** | $36$ | $66$ |
| **Hidden Dimension ($d_{\text{model}}$)** | $1024$ | $1280$ |
| **Attention Heads ($H_q$ / $H_{kv}$) & $d_h$** | $8$ query / $2$ KV heads, $d_h = 64$ | $6$ query / $2$ KV heads, $d_h = 128$ |
| **Mamba-2 Heads ($H_{\text{ssm}}$) & $d_{\text{head,ssm}}$** | $24$ heads, $d_{\text{head,ssm}} = 64$ | $24$ heads, $d_{\text{head,ssm}} = 64$ |
| **Mamba-2 State Dim ($d_{\text{state}}$) & Groups** | $d_{\text{state}} = 128$, $G_{\text{ssm}} = 1$ | $d_{\text{state}} = 256$, $G_{\text{ssm}} = 1$ |
| **KV Floats per Token per Layer** | $2 \times 2 \times 64 = 256\text{ floats}$ ($512\text{ B}$ in FP16) | $2 \times 2 \times 128 = 512\text{ floats}$ ($1024\text{ B}$ in FP16) |
| **SSM State $S_P$ Floats per Layer** | $24 \times 64 \times 128 = 196,608\text{ floats}$ ($768$ KV tokens) | $24 \times 64 \times 256 = 393,216\text{ floats}$ ($768$ KV tokens) |
| **Model Weights VRAM (FP16)** | $\approx 1.06\text{ GB}$ | $\approx 3.10\text{ GB}$ |

### 2.2 Why We Report Budget as `% of Retained KV Cache` + Multi-Stream Serving Throughput

Because `Falcon-H1-1.5B-Deep` uses Grouped-Query Attention with $H_{kv}=2$ and $d_h=128$ ($512$ floats $= 1024\text{ B}$ FP16 per token per layer), a single sequence's KV cache at $P = 16,384$ on `1.5B-Deep` ($L=66$) occupies $66 \times 16,384 \times 1024\text{ B} \approx 1.11\text{ GB}$ (or $8.86\text{ GB}$ at batch size $8$ / multi-agent serving). Therefore:
1. **Standardized Reporting Metric:** Following `SnapKV`, `KVzip`, and `LORE-KV`, all primary curves report accuracy vs. **Retained KV Cache Ratio** $\rho_{\text{kv}} = K_{\text{total}} / P \in \{5\%, 10\%, 20\%, 35\%, 50\%, 100\%\}$, where the fixed SSM state $S_P$ is identical across all methods.
2. **Iso-Memory Consolidation Audit:** When testing `Distilled-Consolidate` on $k_{\text{cons}} \in \{32, 64, 128\}$ evicted tokens at budget $K_{\text{ret}}$, we explicitly compare against **both** `SA-SnapKV(K_ret)` **and** `SA-SnapKV(K_ret + k_cons)` to quantify what fraction of exact KV retention is recovered at zero additional KV memory.
3. **T4 FP16 Overflow Guard:** Because `Falcon-H1` was trained in `bfloat16` while Kaggle T4 GPUs (SM 7.5) only accelerate `float16` in hardware Tensor Cores, all Mamba-2 recurrences, $\mu$P-scaled projections, and state-survival calculations run inside an explicit `torch.float32` hook to guarantee zero FP16 overflow.

---

## 3. Complete Mathematical Specification

### 3.1 Exact Post-Conv / Post-$\mu$P Tensor Extraction in `FalconH1Mixer`

In `transformers.models.falcon_h1.modeling_falcon_h1.FalconH1Mixer`, the input hidden state $h_t \in \mathbb{R}^{d_{\text{model}}}$ undergoes $\mu$P input scaling, linear projection, 1D depthwise causal convolution (`d_conv = 4`), `SiLU` activation, and 5-vector $\mu$P multiplier scaling **before** entering the SSM selective scan. Hooking `in_proj` directly would extract un-convolved, un-activated tensors.

Instead, we extract the exact tensors fed into the selective scan at token $t \in \{0 \dots P-1\}$ in FP32:
1. Apply `in_proj` and `ssm_in_multiplier` to obtain `[gate, xBC_raw, dt_raw]`.
2. Pass `xBC_raw` through the mixer's exact causal `conv1d` and `act` (`SiLU`), then split into $(x_t, B_t, C_t)$ and apply the mixer's `ssm_multipliers`:
   $$x_{t,m} \in \mathbb{R}^{d_{\text{head,ssm}}}, \qquad B_t \in \mathbb{R}^{d_{\text{state}}}, \qquad C_t \in \mathbb{R}^{d_{\text{state}}}, \qquad \Delta_{t,m} = \text{softplus}(\text{dt}_{t,m} + b_{\text{dt},m}) \in \mathbb{R}_+$$
3. **Unit-Key Canonicalization of Mamba-2 State Writes:**
   In Mamba-2, the state recurrence for head $m \in \{1 \dots H_{\text{ssm}}\}$ is:
   $$S_{t,m} = a_{t,m} S_{t-1,m} + B_t^\top (\Delta_{t,m} x_{t,m}) \in \mathbb{R}^{d_{\text{state}} \times d_{\text{head,ssm}}}, \qquad a_{t,m} = \exp(-\Delta_{t,m} A_m) \in (0, 1)$$
   Define the unit-norm write key $\hat{b}_t$, effective value vector $u_{t,m}$, write energy $w_{t,m}$, and cumulative survival decay $\alpha_{t \to P, m}$:
   $$\hat{b}_t = \frac{B_t}{\|B_t\|_2 + \epsilon} \in \mathbb{R}^{d_{\text{state}}}, \qquad u_{t,m} = \Delta_{t,m} \|B_t\|_2 x_{t,m} \in \mathbb{R}^{d_{\text{head,ssm}}}$$
   $$w_{t,m} = \|u_{t,m}\|_2 = \Delta_{t,m} \|B_t\|_2 \|x_{t,m}\|_2, \qquad \alpha_{t \to P, m} = \exp\left(\sum_{\tau=t+1}^{P-1} \log a_{\tau,m}\right) \in (0, 1]$$
   Because $\hat{b}_t \hat{b}_t^\top = 1$, the terminal prefill state $S_{P,m}$ decomposes identically as:
   $$S_{P,m} = \sum_{t=0}^{P-1} \alpha_{t \to P, m} \hat{b}_t^\top u_{t,m}$$

---

### 3.2 Pillar 1: Realized State-Readout Decomposition & Calibrated Survival ($\hat{p}_{\text{surv}}$)

Multiplying $S_{P,m}$ on the left by token $t$'s unit write key $\hat{b}_t$ yields the **Exact Realized State-Readout Identity**:
$$\hat{b}_t S_{P,m} = \underbrace{\alpha_{t \to P, m} u_{t,m}}_{\text{Decayed Target Signal } s_{t,m}(P)} + \underbrace{\sum_{\tau \neq t} \alpha_{\tau \to P, m} \langle \hat{b}_t, \hat{b}_\tau \rangle u_{\tau, m}}_{\text{Realized Superposition Interference } \nu_{t,m}(P)}$$

Because $\hat{b}_t S_{P,m}$ and $\alpha_{t \to P, m} u_{t,m}$ are both known at the end of prefill $P$, the exact interference vector $\nu_{t,m}(P)$ is obtained via a single vector subtraction:
$$\nu_{t,m}(P) = \hat{b}_t S_{P,m} - \alpha_{t \to P, m} u_{t,m}$$

#### Three Time-Consistent Per-Token, Per-Head Survival Features
To avoid mixing undecayed and decayed quantities, we define three dimensionless features at position $P$:
1. **Decayed Signal-to-Interference Ratio ($\text{SIR}_{t,m}(P)$):**
   $$\text{SIR}_{t,m}(P) = \frac{\|\alpha_{t \to P, m} u_{t,m}\|_2^2}{\|\nu_{t,m}(P)\|_2^2 + \epsilon} = \frac{\alpha_{t \to P, m}^2 \|u_{t,m}\|_2^2}{\|\hat{b}_t S_{P,m} - \alpha_{t \to P, m} u_{t,m}\|_2^2 + \epsilon}$$
2. **Directional Readout Cosine ($c_{t,m}(P)$):**
   $$c_{t,m}(P) = \frac{\langle \hat{b}_t S_{P,m},\; u_{t,m} \rangle}{\|\hat{b}_t S_{P,m}\|_2 \|u_{t,m}\|_2 + \epsilon} \in [-1, 1]$$
3. **Log-Survival Factor ($\ell_{t,m}(P)$):**
   $$\ell_{t,m}(P) = \log(\alpha_{t \to P, m} + \epsilon) = \sum_{\tau=t+1}^{P-1} \log a_{\tau,m} \le 0$$

#### Calibrated Survival Probability ($\hat{p}_{\text{surv}}(t, l, m \mid P)$)
Rather than assuming an arbitrary heuristic formula, we fit a 4-parameter logistic calibration model $(w_1, w_2, w_3, b_0)$ on held-out state-only recall probes ($K_{\text{ret}}=0$) to predict the exact probability that a fact written at token $t$ can be recalled from $S_P$ alone:
$$\hat{p}_{\text{surv}}(t,l,m \mid P) = \sigma\Big( w_1 \log\big(\text{SIR}_{t,l,m}(P) + \epsilon\big) + w_2 c_{t,l,m}(P) + w_3 \ell_{t,l,m}(P) + b_0 \Big) \in (0, 1)$$
* **Zero-Parameter Training-Free Default:** When $(w_1=1, w_2=0, w_3=0, b_0=0)$, $\hat{p}_{\text{surv}}$ reduces analytically to the closed-form ratio:
  $$\rho_{\text{surv}}^{\text{closed}}(t,l,m \mid P) = \frac{\text{SIR}_{t,l,m}(P)}{1 + \text{SIR}_{t,l,m}(P)} = \frac{\alpha_{t \to P,m}^2 \|u_{t,m}\|_2^2}{\alpha_{t \to P,m}^2 \|u_{t,m}\|_2^2 + \|\nu_{t,m}(P)\|_2^2 + \epsilon} \in [0, 1)$$
* **Layer-Aggregated Write-Weighted Survival Probability ($\bar{p}_{\text{surv}, t, l}(P)$):**
  Across the $H_{\text{ssm}}$ Mamba-2 heads in layer $l$, we weight each head's survival probability by how strongly token $t$ was written into that head ($w_{t,l,m} = \|u_{t,l,m}\|_2$):
  $$\bar{p}_{\text{surv}, t, l}(P) = \frac{\sum_{m=1}^{H_{\text{ssm}}} w_{t,l,m} \, \hat{p}_{\text{surv}}(t,l,m \mid P)}{\sum_{m=1}^{H_{\text{ssm}}} w_{t,l,m} + \epsilon} \in (0, 1)$$

---

### 3.3 Pillar 2: Token-Level State-Aware KV Eviction (`SA-SnapKV`) & Risk-Adaptive Budgets

#### 1. Token-Level Retention Score ($s_{t,l}$)
At the switch point $P$ (end of document context prefill), we always retain the 4 attention sinks $\{0,1,2,3\}$ and the local observation window $\mathcal{W}_{\text{obs}} = [P - W_{\text{obs}}, P - 1]$ ($W_{\text{obs}} = 64$). For every candidate prefix token $t \in \{4, \dots, P - W_{\text{obs}} - 1\}$ in layer $l$:
* Compute the **Observation-Window Attention Utility ($D_{t,l}$)** using value-norm-weighted attention mass from queries $q \in \mathcal{W}_{\text{obs}}$ pooled over query groups (following `SnapKV` + value-norm scaling):
  $$D_{t,l} = \frac{1}{W_{\text{obs}} H_q} \sum_{q \in \mathcal{W}_{\text{obs}}} \sum_{h=1}^{H_q} A_{q \to t}^{(l,h)} \|v_{t, l, g(h)}\|_2$$
* Multiply by the token's **State-Survival Deficit** $1 - \bar{p}_{\text{surv}, t, l}(P)$:
  $$s_{t,l} = D_{t,l} \cdot \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$

#### 2. Risk-Adaptive Layerwise KV Budget Allocation ($K_l$)
In hybrid models, some layers have long-horizon SSM heads ($\bar{p}_{\text{surv}, t, l}(P) \approx 0.8$) while other layers decay rapidly ($\bar{p}_{\text{surv}, t, l}(P) \approx 0.05$). Given a total target KV token budget $K_{\text{total}} = L \cdot K_{\text{avg}}$ (or a target residual risk ceiling $\tau_{\text{risk}}$):
* Define Layer $l$'s **Expected State Forgetting Risk**:
  $$\mathcal{R}_l(P) = \sum_{t=4}^{P - W_{\text{obs}} - 1} D_{t,l} \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$
* Allocate layer $l$'s retained candidate budget $K_l$ (clamped to $[K_{\min}, P - W_{\text{obs}} - 4]$) proportionally to $\mathcal{R}_l(P)$:
  $$K_l = \text{Round}\left( K_{\text{total}} \cdot \frac{\mathcal{R}_l(P)}{\sum_{l'=1}^L \mathcal{R}_{l'}(P) + \epsilon} \right)$$
* Retain the top-$K_l$ tokens $\mathcal{T}_{\text{ret}, l} = \text{TopK}_{t}(s_{t,l}; K_l)$ in layer $l$'s Attention KV cache.

#### 3. Streaming Chunked-Prefill Variant (`SA-SnapKV-Stream`, Stretch Module)
To answer the systems question *"Why evict after materializing the full prefill KV cache?"*, during chunked prefill of length $P$ with chunk size $C_{\text{prefill}} = 2,048$:
* At the end of each prefill chunk $c$, compute a conservative lower bound on terminal survival using the running SSM state $S_{c,l,m}$ and minimum per-head decay rate, and immediately evict low-$s_{t,l}$ tokens so **peak prefill KV memory never exceeds $\max(K_l + C_{\text{prefill}})$**.

---

### 3.4 Pillar 3: Teacher-Distilled Ridge State Consolidation (`Distilled-Consolidate`)

Let $\mathcal{T}_{\text{cons}, l} = \{t_1, \dots, t_k\}$ be the **top-$k$ evicted runner-up tokens** in layer $l$ (tokens ranked $K_l + 1 \dots K_l + k$ by $s_{t,l}$ that must be evicted from the Attention KV cache, with $k \in \{32, 64\}$).

#### 1. Batched Ridge-Regularized Delta-Rule Consolidation
Stack the unit write keys $\hat{B}_l = [\hat{b}_{t_1}; \dots; \hat{b}_{t_k}] \in \mathbb{R}^{k \times d_{\text{state}}}$ and effective values $U_{l,m} = [u_{t_1, l, m}; \dots; u_{t_k, l, m}] \in \mathbb{R}^{k \times d_{\text{head,ssm}}}$.
For each layer $l$ and SSM head $m$, let $\gamma_{l,m} = \sigma(\tilde{\gamma}_{l,m}) \in (0, 1)$ be a learnable consolidation gate and $\lambda_{l,m} = \exp(\tilde{\lambda}_{l,m}) > 0$ be a learnable ridge parameter. Define the deficit-weighted diagonal gain matrix:
$$\Gamma_{l,m} = \gamma_{l,m} \cdot \text{diag}\Big(1 - \hat{p}_{\text{surv}}(t_1, l, m \mid P), \dots, 1 - \hat{p}_{\text{surv}}(t_k, l, m \mid P)\Big) \in [0, 1]^{k \times k}$$
Update $S_{P,l,m}$ at position $P$ via a single $k \times k$ Cholesky solve (`torch.linalg.solve`):
$$S_{P,l,m}^{\text{cons}} = S_{P,l,m} + \hat{B}_l^\top \big(\hat{B}_l \hat{B}_l^\top + \lambda_{l,m} I_k\big)^{-1} \Gamma_{l,m} \big(U_{l,m} - \hat{B}_l S_{P,l,m}\big)$$

#### 2. Proposition 1 (Exact Spectral Perturbation & Collateral Damage Bound — Corrected)
Let $\hat{B}_l = W \Sigma V^\top$ be the singular value decomposition of $\hat{B}_l \in \mathbb{R}^{k \times d_{\text{state}}}$ with singular values $\sigma_1 \ge \dots \ge \sigma_k > 0$, and let $\gamma_{\max, l, m} = \max_{r \in \{1\dots k\}} (\Gamma_{l,m})_{r,r} \le \gamma_{l,m}$. Then:
1. **Exact Pseudo-Inverse Operator Norm:**
   $$\left\| \hat{B}_l^\top \big(\hat{B}_l \hat{B}_l^\top + \lambda_{l,m} I_k\big)^{-1} \right\|_2 = \max_{i \in \{1 \dots k\}} \frac{\sigma_i}{\sigma_i^2 + \lambda_{l,m}} \le \frac{1}{2\sqrt{\lambda_{l,m}}}$$
2. **Zero Perturbation on Orthogonal Memories:** For any stored token $\tau \notin \mathcal{T}_{\text{cons}, l}$ whose write key $\hat{b}_\tau$ is orthogonal to $\text{rowspan}(\hat{B}_l)$ ($V^\top \hat{b}_\tau^\top = 0$), readout is invariant: $\hat{b}_\tau S_{P,l,m}^{\text{cons}} = \hat{b}_\tau S_{P,l,m}$.
3. **Worst-Case Collateral Perturbation Bound on Arbitrary Stored Memories:** For any stored token $\tau \notin \mathcal{T}_{\text{cons}, l}$ with subspace overlap $\theta_\tau = \|V^\top \hat{b}_\tau^\top\|_2 \in [0, 1]$, the readout perturbation is strictly bounded by:
   $$\big\| \hat{b}_\tau S_{P,l,m}^{\text{cons}} - \hat{b}_\tau S_{P,l,m} \big\|_2 \le \theta_\tau \cdot \gamma_{\max, l, m} \left(\max_{i \in \{1 \dots k\}} \frac{\sigma_i}{\sigma_i^2 + \lambda_{l,m}}\right) \big\| U_{l,m} - \hat{B}_l S_{P,l,m} \big\|_2$$

#### 3. 300-Step Full-KV Teacher Distillation of $\{\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m}\}$
Instead of hand-tuning $\gamma$ and $\lambda$, we freeze 100% of the pretrained `Falcon-H1` backbone and optimize only $\{\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m}\}$ ($36 \times 24 \times 2 = 1,728$ scalars on `0.5B`; $66 \times 24 \times 2 = 3,168$ scalars on `1.5B-Deep`) for 300 steps (~15 minutes on 2× T4) on prospective context-question splits by minimizing the token-level KL divergence between the Full-KV teacher and the Evicted+Consolidated student over suffix tokens $t \ge P$:
$$\mathcal{L}_{\text{distill}}(\tilde{\gamma}, \tilde{\lambda}) = \frac{1}{|T_{\text{suffix}}|} \sum_{t \in T_{\text{suffix}}} \text{KL}\Big( p_{\text{Full-KV}}(\cdot \mid x_{\le t}) \;\Big\|\; p_{\text{Evict+Cons}}(\cdot \mid x_{\le t}; \tilde{\gamma}, \tilde{\lambda}) \Big)$$
* **Automatic Head Selection:** On long-horizon heads where $S_{P,l,m}$ already holds global context and consolidation causes collateral interference ($\theta_\tau > 0$), $\nabla \mathcal{L}_{\text{distill}}$ automatically drives $\gamma_{l,m} \to 0$. On fast-decaying heads where evicted tokens were lost, $\gamma_{l,m}$ stays positive.

---

## 4. Benchmark Suite & Prospective Multi-Query Evaluation Protocol

### 4.1 Fixing the RULER "Question-in-Prefill" Trap (Prospective Split)
In standard single-turn RULER/LongBench prompts, the query sits at the very end of the prompt ($\le P$), allowing `SnapKV` to see the exact test question before evicting the KV cache. To evaluate both **prospective (blind/multi-turn)** and **query-aware** eviction honestly, we enforce two strict evaluation modes:
1. **Mode A — Prospective / Multi-Turn Eviction (`SCBench` Protocol, Primary):**
   The document context ($0 \dots P-1$) is prefilled first; **KV eviction and state consolidation occur at $P$ before any test question is revealed**; then $Q \in \{1, \dots, 4\}$ sequential follow-up questions arrive in the suffix ($t \ge P$).
2. **Mode B — Standard Prompt-End Eviction:**
   Standard single-turn RULER/LongBench split (reported for direct comparison against published `SnapKV` and `PyramidKV` numbers).

### 4.2 Complete Benchmark & Baseline Matrix

#### Four Diagnostic & Real-World Benchmarks
1. **Gate 0 Diagnostic Suite (`MQAR` + `R5a Decay Trap` + `R5b Key-Collision Trap` + `FineWeb-Edu Natural Prefix Recall`):**
   * **R5a (Decay Trap):** Set A of 16 KV pairs placed at $t \in [64, 256]$, Set B of 16 KV pairs placed at $t \in [P-256, P-64]$, with high-$\Delta_t$ filler in between. Write-time surprise (`HOLA`) and chunk drift (`DeltaS`) score Set A and Set B equally, wasting 50% of the KV budget on Set B (which $S_P$ already remembers).
   * **R5b (Key-Collision Interference Trap):** Set A and Set B placed at the **exact same intermediate distance** from $P$ (controlling for $\alpha_{t \to P}$ and token age!), where Set A's write keys are perturbed by subsequent high-cosine distractor writes ($\langle \hat{b}_t, \hat{b}_\tau \rangle > 0.7$) while Set B's write keys lie in an orthogonal subspace. Decay-only ($1 - \alpha_{t \to P}$) scores Set A and Set B identically, whereas $\text{SIR}_{t,m}(P)$ and $c_{t,m}(P)$ detect Set A's superposition corruption $\nu_{t,m}(P)$.
   * **Natural-Text Age-Controlled Recall:** Sample 500 natural FineWeb-Edu passages ($P=4,096$), identify rare entity/number tokens at matched token ages $P - t \in \{256, 512, 1024, 2048\}$, and measure partial correlation between $\hat{p}_{\text{surv}}(t \mid P)$ and state-only cloze recall accuracy **controlling for token age $(P - t)$**.
2. **Prospective `RULER` (4K, 8K, 16K on `Falcon-H1-1.5B-Deep` & `0.5B`):** Single/Multi-Needle (`NIAH-S1/S2/MK1/MK2`), Variable Tracking (`VT`), and Multi-Query Aggregation (`CWE`).
3. **`SCBench` Multi-Turn / Multi-Request Shared-Context Benchmark (`arXiv:2412.10319`):** Multi-turn QA and code/log needle retrieval over a shared prefilled context.
4. **`BABILong` / `LongBench-E` Subset:** Multi-hop reasoning (`QA1–QA3`) and long-document QA at 4K–16K.

#### Ten Policies Compared Head-to-Head on Identical Backbones
1. **`Full-KV` (100% KV Cache)** — Upper bound.
2. **`State-Only` (0% Prefix KV Cache, $S_P$ + Sinks + Local Window only)** — Parametric lower bound.
3. **`Recent / Sliding Window` (SWA)** — Recency baseline.
4. **`H2O` (Heavy-Hitter Oracle)** — Cumulative attention baseline.
5. **`SnapKV` (Observation-Window Attention, `Demand-Only`)** — Primary attention-only baseline.
6. **`KVzip-Proxy` (`arXiv:2505.23416`) / Context-Reconstruction Scoring** — Query-agnostic KV eviction baseline.
7. **`LORE-KV` ($M=1$ Greedy Lookahead Continuation, `arXiv:2610.07643`)** — Lookahead query baseline.
8. **`HOLA-Write` (`arXiv:2607.02303`) & `DeltaS-Port` (`arXiv:2609.27470`)** — Write-time surprise $\|u_{t,m} - \hat{b}_t S_{t-1,m}\|_2$ and normalized state drift $\|S_{c,m} - S_{c-1,m}\|_F / \|S_{c-1,m}\|_F$.
9. **`DASC-Decay` (`arXiv:2608.30386`, `Decay-Only`)** — Pure decay deficit $D_{t,l} \cdot (1 - \bar{\alpha}_{t \to P, l})$.
10. **Ours (`SA-SnapKV` & `SA-SnapKV + Distilled-Consolidate`)** — Evaluated with uniform vs. risk-adaptive budgets $K_l$, and directly compared against **`SA-SnapKV(K_l + k_eq)`** (iso-memory extra KV tokens).

---

## 5. Mandatory 10-Test Verification Suite (`pytest tests/`)

Before running any benchmark sweep, all 10 unit tests must pass 100% green:

1. **Test 1 — Exact Realized State-Readout Identity:** Verify on `FalconH1Mixer` in FP32 that $\hat{b}_t S_{P,m}$ equals $\alpha_{t \to P, m} u_{t,m} + \sum_{\tau \neq t} \alpha_{\tau \to P, m} \langle \hat{b}_t, \hat{b}_\tau \rangle u_{\tau, m}$ within $< 10^{-5}$ max absolute error.
2. **Test 2 — `FalconH1Mixer` Post-Conv / Post-$\mu$P Hook Verification:** Verify that running our FP32 step-by-step recurrence using the extracted $(x_t, B_t, \Delta_t, A_m)$ reproduces `cache_params.ssm_states[layer_idx]` at position $P$ within $< 10^{-4}$ relative Frobenius error.
3. **Test 3 — Time-Consistent $\text{SIR}_{t,m}(P)$ & Single-Write Limit:** Verify that for an isolated write at $t_0$ followed by zero writes ($u_\tau = 0$ for $\tau > t_0$), $\|\nu_{t_0, m}(P)\|_2 = 0$, $c_{t_0, m}(P) = 1.0$, and $\rho_{\text{surv}}^{\text{closed}}(t_0, m \mid P) \to 1.0$ regardless of $\alpha_{t_0 \to P, m} > 0$.
4. **Test 4 — R5b Equal-Age Interference Separation:** Verify on synthetic equal-age pairs ($P - t_A = P - t_B$) where $t_A$ suffers later key collision ($\langle \hat{b}_{t_A}, \hat{b}_{\text{dist}} \rangle = 0.8$) that $\alpha_{t_A \to P} = \alpha_{t_B \to P}$, while $\text{SIR}_{t_A}(P) < 0.2 \cdot \text{SIR}_{t_B}(P)$ and $c_{t_A}(P) < c_{t_B}(P)$.
5. **Test 5 — Proposition 1 Spectral Perturbation Bound:** Verify numerically across 100 random key matrices $\hat{B}_l \in \mathbb{R}^{k \times 128}$ that $\|\hat{B}_l^\top (\hat{B}_l \hat{B}_l^\top + \lambda I_k)^{-1}\|_2 = \max_i \frac{\sigma_i}{\sigma_i^2 + \lambda}$, that orthogonal queries $\hat{b}_\tau \perp \text{rowspan}(\hat{B}_l)$ experience $< 10^{-6}$ perturbation, and that non-orthogonal queries strictly obey the Proposition 1 inequality.
6. **Test 6 — Risk-Adaptive Budget Conservation:** Verify that $\sum_{l=1}^L K_l = K_{\text{total}}$ and $K_l \in [K_{\min}, P - W_{\text{obs}} - 4]$ for all layers.
7. **Test 7 — Prospective Eviction Causality (Mode A):** Verify that perturbing any suffix question/answer token $t \ge P$ leaves $\hat{p}_{\text{surv}}(t', l, m \mid P)$, $s_{t', l}$, $\mathcal{T}_{\text{ret}, l}$, and $S_{P,l,m}^{\text{cons}}$ bit-identical.
8. **Test 8 — Teacher KL Distillation Gradient Isolation:** Verify during the 300-step consolidation distillation that $\nabla \mathcal{L}_{\text{distill}}$ is non-zero on $\{\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m}\}$ and strictly `None` / $0$ on all pretrained `Falcon-H1` parameters.
9. **Test 9 — FP16 Inference / FP32 Hook Stability on T4:** Run 50 long sequences ($P=8,192$) through `Falcon-H1-0.5B` and `1.5B-Deep` on T4 in `float16` with the FP32 survival/consolidation hook and verify zero `NaN`/`Inf` values in logits or states.
10. **Test 10 — Simulated Masked Eviction vs. Physical KV Cache Slicing Equivalence:** Verify that indexing `past_key_values` via `torch.gather` on $\mathcal{T}_{\text{ret}, l} \cup \mathcal{S}_{\text{sink}} \cup \mathcal{W}_{\text{obs}}$ produces bit-identical suffix logits to applying an additive `-inf` attention mask over evicted tokens.

---

## 6. Phased Gated Execution Roadmap (24 Days Total on Kaggle $2 \times \text{T4}$)

### Phase 0: Environment, `FalconH1Mixer` Hook Verification & 10-Test Gate (Days 1–3)
* **Tasks:**
  1. Implement `falcon_h1_state_hook.py` to extract post-`conv1d`, post-`SiLU`, post-$\mu$P $(x_t, B_t, C_t, \Delta_t)$ and verify exact agreement with `cache_params.ssm_states` at $P$ (Test 2).
  2. Run `pytest tests/test_v13_suite.py` and pass **10/10 unit tests**.
  3. Measure Full-KV baseline accuracy and VRAM on `Falcon-H1-0.5B-Instruct` and `Falcon-H1-1.5B-Deep-Instruct` at $P \in \{4\text{K}, 8\text{K}, 16\text{K}\}$.

### Phase 1 (GATE 0 & GATE 1): Survival Calibration & Age-Controlled Audit (Days 4–8)
* **Tasks:**
  1. **Gate 0 (Diagnostic Separation):** Evaluate `SIR + Cosine` ($\hat{p}_{\text{surv}}$), `DASC-Decay` ($\alpha_{t \to P}$), `DeltaS-Port`, and `HOLA-Write` on **MQAR**, **R5a (Decay Trap)**, and **R5b (Equal-Age Key-Collision Trap)**.
  2. **Gate 1 (Natural-Text Calibration & Age Control on `Falcon-H1`):**
     * Run `State-Only` ($0\%$ prefix KV) on 500 natural prefix-recall passages and multi-needle prompts on `Falcon-H1-0.5B` & `1.5B-Deep`.
     * Fit the 4-parameter logistic calibrator $\hat{p}_{\text{surv}}(t,l,m \mid P)$ and report **per-head AUROC, ECE Calibration Curves, Brier Score, and Partial Correlation with state-only recall controlling for token age $(P - t)$**.
* **Hard Decision Gate 1:**
  * Does $\hat{p}_{\text{surv}}$ beat `DeltaS-Port` and `HOLA-Write` by $>15\%$ recall on R5a/R5b AND achieve statistically significant positive partial correlation ($p < 0.01$) with state-only recall when controlling for token age $(P - t)$?
  * *If $\text{SIR}_{t,m}(P)$ adds $<2\%$ AUROC over $\log \alpha_{t \to P, m}$ on natural text, simplify $\hat{p}_{\text{surv}}$ to calibrated per-head log-decay $\sigma(w_3 \ell_{t,l,m}(P) + b_0)$ and report the interference analysis as an empirical finding.*

### Phase 2: Token-Level `SA-SnapKV` & Risk-Adaptive Budget Allocation (Days 9–14)
* **Tasks:**
  1. Evaluate the 4-way token-level ablation (`Demand-Only`, `Decay-Only`, `Survival-Only`, and `Demand × Survival Deficit`) across KV budgets $\rho_{\text{kv}} \in \{5\%, 10\%, 20\%, 35\%, 50\%\}$ on `Falcon-H1-0.5B` and `1.5B-Deep` across Prospective `RULER`, `SCBench`, and `BABILong`.
  2. Compare **Risk-Adaptive Layer Budgets ($K_l \propto \mathcal{R}_l(P)$)** against **Uniform Layer Budgets ($K_l = K_{\text{avg}}$)** and `PyramidKV`.
  3. Benchmark the **Streaming Chunked-Prefill Variant (`SA-SnapKV-Stream`)** on 16K context to demonstrate peak prefill KV memory reduction.
* **Milestone Guarantee:** At the end of Day 14, Pillar 1 + Pillar 2 already constitute a complete, self-contained paper (*"Calibrated State-Survival and Risk-Adaptive KV Eviction in Hybrid Language Models"*).

### Phase 3 (GATE 2): Teacher-Distilled State Consolidation & Iso-Memory Audit (Days 15–20)
* **Tasks:**
  1. **Gate 2 Collateral Damage & $(k, \gamma, \lambda)$ Audit:** Before training gates, sweep $k \in \{16, 32, 64, 128\}$ on MQAR and RULER-MK2 and directly measure **both** (a) recall recovery on the $k$ consolidated runner-up tokens and (b) **empirical collateral recall drop on untouched stored tokens** $\tau \notin \mathcal{T}_{\text{cons}, l}$ vs. Proposition 1's bound.
  2. **300-Step Teacher KL Distillation:** Train the $1,728$ (`0.5B`) / $3,168$ (`1.5B-Deep`) scalar parameters $\{\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m}\}$ on held-out prospective context-question splits (~15 minutes on 2× T4).
  3. **Iso-Memory Fair Comparison:** Compare `SA-SnapKV(K_ret) + Distilled-Consolidate(k)` against **both** `SA-SnapKV(K_ret)` and **`SA-SnapKV(K_ret + k)`** (spending the extra memory on $k$ additional KV tokens).
* **Hard Decision Gate 2:**
  * Keep `Distilled-Consolidate` as a headline contribution **only if** it beats `SA-SnapKV(K_ret)` with statistical significance ($p < 0.05$ paired bootstrap) at low budgets ($\rho_{\text{kv}} \in \{5\%, 10\%, 20\%\}$) without degrading untouched state memories. If consolidation gains are marginal, report Gate 2 as a rigorous negative/capacity-limit study and lead with Pillar 1 + Pillar 2.

### Phase 4: Paper Figures, Tables & Reproducible Code Release (Days 21–24)
* **Tasks:**
  1. Generate the 6 primary paper figures:
     * **Figure 1:** Layer-by-head survival heatmaps $\hat{p}_{\text{surv}}(t,l,m \mid P)$ on `Falcon-H1-1.5B-Deep` showing fast-decaying vs. long-horizon heads and why `DeltaS`/`HOLA` fail on R5a/R5b.
     * **Figure 2:** Reliability diagrams (ECE), AUROC, and age-controlled partial correlation of $\hat{p}_{\text{surv}}$ vs. state-only recall.
     * **Figure 3:** Prospective `RULER` & `SCBench` accuracy vs. Retained KV Ratio ($\rho_{\text{kv}} \in \{5\% \dots 50\%\}$) across all 10 policies on `Falcon-H1-1.5B-Deep` and `0.5B`.
     * **Figure 4:** Risk-Adaptive Layer Budget Allocation $K_l$ profile across all 66 layers of `Falcon-H1-1.5B-Deep` vs. Uniform and `PyramidKV`.
     * **Figure 5:** Gate 2 Consolidation Curve — Consolidated token recovery vs. empirical collateral perturbation on stored memories + learned $\gamma_{l,m}$ distribution vs. head decay rate $A_m$.
     * **Figure 6:** Peak VRAM & Throughput of `SA-SnapKV-Stream` during 16K–32K chunked prefill on a single 16GB T4 GPU.

---

## 7. Repository Structure (Lean, Zero-Bloat V13 Codebase)

```text
hybrid_state_eviction_v13/
├── configs/
│   ├── falcon_h1_0.5b.yaml          # 36-layer Falcon-H1-0.5B-Instruct config
│   └── falcon_h1_1.5b_deep.yaml     # 66-layer Falcon-H1-1.5B-Deep-Instruct config
├── hooks/
│   ├── falcon_h1_state_hook.py      # Exact post-conv1d / post-SiLU / post-muP FP32 tensor & S_P extractor
│   └── survival_metrics.py          # Exact realized interference nu_{t,m}(P), SIR, cosine c_{t,m}, and Platt calibrator
├── eviction/
│   ├── sa_snapkv.py                 # Token-level Demand x (1 - p_surv) + 4-way ablation modes
│   ├── adaptive_budget.py           # Risk-adaptive per-layer KV budget allocator K_l
│   ├── streaming_prefill.py         # Chunked-prefill streaming eviction (SA-SnapKV-Stream)
│   └── baselines.py                 # SnapKV, H2O, KVzip-proxy, LORE-KV (M=1), HOLA-Write, DeltaS-Port, DASC-Decay
├── consolidation/
│   ├── ridge_consolidate.py         # Batched Cholesky ridge delta-rule update + Proposition 1 bound checker
│   └── distill_gates.py             # 300-step Full-KV teacher KL distillation for {gamma_{l,m}, lambda_{l,m}}
├── benchmarks/
│   ├── gate0_diagnostics.py         # MQAR, R5a (Decay Trap), R5b (Equal-Age Interference), Natural Age-Controlled Recall
│   ├── prospective_ruler.py         # Context-first -> Evict at P -> Multi-query suffix RULER harness
│   └── scbench_babilong.py          # SCBench multi-turn & BABILong evaluation harness
└── tests/
    └── test_v13_suite.py            # Mandatory 10 unit tests (Tests 1-10)
```