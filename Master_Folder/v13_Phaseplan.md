# MASTER PLAN V13 — PRE-BUILD IMPLEMENTATION & EXECUTION BLUEPRINT

**Purpose:** This document is the step-by-step engineering reference manual for building, verifying, and running the `hybrid_state_eviction_v13` codebase on Kaggle Free Tier ($2 \times \text{NVIDIA T4 16GB GPUs}$). When writing code for any phase, refer directly to that phase's section for the exact file targets, tensor shapes, mathematical formulas, logical steps, self-correction rules, and handover verification gates.

---

## Global Rule: Agent Mathematical Self-Correction & Comment Protocol

Every phase in this blueprint enforces a strict **Mathematical Upgrade & Non-Breaking Comment Protocol** for any human developer or coding agent writing the code:

1. **Authorization to Upgrade or Correct Logic:** If, while writing the code for a phase, the developer or coding agent identifies a mathematical improvement, numerical stability refinement (e.g., log-sum-exp stabilization, Cholesky diagonal jitter, exact GQA head broadcasting, or integer rounding conservation), or logical correction that improves accuracy, diagnostic precision, or downstream performance, the agent is **explicitly authorized and required** to implement the improvement.
2. **Mandatory In-Code Comment Isolation:** Whenever a mathematical or logical adjustment is made—or when an original formula is superseded—the agent **must** keep the original logic commented out or documented right above the active code using the following standardized comment block so that no broken or mismatched assumptions ever propagate into the next phase:

```python
# ==============================================================================
# [MATH-UPGRADE | Phase X -> Phase Y]
# Original Spec: <Write the original formula or logic here>
# Upgraded Spec: <Write the improved/corrected mathematical formula here>
# Reason:        <Explain the numerical/mathematical/diagnostic improvement>
# Contract Check: <Confirm output tensor shapes, dtypes, and invariants remain intact>
# ==============================================================================
```

3. **Zero Silent Breaking Changes:** No mathematical upgrade in Phase $X$ is permitted to alter the function signatures, dictionary keys, tensor shapes, or data types expected by Phase $X+1$.

---

## Phase 0: Environment Setup, Hardware Guards & Configuration Scaffolding

### 0.1 Focus, Features & Techniques
* **Primary Focus:** Establish the directory tree, model configuration YAML files for `Falcon-H1-0.5B-Instruct` and `Falcon-H1-1.5B-Deep-Instruct`, exact memory accounting utilities, and the T4 mixed-precision execution guard.
* **Key Technique — T4 FP16/FP32 Split Execution Guard:** `Falcon-H1` models were pretrained in `bfloat16`, whereas Kaggle NVIDIA T4 GPUs (Turing SM 7.5) only support hardware Tensor Core acceleration in `float16`. Running Mamba-2 selective scans and $\mu$P multipliers directly in `float16` causes exponent overflow. We load model weights in `torch.float16` (`device_map="auto"` across $2 \times \text{T4}$) while forcing all Mamba-2 state recurrences, $\mu$P-scaled projections, and survival computations into `torch.float32`.

### 0.2 Files to Create in Phase 0
1. `configs/falcon_h1_0.5b.yaml`
2. `configs/falcon_h1_1.5b_deep.yaml`
3. `hooks/__init__.py`, `eviction/__init__.py`, `consolidation/__init__.py`, `benchmarks/__init__.py`, `tests/__init__.py`

### 0.3 Mathematical & Architectural Specifications
For any sequence prefix of length $P$, our configuration and memory accounting module must enforce the exact architectural constants:

* **`Falcon-H1-0.5B-Instruct` (Fast Dev & Gate 1–2):**
  $$L = 36, \quad d_{\text{model}} = 1024, \quad H_q = 8, \quad H_{kv} = 2, \quad d_h = 64$$
  $$H_{\text{ssm}} = 24, \quad d_{\text{head,ssm}} = 64, \quad d_{\text{state}} = 128, \quad G_{\text{ssm}} = 1, \quad d_{\text{conv}} = 4$$
* **`Falcon-H1-1.5B-Deep-Instruct` (Primary Benchmark):**
  $$L = 66, \quad d_{\text{model}} = 1280, \quad H_q = 6, \quad H_{kv} = 2, \quad d_h = 128$$
  $$H_{\text{ssm}} = 24, \quad d_{\text{head,ssm}} = 64, \quad d_{\text{state}} = 256, \quad G_{\text{ssm}} = 1, \quad d_{\text{conv}} = 4$$

#### Exact Per-Layer Memory Accounting Formulas
1. **Attention KV Floats per Token per Layer:**
   $$\text{Floats}_{\text{KV/token/layer}} = 2 \times H_{kv} \times d_h = \begin{cases} 2 \times 2 \times 64 = 256 \text{ floats } (512 \text{ Bytes in FP16}) & \text{for } 0.5\text{B} \\ 2 \times 2 \times 128 = 512 \text{ floats } (1024 \text{ Bytes in FP16}) & \text{for } 1.5\text{B-Deep} \end{cases}$$
2. **Recurrent State $S_P$ Floats per Layer & Token-Equivalent Size:**
   $$\text{Floats}_{S_P\text{/layer}} = H_{\text{ssm}} \times d_{\text{head,ssm}} \times d_{\text{state}} = \begin{cases} 24 \times 64 \times 128 = 196,608 \text{ floats } (768 \text{ KV tokens}) & \text{for } 0.5\text{B} \\ 24 \times 64 \times 256 = 393,216 \text{ floats } (768 \text{ KV tokens}) & \text{for } 1.5\text{B-Deep} \end{cases}$$
3. **Retained KV Cache Ratio ($\rho_{\text{kv}}$):**
   Given a prefix length $P$, sink count $N_{\text{sink}} = 4$, observation window $W_{\text{obs}} = 64$, and retained candidate budget $K_{\text{total}} = \sum_{l=1}^L K_l$, the standardized retention ratio is:
   $$\rho_{\text{kv}} = \frac{K_{\text{total}}}{L \cdot P} \in \{0.05, 0.10, 0.20, 0.35, 0.50, 1.00\}$$

### 0.4 Phase 0 Agent Self-Correction Instruction
* **Instruction:** When implementing the YAML loader and T4 model initializer, if the agent detects that `transformers` version differences alter how `FalconH1ForCausalLM` initializes `HybridMambaAttentionDynamicCache` or casts `A_log` / `dt_bias`, the agent must upgrade the initialization wrapper to explicitly keep `A_log`, `D`, and `dt_bias` in `torch.float32` and log the adjustment using a `# [MATH-UPGRADE | Phase 0 -> Phase 1]` comment block.

### 0.5 Connection & Handover to Phase 1
* **Exit Gate:** Both models load cleanly on Kaggle $2 \times \text{T4}$ in `torch.float16` without OOM at $P = 8,192$ (Test 9 baseline check).
* **Handover Artifact:** Verified config dictionaries and initialized `FalconH1ForCausalLM` + `HybridMambaAttentionDynamicCache` objects passed into Phase 1's `FalconH1Mixer` hook.

---

## Phase 1: Exact Post-Conv / Post-$\mu$P `FalconH1Mixer` State Hook & Canonicalization

### 1.1 Focus, Features & Techniques
* **Primary Focus:** Build `hooks/falcon_h1_state_hook.py` to intercept every layer's `FalconH1Mixer` forward pass during prefill ($t \in \{0, \dots, P-1\}$) and extract the exact post-convolution, post-`SiLU`, post-$\mu$P-scaled SSM tensors $(x_t, B_t, C_t, \Delta_t)$ alongside the terminal recurrent state $S_P$.
* **Key Technique — Post-Conv / Post-$\mu$P Interception:** Hooking `in_proj` directly is mathematically wrong because `FalconH1Mixer` applies `ssm_in_multiplier`, 1D depthwise causal convolution (`conv1d` with kernel size $d_{\text{conv}}=4$), `SiLU` activation, and a 5-vector `ssm_multipliers` scaling *before* the selective scan. We hook immediately after those transformations and canonicalize the Mamba-2 state recurrence into unit-norm write keys $\hat{b}_t$ and effective value vectors $u_{t,m}$.

### 1.2 File to Create in Phase 1
* `hooks/falcon_h1_state_hook.py`

### 1.3 Mathematical & Logical Specification

#### Step 1: Exact Post-Conv / Post-$\mu$P Tensor Extraction (in `torch.float32`)
For each layer $l \in \{1, \dots, L\}$ and token position $t \in \{0, \dots, P-1\}$, given input hidden state $h_t \in \mathbb{R}^{d_{\text{model}}}$:
1. Apply the $\mu$P input multiplier and linear projection:
   $$z_{\text{proj}, t} = \text{in\_proj}\big(\mu_{\text{in}} \cdot h_t\big)$$
2. Split $z_{\text{proj}, t}$ into gating branch, convolution branch $\text{xBC}_{\text{raw}, t}$, and time-step branch $\text{dt}_{\text{raw}, t}$.
3. Apply causal 1D depthwise convolution (`d_conv = 4`) followed by `SiLU` activation on the full sequence $\text{xBC}_{\text{raw}, 0:P-1}$:
   $$\text{xBC}_{\text{conv}, t} = \text{SiLU}\Big(\text{conv1d}(\text{xBC}_{\text{raw}})_t\Big)$$
4. Split $\text{xBC}_{\text{conv}, t}$ into $(x_{\text{raw}, t}, B_{\text{raw}, t}, C_{\text{raw}, t})$ and apply the mixer's exact $\mu$P `ssm_multipliers` $(\mu_z, \mu_x, \mu_B, \mu_C, \mu_{\text{dt}})$:
   $$x_{t,m} = \mu_x \cdot x_{\text{raw}, t, m} \in \mathbb{R}^{d_{\text{head,ssm}}}, \quad B_t = \mu_B \cdot B_{\text{raw}, t} \in \mathbb{R}^{d_{\text{state}}}, \quad C_t = \mu_C \cdot C_{\text{raw}, t} \in \mathbb{R}^{d_{\text{state}}}$$
   $$\Delta_{t,m} = \text{softplus}\big(\mu_{\text{dt}} \cdot \text{dt}_{\text{raw}, t, m} + b_{\text{dt}, m}\big) \in \mathbb{R}_+$$
   where $m \in \{1, \dots, H_{\text{ssm}}\}$ indexes the Mamba-2 heads and $A_m = \exp(\text{A\_log}_m) > 0$ is the positive decay parameter for head $m$.

#### Step 2: Mamba-2 State Recurrence & Unit-Key Canonicalization
The native Mamba-2 state recurrence for head $m$ at step $t$ is:
$$S_{t,m} = a_{t,m} S_{t-1,m} + B_t^\top (\Delta_{t,m} x_{t,m}) \in \mathbb{R}^{d_{\text{state}} \times d_{\text{head,ssm}}}, \qquad a_{t,m} = \exp(-\Delta_{t,m} A_m) \in (0, 1)$$
(Note: if `cache_params.ssm_states[l]` stores transposed shape $[B_{\text{batch}}, H_{\text{ssm}}, d_{\text{head,ssm}}, d_{\text{state}}]$, transpose the last two dimensions so our canonical mathematical representation is always $S_{t,m} \in \mathbb{R}^{d_{\text{state}} \times d_{\text{head,ssm}}}$).

To decouple key direction from write magnitude, we define:
1. **Unit-Norm Write Key ($\hat{b}_t$):**
   $$\hat{b}_t = \frac{B_t}{\|B_t\|_2 + \epsilon} \in \mathbb{R}^{1 \times d_{\text{state}}}, \qquad \epsilon = 10^{-8}$$
2. **Effective Value Vector ($u_{t,m}$) and Write Energy ($w_{t,m}$):**
   $$u_{t,m} = \Delta_{t,m} \|B_t\|_2 x_{t,m} \in \mathbb{R}^{1 \times d_{\text{head,ssm}}}, \qquad w_{t,m} = \|u_{t,m}\|_2 = \Delta_{t,m} \|B_t\|_2 \|x_{t,m}\|_2 \ge 0$$
3. **Cumulative Log-Decay ($\ell_{t,m}(P)$) and Survival Factor ($\alpha_{t \to P, m}$):**
   Using reverse cumulative sums over $\log a_{\tau,m} = -\Delta_{\tau,m} A_m$ for $\tau \in \{t+1, \dots, P-1\}$ (with $\ell_{P-1, m}(P) = 0$ and $\alpha_{P-1 \to P, m} = 1$):
   $$\ell_{t,m}(P) = \sum_{\tau=t+1}^{P-1} \log a_{\tau,m} = -\sum_{\tau=t+1}^{P-1} \Delta_{\tau,m} A_m \le 0, \qquad \alpha_{t \to P, m} = \exp\big(\ell_{t,m}(P)\big) \in (0, 1]$$
4. **Exact Terminal State Decomposition:**
   Because $\hat{b}_t^\top u_{t,m} = \frac{B_t^\top}{\|B_t\|_2} (\Delta_{t,m} \|B_t\|_2 x_{t,m}) = B_t^\top (\Delta_{t,m} x_{t,m})$ as $\|B_t\|_2 \gg \epsilon$, the terminal prefill state $S_{P,m}$ satisfies:
   $$S_{P,m} = \sum_{t=0}^{P-1} \alpha_{t \to P, m} \hat{b}_t^\top u_{t,m} \in \mathbb{R}^{d_{\text{state}} \times d_{\text{head,ssm}}}$$

### 1.4 Phase 1 Agent Self-Correction Instruction
* **Instruction:** When implementing `falcon_h1_state_hook.py`, verify the exact ordering of `ssm_multipliers` in `transformers.models.falcon_h1.modeling_falcon_h1.FalconH1Mixer` and check whether the CUDA Mamba-2 selective scan kernel includes or excludes normalization by $\|B_t\|_2 + \epsilon$. To guarantee exact numerical identity ($<10^{-5}$ error), define $u_{t,m} = \Delta_{t,m} (\|B_t\|_2 + \epsilon) x_{t,m}$ so that $\hat{b}_t^\top u_{t,m} = B_t^\top (\Delta_{t,m} x_{t,m})$ holds *identically* without an $O(\epsilon)$ discrepancy. Document this or any tensor-layout transpose in a `# [MATH-UPGRADE | Phase 1 -> Phase 2]` comment block.

### 1.5 Connection & Handover to Phase 2
* **Exit Gate:** Must pass **Test 1** (Realized State-Readout Identity $< 10^{-5}$ max absolute error) and **Test 2** (Step-by-step recurrence matches `cache_params.ssm_states[layer_idx]` within $< 10^{-4}$ relative Frobenius error).
* **Handover Artifact:** `FalconH1StateHook` outputs a per-layer dictionary containing `b_hat` ($[B, P, d_{\text{state}}]$), `u` ($[B, P, H_{\text{ssm}}, d_{\text{head,ssm}}]$), `w` ($[B, P, H_{\text{ssm}}]$), `log_alpha` ($[B, P, H_{\text{ssm}}]$), `alpha` ($[B, P, H_{\text{ssm}}]$), and `S_P` ($[B, H_{\text{ssm}}, d_{\text{state}}, d_{\text{head,ssm}}]$) directly to `hooks/survival_metrics.py`.

---

## Phase 2: Realized State-Readout Decomposition & Calibrated Survival Predictor

### 2.1 Focus, Features & Techniques
* **Primary Focus:** Build `hooks/survival_metrics.py` to compute the exact realized superposition interference vector $\nu_{t,m}(P)$, extract the three time-consistent survival features ($\text{SIR}_{t,m}(P)$, $c_{t,m}(P)$, $\ell_{t,m}(P)$), and compute the calibrated per-head survival probability $\hat{p}_{\text{surv}}(t,l,m \mid P)$ and write-weighted layer survival $\bar{p}_{\text{surv}, t, l}(P)$.
* **Key Technique — O(P) Realized Interference via Single Matrix Multiplication:** Computing pairwise interference $\sum_{\tau \neq t} \alpha_{\tau \to P, m} \langle \hat{b}_t, \hat{b}_\tau \rangle u_{\tau, m}$ naively takes $O(P^2)$ time. Instead, we multiply $\hat{b}_t$ by $S_{P,m}$ in $O(P \cdot d_{\text{state}} \cdot d_{\text{head,ssm}})$ time and subtract the decayed target signal $\alpha_{t \to P, m} u_{t,m}$ in a single vectorized operation.

### 2.2 File to Create in Phase 2
* `hooks/survival_metrics.py`

### 2.3 Mathematical & Logical Specification

#### Step 1: Realized State-Readout Decomposition at Position $P$
Multiplying $S_{P,m}$ on the left by $\hat{b}_t$ gives the readout vector $r_{t,m}(P) \in \mathbb{R}^{d_{\text{head,ssm}}}$:
$$r_{t,m}(P) = \hat{b}_t S_{P,m} = \underbrace{\alpha_{t \to P, m} u_{t,m}}_{\text{Decayed Target Signal } s_{t,m}(P)} + \underbrace{\sum_{\tau \neq t} \alpha_{\tau \to P, m} \langle \hat{b}_t, \hat{b}_\tau \rangle u_{\tau, m}}_{\text{Realized Superposition Interference } \nu_{t,m}(P)}$$
Therefore, we compute the exact realized interference vector $\nu_{t,m}(P)$ in $O(1)$ per token-head via subtraction:
$$s_{t,m}(P) = \alpha_{t \to P, m} u_{t,m}, \qquad \nu_{t,m}(P) = r_{t,m}(P) - s_{t,m}(P) = \hat{b}_t S_{P,m} - \alpha_{t \to P, m} u_{t,m}$$

#### Step 2: Three Time-Consistent Dimensionless Survival Features
For every token $t \in \{0, \dots, P-1\}$, layer $l$, and SSM head $m$:
1. **Decayed Signal-to-Interference Ratio ($\text{SIR}_{t,l,m}(P)$):**
   $$\text{SIR}_{t,l,m}(P) = \frac{\|s_{t,l,m}(P)\|_2^2}{\|\nu_{t,l,m}(P)\|_2^2 + \epsilon} = \frac{\alpha_{t \to P, l, m}^2 \|u_{t,l,m}\|_2^2}{\|\hat{b}_{t,l} S_{P,l,m} - \alpha_{t \to P, l, m} u_{t,l,m}\|_2^2 + \epsilon}$$
2. **Directional Readout Cosine Alignment ($c_{t,l,m}(P)$):**
   $$c_{t,l,m}(P) = \frac{\langle \hat{b}_{t,l} S_{P,l,m},\; u_{t,l,m} \rangle}{\|\hat{b}_{t,l} S_{P,l,m}\|_2 \|u_{t,l,m}\|_2 + \epsilon} \in [-1, 1]$$
3. **Cumulative Log-Survival Factor ($\ell_{t,l,m}(P)$):**
   $$\ell_{t,l,m}(P) = \log(\alpha_{t \to P, l, m} + \epsilon) = \sum_{\tau=t+1}^{P-1} \log a_{\tau, l, m} \le 0$$

#### Step 3: Calibrated Survival Probability ($\hat{p}_{\text{surv}}$) & Layer Aggregation
1. **4-Parameter Logistic Calibrator:**
   $$\hat{p}_{\text{surv}}(t,l,m \mid P) = \sigma\Big( w_1 \log\big(\text{SIR}_{t,l,m}(P) + \epsilon\big) + w_2 c_{t,l,m}(P) + w_3 \ell_{t,l,m}(P) + b_0 \Big) \in (0, 1)$$
   where $\sigma(z) = (1 + e^{-z})^{-1}$.
2. **Zero-Parameter Closed-Form Default ($w_1=1, w_2=0, w_3=0, b_0=0$):**
   When running training-free before Platt calibration:
   $$\rho_{\text{surv}}^{\text{closed}}(t,l,m \mid P) = \frac{\text{SIR}_{t,l,m}(P)}{1 + \text{SIR}_{t,l,m}(P)} = \frac{\alpha_{t \to P, l, m}^2 \|u_{t,l,m}\|_2^2}{\alpha_{t \to P, l, m}^2 \|u_{t,l,m}\|_2^2 + \|\nu_{t,l,m}(P)\|_2^2 + \epsilon} \in [0, 1)$$
3. **Write-Weighted Layer-Aggregated Survival Probability ($\bar{p}_{\text{surv}, t, l}(P)$):**
   Weighting each head $m$ by its write energy $w_{t,l,m} = \|u_{t,l,m}\|_2$:
   $$\bar{p}_{\text{surv}, t, l}(P) = \frac{\sum_{m=1}^{H_{\text{ssm}}} w_{t,l,m} \, \hat{p}_{\text{surv}}(t,l,m \mid P)}{\sum_{m=1}^{H_{\text{ssm}}} w_{t,l,m} + \epsilon} \in (0, 1)$$

### 2.4 Phase 2 Agent Self-Correction Instruction
* **Instruction:** Notice that in $\text{SIR}_{t,l,m}(P)$, when a fast-decaying head has $\alpha_{t \to P, l, m} < 10^{-6}$, squaring $\alpha_{t \to P, l, m}$ in FP32 can underflow below $\epsilon = 10^{-8}$ even if $\nu_{t,l,m}(P) = 0$ (e.g., in Test 3's isolated single-write limit). If the agent observes numerical underflow when $\alpha_{t \to P, l, m}^2 \|u_{t,l,m}\|_2^2 \ll \epsilon$, the agent is authorized to scale the regularizer in the denominator proportionally to total readout energy or compute $\log \text{SIR}_{t,l,m}(P) = 2 \ell_{t,l,m}(P) + 2 \log(\|u_{t,l,m}\|_2 + \epsilon) - \log(\|\nu_{t,l,m}(P)\|_2^2 + \epsilon_{\text{rel}} \|s_{t,l,m}(P)\|_2^2 + 10^{-30})$ in log-space, documenting the exact stabilization with `# [MATH-UPGRADE | Phase 2 -> Phase 3]`.

### 2.5 Connection & Handover to Phase 3
* **Exit Gate:** Must pass **Test 3** (Single-write isolated limit gives $\|\nu_{t_0,m}(P)\|_2 = 0$, $c_{t_0,m}(P) = 1.0$, and $\rho_{\text{surv}}^{\text{closed}} \to 1.0$) and **Test 4** (R5b equal-age key-collision separation: $\alpha_{t_A \to P} = \alpha_{t_B \to P}$ while $\text{SIR}_{t_A}(P) < 0.2 \cdot \text{SIR}_{t_B}(P)$ and $c_{t_A}(P) < c_{t_B}(P)$).
* **Handover Artifact:** `SurvivalPredictor` class providing `compute_survival_features()` and `predict_survival()` to Phase 3 (Gate 0/1 calibration) and Phase 4 (`SA-SnapKV` eviction).

---

## Phase 3: Gate 0 & Gate 1 Diagnostic Suite & Age-Controlled Calibration

### 3.1 Focus, Features & Techniques
* **Primary Focus:** Build `benchmarks/gate0_diagnostics.py` to execute Gate 0 (`MQAR`, `R5a Decay Trap`, `R5b Equal-Age Key-Collision Trap`) and Gate 1 (`FineWeb-Edu Natural Prefix Recall` with Age-Controlled Partial Correlation, AUROC, ECE, and Brier Score), and fit the 4-parameter logistic calibrator $(w_1, w_2, w_3, b_0)$.
* **Key Technique — De-Circularized Age-Controlled Evaluation:** Because cumulative decay $\alpha_{t \to P}$ naturally correlates with token age $d_t = P - t$, a reviewer could ask whether $\hat{p}_{\text{surv}}$ is just a recency proxy. We eliminate this confounder in two ways: (1) **R5b** places Set A and Set B at the *exact same intermediate distance* from $P$ and perturbs only Set A's write keys with high-cosine distractors ($\langle \hat{b}_t, \hat{b}_\tau \rangle > 0.7$), and (2) **Natural Prefix Recall** evaluates partial correlation between $\hat{p}_{\text{surv}}(t \mid P)$ and state-only recall accuracy while statistically controlling for token age $d_t = P - t$.

### 3.2 File to Create in Phase 3
* `benchmarks/gate0_diagnostics.py`

### 3.3 Mathematical & Logical Specification

#### 1. R5a (Decay Trap) & R5b (Key-Collision Interference Trap)
* **R5a Construction:** In a sequence of length $P$, place Set A ($16$ key-value needle pairs) at early indices $t_A \in [64, 256]$ and Set B ($16$ key-value needle pairs) at late indices $t_B \in [P-256, P-64]$, separated by high-$\Delta_t$ filler tokens. At position $P$, restrict the KV budget to $K = 16$ pairs.
  * *Expected Physics:* Write-time surprise (`HOLA`: $\|u_{t,m} - \hat{b}_t S_{t-1,m}\|_2$) and chunk drift (`DeltaS`: $\|S_{c,m} - S_{c-1,m}\|_F / \|S_{c-1,m}\|_F$) score Set A and Set B similarly at write time, wasting ~50% of the KV budget on Set B (which $S_P$ already remembers), whereas $1 - \bar{p}_{\text{surv}, t, l}(P)$ selectively retains Set A in the KV cache and recalls Set B from $S_P$.
* **R5b Construction:** Place Set A ($16$ pairs) and Set B ($16$ pairs) interleaved at matched intermediate positions $t_A, t_B \in [P/2 - 128, P/2 + 128]$ so that $P - t_A \approx P - t_B$ and $\alpha_{t_A \to P, m} \approx \alpha_{t_B \to P, m}$. In the second half of the prefix $\tau \in (P/2 + 128, P - 64)$, inject distractor tokens designed to have high write-key cosine similarity with Set A ($\langle \hat{b}_{t_A}, \hat{b}_\tau \rangle > 0.7$) and near-zero similarity with Set B ($\langle \hat{b}_{t_B}, \hat{b}_\tau \rangle \approx 0$).
  * *Expected Physics:* Pure decay `DASC-Decay` ($1 - \bar{\alpha}_{t \to P, l}$) assigns identical scores to Set A and Set B, whereas $\text{SIR}_{t,m}(P)$ and $c_{t,m}(P)$ detect the large interference norm $\|\nu_{t_A, m}(P)\|_2^2$ on Set A and prioritize Set A for KV retention.

#### 2. Fitting the 4-Parameter Logistic Calibrator (Gate 1)
On a held-out calibration split of $N_{\text{cal}} = 250$ natural FineWeb-Edu passages ($P = 4,096$) run under `State-Only` ($K_{\text{ret}} = 0$, retaining only the 4 sinks + 64 local window tokens), let $y_i \in \{0, 1\}$ denote whether target prefix fact $i$ at token $t_i$ is correctly recalled from $S_P$ alone. We optimize $(w_1, w_2, w_3, b_0)$ via L-BFGS on binary cross-entropy loss with weak $L_2$ regularization ($\beta = 10^{-4}$):
$$\mathcal{L}_{\text{cal}}(w_1, w_2, w_3, b_0) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \log \bar{p}_{\text{surv}, t_i}(P) + (1 - y_i) \log\big(1 - \bar{p}_{\text{surv}, t_i}(P)\big) \Big] + \beta (w_1^2 + w_2^2 + w_3^2)$$

#### 3. Age-Controlled Partial Correlation & Calibration Metrics
On the $250$ held-out test passages:
1. **Expected Calibration Error (ECE across $M=10$ bins $B_1, \dots, B_{10}$):**
   $$\text{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{N} \big| \text{acc}(B_b) - \text{conf}(B_b) \big|$$
2. **Age-Controlled Partial Correlation ($r_{\hat{p}, y \mid d}$):**
   Let $x_i = \bar{p}_{\text{surv}, t_i}(P)$, $y_i \in \{0, 1\}$ (or target token log-probability under `State-Only`), and $d_i = \log(P - t_i)$ be the log token age. Compute the sample Pearson/Spearman correlations $r_{xy}, r_{xd}, r_{yd}$ and evaluate:
   $$r_{xy \mid d} = \frac{r_{xy} - r_{xd} r_{yd}}{\sqrt{(1 - r_{xd}^2)(1 - r_{yd}^2)}}$$
   along with its two-sided $t$-test $p$-value:
   $$t_{\text{stat}} = r_{xy \mid d} \sqrt{\frac{N - 3}{1 - r_{xy \mid d}^2}}$$

### 3.4 Phase 3 Agent Self-Correction Instruction
* **Instruction:** Look closely at the Gate 1 decision rule: if fitting a single global $(w_1, w_2, w_3, b_0)$ across all layers underfits because early vs. deep layers have different feature scales, the agent is authorized to standardize the three features $(\log \text{SIR}, c, \ell)$ to zero mean and unit variance on the calibration set before logistic regression, and/or implement the Hard Decision Gate 1 fallback check automatically (if $\text{SIR}$ + $c$ adds $< 2\%$ AUROC over $\ell_{t,l,m}(P)$ on natural text, expose a `mode="calibrated_log_decay"` toggle). Document any standardization parameters in `# [MATH-UPGRADE | Phase 3 -> Phase 4]`.

### 3.5 Connection & Handover to Phase 4
* **Exit Gate (Hard Decision Gate 1):** $\hat{p}_{\text{surv}}$ beats `DeltaS-Port` and `HOLA-Write` by $>15\%$ recall on R5a/R5b AND achieves statistically significant positive partial correlation ($p < 0.01$) with state-only recall controlling for token age $P - t$.
* **Handover Artifact:** Fitted calibration weights `(w_1, w_2, w_3, b_0)` saved to `configs/` and loaded automatically by `eviction/sa_snapkv.py` in Phase 4.

---

## Phase 4: Token-Level `SA-SnapKV`, Risk-Adaptive Layer Budgets, Streaming Prefill & Baselines

### 4.1 Focus, Features & Techniques
* **Primary Focus:** Build the entire `eviction/` module (`sa_snapkv.py`, `adaptive_budget.py`, `streaming_prefill.py`, and `baselines.py`) implementing our training-free token-level ($B=1$) state-aware KV cache eviction, risk-adaptive per-layer budget allocation $K_l$, chunked-prefill streaming eviction (`SA-SnapKV-Stream`), and all 10 comparison policies.
* **Key Technique — Prospective Utility $\times$ State-Survival Deficit:** At switch point $P$, we retain tokens that simultaneously have high expected future attention utility $D_{t,l}$ AND high state-forgetting deficit $1 - \bar{p}_{\text{surv}, t, l}(P)$, while allocating larger KV budgets $K_l$ to layers whose SSM heads suffer higher total forgetting risk $\mathcal{R}_l(P)$.

### 4.2 Files to Create in Phase 4
1. `eviction/sa_snapkv.py`
2. `eviction/adaptive_budget.py`
3. `eviction/streaming_prefill.py`
4. `eviction/baselines.py`

### 4.3 Mathematical & Logical Specification

#### 1. Protected Token Partitions at Switch Point $P$
For any prefix $0, \dots, P-1$:
* **Attention Sinks (Always Retained):** $\mathcal{S}_{\text{sink}} = \{0, 1, 2, 3\}$ ($N_{\text{sink}} = 4$).
* **Observation Window (Always Retained):** $\mathcal{W}_{\text{obs}} = \{P - W_{\text{obs}}, \dots, P - 1\}$ ($W_{\text{obs}} = 64$).
* **Candidate Eviction Pool:** $\mathcal{C}_{\text{cand}} = \{4, 5, \dots, P - W_{\text{obs}} - 1\}$ of size $P_{\text{cand}} = P - W_{\text{obs}} - 4$.

#### 2. Observation-Window Attention Utility ($D_{t,l}$) & 4-Way Ablation Modes
For every candidate token $t \in \mathcal{C}_{\text{cand}}$ in layer $l$, let $A_{q \to t}^{(l,h)}$ be the causal softmax attention weight from query $q \in \mathcal{W}_{\text{obs}}$ to key $t$ in query head $h \in \{1, \dots, H_q\}$, and let $g(h) \in \{1, \dots, H_{kv}\}$ be its GQA key-value group index:
$$D_{t,l} = \frac{1}{W_{\text{obs}} H_q} \sum_{q \in \mathcal{W}_{\text{obs}}} \sum_{h=1}^{H_q} A_{q \to t}^{(l,h)} \|v_{t, l, g(h)}\|_2$$
(Following `SnapKV`, apply 1D average pooling of kernel size $k_{\text{pool}} = 5$, stride $1$ along the sequence dimension $t$ of $D_{t,l}$ to preserve local phrase coherence).

We support all four ablation scoring rules in `sa_snapkv.py`:
1. **`Demand-Only` (`SnapKV` + Value Norm):** $s_{t,l}^{\text{demand}} = D_{t,l}$
2. **`Decay-Only` (`DASC-Decay`):** $s_{t,l}^{\text{decay}} = D_{t,l} \cdot \big(1 - \bar{\alpha}_{t \to P, l}\big)$, where $\bar{\alpha}_{t \to P, l} = \frac{\sum_m w_{t,l,m} \alpha_{t \to P, l, m}}{\sum_m w_{t,l,m} + \epsilon}$
3. **`Survival-Only`:** $s_{t,l}^{\text{surv-only}} = 1 - \bar{p}_{\text{surv}, t, l}(P)$
4. **`SA-SnapKV` (Ours — `Demand × Survival Deficit`):**
   $$s_{t,l} = D_{t,l} \cdot \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$

#### 3. Risk-Adaptive Layerwise KV Budget Allocation (`adaptive_budget.py`)
Given total candidate budget $K_{\text{total}} = L \cdot K_{\text{avg}}$ across all $L$ layers:
1. Compute Layer $l$'s **Expected State Forgetting Risk**:
   $$\mathcal{R}_l(P) = \sum_{t=4}^{P - W_{\text{obs}} - 1} D_{t,l} \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$
2. Compute raw proportional budgets clamped to $[K_{\min}, P_{\text{cand}}]$ (with $K_{\min} = \min(32, K_{\text{avg}})$):
   $$\tilde{K}_l = \text{clip}\left( K_{\text{total}} \cdot \frac{\mathcal{R}_l(P)}{\sum_{l'=1}^L \mathcal{R}_{l'}(P) + \epsilon},\; K_{\min},\; P_{\text{cand}} \right)$$
3. Apply exact integer conservation (Largest Remainder Method) so that:
   $$K_l \in \mathbb{Z}, \qquad K_{\min} \le K_l \le P_{\text{cand}}, \qquad \sum_{l=1}^L K_l = K_{\text{total}}$$
4. Select the top-$K_l$ candidate tokens $\mathcal{T}_{\text{ret}, l} = \text{TopK}_{t \in \mathcal{C}_{\text{cand}}}(s_{t,l}; K_l)$ and also record the **top-$k$ evicted runner-up tokens** $\mathcal{T}_{\text{cons}, l}$ (ranked $K_l + 1$ to $K_l + k$ by $s_{t,l}$) for Phase 5 state consolidation.

#### 4. Physical KV Cache Slicing (`torch.gather`)
Once $\mathcal{I}_{\text{keep}, l} = \text{sort}\big(\mathcal{S}_{\text{sink}} \cup \mathcal{T}_{\text{ret}, l} \cup \mathcal{W}_{\text{obs}}\big)$ of length $4 + K_l + 64$ is determined, physically slice layer $l$'s attention cache in `past_key_values`:
$$\mathcal{K}_{P,l}^{\text{evict}} = \mathcal{K}_{P,l}[:, :, \mathcal{I}_{\text{keep}, l}, :], \qquad \mathcal{V}_{P,l}^{\text{evict}} = \mathcal{V}_{P,l}[:, :, \mathcal{I}_{\text{keep}, l}, :]$$
Crucially, **preserve original RoPE position IDs** during suffix generation ($t \ge P$) by passing `cache_position` / `position_ids` matching the true sequence length $P, P+1, \dots$ so relative RoPE distances on prefill keys (which already had RoPE applied at their true positions $t < P$) remain exact.

#### 5. Streaming Chunked-Prefill Variant (`streaming_prefill.py`)
For long contexts ($P \in \{16\text{K}, 32\text{K}\}$) processed in chunks of size $C_{\text{prefill}} = 2,048$:
* At the end of chunk $c$ (current prefix length $P_c$), compute local survival $\bar{p}_{\text{surv}, t, l}(P_c)$ against the current running state $S_{P_c, l, m}$ and current chunk utility $D_{t,l}^{(c)}$, and immediately evict candidate tokens outside $\mathcal{S}_{\text{sink}} \cup \mathcal{T}_{\text{ret}, l}^{(c)} \cup \mathcal{W}_{\text{obs}}^{(c)}$.
* Peak KV memory at any point during prefill is bounded by $L \times (K_{\text{avg}} + C_{\text{prefill}} + 68)$ tokens instead of $L \times P$.

#### 6. All Baselines in `eviction/baselines.py`
Implement clean, unified interfaces for:
1. `Full-KV` ($100\%$ KV cache)
2. `State-Only` ($K_l = 0$, retaining only $\mathcal{S}_{\text{sink}} \cup \mathcal{W}_{\text{obs}} + S_P$)
3. `SWA` (Recent / Sliding Window of size $4 + K_l + 64$)
4. `H2O` (Cumulative attention score across the prefix)
5. `SnapKV` (`Demand-Only`, $s_{t,l} = D_{t,l}$)
6. `KVzip-Proxy` (Context reconstruction self-attention utility score)
7. `LORE-KV` ($M=1$ greedy lookahead token continuation appended to $\mathcal{W}_{\text{obs}}$ before computing $D_{t,l}$)
8. `HOLA-Write` (Write-time surprise score $\bar{u}_{\text{surprise}, t, l} = \frac{1}{H_{\text{ssm}}} \sum_m \|u_{t,l,m} - \hat{b}_{t,l} S_{t-1,l,m}\|_2$)
9. `DeltaS-Port` (Write-time normalized state drift $\|S_{t,l,m} - S_{t-1,l,m}\|_F / (\|S_{t-1,l,m}\|_F + \epsilon)$)
10. `DASC-Decay` (`Decay-Only`, $D_{t,l} \cdot (1 - \bar{\alpha}_{t \to P, l})$)

### 4.4 Phase 4 Agent Self-Correction Instruction
* **Instruction:** In `adaptive_budget.py`, when clamping $\tilde{K}_l$ to $[K_{\min}, P_{\text{cand}}]$, simple rounding can cause $\sum_{l=1}^L K_l \neq K_{\text{total}}$ by a few tokens, which would fail **Test 6**. The agent must implement iterative water-filling + Largest Remainder redistribution over unclamped layers so that $\sum_{l=1}^L K_l == K_{\text{total}}$ holds with **zero discrepancy**. Furthermore, in `sa_snapkv.py`, normalize $D_{t,l}$ within each layer ($\hat{D}_{t,l} = D_{t,l} / (\sum_{t'} D_{t',l} + \epsilon)$) when computing cross-layer risk $\mathcal{R}_l(P)$ if deep layers have systematically different value norms $\|v_{t,l}\|_2$, and document this in a `# [MATH-UPGRADE | Phase 4 -> Phase 5]` comment block.

### 4.5 Connection & Handover to Phase 5
* **Exit Gate:** Must pass **Test 6** (Exact budget conservation $\sum_l K_l = K_{\text{total}}$), **Test 7** (Prospective eviction causality—suffix perturbations at $t \ge P$ leave all prefill scores and retained indices bit-identical), and **Test 10** (Physical KV slicing via `torch.gather` matches additive `-inf` attention masking to machine precision).
* **Handover Artifact:** `SASnapKVEvictor` returns `(evicted_past_key_values, runner_up_indices_T_cons, survival_probs_at_T_cons)` directly to Phase 5 (`consolidation/ridge_consolidate.py`).

---

## Phase 5: Batched Ridge State Consolidation & 300-Step Teacher KL Distillation

### 5.1 Focus, Features & Techniques
* **Primary Focus:** Build `consolidation/ridge_consolidate.py` and `consolidation/distill_gates.py` to rescue the top-$k$ evicted runner-up tokens $\mathcal{T}_{\text{cons}, l} = \{t_1, \dots, t_k\}$ ($k \in \{32, 64\}$) by writing their unrecovered information into the recurrent state $S_{P,l,m}$ via a batched ridge-regularized delta-rule update, verify Proposition 1's exact spectral collateral damage bound, and optimize the per-layer, per-head scalar gates $\{\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m}\}$ via 300-step Full-KV teacher KL distillation.
* **Key Technique — Deficit-Weighted Cholesky Delta Rule + Teacher Self-Distillation:** Instead of uniformly overwriting $S_{P,l,m}$ (which would corrupt long-horizon heads where $S_{P,l,m}$ already holds intact memories), each token $t_r \in \mathcal{T}_{\text{cons}, l}$ is weighted by its head-specific deficit $1 - \hat{p}_{\text{surv}}(t_r, l, m \mid P)$ and gated by $\gamma_{l,m} = \sigma(\tilde{\gamma}_{l,m})$, solved via a tiny $k \times k$ symmetric positive-definite linear system.

### 5.2 Files to Create in Phase 5
1. `consolidation/ridge_consolidate.py`
2. `consolidation/distill_gates.py`

### 5.3 Mathematical & Logical Specification

#### 1. Batched Ridge-Regularized Delta-Rule State Update
For layer $l$ and SSM head $m$, given the $k$ runner-up evicted token indices $\mathcal{T}_{\text{cons}, l} = \{t_1, \dots, t_k\}$:
1. Stack the unit write keys $\hat{B}_l \in \mathbb{R}^{k \times d_{\text{state}}}$ and effective values $U_{l,m} \in \mathbb{R}^{k \times d_{\text{head,ssm}}}$:
   $$\hat{B}_l = \begin{bmatrix} \hat{b}_{t_1, l} \\ \vdots \\ \hat{b}_{t_k, l} \end{bmatrix} \in \mathbb{R}^{k \times d_{\text{state}}}, \qquad U_{l,m} = \begin{bmatrix} u_{t_1, l, m} \\ \vdots \\ u_{t_k, l, m} \end{bmatrix} \in \mathbb{R}^{k \times d_{\text{head,ssm}}}$$
2. Parameterize the learnable consolidation gate $\gamma_{l,m}$ and ridge regularizer $\lambda_{l,m}$ from unconstrained scalars $(\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m})$:
   $$\gamma_{l,m} = \sigma(\tilde{\gamma}_{l,m}) \in (0, 1), \qquad \lambda_{l,m} = \exp(\tilde{\lambda}_{l,m}) > 0$$
   (Initialized so $\gamma_{l,m}^{(0)} \approx 0.10 \implies \tilde{\gamma}_{l,m}^{(0)} = -2.197$, and $\lambda_{l,m}^{(0)} = 1.0 \implies \tilde{\lambda}_{l,m}^{(0)} = 0.0$).
3. Construct the $k \times k$ diagonal deficit-gain matrix $\Gamma_{l,m}$:
   $$\Gamma_{l,m} = \gamma_{l,m} \cdot \text{diag}\Big(1 - \hat{p}_{\text{surv}}(t_1, l, m \mid P), \dots, 1 - \hat{p}_{\text{surv}}(t_k, l, m \mid P)\Big) \in [0, 1]^{k \times k}$$
4. Compute the state prediction residual on $\mathcal{T}_{\text{cons}, l}$:
   $$E_{l,m} = U_{l,m} - \hat{B}_l S_{P,l,m} \in \mathbb{R}^{k \times d_{\text{head,ssm}}}$$
5. Form the $k \times k$ Gram matrix $G_l = \hat{B}_l \hat{B}_l^\top \in \mathbb{R}^{k \times k}$ and solve the regularized SPD system in `torch.float32` using `torch.linalg.solve`:
   $$Z_{l,m} = \big(G_l + \lambda_{l,m} I_k\big)^{-1} \big(\Gamma_{l,m} E_{l,m}\big) \in \mathbb{R}^{k \times d_{\text{head,ssm}}}$$
6. Update the recurrent state $S_{P,l,m}$:
   $$S_{P,l,m}^{\text{cons}} = S_{P,l,m} + \hat{B}_l^\top Z_{l,m} = S_{P,l,m} + \hat{B}_l^\top \big(\hat{B}_l \hat{B}_l^\top + \lambda_{l,m} I_k\big)^{-1} \Gamma_{l,m} \big(U_{l,m} - \hat{B}_l S_{P,l,m}\big)$$

#### 2. Proposition 1 Verification (Exact Spectral Perturbation & Collateral Damage Bound)
Let $\hat{B}_l = W \Sigma V^\top$ be the SVD of $\hat{B}_l$ with singular values $\sigma_1 \ge \dots \ge \sigma_k > 0$, and let $\gamma_{\max, l, m} = \max_{r \in \{1 \dots k\}} (\Gamma_{l,m})_{r,r}$. Our bound checker function `verify_proposition1_bound()` in `ridge_consolidate.py` computes and verifies:
1. **Exact Operator Norm Identity:**
   $$\left\| \hat{B}_l^\top \big(\hat{B}_l \hat{B}_l^\top + \lambda_{l,m} I_k\big)^{-1} \right\|_2 = \max_{i \in \{1 \dots k\}} \frac{\sigma_i}{\sigma_i^2 + \lambda_{l,m}} \le \frac{1}{2\sqrt{\lambda_{l,m}}}$$
2. **Orthogonal Subspace Invariance:**
   For any stored key $\hat{b}_\tau$ with $V^\top \hat{b}_\tau^\top = 0$, $\|\hat{b}_\tau S_{P,l,m}^{\text{cons}} - \hat{b}_\tau S_{P,l,m}\|_2 = 0$.
3. **Worst-Case Collateral Perturbation Bound on Stored Token $\tau \notin \mathcal{T}_{\text{cons}, l}$:**
   With subspace overlap $\theta_\tau = \|V^\top \hat{b}_\tau^\top\|_2 \in [0, 1]$:
   $$\big\| \hat{b}_\tau S_{P,l,m}^{\text{cons}} - \hat{b}_\tau S_{P,l,m} \big\|_2 \le \theta_\tau \cdot \gamma_{\max, l, m} \left(\max_{i \in \{1 \dots k\}} \frac{\sigma_i}{\sigma_i^2 + \lambda_{l,m}}\right) \big\| U_{l,m} - \hat{B}_l S_{P,l,m} \big\|_2$$

#### 3. 300-Step Full-KV Teacher KL Distillation (`distill_gates.py`)
1. Freeze 100% of pretrained `Falcon-H1` parameters (`param.requires_grad = False`).
2. Instantiate learnable parameter tensors `tilde_gamma` ($\mathbb{R}^{L \times H_{\text{ssm}}}$) and `tilde_lambda` ($\mathbb{R}^{L \times H_{\text{ssm}}}$) with `requires_grad = True` ($1,728$ total scalars on `0.5B`; $3,168$ total scalars on `1.5B-Deep` [$66 \times 24 \times 2$]).
3. For 300 steps using AdamW ($\text{lr} = 3 \times 10^{-2}$) on prospective context-question sequences:
   * Run prefill $0 \dots P-1$ once to obtain the Full-KV teacher logits $z_{\text{Full-KV}, t}$ on suffix tokens $t \in T_{\text{suffix}}$ ($t \ge P$).
   * Apply `SA-SnapKV` eviction at position $P$, apply differentiable `ridge_consolidate(S_P, tilde_gamma, tilde_lambda)` to obtain $S_P^{\text{cons}}(\tilde{\gamma}, \tilde{\lambda})$, and run the student suffix forward pass $t \in T_{\text{suffix}}$ to obtain $z_{\text{Evict+Cons}, t}$.
   * Minimize the token-level forward KL divergence:
     $$\mathcal{L}_{\text{distill}}(\tilde{\gamma}, \tilde{\lambda}) = \frac{1}{|T_{\text{suffix}}|} \sum_{t \in T_{\text{suffix}}} \text{KL}\Big( \text{softmax}(z_{\text{Full-KV}, t}) \;\Big\|\; \text{softmax}\big(z_{\text{Evict+Cons}, t}(\tilde{\gamma}, \tilde{\lambda})\big) \Big)$$

### 5.4 Phase 5 Agent Self-Correction Instruction
* **Instruction:** Look closely at step 6 of the consolidation update: in Mamba-2, when suffix tokens $t \ge P$ read from $S_{P,l,m}^{\text{cons}}$, the readout is scaled by the SSM output vector $C_t$ and `D` skip connection. If the effective value $U_{l,m} = \Delta_{t,m} \|B_t\|_2 x_{t,m}$ was accumulated over $\tau \le P$ with step size $\Delta_{t,m}$, overwriting $\hat{b}_{t_r} S_{P,l,m}^{\text{cons}} \to u_{t_r, l, m}$ assumes a single-step write scale. If during Gate 2 diagnostics the residual norm $\|U_{l,m} - \hat{B}_l S_{P,l,m}\|_2$ has a scale mismatch relative to the running state norm $\|S_{P,l,m}\|_F$, the agent is authorized to bound the update norm or normalize $U_{l,m}$ to match the target state readout scale, and clamp $\lambda_{l,m} = \exp(\text{clip}(\tilde{\lambda}_{l,m}, -4.0, 6.0)) + 10^{-4}$ so the Cholesky solve never becomes ill-conditioned. Document any adjustment in `# [MATH-UPGRADE | Phase 5 -> Phase 6]`.

### 5.5 Connection & Handover to Phase 6
* **Exit Gate:** Must pass **Test 5** (Proposition 1 spectral bound holds across 100 random matrices and orthogonal queries have $<10^{-6}$ perturbation) and **Test 8** (Teacher KL distillation produces non-zero gradients on `tilde_gamma` and `tilde_lambda` while all `Falcon-H1` backbone gradients are strictly `None` or $0$).
* **Handover Artifact:** Trained `distilled_gates_0.5b.pt` and `distilled_gates_1.5b.pt` checkpoints and the `DistilledConsolidator` module passed to Phase 6's end-to-end evaluation pipeline.

---

## Phase 6: Mandatory 10-Test Verification Suite & Running the Complete Project

### 6.1 Focus, Features & Techniques
* **Primary Focus:** Build `tests/test_v13_suite.py` (all 10 mandatory unit tests) and the two primary benchmark harnesses `benchmarks/prospective_ruler.py` (Mode A Prospective Split & Mode B Standard Split) and `benchmarks/scbench_babilong.py`, then execute the gated experimental roadmap to generate all 6 paper figures and benchmark tables.
* **Key Technique — Prospective Multi-Query Evaluation (Mode A vs. Mode B):** Standard single-turn RULER places the question at the end of the prompt ($\le P$), leaking the test question into the observation window $\mathcal{W}_{\text{obs}}$ before eviction. `prospective_ruler.py` implements **Mode A (Primary Prospective Split)**: prefill the document context $0 \dots P-1$, perform KV eviction and state consolidation at position $P$ *before any question is seen*, and then evaluate $Q \in \{1, \dots, 4\}$ sequential queries arriving at $t \ge P$.

### 6.2 Files to Create in Phase 6
1. `tests/test_v13_suite.py`
2. `benchmarks/prospective_ruler.py`
3. `benchmarks/scbench_babilong.py`

### 6.3 Exact Specification of the 10 Mandatory Unit Tests (`tests/test_v13_suite.py`)

| Test ID | Test Function Name | Mathematical & Logical Verification Condition |
| :--- | :--- | :--- |
| **Test 1** | `test_01_realized_readout_identity` | Verify in FP32 that $\max_{t,m} \|\hat{b}_t S_{P,m} - (\alpha_{t \to P, m} u_{t,m} + \sum_{\tau \neq t} \alpha_{\tau \to P, m} \langle \hat{b}_t, \hat{b}_\tau \rangle u_{\tau, m})\|_\infty < 10^{-5}$. |
| **Test 2** | `test_02_falcon_h1_post_conv_mup_hook` | Verify on `FalconH1Mixer` that the step-by-step FP32 recurrence from extracted $(x_t, B_t, \Delta_t, A_m)$ matches `cache_params.ssm_states[layer_idx]` within $< 10^{-4}$ relative Frobenius error. |
| **Test 3** | `test_03_sir_single_write_limit` | Verify for a single isolated write at $t_0$ ($u_\tau = 0$ for $\tau \neq t_0$) that $\|\nu_{t_0, m}(P)\|_2 = 0$, $c_{t_0, m}(P) = 1.0$, and $\rho_{\text{surv}}^{\text{closed}}(t_0, m \mid P) > 1 - 10^{-5}$ for any $\alpha_{t_0 \to P, m} > 0$. |
| **Test 4** | `test_04_r5b_equal_age_interference` | Verify on equal-age pairs ($P - t_A = P - t_B$) where $t_A$ suffers later key collision ($\langle \hat{b}_{t_A}, \hat{b}_{\text{dist}} \rangle = 0.8$) that $\alpha_{t_A \to P} = \alpha_{t_B \to P}$, while $\text{SIR}_{t_A}(P) < 0.2 \cdot \text{SIR}_{t_B}(P)$ and $c_{t_A}(P) < c_{t_B}(P)$. |
| **Test 5** | `test_05_proposition1_spectral_bound` | Verify across 100 random key matrices $\hat{B}_l \in \mathbb{R}^{k \times 128}$ that $\|\hat{B}_l^\top(\hat{B}_l \hat{B}_l^\top + \lambda I_k)^{-1}\|_2 = \max_i \frac{\sigma_i}{\sigma_i^2 + \lambda}$, orthogonal keys have $< 10^{-6}$ perturbation, and non-orthogonal keys strictly satisfy Proposition 1. |
| **Test 6** | `test_06_risk_adaptive_budget_conservation` | Verify that $\sum_{l=1}^L K_l == K_{\text{total}}$ and $K_{\min} \le K_l \le P - W_{\text{obs}} - 4$ across all layers $l \in \{1, \dots, L\}$. |
| **Test 7** | `test_07_prospective_eviction_causality` | Verify in Mode A that replacing suffix tokens $t \ge P$ leaves $\hat{p}_{\text{surv}}(t', l, m \mid P)$, $s_{t', l}$, $\mathcal{T}_{\text{ret}, l}$, and $S_{P,l,m}^{\text{cons}}$ bit-identical (`torch.equal`). |
| **Test 8** | `test_08_distillation_gradient_isolation` | Verify during a KL distillation step that `tilde_gamma.grad` and `tilde_lambda.grad` are non-zero while every parameter in `FalconH1ForCausalLM` has `grad is None` or `torch.all(grad == 0)`. |
| **Test 9** | `test_09_t4_fp16_fp32_hook_stability` | Run long sequences through the FP16 model + FP32 hook and verify zero `NaN` or `Inf` values in logits, attention weights, or $S_P^{\text{cons}}$. |
| **Test 10** | `test_10_physical_slicing_vs_mask_equivalence` | Verify that slicing `past_key_values` via `torch.gather` on $\mathcal{I}_{\text{keep}, l}$ yields suffix logits matching an additive `-inf` attention mask on evicted tokens within $< 10^{-5}$. |

### 6.4 Step-by-Step Guide to Running the Project End-to-End

Once Phases 0–6 are coded, execute the project in the following strict sequence:

```bash
# Step 1: Run the Mandatory 10-Test Suite (Must pass 10/10 green before any benchmark)
pytest tests/test_v13_suite.py -v

# Step 2: Run Gate 0 & Gate 1 Diagnostics + Fit 4-Parameter Calibrator (Days 4-8)
python -m benchmarks.gate0_diagnostics \
    --config configs/falcon_h1_0.5b.yaml \
    --tasks mqar r5a r5b fineweb_age_recall \
    --save_calibrator configs/calibrator_0.5b.json

python -m benchmarks.gate0_diagnostics \
    --config configs/falcon_h1_1.5b_deep.yaml \
    --tasks mqar r5a r5b fineweb_age_recall \
    --save_calibrator configs/calibrator_1.5b.json

# Step 3: Run Pillar 2 Token-Level SA-SnapKV & Risk-Adaptive Budget Sweeps (Days 9-14)
python -m benchmarks.prospective_ruler \
    --config configs/falcon_h1_1.5b_deep.yaml \
    --mode prospective_mode_a \
    --lengths 4096 8192 16384 \
    --budgets 0.05 0.10 0.20 0.35 0.50 1.00 \
    --policies full_kv state_only swa h2o snapkv kvzip lore_kv hola deltas dasc sa_snapkv

# Step 4: Run Gate 2 Collateral Audit & 300-Step Teacher KL Distillation (Days 15-18)
python -m consolidation.distill_gates \
    --config configs/falcon_h1_1.5b_deep.yaml \
    --steps 300 --k_cons 64 \
    --save_gates checkpoints/distilled_gates_1.5b.pt

# Step 5: Run Iso-Memory Consolidation Comparison & Multi-Turn SCBench/BABILong (Days 19-24)
python -m benchmarks.scbench_babilong \
    --config configs/falcon_h1_1.5b_deep.yaml \
    --gates checkpoints/distilled_gates_1.5b.pt \
    --compare_iso_memory_kv_plus_k
```

### 6.5 Phase 6 Agent Self-Correction Instruction
* **Instruction:** During `prospective_ruler.py` and `scbench_babilong.py` evaluation, when comparing `SA-SnapKV(K_ret) + Distilled-Consolidate(k)` against `SA-SnapKV(K_ret)` and `SA-SnapKV(K_ret + k)`, compute paired bootstrap $95\%$ confidence intervals and two-sided $p$-values across examples. If any baseline or attention-mask indexing nuance in HuggingFace `FalconH1ForCausalLM` requires custom `position_ids` or `cache_position` handling during multi-turn suffix generation ($Q \in \{1 \dots 4\}$), enforce the exact fix verified in **Test 10** and document it with `# [MATH-UPGRADE | Phase 6 -> Final Evaluation]`.