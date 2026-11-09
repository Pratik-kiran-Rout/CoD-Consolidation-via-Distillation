# Comprehensive Project Guide: Calibrated State-Survival KV Eviction & Distilled State Consolidation (V13)

---

## 1. The Core Problem LLMs Face: The Memory Paradox

To understand this project, imagine you are reading a 500-page mystery novel and taking notes to answer questions at the very end.

In current Large Language Models (LLMs), there are two major ways architectures remember information over long texts:

```
┌────────────────────────────────────────────────────────────────────────┐
│                          THE DUAL MEMORY REALITY                       │
├──────────────────────────────────┬─────────────────────────────────────┤
│   Attention KV Cache             │   Recurrent SSM State               │
│   (Nonparametric "Scratchpad")   │   (Parametric "Mental Summary")     │
├──────────────────────────────────┼─────────────────────────────────────┤
│ • Exact, verbatim record         │ • Fixed-size vector matrix          │
│ • Perfect recall of details      │ • Constant memory footprint         │
│ • Memory grows linearly O(P)     │ • Suffers decay & interference      │
│ • Explodes VRAM on long inputs   │ • Forgets older / collided items    │
└──────────────────────────────────┴─────────────────────────────────────┘
```

### 1.1 The Pure Transformer Problem (VRAM Explosion)
Traditional Transformers (like GPT-4 or Llama) store a **Key-Value (KV) Cache** for every single word. As the document grows to 16,000 or 32,000 tokens:
* The KV cache grows linearly ($O(P)$).
* Serving multiple users or long documents causes GPUs to run out of memory (Out-Of-Memory / OOM).
* The cost of serving long-context LLMs skyrockets.

### 1.2 The Hybrid Architecture Revolution
To fix this, modern state-of-the-art models like **Falcon-H1**, **Hymba**, and **Zamba2** use a **Hybrid Architecture**:
* In every single layer, they run **Attention** alongside a **State-Space Model / Mamba-2 (SSM)**.
* The SSM maintains a **fixed-size recurrent state matrix** ($S_P$). Whether the text is 100 words or 100,000 words, $S_P$ never takes up a single byte more memory.

### 1.3 The Unsolved Dilemma
Because GPU memory is still limited, we cannot keep the full Attention KV cache forever. We must **evict (throw away)** unnecessary KV tokens.

Here is the central question:
> **"If our model already has a recurrent memory ($S_P$) running alongside Attention, why are we paying expensive GPU memory to store KV tokens that the SSM state already remembers? And why are we throwing away KV tokens that the SSM state has completely forgotten?"**

Existing eviction algorithms are completely blind to this. They treat hybrid models as if they were pure Transformers, creating massive memory waste or severe information loss.

---

## 2. Existing Solutions, Technologies Used, and Why They Fail

### 2.1 The Technologies Currently in Play
1. **Standard Attention KV Eviction (`SnapKV`, `H2O`, `PyramidKV`):**
   * *How they work:* They look only at recent attention weights (an "observation window"). If a past token received high attention recently, they keep it. If not, they discard it.
   * *Why they fail in hybrids:* They are **State-Blind**. If the SSM state $S_P$ already remembers a token with 100% precision, `SnapKV` still hoards it in the KV cache, wasting precious memory budget.
2. **Write-Time State Surprisal (`HOLA`, `DeltaS`):**
   * *How they work:* When token $t$ is first processed, they calculate how surprised the SSM state was (`HOLA`), or how much the state drifted when writing that token (`DeltaS`). If surprise was high, they save it to the KV cache.
   * *Why they fail:* They suffer from **Time Inconsistency**. A fact written at page 10 might have produced a huge surprise *at that moment*, but by page 400, exponential decay has erased it. Conversely, a high-surprise write on page 399 is still completely intact in $S_P$, yet write-time heuristics waste KV budget saving it.
3. **Pure Decay Eviction (`DASC`):**
   * *How they work:* They check the theoretical exponential decay rate ($\alpha_{t \to P}$) of the SSM to see if a token has decayed.
   * *Why they fail:* They ignore **Superposition Interference / Key Collisions**. In an SSM, memories are packed into fixed-dimensional vectors. Even if decay is low, if subsequent tokens share similar write directions, they overwrite and corrupt the earlier memory. Decay alone cannot detect this corruption.

---

## 3. What Our Project Implements and How It Solves the Problem

Our project introduces **`hybrid_state_eviction_v13`**, a unified framework operating on three distinct, mathematically rigorous mechanisms:

```
                               ┌────────────────────────────────────────────────────────┐
                               │                    PREFILL PREFIX                      │
                               │                Tokens 0, 1, ..., P-1                   │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
             ┌────────────────────────────────────────────────────────────────────────────────────────┐
             │ PILLAR 1: Realized State-Survival Readout & Logistic Calibration                      │
             │                                                                                        │
             │   Exact Readout Decomposition:                                                         │
             │        b̂_t · S_P  =  α_{t→P} · u_t   +   ν_t(P)                                        │
             │                     [Target Signal]    [Realized Interference]                         │
             │                                                                                        │
             │   Computes: SIR_{t,m}(P),  Cosine c_{t,m}(P),  Log-Decay ℓ_{t,m}(P)                    │
             │   Predicts: p̂_surv(t, l, m | P)  →  State-Survival Probability                         │
             └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                          │
                                                          ▼
             ┌────────────────────────────────────────────────────────────────────────────────────────┐
             │ PILLAR 2: Token-Level SA-SnapKV & Risk-Adaptive Layer Allocation                       │
             │                                                                                        │
             │   Retention Score:  s_{t,l} = Attention_Demand(D_{t,l}) × State_Deficit(1 - p̄_surv)   │
             │                                                                                        │
             │   Layer Budgets:    Allocates K_l proportionally to layer forgetting risk R_l(P)       │
             │   Outcome:          Top-K_l tokens kept in KV Cache; Sinks & Local Window protected     │
             └────────────────────────────────────────────┬───────────────────────────────────────────┘
                                                          │
                                                          ▼
             ┌────────────────────────────────────────────────────────────────────────────────────────┐
             │ PILLAR 3: Distilled Ridge State Consolidation                                          │
             │                                                                                        │
             │   Rescues top-k runner-up evicted tokens (T_cons) by writing them directly into S_P    │
             │   via regularized Cholesky Delta-Rule:                                                 │
             │                                                                                        │
             │        S_P^{cons} = S_P + B̂ᵀ (B̂ B̂ᵀ + λ I)^{-1} Γ (U - B̂ S_P)                           │
             │                                                                                        │
             │   Safety: 300-step Full-KV KL distillation tunes tiny scalar gates {γ_{l,m}, λ_{l,m}}  │
             │           to eliminate collateral damage on untouched memories.                        │
             └────────────────────────────────────────────────────────────────────────────────────────┘
```

### What It Solves:
1. **No Redundant Storage:** If $S_P$ already knows a fact, its state deficit $(1 - \bar{p}_{\text{surv}})$ is near $0$. We do not waste KV cache on it.
2. **Interference Detection:** If a token was overwritten by later text, our realized interference vector $\nu_t(P)$ catches it immediately, ensuring it stays in the KV cache.
3. **Layer Specialization:** Mamba-2 layers have vastly different decay rates. Our algorithm gives larger KV budgets to fast-decaying layers and smaller budgets to long-horizon memory layers.
4. **Rescuing Borderline Tokens:** Tokens that just missed the KV cut are safely consolidated into the SSM state without increasing KV cache size.

---

## 4. A Concrete Walkthrough Example

Consider an LLM reading an executive business report ($P = 8,192$ tokens):
* **Token 120:** *"Acquisition Price: \$4.2 Billion"* (High importance).
* **Token 2,500:** *"Q2 Widget Shipping Count: 140,000 units"* (High importance).
* **Tokens 3,000–8,000:** Extensive operational, shipping, and logistics discussion.
* **Token 8,192:** End of document prefill. The user is about to ask questions.

```
Token Position:   t=120 ($4.2B Price)               t=2,500 (140K Units)             t=8,192 (End of Prefill)
                  ┌──────────────────────┐          ┌──────────────────────┐          ┌──────────────────────┐
Attention KV:     │ Exact KV Record      │          │ Exact KV Record      │          │ Memory limit reached!│
                  └──────────────────────┘          └──────────────────────┘          │ Must evict 80% of KV │
SSM State S:      │ Decay: α_{120→P} ≈ 0 │          │ Decay: α_{2500→P}≈0.9│          └──────────────────────┘
                  │ Status: FORGOTTEN    │          │ Collision: Overwritten│
                  └──────────────────────┘          └──────────────────────┘
```

### How Different Policies Behave at Token 8,192:

1. **Standard Attention Eviction (`SnapKV`):**
   * Sees low recent attention on Token 120 and Token 2,500 during the last 64 tokens.
   * **Action:** Discards both tokens from the KV cache.
   * **Result:** Token 120 is lost forever because the SSM had decayed it, causing a hallucination.
2. **Write-Time Surprise (`HOLA` / `DeltaS`):**
   * Checks how surprising the tokens were when written. Both were unexpected numbers, so both get high surprise scores.
   * **Action:** Keeps both tokens, or prioritizes whichever had higher write-time surprise.
   * **Result:** Wastes budget keeping tokens that the SSM state never decayed, while dropping others that were silently overwritten later.
3. **Pure Decay (`DASC`):**
   * Notices Token 120 decayed ($\alpha_{120 \to P} \approx 0$) $\implies$ Keeps Token 120.
   * Calculates Token 2,500 had low decay ($\alpha_{2500 \to P} \approx 0.90$) $\implies$ Discards Token 2,500.
   * **Result:** Token 2,500 was heavily overwritten during tokens 3,000–8,000 by similar logistics tokens (key collision). Its actual readout in $S_P$ is garbled noise, but decay alone couldn't see that. Token 2,500 is lost.
4. **Our Method (`SA-SnapKV` + Distilled Consolidation):**
   * Evaluates $\hat{b}_t S_P$ directly at position $P$:
     * Token 120: Target signal is $0$. State deficit is $1.0$. **Action: Must retain in KV cache.**
     * Token 2,500: Decay is low, but interference norm $\|\nu_{2500}(P)\|_2^2$ is very high. State deficit is $0.85$. **Action: Retain in KV cache.**
     * Unneeded filler tokens that $S_P$ still recalls with high $\text{SIR}$ have deficit $\approx 0$. **Action: Evicted.**
   * Runner-up tokens that just missed the KV cutoff are written directly into $S_P$ via ridge consolidation without costing any KV memory.

---

## 5. Critical Catches, Flaws Fixed, and Creative Innovations in V13

During technical audits leading up to V13, eight major flaws were caught and completely redesigned:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE 8 CRITICAL V13 CORRECTIONS                            │
├────┬─────────────────────────────┬─────────────────────────────────────────────────────┤
│ #  │ Flaw Caught in Audit        │ Exact Resolution Implemented in V13                 │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 1  │ Block Granularity (B=32)    │ Switched strictly to Token-Level (B=1) eviction.    │
│    │ smeared out single needles  │ Preserves exact individual key-value pairs.         │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 2  │ Naive Linear Projection     │ Hooks FalconH1Mixer strictly AFTER depthwise        │
│    │ Hook missed conv1d & μP     │ conv1d, SiLU activation, and μP multiplier vectors. │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 3  │ False Memory Accounting     │ Corrected SSM state size math (196K-262K floats/    │
│    │ (assumed S_P was 32 tokens) │ layer = 768-1024 token-equivalents). Reported as    │
│    │                             │ standardized Retained KV Cache Ratio (5% to 50%).   │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 4  │ Inconsistent SIR Metric     │ Replaced mismatched ratios with time-consistent     │
│    │ (mixed decayed & undecayed) │ Decayed Signal-to-Interference Ratio (SIR) + Cosine.│
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 5  │ Loose Proposition 2 Bound   │ Upgraded to exact spectral singular-value bound:    │
│    │ on spectral perturbation    │ ||B̂ᵀ(B̂ B̂ᵀ + λI)⁻¹||₂ = max_i σ_i / (σ_i² + λ).     │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 6  │ Hand-Tuned Consolidation    │ Replaced dangerous manual tuning with a 300-step    │
│    │ risked corrupting memory    │ Full-KV Teacher KL Distillation on tiny scalar gates│
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 7  │ Recency Confounder in R5    │ Added Age-Controlled Partial Correlation (r_{p,y|d})│
│    │ (decay correlates with age) │ on natural text to prove signal value beyond age.   │
├────┼─────────────────────────────┼─────────────────────────────────────────────────────┤
│ 8  │ Question Leakage in RULER   │ Split into Mode A (Prospective Context-First) vs    │
│    │ (prompt end saw question)   │ Mode B (Standard) to prevent prefill leakage.       │
└────┴─────────────────────────────┴─────────────────────────────────────────────────────┘
```

---

## 6. The Three Pillars Explained in Depth

### Pillar 1: The Exact Tensor Hook & Calibrated Survival Predictor ($\hat{p}_{\text{surv}}$)

In `FalconH1Mixer`, the model processes tokens through several stages before the Mamba-2 selective scan:

$$\text{Input } h_t \xrightarrow{\mu\text{P}} \text{in\_proj} \xrightarrow{\text{Split}} \text{xBC}_{\text{raw}} \xrightarrow{\text{conv1d} + \text{SiLU}} \xrightarrow{\mu\text{P multipliers}} (x_t, B_t, C_t, \Delta_t)$$

#### The Mathematical Innovation
We extract $(x_t, B_t, C_t, \Delta_t)$ strictly after these operations. We then normalize the write key into a unit direction $\hat{b}_t = B_t / (\|B_t\|_2 + \epsilon)$ and scale the effective value vector $u_{t,m} = \Delta_{t,m} (\|B_t\|_2 + \epsilon) x_{t,m}$.

At the end of prefill $P$, multiplying the final recurrent state $S_{P,m}$ by $\hat{b}_t$ yields:

$$\hat{b}_t S_{P,m} = \underbrace{\alpha_{t \to P, m} u_{t,m}}_{\text{Target Signal } s_{t,m}(P)} + \underbrace{\sum_{\tau \neq t} \alpha_{\tau \to P, m} \langle \hat{b}_t, \hat{b}_\tau \rangle u_{\tau, m}}_{\text{Realized Superposition Interference } \nu_{t,m}(P)}$$

Instead of computing pairwise cross-terms ($O(P^2)$ complexity), we compute the realized interference vector $\nu_{t,m}(P)$ in **$O(1)$ per token** via a single subtraction:

$$\nu_{t,m}(P) = \hat{b}_t S_{P,m} - \alpha_{t \to P, m} u_{t,m}$$

From this, we construct three dimensionless features:
1. **Decayed Signal-to-Interference Ratio ($\text{SIR}_{t,m}(P)$):** $\|s_{t,m}(P)\|_2^2 / (\|\nu_{t,m}(P)\|_2^2 + \epsilon)$
2. **Readout Cosine Alignment ($c_{t,m}(P)$):** Directional alignment between $\hat{b}_t S_{P,m}$ and $u_{t,m}$.
3. **Cumulative Log-Survival ($\ell_{t,m}(P)$):** $-\sum_{\tau=t+1}^{P-1} \Delta_{\tau,m} A_m$

These feed into a calibrated 4-parameter logistic model to produce $\hat{p}_{\text{surv}}(t,l,m \mid P) \in (0, 1)$, which is weighted across heads by write energy to get $\bar{p}_{\text{surv}, t, l}(P)$.

---

### Pillar 2: Token-Level State-Aware KV Eviction (`SA-SnapKV`) & Risk-Adaptive Budgets

#### 1. Token Retention Scoring
At the eviction point, we retain the 4 attention sinks and the 64-token local observation window. For all candidate tokens:

$$s_{t,l} = D_{t,l} \cdot \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$

* If a token has **high attention demand** ($D_{t,l}$) AND the SSM state has **forgotten it** ($(1 - \bar{p}_{\text{surv}}) \approx 1$), the score is maximized $\implies$ **Retain in KV cache**.
* If a token is heavily attended to, but the SSM state **remembers it cleanly** ($\bar{p}_{\text{surv}} \approx 1$), the deficit is $\approx 0 \implies$ **Safe to evict from KV cache**.

#### 2. Risk-Adaptive Layerwise Budgeting
Different layers have different memory responsibilities. We calculate each layer's total expected forgetting risk:

$$\mathcal{R}_l(P) = \sum_{t} D_{t,l} \big(1 - \bar{p}_{\text{surv}, t, l}(P)\big)$$

We allocate each layer's KV budget $K_l$ proportionally to $\mathcal{R}_l(P)$ using the Largest Remainder Method, ensuring exact integer budget conservation:

$$\sum_{l=1}^L K_l = K_{\text{total}}$$

---

### Pillar 3: Distilled Ridge State Consolidation

#### Why We Call It "Consolidation"
In cognitive neuroscience, **memory consolidation** is the biological process where temporary, short-term memories stored in the hippocampus are transferred into stable, long-term representations in the neocortex.
* In our hybrid model, the **Attention KV Cache** is the hippocampus (exact, fragile, high-capacity short-term buffer).
* The **Recurrent SSM State ($S_P$)** is the neocortex (compact, long-term parametric state).
* When we evict runner-up tokens from the KV cache, we **consolidate** their missing representations into $S_P$ so they are not permanently discarded.

#### Why We Use "Distillation" to Govern Consolidation
Writing new information into a recurrent state carries a major risk called **collateral damage**: if you write carelessly into $S_P$, you might overwrite and destroy an earlier memory that the model was preserving.

To prevent this, each layer and head has a consolidation gate $\gamma_{l,m} = \sigma(\tilde{\gamma}_{l,m})$ and a ridge parameter $\lambda_{l,m} = \exp(\tilde{\lambda}_{l,m})$.

Instead of guessing these parameters by hand, we perform a **300-step Full-KV Teacher KL Distillation**:
1. We freeze the entire LLM backbone (zero weights updated).
2. We only update the scalar gates $\{\tilde{\gamma}_{l,m}, \tilde{\lambda}_{l,m}\}$ ($1,728$ parameters on `0.5B`; $3,168$ parameters on `1.5B-Deep`).
3. The model compares its evicted+consolidated output logits against the untouched Full-KV teacher model using KL divergence.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   WHY DISTILLATION FITS THE PROJECT                    │
├────────────────────────────────────────────────────────────────────────┤
│ • Automatic Safety: If writing to a long-horizon head hurts recall,    │
│   the KL loss gradient instantly drives γ_{l,m} → 0 (closes the gate). │
│ • Fast Convergence: With only ~1,700 scalar parameters to train,       │
│   distillation converges in ~15 minutes on free Kaggle T4 GPUs.        │
│ • No Manual Hyperparameter Sweeps: The model automatically discovers   │
│   which layers and heads can accept consolidated memory safely.        │
└────────────────────────────────────────────────────────────────────────┘
```

The mathematical update is solved in closed-form via a tiny $k \times k$ positive-definite linear system ($k \in \{32, 64\}$ runner-up tokens):

$$S_{P,l,m}^{\text{cons}} = S_{P,l,m} + \hat{B}_l^\top \big(\hat{B}_l \hat{B}_l^\top + \lambda_{l,m} I_k\big)^{-1} \Gamma_{l,m} \big(U_{l,m} - \hat{B}_l S_{P,l,m}\big)$$

---

## 7. Complete System Architecture & Execution Pipeline

The complete end-to-end execution lifecycle of `hybrid_state_eviction_v13` progresses through four sequential stages:

```
===================================================================================================
                                  END-TO-END EXECUTION PIPELINE
===================================================================================================

STAGE 1: Prefill & State Interception (Tokens 0 ... P-1)
---------------------------------------------------------------------------------------------------
  Input Tokens  ──►  FalconH1ForCausalLM (FP16 Backbone on Kaggle 2×T4)
                           │
                           ▼
                     FalconH1Mixer Layer l
                     ├── in_proj + ssm_in_multiplier
                     ├── conv1d (causal, kernel=4) + SiLU
                     └── μP multipliers: (μ_x, μ_B, μ_C, μ_dt)
                           │
                           ├──► [FP32 State Hook]
                           │      Extracts: x_{t,m}, B_t, C_t, Δ_{t,m}, A_m
                           │      Computes: Unit keys b̂_t, Values u_{t,m}, Decay α_{t→P, m}
                           │      Reads: Terminal recurrent state S_{P,l,m}
                           │
                           └──► Full Prefill Key-Values K_{P,l}, V_{P,l} materialized in cache

STAGE 2: Realized Survival Analysis & Metric Extraction
---------------------------------------------------------------------------------------------------
  For each candidate token t ∈ [4, P - W_obs - 1] across all layers l and heads m:
    1. Readout projection:          r_{t,m}(P) = b̂_t · S_{P,m}
    2. Realized interference:       ν_{t,m}(P) = r_{t,m}(P) - α_{t→P,m} u_{t,m}
    3. Three features computed:     SIR_{t,m}(P), Cosine c_{t,m}(P), Log-Decay ℓ_{t,m}(P)
    4. Calibrated survival:         p̂_surv(t,l,m | P) = σ(w₁ log(SIR) + w₂ c + w₃ ℓ + b₀)
    5. Write-weighted layer avg:    p̄_surv, t, l(P) = Σ_m (w_{t,l,m} p̂_surv) / (Σ_m w_{t,l,m})

STAGE 3: Risk-Adaptive Eviction & Physical Cache Slicing
---------------------------------------------------------------------------------------------------
  For each layer l ∈ {1, ..., L}:
    1. Compute Attention Demand:    D_{t,l} from observation window W_obs
    2. Retention Score:             s_{t,l} = D_{t,l} × (1 - p̄_surv, t, l(P))
    3. Layer Forgetting Risk:       R_l(P) = Σ_t D_{t,l} (1 - p̄_surv, t, l(P))
    4. Proportional Budget:         K_l = LargestRemainder(K_total · R_l / Σ R)
    5. Partition tokens:
       - Retained KV:               T_ret,l = TopK(s_{t,l}; K_l)
       - Protected:                 Sinks {0,1,2,3} ∪ Observation Window [P-64, P-1]
       - Runner-Up Evicted:         T_cons,l = Rank_{K_l+1 ... K_l+k}(s_{t,l}) (k = 32 or 64)
    6. Physical Slicing:            torch.gather on past_key_values (original RoPE IDs preserved)

STAGE 4: Distilled State Consolidation & Suffix Generation
---------------------------------------------------------------------------------------------------
  For runner-up tokens T_cons,l:
    1. Stack unit write keys B̂_l and residual errors E_{l,m} = U_{l,m} - B̂_l S_{P,l,m}
    2. Modulate by deficit gain:    Γ_{l,m} = γ_{l,m} · diag(1 - p̂_surv(t_r, l, m | P))
    3. Solve regularized system:    Z_{l,m} = (B̂_l B̂_lᵀ + λ_{l,m} I_k)⁻¹ (Γ_{l,m} E_{l,m})
    4. Update SSM state:            S_{P,l,m}^{cons} = S_{P,l,m} + B̂_lᵀ Z_{l,m}
    5. Suffix Generation (t ≥ P):   Model generates responses using S_{P}^{cons} and sliced KV cache.
===================================================================================================
```

---

## 8. Summary of Theoretical Guarantees & Hardware Feasibility

| Dimension | Engineering Specification | Verification Status |
| :--- | :--- | :--- |
| **Compute Environment** | Kaggle Free Tier: $2 \times \text{NVIDIA T4 16GB GPUs}$ | Guaranteed fitting within 16GB per GPU via FP16 backbone + FP32 state hooks. |
| **Backbone Models** | `Falcon-H1-0.5B-Instruct` & `Falcon-H1-1.5B-Deep-Instruct` | Verified official architectures; $L=36$ and $L=66$. |
| **Pillar 1 Complexity** | $O(P \cdot d_{\text{state}} \cdot d_{\text{head}})$ matrix multiplication | Avoids naive $O(P^2)$ pairwise cross-terms; executes in $<50\text{ ms}$. |
| **Pillar 2 Invariant** | Exact integer budget conservation $\sum_l K_l = K_{\text{total}}$ | Enforced via Largest Remainder Method; verified in Test 6. |
| **Pillar 3 Bounds** | Operator norm $\|\hat{B}_l^\top (\hat{B}_l \hat{B}_l^\top + \lambda I)^{-1}\|_2 \le \frac{1}{2\sqrt{\lambda}}$ | Verified via Proposition 1 and Test 5 across random and empirical key matrices. |
| **Distillation Footprint** | $1,728$ scalars (`0.5B`) / $3,168$ scalars (`1.5B-Deep`) | Backpropagation touches only scalar gates; zero gradients on backbone weights (Test 8). |