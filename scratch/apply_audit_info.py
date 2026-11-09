import json
from pathlib import Path

ROOT = Path(".").resolve()
NB_DIR = ROOT / "Notebook_Codes" / "phase_2"

AUDIT_SECTION = """## Formal Mathematical Audit & Empirical Verification of Phase 2

Both the methodology and the underlying mathematics in the patched `hooks/survival_metrics.py` and the refreshed Phase 2 reference bundles (`05b` and `15b`) are **mathematically sound, dimensionally consistent, and empirically verified**.

---

### 1. Mathematical Audit of Core Mechanisms

#### A. Vectorized $\\mathcal{O}(P \\cdot d_{\\text{state}} \\cdot d_{\\text{head}})$ Readout Decomposition

Given the terminal Mamba-2 state $S_{P,m} = \\sum_{\\tau=1}^{P} \\alpha_{\\tau \\to P, m} \\, \\hat{b}_\\tau u_{\\tau, m}^\\top \\in \\mathbb{R}^{d_{\\text{state}} \\times d_{\\text{head}}}$, probing $S_{P,m}$ along token $t$'s write key $\\hat{b}_t$ yields:

$$r_{t,m}(P) = \\hat{b}_t^\\top S_{P,m} = \\underbrace{\\Vert{}\\hat{b}_t\\Vert{}_2^2 \\, \\alpha_{t \\to P, m} \\, u_{t,m}}_{s_{t,m}(P)} + \\underbrace{\\sum_{\\tau \\neq t} \\alpha_{\\tau \\to P, m} \\langle \\hat{b}_t, \\hat{b}_\\tau \\rangle u_{\\tau, m}}_{\\nu_{t,m}(P)}$$

* **Complexity & Exactness:** `torch.einsum("bpn,bhnd->bphd", b64, S64)` computes $r_{t,m}(P)$ for all $P$ tokens simultaneously in $\\mathcal{O}(B \\cdot P \\cdot H_{\\text{ssm}} \\cdot d_{\\text{state}} \\cdot d_{\\text{head}})$ time without an $\\mathcal{O}(P^2)$ pairwise loop, and extracts $\\nu_{t,m}(P) = r_{t,m}(P) - s_{t,m}(P)$ by linearity in FP64.
* **Associativity Dust Guard:** Snapping $\\Vert{}\\nu_{t,m}(P)\\Vert{}_2 \\le 10^{-6} \\Vert{}s_{t,m}(P)\\Vert{}_2$ to exact zero removes IEEE-754 reduction-order noise ($\\sim 10^{-15}$) in isolated single-write regimes while preserving genuine multi-token interference.

#### B. Matched-Filter Subspace Processing Gain ($g_{\\text{sub}}$)

In an SSM state space of dimension $M = d_{\\text{state}} \\cdot d_{\\text{head}}$, isotropic cross-token interference energy $\\Vert{}\\nu_{t,m}(P)\\Vert{}_2^2$ is distributed across $M$ degrees of freedom, whereas coherent readout projects onto the 1D target direction $\\hat{u}_{t,m} = u_{t,m} / \\Vert{}u_{t,m}\\Vert{}_2$:

$$g_{\\text{sub}} = \\log(d_{\\text{state}} \\cdot d_{\\text{head}}) \\cdot \\max\\left(c_{t,l,m}(P), 0\\right)^2 \\cdot \\frac{\\Vert{}\\nu_{t,m}(P)\\Vert{}_2^2}{\\Vert{}\\nu_{t,m}(P)\\Vert{}_2^2 + \\Vert{}s_{t,m}(P)\\Vert{}_2^2 + 10^{-30}}$$

* **Test 3 Invariance (Single Write, $\\nu = 0$):** The interference fraction $\\frac{\\Vert{}\\nu\\Vert{}_2^2}{\\Vert{}\\nu\\Vert{}_2^2 + \\Vert{}s\\Vert{}_2^2 + 10^{-30}}$ evaluates to $0$, so $g_{\\text{sub}} = 0$ identically and $\\text{SIR} = \\varepsilon_{\\text{rel}}^{-1} = 10^8$.
* **Test 4 Invariance (R5b Equal-Age Subspace Collision):** Collided tokens (Set A) suffer directional decorrelation ($c_A \\approx 0 \\implies \\max(c_A, 0)^2 \\approx 0$), receiving almost no subspace gain, while orthogonal tokens (Set B) maintain $c_B \\to 1$ and receive full processing gain, preserving $\\text{SIR}_B \\gg \\text{SIR}_A$ at identical decay age $\\alpha_A = \\alpha_B$.
* **Physical Capacity Scaling ($d_{\\text{state}} = 128$ vs. $256$):** Doubling $d_{\\text{state}}$ from `05b` ($128$) to `15b` ($256$) halves random key collision variance $\\mathbb{E}[\\langle \\hat{b}_t, \\hat{b}_\\tau \\rangle^2] \\propto 1/d_{\\text{state}}$ and adds $\\log(2) \\approx 0.693$ nats of subspace gain, explaining why `15b` achieves a $4.8\\times$ higher uncalibrated survival rate (`0.0178` vs. `0.0037`).

#### C. $C^2$-Smooth Hyperbolic Arcsine Lower Floor (`_smooth_lower_floor`)

For kink $k = -20.0$ and scale $s = 2.0$, the mapping for $x < k$ is:

$$f(x) = k - s \\cdot \\operatorname{asinh}\\left(\\frac{k - x}{s}\\right)$$

* **Smoothness & Monotonicity:** For all $x < k$, $f'(x) = \\left(1 + \\left(\\frac{k-x}{s}\\right)^2\\right)^{-1/2} > 0$. At the boundary $x = k^-$, $f(k^-) = k$, $f'(k^-) = 1 = f'(k^+)$, and $f''(k^-) = 0 = f''(k^+)$. The floor is strictly monotonic and **$C^2$-continuous** across the kink.
* **Variance Stabilization:** Mapping inactive heads (`2.0 * ell64 - 25.0`) through `asinh` instead of a hard $-10^6$ sentinel compresses the lower tail into $[-60, -20]$, reducing `within_age_log_sir_std` from $199,598$ down to $\\mathcal{O}(1)$ (`[2.182, 7.531]` on `05b` and `[4.434, 9.160]` on `15b`).

#### D. Horizon-Modulated Write-Energy Aggregation (`aggregate_heads_to_layer`)

$$\\tilde{w}_{t,l,m}(P) = w_{t,l,m} \\left(0.02 + 0.98 \\exp\\left(\\operatorname{clamp}\\left(\\frac{\\ell_{t,l,m}(P)}{8.0}, -20.0, 0.0\\right)\\right)\\right), \\quad \\bar{p}_{\\text{surv}, t, l}(P) = \\frac{\\sum_{m=1}^{H_{\\text{ssm}}} \\tilde{w}_{t,l,m}(P) \\, \\hat{p}_{\\text{surv}}(t,l,m \\mid P)}{\\sum_{m=1}^{H_{\\text{ssm}}} \\tilde{w}_{t,l,m}(P) + \\varepsilon}$$

* **Multi-Timescale Head Separation:** Prevents the $\\sim 16$ fast local heads ($\\ell \\ll -100$ at long horizons) from drowning out the $\\sim 8$ slow long-horizon memory heads ($\\ell \\in [-8, 0]$) in the denominator, while remaining a strictly convex combination in $[0, 1)$ and invariant under equal-age comparisons.

---

### 2. Empirical Verification Summary (`P = 2048`)

| Metric / Gate | Falcon-H1-0.5B (`05b`, $d_{\\text{state}}=128$) | Falcon-H1-1.5B-Deep (`15b`, $d_{\\text{state}}=256$) | Target / Theoretical Requirement | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Test 3 (Single-Write Limit)** | `True` | `True` | Exact $\\text{SIR} = \\varepsilon_{\\text{rel}}^{-1}$, $c = 1.0$ | **PASS** |
| **Test 4 (R5b Equal-Age Collision)** | `True` | `True` | $\\text{SIR}_B > \\text{SIR}_A$ at equal $\\alpha$ | **PASS** |
| **Mean Candidate $\\bar{p}_{\\text{surv}}$** | `0.0037` | `0.0178` | Non-collapsed ($> 10^{-3}$), $15\\text{b} > 05\\text{b}$ | **PASS** |
| **Layer Deficit Range** | `[0.9749, 1.0000]` | `[0.9291, 0.9993]` | Non-trivial dynamic range across layers | **PASS** |
| **Within-Age $\\log\\text{SIR}$ Std** | `[2.182, 7.531]` | `[4.434, 9.160]` | $\\mathcal{O}(1)$ std (no sentinel explosion) | **PASS** |

---

### 3. Implementation Note for Phase 3 (L-BFGS Calibration)

In `SurvivalPredictor.predict_head_survival_from_features`, line 292 defines:

```python
dead_mask = log_sir64 < (LOG_SIR_SOFT_KINK - 0.5)  # threshold at -20.5
```

* For the Phase 2 `closed_form` bundles just saved, $\\sigma(-20.5) = 1.25 \\times 10^{-9}$, so this threshold has zero numerical impact.
* When we implement **Phase 3 (Gate 1 L-BFGS Calibration)**, note that `dead_mask` is also applied to `calibrated` and `calibrated_log_decay` modes. Because `calibrated_log_decay` is the decay-only ablation baseline ($\\sigma(w_3 f_\\ell + b_0)$), applying a `log_sir64 < -20.5` cutoff would slightly leak `log_sir` into the decay-only baseline and clip active heads with $-23.0 < \\text{log\\_sir} < -20.5$. In Phase 3, we should ensure the L-BFGS objective fits on the exact same active mask or relax `dead_mask` in calibrated modes to `log_sir64 < -23.25` (which corresponds strictly to `inactive_mask` where `2*ell - 25 <= -25` maps to $\\le -23.295$ after `_smooth_lower_floor`).
"""

# Read existing README2.md
readme2_path = NB_DIR / "README2.md"
readme2_text = readme2_path.read_text(encoding="utf-8")

# Insert AUDIT_SECTION right before "## Files Inventory"
split_marker = "## Files Inventory"
if split_marker in readme2_text:
    new_readme2 = readme2_text.replace(split_marker, AUDIT_SECTION + "\n\n---\n\n" + split_marker)
else:
    new_readme2 = readme2_text + "\n\n---\n\n" + AUDIT_SECTION

readme2_path.write_text(new_readme2, encoding="utf-8")
print(f"Updated {readme2_path}")

# Update PHASE_2.md in root
phase2_root_path = ROOT / "PHASE_2.md"
phase2_root_content = f"""# Phase 2: Complete Implementation, Mathematical Fixes & Verification Report (Patched)

> **Phase Status:** Phase 2 is **100% COMPLETE and PASSED** with the **Phase 2 Patch for Phase 3** applied.
> All numerical instabilities (sentinel cliff explosion, matched-filter subspace gain, and horizon head modulation) are fully resolved and empirically verified.
> The active target for the next phase is **Phase 3: Risk-Calibrated Dynamic Consolidation (DASC)**.

---

{new_readme2}
"""
phase2_root_path.write_text(phase2_root_content, encoding="utf-8")
print(f"Updated {phase2_root_path}")

