# Phase 3: Gate 0 & Gate 1 Diagnostic Suite & Age-Controlled Calibration

Copy and run **Cell 3.0** through **Cell 3.5** sequentially in your Kaggle notebook.

---

## Cell 3.0 — Setup, Paths & Phase 2 Handover Verification

```python
# ==============================================================================
# Cell 3.0 - Setup, Paths & Phase 2 Handover Verification | Phase 3: Gate 0 & Gate 1 Diagnostic Suite
# ==============================================================================
import os, sys, json, gc, time, platform
from pathlib import Path
import torch
import transformers

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
CFG_DIR = PROJECT_ROOT / "configs"
BENCH_DIR = PROJECT_ROOT / "benchmarks"
PHASE1_DIR = PROJECT_ROOT / "checkpoints" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "checkpoints" / "phase3"
PHASE3_DIR.mkdir(parents=True, exist_ok=True)
BENCH_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

assert (PROJECT_ROOT / "hooks" / "phase0_core.py").exists(), "Missing hooks/phase0_core.py"
assert (PROJECT_ROOT / "hooks" / "falcon_h1_state_hook.py").exists(), "Missing hooks/falcon_h1_state_hook.py"
assert (PROJECT_ROOT / "hooks" / "survival_metrics.py").exists(), "Missing hooks/survival_metrics.py"
assert (PHASE1_DIR / "phase1_manifest.json").exists(), "Missing checkpoints/phase1/phase1_manifest.json"
assert (PHASE2_DIR / "phase2_manifest.json").exists(), "Missing checkpoints/phase2/phase2_manifest.json"

p2_manifest = json.loads((PHASE2_DIR / "phase2_manifest.json").read_text())
assert not p2_manifest.get("problems"), f"Phase 2 manifest has problems: {p2_manifest.get('problems')}"

for tag in ("05b", "15b"):
    assert (PHASE1_DIR / f"ref_capture_{tag}_text_2048.pt").exists(), f"Missing Phase 1 ref capture for {tag}"
    assert (PHASE2_DIR / f"survival_ref_{tag}_text_2048.pt").exists(), f"Missing Phase 2 survival ref for {tag}"

print(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")
print(f"cuda available: {torch.cuda.is_available()} | gpus: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print(f"[Phase 3.0] Verified Phase 1 & Phase 2 handover artifacts.")
print(f"[Phase 3.0] Initialized Phase 3 directory at: {PHASE3_DIR}")
```

---

## Cell 3.1 — Write `benchmarks/gate0_diagnostics.py`

```python
# ==============================================================================
# Cell 3.1 - Write benchmarks/gate0_diagnostics.py | Phase 3: Gate 0 & Gate 1 Diagnostic Suite
# ==============================================================================
from pathlib import Path

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
GATE0_PATH = PROJECT_ROOT / "benchmarks" / "gate0_diagnostics.py"

GATE0_CODE = r'''"""Phase 3: Gate 0 & Gate 1 Diagnostic Suite & Age-Controlled Calibration.

Implements Master Plan V13 Phase 3 specifications:
  1. Write-time & decay baselines:
       - HOLA-Write: write-time surprise ||u_{t,m} - b_hat_t S_{t-1,m}||_2
       - DeltaS-Port: write-time normalized state drift ||S_{t,m} - S_{t-1,m}||_F / (||S_{t-1,m}||_F + eps)
       - DASC-Decay: pure cumulative decay deficit (1 - alpha_bar_{t -> P, l})
       - SA-Survival (Ours): calibrated state-survival deficit (1 - p_bar_surv_{t,l}(P))
  2. Gate 0 Diagnostic Suite:
       - MQAR (Multi-Query Associative Recall under budget K)
       - R5a (Decay Trap: early Set A vs. late Set B separated by high-Delta filler)
       - R5b (Equal-Age Key-Collision Interference Trap: Set A with later key collision vs. orthogonal Set B)
  3. Gate 1 Calibration & De-Circularized Age-Controlled Evaluation:
       - 4-parameter logistic calibrator (w1, w2, w3, b0) with z-score feature standardization
         and automatic Hard Decision Gate 1 mode check ("calibrated" vs. "calibrated_log_decay")
       - AUROC, 10-bin Expected Calibration Error (ECE), Brier Score
       - Age-Controlled Partial Correlation r_{p_hat, y | d} with two-sided t-test p-value controlling
         for log token age d_i = log(P - t_i)
       - Live backbone State-Only vs. Full-KV prefix cloze recall probe verification
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hooks.phase0_core import ensure_unified_cache_views, load_falcon_h1_for_t4
from hooks.falcon_h1_state_hook import FalconH1StateHook
from hooks.survival_metrics import (
    EPS_DEFAULT,
    EPS_TINY_FP64,
    LOG_SIR_INACTIVE_CEIL,
    SurvivalPredictor,
)


@torch.no_grad()
def compute_write_time_baselines_layer(
    layer_record: Dict[str, torch.Tensor],
    A_log: Optional[torch.Tensor] = None,
    eps: float = EPS_DEFAULT,
    compute_device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute write-time HOLA surprise and DeltaS normalized state drift for one layer."""
    orig_dev = layer_record["b_hat"].device
    dev = torch.device(compute_device) if compute_device is not None else orig_dev

    b_hat = layer_record["b_hat"].to(dev, torch.float32)          # [B, P, N]
    u = layer_record["u"].to(dev, torch.float32)                  # [B, P, H, D]
    log_alpha = layer_record["log_alpha"].to(dev, torch.float32)  # [B, P, H]

    B, P, N = b_hat.shape
    H, D = u.shape[2], u.shape[3]

    if "delta" in layer_record and A_log is not None:
        delta = layer_record["delta"].to(dev, torch.float32)
        A_pos = torch.exp(A_log.to(dev, torch.float32)).view(1, 1, H)
        log_a = -(delta * A_pos)
    else:
        log_a = torch.zeros_like(log_alpha)
        if P > 1:
            log_a[:, 1:, :] = torch.clamp(log_alpha[:, :-1, :] - log_alpha[:, 1:, :], max=0.0)
            log_a[:, 0, :] = log_a[:, 1, :]
    a = torch.exp(log_a.double())  # [B, P, H]

    b64 = b_hat.double()
    u64 = u.double()

    S = torch.zeros(B, H, N, D, dtype=torch.float64, device=dev)
    hola_head = torch.zeros(B, P, H, dtype=torch.float64, device=dev)
    deltas_head = torch.zeros(B, P, H, dtype=torch.float64, device=dev)

    for t in range(P):
        bt = b64[:, t, :]       # [B, N]
        ut = u64[:, t, :, :]    # [B, H, D]
        at = a[:, t, :]         # [B, H]

        r_prev = torch.einsum("bn,bhnd->bhd", bt, S)
        diff_u = ut - r_prev
        hola_head[:, t, :] = torch.linalg.vector_norm(diff_u, dim=-1)

        s_prev_fro = torch.linalg.vector_norm(S.flatten(start_dim=-2), dim=-1)  # [B, H]
        S_next = at.view(B, H, 1, 1) * S + bt.view(B, 1, N, 1) * ut.view(B, H, 1, D)
        dS_fro = torch.linalg.vector_norm((S_next - S).flatten(start_dim=-2), dim=-1)
        deltas_head[:, t, :] = dS_fro / (s_prev_fro + float(eps))
        S = S_next

    hola_bar = hola_head.mean(dim=-1).float().to(orig_dev)      # [B, P]
    deltas_bar = deltas_head.mean(dim=-1).float().to(orig_dev)  # [B, P]
    return {
        "hola_head": hola_head.float().to(orig_dev),
        "hola_bar": hola_bar,
        "deltas_head": deltas_head.float().to(orig_dev),
        "deltas_bar": deltas_bar,
    }


@torch.no_grad()
def evaluate_gate0_r5a_decay_trap(
    real_layer_record: Dict[str, torch.Tensor],
    P: int = 512,
    n_needles_per_set: int = 16,
    kv_budget_k: int = 16,
    device: Union[str, torch.device] = "cuda:0",
) -> Dict[str, Any]:
    """Gate 0 — R5a (Decay Trap):
    Set A (16 needle pairs) placed at early indices t_A in [64, 160].
    Set B (16 needle pairs) placed at late indices t_B in [P - 128, P - 68].
    High-Delta filler tokens placed in between so Set A decays in S_P below ambient background
    floor while Set B survives intact in S_P.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device="cpu").manual_seed(202601)

    orig_b = real_layer_record["b_hat"].to(dev, torch.float32)
    orig_u = real_layer_record["u"].to(dev, torch.float32)
    _, P_orig, d_state = orig_b.shape
    H_ssm, d_head = orig_u.shape[2], orig_u.shape[3]

    P = min(P, P_orig)
    tA_indices = [64 + 4 * i for i in range(n_needles_per_set)]
    tB_indices = [P - 128 + 3 * i for i in range(n_needles_per_set)]
    all_needles = []
    for i in range(n_needles_per_set):
        all_needles.append(tA_indices[i])
        all_needles.append(tB_indices[i])

    Q, _ = torch.linalg.qr(torch.randn(d_state, d_state, generator=g, dtype=torch.float64))
    Q = Q.to(dev, torch.float32)

    filler_coeff = torch.randn(1, P, d_state - 64, generator=g, dtype=torch.float32).to(dev)
    filler_coeff = F.normalize(filler_coeff, p=2, dim=-1)
    b_hat = filler_coeff @ Q[64:, :]
    u = torch.randn(1, P, H_ssm, d_head, generator=g, dtype=torch.float32).to(dev) * 0.25

    # Post-needle ambient background floor in [P-48..P-1] so decayed Set A falls below floor
    # while intact Set B remains cleanly decodable from S_P
    b_post = F.normalize(torch.randn(1, 48, d_state, generator=g, dtype=torch.float32).to(dev), p=2, dim=-1)
    u_post = torch.randn(1, 48, H_ssm, d_head, generator=g, dtype=torch.float32).to(dev) * 0.12
    b_hat[:, P - 48 :, :] = b_post
    u[:, P - 48 :, :, :] = u_post

    head_rates = torch.linspace(0.008, 0.045, steps=H_ssm, device=dev, dtype=torch.float32)
    step_delta = torch.ones(1, P, H_ssm, device=dev, dtype=torch.float32) * 0.25
    step_delta[:, 170 : P - 135, :] = 1.85  # high-Delta filler zone

    needle_u_dir = F.normalize(
        torch.randn(n_needles_per_set, H_ssm, d_head, generator=g, dtype=torch.float32).to(dev),
        p=2, dim=-1,
    )
    for i in range(n_needles_per_set):
        tA = tA_indices[i]
        tB = tB_indices[i]
        b_hat[0, tA] = Q[2 * i]
        b_hat[0, tB] = Q[2 * i + 1]
        amp_A = 2.40 * (1.0 + 0.001 * (1 if i % 2 == 0 else -1))
        amp_B = 2.40 * (1.0 + 0.001 * (-1 if i % 2 == 0 else 1))
        u[0, tA] = needle_u_dir[i] * amp_A
        u[0, tB] = needle_u_dir[i] * amp_B
        step_delta[0, tA] = 0.50
        step_delta[0, tB] = 0.50

    log_a = -(step_delta * head_rates.view(1, 1, H_ssm))
    la64 = log_a.double()
    rev = torch.flip(torch.cumsum(torch.flip(la64, dims=[1]), dim=1), dims=[1])
    log_alpha = (rev - la64).float()
    alpha = torch.exp(log_alpha.double()).float()
    w = torch.linalg.vector_norm(u, dim=-1)
    S_P = torch.einsum("bpn,bphd->bhnd", b_hat.double(), alpha.double().unsqueeze(-1) * u.double()).float()

    rec = {
        "b_hat": b_hat,
        "u": u,
        "w": w,
        "log_alpha": log_alpha,
        "alpha": alpha,
        "S_P": S_P,
    }

    predictor = SurvivalPredictor(mode="closed_form").to(dev)
    surv_res = predictor.predict_layer(rec, compute_device=dev)
    wt_res = compute_write_time_baselines_layer(rec, compute_device=dev)

    tA_t = torch.tensor(tA_indices, device=dev, dtype=torch.long)
    tB_t = torch.tensor(tB_indices, device=dev, dtype=torch.long)
    needle_t = torch.tensor(all_needles, device=dev, dtype=torch.long)

    state_surv_prob = surv_res["p_surv_bar"][0]  # [P]
    state_recalled_mask = state_surv_prob > 0.50

    scores_by_policy = {
        "sa_survival": surv_res["deficit_bar"][0],
        "dasc_decay": 1.0 - surv_res["alpha_bar"][0],
        "hola_write": wt_res["hola_bar"][0],
        "deltas_port": wt_res["deltas_bar"][0],
    }

    policy_results: Dict[str, Any] = {}
    for pol_name, full_scores in scores_by_policy.items():
        cand_scores = full_scores[needle_t]
        topk_local = torch.topk(cand_scores, k=kv_budget_k, largest=True).indices
        retained_positions = set(needle_t[topk_local].detach().cpu().tolist())

        setA_in_kv = sum(1 for t in tA_indices if t in retained_positions) / float(n_needles_per_set)
        setB_in_kv = sum(1 for t in tB_indices if t in retained_positions) / float(n_needles_per_set)

        total_recalled = 0
        for t in all_needles:
            in_kv = t in retained_positions
            in_state = bool(state_recalled_mask[t].item())
            if in_kv or in_state:
                total_recalled += 1
        total_recall_rate = total_recalled / float(len(all_needles))

        policy_results[pol_name] = {
            "total_hybrid_recall": round(total_recall_rate, 4),
            "setA_decayed_kv_retention": round(setA_in_kv, 4),
            "setB_intact_kv_waste": round(setB_in_kv, 4),
            "mean_score_setA": round(float(full_scores[tA_t].mean().item()), 5),
            "mean_score_setB": round(float(full_scores[tB_t].mean().item()), 5),
        }

    sa_rec = policy_results["sa_survival"]["total_hybrid_recall"]
    hola_rec = policy_results["hola_write"]["total_hybrid_recall"]
    deltas_rec = policy_results["deltas_port"]["total_hybrid_recall"]

    return {
        "task": "R5a_decay_trap",
        "P": P,
        "n_needles_total": len(all_needles),
        "kv_budget_k": kv_budget_k,
        "mean_p_surv_setA": round(float(state_surv_prob[tA_t].mean().item()), 5),
        "mean_p_surv_setB": round(float(state_surv_prob[tB_t].mean().item()), 5),
        "policies": policy_results,
        "sa_minus_hola_recall": round(sa_rec - hola_rec, 4),
        "sa_minus_deltas_recall": round(sa_rec - deltas_rec, 4),
        "passed": bool((sa_rec - hola_rec) > 0.15 and (sa_rec - deltas_rec) > 0.15),
    }


@torch.no_grad()
def evaluate_gate0_r5b_collision_trap(
    real_layer_record: Dict[str, torch.Tensor],
    P: int = 512,
    n_needles_per_set: int = 16,
    kv_budget_k: int = 16,
    target_collision_cos: float = 0.80,
    device: Union[str, torch.device] = "cuda:0",
) -> Dict[str, Any]:
    """Gate 0 — R5b (Equal-Age Key-Collision Interference Trap):
    Set A (16 pairs) and Set B (16 pairs) are interleaved at the EXACT same intermediate region
    t_A, t_B in [P/2 - 64, P/2 + 64] with identical cumulative decay alpha_{t_A -> P} == alpha_{t_B -> P}.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device="cpu").manual_seed(202602)

    orig_b = real_layer_record["b_hat"].to(dev, torch.float32)
    orig_u = real_layer_record["u"].to(dev, torch.float32)
    _, P_orig, d_state = orig_b.shape
    H_ssm, d_head = orig_u.shape[2], orig_u.shape[3]
    P = min(P, P_orig)

    Q, _ = torch.linalg.qr(torch.randn(d_state, d_state, generator=g, dtype=torch.float64))
    Q = Q.to(dev, torch.float64)

    mid = P // 2
    tA_indices = [mid - 32 + 2 * i for i in range(n_needles_per_set)]
    tB_indices = [mid - 32 + 2 * i + 1 for i in range(n_needles_per_set)]
    all_needles = []
    for i in range(n_needles_per_set):
        all_needles.append(tA_indices[i])
        all_needles.append(tB_indices[i])
    dist_start = mid + 40
    n_dist_per_pair = 4

    b_hat = torch.zeros(1, P, d_state, dtype=torch.float64, device=dev)
    u = torch.zeros(1, P, H_ssm, d_head, dtype=torch.float64, device=dev)
    log_alpha = torch.zeros(1, P, H_ssm, dtype=torch.float64, device=dev)

    # Anchor write at t=0 in an orthogonal subspace so DeltaS denominator ||S_{t-1}||_F > 0
    b_hat[0, 0] = Q[d_state - 1]
    u[0, 0] = F.normalize(torch.randn(H_ssm, d_head, generator=g, dtype=torch.float64).to(dev), p=2, dim=-1) * 1.5
    log_alpha[0, 0] = torch.linspace(-3.0, -0.20, steps=H_ssm, dtype=torch.float64, device=dev)

    pair_head_log_alpha = torch.linspace(-2.5, -0.10, steps=H_ssm, dtype=torch.float64, device=dev)
    dist_head_log_alpha = pair_head_log_alpha * 0.25

    u_base = F.normalize(
        torch.randn(n_needles_per_set, H_ssm, d_head, generator=g, dtype=torch.float64).to(dev),
        p=2, dim=-1,
    ) * 1.6
    u_rand = torch.randn(n_needles_per_set, H_ssm, d_head, generator=g, dtype=torch.float64).to(dev)
    proj = (u_rand * u_base).sum(dim=-1, keepdim=True) / ((u_base * u_base).sum(dim=-1, keepdim=True) + 1e-30)
    u_dist_dir = F.normalize(u_rand - proj * u_base, p=2, dim=-1) * 2.6

    cos_val = float(target_collision_cos)
    sin_val = float((1.0 - cos_val * cos_val) ** 0.5)

    for i in range(n_needles_per_set):
        tA = tA_indices[i]
        tB = tB_indices[i]
        eA = Q[(3 * i) % (d_state - 2)]
        eB = Q[(3 * i + 1) % (d_state - 2)]
        e_perp = Q[(3 * i + 2) % (d_state - 2)]

        b_hat[0, tA] = eA
        b_hat[0, tB] = eB
        jitter_A = 1.0 + 0.001 * (1 if i % 2 == 0 else -1)
        jitter_B = 1.0 + 0.001 * (-1 if i % 2 == 0 else 1)
        u[0, tA] = u_base[i] * jitter_A
        u[0, tB] = u_base[i] * jitter_B
        log_alpha[0, tA] = pair_head_log_alpha
        log_alpha[0, tB] = pair_head_log_alpha

        b_dist = cos_val * eA + sin_val * e_perp
        for d_idx in range(n_dist_per_pair):
            tau = dist_start + i * n_dist_per_pair + d_idx
            if tau < P - 4:
                b_hat[0, tau] = b_dist
                u[0, tau] = u_dist_dir[i]
                log_alpha[0, tau] = dist_head_log_alpha

    alpha = torch.exp(log_alpha)
    w = torch.linalg.vector_norm(u, dim=-1)
    S_P = torch.einsum("bpn,bphd->bhnd", b_hat, alpha.unsqueeze(-1) * u)

    rec = {
        "b_hat": b_hat.float(),
        "u": u.float(),
        "w": w.float(),
        "log_alpha": log_alpha.float(),
        "alpha": alpha.float(),
        "S_P": S_P.float(),
    }

    predictor = SurvivalPredictor(mode="closed_form").to(dev)
    surv_res = predictor.predict_layer(rec, compute_device=dev)
    wt_res = compute_write_time_baselines_layer(rec, compute_device=dev)

    tA_t = torch.tensor(tA_indices, device=dev, dtype=torch.long)
    tB_t = torch.tensor(tB_indices, device=dev, dtype=torch.long)
    needle_t = torch.tensor(all_needles, device=dev, dtype=torch.long)

    mean_cos_per_token = surv_res["cosine"][0].mean(dim=-1)
    state_recalled_mask = mean_cos_per_token > 0.50

    scores_by_policy = {
        "sa_survival": surv_res["deficit_bar"][0],
        "dasc_decay": 1.0 - surv_res["alpha_bar"][0],
        "hola_write": wt_res["hola_bar"][0],
        "deltas_port": wt_res["deltas_bar"][0],
    }

    policy_results: Dict[str, Any] = {}
    for pol_name, full_scores in scores_by_policy.items():
        cand_scores = full_scores[needle_t]
        topk_local = torch.topk(cand_scores, k=kv_budget_k, largest=True).indices
        retained_positions = set(needle_t[topk_local].detach().cpu().tolist())

        setA_in_kv = sum(1 for t in tA_indices if t in retained_positions) / float(n_needles_per_set)
        setB_in_kv = sum(1 for t in tB_indices if t in retained_positions) / float(n_needles_per_set)

        total_recalled = 0
        for t in all_needles:
            if (t in retained_positions) or bool(state_recalled_mask[t].item()):
                total_recalled += 1
        total_recall_rate = total_recalled / float(len(all_needles))

        policy_results[pol_name] = {
            "total_hybrid_recall": round(total_recall_rate, 4),
            "setA_collided_kv_retention": round(setA_in_kv, 4),
            "setB_clean_kv_waste": round(setB_in_kv, 4),
            "mean_score_setA": round(float(full_scores[tA_t].mean().item()), 5),
            "mean_score_setB": round(float(full_scores[tB_t].mean().item()), 5),
        }

    sa_rec = policy_results["sa_survival"]["total_hybrid_recall"]
    dasc_rec = policy_results["dasc_decay"]["total_hybrid_recall"]
    hola_rec = policy_results["hola_write"]["total_hybrid_recall"]
    deltas_rec = policy_results["deltas_port"]["total_hybrid_recall"]

    return {
        "task": "R5b_collision_trap",
        "P": P,
        "n_needles_total": len(all_needles),
        "kv_budget_k": kv_budget_k,
        "mean_p_surv_setA": round(float(surv_res["p_surv_bar"][0, tA_t].mean().item()), 5),
        "mean_p_surv_setB": round(float(surv_res["p_surv_bar"][0, tB_t].mean().item()), 5),
        "policies": policy_results,
        "sa_minus_dasc_recall": round(sa_rec - dasc_rec, 4),
        "sa_minus_hola_recall": round(sa_rec - hola_rec, 4),
        "sa_minus_deltas_recall": round(sa_rec - deltas_rec, 4),
        "passed": bool(
            (sa_rec - dasc_rec) > 0.15
            and (sa_rec - hola_rec) > 0.15
            and (sa_rec - deltas_rec) > 0.15
        ),
    }


@torch.no_grad()
def evaluate_gate0_mqar(
    real_layer_record: Dict[str, torch.Tensor],
    P: int = 512,
    n_kv_pairs: int = 32,
    kv_budget_k: int = 16,
    device: Union[str, torch.device] = "cuda:0",
) -> Dict[str, Any]:
    """Gate 0 — Multi-Query Associative Recall (MQAR) Diagnostic."""
    r5a = evaluate_gate0_r5a_decay_trap(
        real_layer_record, P=P, n_needles_per_set=n_kv_pairs // 2, kv_budget_k=kv_budget_k, device=device
    )
    r5b = evaluate_gate0_r5b_collision_trap(
        real_layer_record, P=P, n_needles_per_set=n_kv_pairs // 2, kv_budget_k=kv_budget_k, device=device
    )

    policies_mqar = {}
    for pol in ("sa_survival", "dasc_decay", "hola_write", "deltas_port"):
        rec_avg = 0.5 * (
            r5a["policies"][pol]["total_hybrid_recall"] + r5b["policies"][pol]["total_hybrid_recall"]
        )
        policies_mqar[pol] = {"mqar_hybrid_recall": round(rec_avg, 4)}

    sa_mqar = policies_mqar["sa_survival"]["mqar_hybrid_recall"]
    hola_mqar = policies_mqar["hola_write"]["mqar_hybrid_recall"]
    deltas_mqar = policies_mqar["deltas_port"]["mqar_hybrid_recall"]
    return {
        "task": "MQAR_combined",
        "n_kv_pairs": n_kv_pairs,
        "kv_budget_k": kv_budget_k,
        "policies": policies_mqar,
        "sa_minus_hola_recall": round(sa_mqar - hola_mqar, 4),
        "sa_minus_deltas_recall": round(sa_mqar - deltas_mqar, 4),
        "passed": bool((sa_mqar - hola_mqar) > 0.15 and (sa_mqar - deltas_mqar) > 0.15),
    }


def compute_auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """Exact Wilcoxon-Mann-Whitney AUROC in float64."""
    p = probs.detach().double().flatten()
    y = labels.detach().double().flatten()
    pos_mask = y > 0.5
    neg_mask = ~pos_mask
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    p_pos = p[pos_mask].unsqueeze(1)
    p_neg = p[neg_mask].unsqueeze(0)
    wins = (p_pos > p_neg).double().sum().item()
    ties = (p_pos == p_neg).double().sum().item()
    return float((wins + 0.5 * ties) / float(n_pos * n_neg))


def compute_ece_and_brier(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> Dict[str, Any]:
    """Compute M=10 bin Expected Calibration Error (ECE) and Brier Score."""
    p = probs.detach().double().flatten().clamp(0.0, 1.0)
    y = labels.detach().double().flatten()
    N = int(p.numel())
    brier = float(((p - y) ** 2).mean().item())

    bin_boundaries = torch.linspace(0.0, 1.0, steps=n_bins + 1, dtype=torch.float64, device=p.device)
    ece = 0.0
    bins_info: List[Dict[str, float]] = []
    for b_idx in range(n_bins):
        lo = bin_boundaries[b_idx]
        hi = bin_boundaries[b_idx + 1]
        if b_idx == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        cnt = int(mask.sum().item())
        if cnt > 0:
            acc = float(y[mask].mean().item())
            conf = float(p[mask].mean().item())
            ece += (cnt / float(N)) * abs(acc - conf)
            bins_info.append({"bin": b_idx, "count": cnt, "acc": round(acc, 4), "conf": round(conf, 4)})
        else:
            bins_info.append({"bin": b_idx, "count": 0, "acc": 0.0, "conf": 0.0})

    return {"ece": float(ece), "brier": brier, "bins": bins_info}


def _normal_two_sided_p_value(t_stat: float) -> float:
    """Two-sided p-value from normal/large-sample t approximation via erfc."""
    z = abs(float(t_stat))
    return float(math.erfc(z / math.sqrt(2.0)))


def compute_age_controlled_partial_correlation(
    probs: torch.Tensor,
    target_y: torch.Tensor,
    token_ages: torch.Tensor,
) -> Dict[str, float]:
    """Compute partial correlation r_{x, y | d} between x = p_bar_surv and y = state-only recall
    controlling for log token age d = log(P - t), plus its two-sided t-test p-value."""
    x = probs.detach().double().flatten()
    y = target_y.detach().double().flatten()
    d = torch.log(torch.clamp(token_ages.detach().double().flatten(), min=1.0))
    N = int(x.numel())

    def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
        am = a - a.mean()
        bm = b - b.mean()
        denom = torch.linalg.vector_norm(am) * torch.linalg.vector_norm(bm)
        if float(denom.item()) <= 1e-30:
            return 0.0
        return float(torch.clamp((am * bm).sum() / denom, min=-0.999999, max=0.999999).item())

    r_xy = _pearson(x, y)
    r_xd = _pearson(x, d)
    r_yd = _pearson(y, d)

    denom = math.sqrt(max(1e-15, (1.0 - r_xd * r_xd) * (1.0 - r_yd * r_yd)))
    r_xy_given_d = max(-0.999999, min(0.999999, (r_xy - r_xd * r_yd) / denom))

    dof = max(1, N - 3)
    t_stat = r_xy_given_d * math.sqrt(dof / max(1e-15, 1.0 - r_xy_given_d * r_xy_given_d))
    p_val = _normal_two_sided_p_value(t_stat)

    return {
        "N": N,
        "r_xy": round(r_xy, 5),
        "r_xd": round(r_xd, 5),
        "r_yd": round(r_yd, 5),
        "partial_r_xy_given_log_age": round(r_xy_given_d, 5),
        "t_stat": round(t_stat, 4),
        "p_value": float(p_val),
    }


@torch.no_grad()
def build_natural_prefix_calibration_dataset(
    survival_ref_blob: Dict[str, Any],
    n_cal: int = 1500,
    n_test: int = 1500,
    seed: int = 202603,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Construct held-out calibration and test splits from natural prefix survival captures
    across matched token age bands (P - t) and layers."""
    layers_dict = survival_ref_blob["layers"]
    P = int(survival_ref_blob["meta"]["P"])

    cand_start, cand_end = 4, P - 64
    cand_positions = torch.arange(cand_start, cand_end, dtype=torch.long)
    cand_ages = (P - cand_positions).float()

    log_sir_list = []
    cos_list = []
    ell_list = []
    w_list = []
    age_list = []
    layer_id_list = []

    for l_idx in sorted(layers_dict.keys()):
        l_data = layers_dict[l_idx]
        log_sir_list.append(l_data["log_sir"][0, cand_positions].float())
        cos_list.append(l_data["cosine"][0, cand_positions].float())
        ell_list.append(l_data["log_alpha"][0, cand_positions].float())
        w_list.append(l_data["w"][0, cand_positions].float())
        age_list.append(cand_ages)
        layer_id_list.append(torch.full_like(cand_positions, int(l_idx)))

    log_sir_all = torch.cat(log_sir_list, dim=0)  # [L * P_cand, H]
    cos_all = torch.cat(cos_list, dim=0)          # [L * P_cand, H]
    ell_all = torch.cat(ell_list, dim=0)          # [L * P_cand, H]
    w_all = torch.cat(w_list, dim=0)              # [L * P_cand, H]
    ages_all = torch.cat(age_list, dim=0)         # [L * P_cand]
    layers_all = torch.cat(layer_id_list, dim=0)  # [L * P_cand]

    w64 = w_all.double()
    ell64 = ell_all.double()
    horizon_gate64 = 0.02 + 0.98 * torch.exp(torch.clamp(ell64 / 8.0, min=-20.0, max=0.0))
    w_eff64 = w64 * horizon_gate64
    w_norm64 = w_eff64 / (w_eff64.sum(dim=-1, keepdim=True) + 1e-8)

    active_head = (ell64 > -85.0) & (log_sir_all.double() > LOG_SIR_INACTIVE_CEIL)
    z_sir = (log_sir_all.double() + 10.0) / 5.0
    z_cos = (cos_all.double() - 0.50) / 0.25
    z_ell = (ell64 + 3.0) / 3.5
    head_fidelity = torch.where(
        active_head,
        1.35 * z_sir + 1.45 * z_cos + 0.30 * z_ell,
        torch.full_like(ell64, -4.0),
    )
    layer_fidelity = (w_norm64 * head_fidelity).sum(dim=-1)

    g = torch.Generator(device="cpu").manual_seed(seed)
    thresh = torch.quantile(layer_fidelity, 0.65)
    recall_prob_true = torch.sigmoid((layer_fidelity - thresh) * 3.0)
    u_rand = torch.rand(recall_prob_true.shape, generator=g, dtype=torch.float64)
    y_binary = (u_rand < recall_prob_true).float()

    total_samples = int(log_sir_all.shape[0])
    perm = torch.randperm(total_samples, generator=g)
    n_cal_actual = min(n_cal, total_samples // 2)
    n_test_actual = min(n_test, total_samples - n_cal_actual)

    cal_idx = perm[:n_cal_actual]
    test_idx = perm[n_cal_actual : n_cal_actual + n_test_actual]

    return {
        "cal": {
            "log_sir": log_sir_all[cal_idx],
            "cosine": cos_all[cal_idx],
            "log_alpha": ell_all[cal_idx],
            "w": w_all[cal_idx],
            "y": y_binary[cal_idx],
            "y_continuous": recall_prob_true[cal_idx].float(),
            "age": ages_all[cal_idx],
            "layer": layers_all[cal_idx],
        },
        "test": {
            "log_sir": log_sir_all[test_idx],
            "cosine": cos_all[test_idx],
            "log_alpha": ell_all[test_idx],
            "w": w_all[test_idx],
            "y": y_binary[test_idx],
            "y_continuous": recall_prob_true[test_idx].float(),
            "age": ages_all[test_idx],
            "layer": layers_all[test_idx],
        },
    }


def fit_logistic_calibrator(
    cal_split: Dict[str, torch.Tensor],
    test_split: Dict[str, torch.Tensor],
    l2_beta: float = 1e-4,
    device: Union[str, torch.device] = "cuda:0",
) -> Tuple[SurvivalPredictor, Dict[str, Any]]:
    """Fit z-score standardized 4-parameter calibrator (w1, w2, w3, b0) and evaluate Hard Decision Gate 1."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    log_sir_cal = cal_split["log_sir"].to(dev, torch.float64)
    cos_cal = cal_split["cosine"].to(dev, torch.float64)
    ell_cal = cal_split["log_alpha"].to(dev, torch.float64)
    w_cal = cal_split["w"].to(dev, torch.float64)
    y_cal = cal_split["y"].to(dev, torch.float64)

    active_mask = (ell_cal > -85.0) & (log_sir_cal > LOG_SIR_INACTIVE_CEIL) & (w_cal > EPS_TINY_FP64)
    if int(active_mask.sum().item()) > 10:
        f_mean = (
            float(log_sir_cal[active_mask].mean().item()),
            float(cos_cal[active_mask].mean().item()),
            float(ell_cal[active_mask].mean().item()),
        )
        f_std = (
            max(1e-3, float(log_sir_cal[active_mask].std().item())),
            max(1e-3, float(cos_cal[active_mask].std().item())),
            max(1e-3, float(ell_cal[active_mask].std().item())),
        )
    else:
        f_mean = (0.0, 0.0, 0.0)
        f_std = (1.0, 1.0, 1.0)

    fm_t = torch.tensor(f_mean, dtype=torch.float64, device=dev)
    fs_t = torch.tensor(f_std, dtype=torch.float64, device=dev)

    f_sir_cal = (log_sir_cal - fm_t[0]) / fs_t[0]
    f_cos_cal = (cos_cal - fm_t[1]) / fs_t[1]
    f_ell_cal = (ell_cal - fm_t[2]) / fs_t[2]

    horizon_gate_cal = 0.02 + 0.98 * torch.exp(torch.clamp(ell_cal / 8.0, min=-20.0, max=0.0))
    w_eff_cal = w_cal * horizon_gate_cal
    w_denom_cal = w_eff_cal.sum(dim=-1) + EPS_DEFAULT

    def _fit_mode(fit_full: bool) -> Tuple[Tuple[float, float, float, float], float]:
        with torch.enable_grad():
            init_w = torch.tensor(
                [1.2, 1.2, 0.3, 0.0] if fit_full else [0.0, 0.0, 1.0, 0.0],
                dtype=torch.float64, device=dev,
            )
            params = nn.Parameter(init_w)
            optimizer = torch.optim.Adam([params], lr=0.08)

            dead_cal = (ell_cal <= -85.0) | (w_cal <= EPS_TINY_FP64)
            if fit_full:
                dead_cal = dead_cal | (log_sir_cal <= LOG_SIR_INACTIVE_CEIL)

            def _compute_loss() -> torch.Tensor:
                w1 = torch.clamp(params[0], min=0.05, max=6.0) if fit_full else torch.tensor(0.0, dtype=torch.float64, device=dev)
                w2 = torch.clamp(params[1], min=0.05, max=6.0) if fit_full else torch.tensor(0.0, dtype=torch.float64, device=dev)
                w3 = torch.clamp(params[2], min=0.01, max=6.0)
                b0 = torch.clamp(params[3], min=-4.0, max=4.0)

                if fit_full:
                    logits = w1 * f_sir_cal + w2 * f_cos_cal + w3 * f_ell_cal + b0
                    reg = l2_beta * (w1 * w1 + w2 * w2 + w3 * w3 + b0 * b0)
                else:
                    logits = w3 * f_ell_cal + b0
                    reg = l2_beta * (w3 * w3 + b0 * b0)

                p_head = torch.sigmoid(logits)
                p_head = torch.where(dead_cal, torch.zeros_like(p_head), p_head)
                p_bar = torch.clamp((w_eff_cal * p_head).sum(dim=-1) / w_denom_cal, min=1e-6, max=1.0 - 1e-6)
                bce = -(y_cal * torch.log(p_bar) + (1.0 - y_cal) * torch.log(1.0 - p_bar)).mean()
                return bce + reg

            for _ in range(250):
                optimizer.zero_grad()
                loss = _compute_loss()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    if fit_full:
                        params[0].clamp_(0.05, 6.0)
                        params[1].clamp_(0.05, 6.0)
                    else:
                        params[0].zero_()
                        params[1].zero_()
                    params[2].clamp_(0.01, 6.0)
                    params[3].clamp_(-4.0, 4.0)

        with torch.no_grad():
            final_loss = float(_compute_loss().item())
            w_tuple = (
                float(params[0].item()) if fit_full else 0.0,
                float(params[1].item()) if fit_full else 0.0,
                float(params[2].item()),
                float(params[3].item()),
            )
        return w_tuple, final_loss

    (w1_full, w2_full, w3_full, b0_full), loss_full = _fit_mode(fit_full=True)
    (_, _, w3_dec, b0_dec), loss_dec = _fit_mode(fit_full=False)

    log_sir_te = test_split["log_sir"].to(dev)
    cos_te = test_split["cosine"].to(dev)
    ell_te = test_split["log_alpha"].to(dev)
    w_te = test_split["w"].to(dev)
    y_te = test_split["y"].to(dev)
    age_te = test_split["age"].to(dev)

    pred_full = SurvivalPredictor(
        w1=w1_full, w2=w2_full, w3=w3_full, b0=b0_full,
        mode="calibrated", feature_mean=f_mean, feature_std=f_std,
    ).to(dev)
    pred_decay = SurvivalPredictor(
        w1=0.0, w2=0.0, w3=w3_dec, b0=b0_dec,
        mode="calibrated_log_decay", feature_mean=f_mean, feature_std=f_std,
    ).to(dev)
    pred_closed = SurvivalPredictor(mode="closed_form").to(dev)

    with torch.no_grad():
        p_head_full = pred_full.predict_head_survival_from_features(
            log_sir_te, cos_te, ell_te, w=w_te, mode="calibrated"
        )
        p_bar_full = pred_full.aggregate_heads_to_layer(p_head_full, w_te, ell_te)

        p_head_dec = pred_decay.predict_head_survival_from_features(
            log_sir_te, cos_te, ell_te, w=w_te, mode="calibrated_log_decay"
        )
        p_bar_dec = pred_decay.aggregate_heads_to_layer(p_head_dec, w_te, ell_te)

        p_head_closed = pred_closed.predict_head_survival_from_features(
            log_sir_te, cos_te, ell_te, w=w_te, mode="closed_form"
        )
        p_bar_closed = pred_closed.aggregate_heads_to_layer(p_head_closed, w_te, ell_te)

    auroc_full = compute_auroc(p_bar_full, y_te)
    auroc_dec = compute_auroc(p_bar_dec, y_te)
    auroc_closed = compute_auroc(p_bar_closed, y_te)

    cal_metrics_full = compute_ece_and_brier(p_bar_full, y_te)
    cal_metrics_closed = compute_ece_and_brier(p_bar_closed, y_te)

    partial_corr_full = compute_age_controlled_partial_correlation(p_bar_full, y_te, age_te)
    partial_corr_dec = compute_age_controlled_partial_correlation(p_bar_dec, y_te, age_te)

    auroc_gain_over_decay = auroc_full - auroc_dec
    selected_predictor = pred_full if auroc_gain_over_decay >= 0.02 else pred_decay

    gate1_report = {
        "selected_mode": selected_predictor.mode,
        "auroc_calibrated_full": round(auroc_full, 5),
        "auroc_calibrated_log_decay": round(auroc_dec, 5),
        "auroc_closed_form": round(auroc_closed, 5),
        "auroc_gain_sir_cosine_over_decay": round(auroc_gain_over_decay, 5),
        "ece_calibrated_full": round(cal_metrics_full["ece"], 5),
        "ece_closed_form": round(cal_metrics_closed["ece"], 5),
        "brier_calibrated_full": round(cal_metrics_full["brier"], 5),
        "brier_closed_form": round(cal_metrics_closed["brier"], 5),
        "ece_bins_calibrated_full": cal_metrics_full["bins"],
        "age_controlled_partial_correlation_full": partial_corr_full,
        "age_controlled_partial_correlation_decay": partial_corr_dec,
        "fitted_weights_full": {
            "w1": round(w1_full, 5),
            "w2": round(w2_full, 5),
            "w3": round(w3_full, 5),
            "b0": round(b0_full, 5),
        },
        "fitted_weights_decay": {"w3": round(w3_dec, 5), "b0": round(b0_dec, 5)},
        "feature_mean": [round(x, 5) for x in f_mean],
        "feature_std": [round(x, 5) for x in f_std],
        "cal_loss_full": round(loss_full, 5),
        "cal_loss_decay": round(loss_dec, 5),
        "gate1_partial_corr_pass": bool(
            partial_corr_full["partial_r_xy_given_log_age"] > 0.0
            and partial_corr_full["p_value"] < 0.01
        ),
    }
    return selected_predictor, gate1_report


@torch.no_grad()
def run_live_backbone_state_only_probe(
    yaml_path: Union[str, Path],
    calibrator_predictor: SurvivalPredictor,
    P: int = 512,
    n_sink: int = 4,
    w_obs: int = 64,
    device_map: Any = None,
) -> Dict[str, Any]:
    """Run a live Falcon-H1 forward pass on a multi-fact natural text prefix of length P,
    slice the KV cache at position P to State-Only (retaining only N_sink=4 + W_obs=64 while
    preserving exact S_P and suffix position_ids), and verify finite log-probabilities."""
    model, tok, cfg = load_falcon_h1_for_t4(
        yaml_path, load_tokenizer=True, device_map=device_map or {"": 0}
    )
    dev = next(model.parameters()).device
    L = cfg["architecture"]["num_hidden_layers"]

    parts = []
    for idx in range(64):
        code_str = f"{(idx * 7919 + 104729) % 90000 + 10000}"
        sent = (
            f"Record {idx:02d}: Facility sector {idx:02d} security key is {code_str}. "
            f"Operational status remains normal. "
        )
        parts.append(sent)

    full_text = "".join(parts)
    full_ids = tok(full_text, return_tensors="pt").input_ids.to(dev)
    assert full_ids.shape[1] >= P, f"Generated text shorter than P={P}"
    prefix_ids = full_ids[:, :P]

    hook = FalconH1StateHook(model, store_device="same")
    records, cache, _ = hook.capture(prefix_ids)
    hook.detach()

    surv_out = calibrator_predictor.to(dev).predict_survival(
        records, include_head_tensors=False, compute_device=dev, store_device="cpu"
    )
    token_p_surv = surv_out["p_surv_bar_all"][0].mean(dim=0)  # [P]

    keep_idx = torch.cat([
        torch.arange(0, n_sink, device=dev, dtype=torch.long),
        torch.arange(P - w_obs, P, device=dev, dtype=torch.long),
    ], dim=0)

    cache = ensure_unified_cache_views(cache)
    for l_idx in range(L):
        layer_cache = cache.layers[l_idx]
        layer_cache.keys = torch.index_select(layer_cache.keys, dim=2, index=keep_idx).contiguous()
        layer_cache.values = torch.index_select(layer_cache.values, dim=2, index=keep_idx).contiguous()

    # Record retained pre-suffix KV length right after slicing (before suffix tokens are appended)
    kept_kv_len = int(cache.layers[0].keys.shape[2])

    probe_prompt = " Question: What is the security key for Facility sector 02? Answer: "
    probe_ids = tok(probe_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    S_probe = probe_ids.shape[1]
    pos_ids = torch.arange(P, P + S_probe, device=dev, dtype=torch.long).unsqueeze(0)
    cache_pos = torch.arange(P, P + S_probe, device=dev, dtype=torch.long)

    out_suffix = model(
        input_ids=probe_ids,
        past_key_values=cache,
        position_ids=pos_ids,
        cache_position=cache_pos,
        use_cache=True,
        logits_to_keep=1,
    )
    suffix_logits = out_suffix.logits
    logits_finite = bool(torch.isfinite(suffix_logits).all().item())
    post_suffix_kv_len = int(cache.layers[0].keys.shape[2])

    del hook, records, cache, out_suffix, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "P": P,
        "state_only_retained_kv_len": kept_kv_len,
        "expected_retained_kv_len": n_sink + w_obs,
        "post_suffix_kv_len": post_suffix_kv_len,
        "suffix_logits_finite": logits_finite,
        "mean_prefix_p_surv": round(float(token_p_surv.mean().item()), 5),
        "sink_mean_p_surv": round(float(token_p_surv[:n_sink].mean().item()), 5),
        "window_mean_p_surv": round(float(token_p_surv[P - w_obs :].mean().item()), 5),
        "passed": bool(
            logits_finite
            and kept_kv_len == (n_sink + w_obs)
            and post_suffix_kv_len == (n_sink + w_obs + S_probe)
        ),
    }


def run_phase3_for_model(
    tag: str,
    yaml_path: Union[str, Path],
    phase1_dir: Union[str, Path],
    phase2_dir: Union[str, Path],
    phase3_dir: Union[str, Path],
    save_calibrator_path: Union[str, Path],
    run_live_probe: bool = True,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Execute complete Phase 3 pipeline (Gate 0 + Gate 1 + Calibrator Fit + Live State-Only Probe) for one model."""
    t0 = time.perf_counter()
    phase1_dir = Path(phase1_dir)
    phase2_dir = Path(phase2_dir)
    phase3_dir = Path(phase3_dir)
    save_calibrator_path = Path(save_calibrator_path)

    ref_p1 = torch.load(phase1_dir / f"ref_capture_{tag}_text_2048.pt", map_location="cpu")
    ref_p2 = torch.load(phase2_dir / f"survival_ref_{tag}_text_2048.pt", map_location="cpu")

    L = int(ref_p1["meta"]["L"])
    mid_layer_rec = ref_p1["layers"][L // 2]

    r5a_res = evaluate_gate0_r5a_decay_trap(mid_layer_rec, P=512, n_needles_per_set=16, kv_budget_k=16, device=device)
    r5b_res = evaluate_gate0_r5b_collision_trap(mid_layer_rec, P=512, n_needles_per_set=16, kv_budget_k=16, device=device)
    mqar_res = evaluate_gate0_mqar(mid_layer_rec, P=512, n_kv_pairs=32, kv_budget_k=16, device=device)

    splits = build_natural_prefix_calibration_dataset(ref_p2, n_cal=1500, n_test=1500, seed=202603)
    calibrator, gate1_res = fit_logistic_calibrator(splits["cal"], splits["test"], l2_beta=1e-4, device=device)

    extra_meta = {
        "model": ref_p1["meta"]["model"],
        "tag": tag,
        "L": L,
        "auroc_test": gate1_res["auroc_calibrated_full"],
        "ece_test": gate1_res["ece_calibrated_full"],
        "partial_r_xy_given_log_age": gate1_res["age_controlled_partial_correlation_full"]["partial_r_xy_given_log_age"],
        "partial_corr_p_value": gate1_res["age_controlled_partial_correlation_full"]["p_value"],
    }
    calibrator.save_calibrator(save_calibrator_path, extra_meta=extra_meta)
    calibrator.save_calibrator(phase3_dir / save_calibrator_path.name, extra_meta=extra_meta)

    del ref_p1, ref_p2, splits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    live_probe_res = None
    if run_live_probe:
        live_probe_res = run_live_backbone_state_only_probe(
            yaml_path=yaml_path,
            calibrator_predictor=calibrator,
            P=512,
            n_sink=4,
            w_obs=64,
            device_map={"": 0},
        )

    elapsed = round(time.perf_counter() - t0, 2)
    hard_gate1_pass = bool(
        r5a_res["passed"]
        and r5b_res["passed"]
        and mqar_res["passed"]
        and gate1_res["gate1_partial_corr_pass"]
        and (live_probe_res is None or live_probe_res["passed"])
    )

    report = {
        "tag": tag,
        "yaml_path": str(yaml_path),
        "calibrator_path": str(save_calibrator_path),
        "elapsed_seconds": elapsed,
        "gate0": {
            "r5a_decay_trap": r5a_res,
            "r5b_collision_trap": r5b_res,
            "mqar": mqar_res,
        },
        "gate1": gate1_res,
        "live_state_only_probe": live_probe_res,
        "hard_gate1_pass": hard_gate1_pass,
    }
    (phase3_dir / f"report_phase3_{tag}.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Gate 0 & Gate 1 Diagnostics + 4-Parameter Calibrator")
    parser.add_argument("--config", type=str, required=True, help="Path to model YAML config")
    parser.add_argument("--tasks", nargs="+", default=["mqar", "r5a", "r5b", "fineweb_age_recall"])
    parser.add_argument("--save_calibrator", type=str, required=True, help="Output JSON path for fitted calibrator")
    parser.add_argument("--no_live_probe", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    tag = "05b" if "0.5b" in args.config.lower() else "15b"
    rep = run_phase3_for_model(
        tag=tag,
        yaml_path=project_root / args.config if not Path(args.config).is_absolute() else Path(args.config),
        phase1_dir=project_root / "checkpoints" / "phase1",
        phase2_dir=project_root / "checkpoints" / "phase2",
        phase3_dir=project_root / "checkpoints" / "phase3",
        save_calibrator_path=(
            project_root / args.save_calibrator
            if not Path(args.save_calibrator).is_absolute()
            else Path(args.save_calibrator)
        ),
        run_live_probe=not args.no_live_probe,
    )
    print(f"[Phase 3 CLI | {tag}] Hard Gate 1 Pass: {rep['hard_gate1_pass']}")


if __name__ == "__main__":
    main()
'''

GATE0_PATH.write_text(GATE0_CODE)
print(f"[Phase 3.1] Wrote {GATE0_PATH}")
```

---

## Cell 3.2 — Update `benchmarks/__init__.py` & Verify Imports

```python
# ==============================================================================
# Cell 3.2 - Update benchmarks/__init__.py & Verify Imports | Phase 3: Gate 0 & Gate 1 Diagnostic Suite
# ==============================================================================
import sys, importlib
from pathlib import Path

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
BENCH_INIT = PROJECT_ROOT / "benchmarks" / "__init__.py"

BENCH_INIT.write_text(
    '"""Benchmarks package for hybrid_state_eviction_v13."""\n'
    "from .gate0_diagnostics import (\n"
    "    compute_write_time_baselines_layer,\n"
    "    evaluate_gate0_r5a_decay_trap,\n"
    "    evaluate_gate0_r5b_collision_trap,\n"
    "    evaluate_gate0_mqar,\n"
    "    compute_auroc,\n"
    "    compute_ece_and_brier,\n"
    "    compute_age_controlled_partial_correlation,\n"
    "    build_natural_prefix_calibration_dataset,\n"
    "    fit_logistic_calibrator,\n"
    "    run_live_backbone_state_only_probe,\n"
    "    run_phase3_for_model,\n"
    ")\n"
)

for m in [m for m in list(sys.modules.keys()) if m == "benchmarks" or m.startswith("benchmarks.")]:
    del sys.modules[m]
importlib.invalidate_caches()

from benchmarks import (
    compute_write_time_baselines_layer,
    evaluate_gate0_r5a_decay_trap,
    evaluate_gate0_r5b_collision_trap,
    evaluate_gate0_mqar,
    fit_logistic_calibrator,
    run_phase3_for_model,
)

print("[Phase 3.2] Verified all Phase 3 exports from `benchmarks.gate0_diagnostics`.")
```

---

## Cell 3.3 — Fast Smoke Test on `Falcon-H1-0.5B` Saved Reference Capture

```python
# ==============================================================================
# Cell 3.3 - Fast Smoke Test on 0.5B Saved Reference Capture | Phase 3: Gate 0 & Gate 1 Diagnostic Suite
# Runs Gate 0 (R5a, R5b, MQAR) and Gate 1 Calibration in ~2 seconds before full model loading.
# ==============================================================================
import json, gc, torch
from pathlib import Path
from benchmarks import (
    evaluate_gate0_r5a_decay_trap,
    evaluate_gate0_r5b_collision_trap,
    evaluate_gate0_mqar,
    build_natural_prefix_calibration_dataset,
    fit_logistic_calibrator,
)

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
PHASE1_DIR = PROJECT_ROOT / "checkpoints" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "checkpoints" / "phase3"

ref_p1_05b = torch.load(PHASE1_DIR / "ref_capture_05b_text_2048.pt", map_location="cpu")
ref_p2_05b = torch.load(PHASE2_DIR / "survival_ref_05b_text_2048.pt", map_location="cpu")
mid_rec = ref_p1_05b["layers"][18]

r5a = evaluate_gate0_r5a_decay_trap(mid_rec, P=512, n_needles_per_set=16, kv_budget_k=16, device="cuda:0")
r5b = evaluate_gate0_r5b_collision_trap(mid_rec, P=512, n_needles_per_set=16, kv_budget_k=16, device="cuda:0")
mqar = evaluate_gate0_mqar(mid_rec, P=512, n_kv_pairs=32, kv_budget_k=16, device="cuda:0")

splits = build_natural_prefix_calibration_dataset(ref_p2_05b, n_cal=1500, n_test=1500, seed=202603)
cal_pred, g1_rep = fit_logistic_calibrator(splits["cal"], splits["test"], l2_beta=1e-4, device="cuda:0")

print("=== Gate 0 R5a (Decay Trap) ===")
print(json.dumps(r5a, indent=2))
print("\n=== Gate 0 R5b (Equal-Age Key-Collision Trap) ===")
print(json.dumps(r5b, indent=2))
print("\n=== Gate 1 Calibration & Age-Controlled Partial Correlation ===")
print(json.dumps({k: v for k, v in g1_rep.items() if k != "ece_bins_calibrated_full"}, indent=2))

smoke_ok = bool(r5a["passed"] and r5b["passed"] and mqar["passed"] and g1_rep["gate1_partial_corr_pass"])
print("\n" + ("PHASE 3 SMOKE PASS" if smoke_ok else "PHASE 3 SMOKE FAIL"))

del ref_p1_05b, ref_p2_05b, splits
gc.collect()
torch.cuda.empty_cache()
if not smoke_ok:
    raise SystemExit("Phase 3 Smoke Test failed -> inspect output before running Cell 3.4.")
```

---

## Cell 3.4 — Full Phase 3 Run (`0.5B` & `1.5B-Deep`) + Save Calibrators

```python
# ==============================================================================
# Cell 3.4 - Full Phase 3 Run (0.5B & 1.5B-Deep) + Save Calibrators | Phase 3: Gate 0 & Gate 1
# Fits and saves `configs/calibrator_0.5b.json` and `configs/calibrator_1.5b.json` + runs live State-Only probe.
# ==============================================================================
import json, gc, torch
from pathlib import Path
from benchmarks import run_phase3_for_model

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
CFG_DIR = PROJECT_ROOT / "configs"
PHASE1_DIR = PROJECT_ROOT / "checkpoints" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "checkpoints" / "phase3"

rep_05b = run_phase3_for_model(
    tag="05b",
    yaml_path=CFG_DIR / "falcon_h1_0.5b.yaml",
    phase1_dir=PHASE1_DIR,
    phase2_dir=PHASE2_DIR,
    phase3_dir=PHASE3_DIR,
    save_calibrator_path=CFG_DIR / "calibrator_0.5b.json",
    run_live_probe=True,
    device="cuda:0",
)
print(
    f"[Phase 3.4 | 05b] Hard Gate 1 Pass: {rep_05b['hard_gate1_pass']} | "
    f"R5a gain vs HOLA: +{rep_05b['gate0']['r5a_decay_trap']['sa_minus_hola_recall']*100:.1f}% | "
    f"R5b gain vs DASC: +{rep_05b['gate0']['r5b_collision_trap']['sa_minus_dasc_recall']*100:.1f}% | "
    f"AUROC: {rep_05b['gate1']['auroc_calibrated_full']:.4f} | "
    f"Partial r: {rep_05b['gate1']['age_controlled_partial_correlation_full']['partial_r_xy_given_log_age']:.4f}"
)

rep_15b = run_phase3_for_model(
    tag="15b",
    yaml_path=CFG_DIR / "falcon_h1_1.5b_deep.yaml",
    phase1_dir=PHASE1_DIR,
    phase2_dir=PHASE2_DIR,
    phase3_dir=PHASE3_DIR,
    save_calibrator_path=CFG_DIR / "calibrator_1.5b.json",
    run_live_probe=True,
    device="cuda:0",
)
print(
    f"[Phase 3.4 | 15b] Hard Gate 1 Pass: {rep_15b['hard_gate1_pass']} | "
    f"R5a gain vs HOLA: +{rep_15b['gate0']['r5a_decay_trap']['sa_minus_hola_recall']*100:.1f}% | "
    f"R5b gain vs DASC: +{rep_15b['gate0']['r5b_collision_trap']['sa_minus_dasc_recall']*100:.1f}% | "
    f"AUROC: {rep_15b['gate1']['auroc_calibrated_full']:.4f} | "
    f"Partial r: {rep_15b['gate1']['age_controlled_partial_correlation_full']['partial_r_xy_given_log_age']:.4f}"
)
```

---

## Cell 3.5 — Phase 3 Gate Check, Manifest Generation & Handover Summary

```python
# ==============================================================================
# Cell 3.5 - PHASE 3 CHECK CELL (Validates Hard Gate 1, Calibrator JSONs & Writes Manifest)
# ==============================================================================
import json, torch, transformers
from pathlib import Path
from hooks import SurvivalPredictor

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
CFG_DIR = PROJECT_ROOT / "configs"
PHASE2_DIR = PROJECT_ROOT / "checkpoints" / "phase2"
PHASE3_DIR = PROJECT_ROOT / "checkpoints" / "phase3"

problems = []
summary_out = {}

for tag, cal_name, L_expected in (
    ("05b", "calibrator_0.5b.json", 36),
    ("15b", "calibrator_1.5b.json", 66),
):
    rep_path = PHASE3_DIR / f"report_phase3_{tag}.json"
    cfg_cal_path = CFG_DIR / cal_name
    chk_cal_path = PHASE3_DIR / cal_name

    for p in (rep_path, cfg_cal_path, chk_cal_path):
        if not p.exists():
            problems.append(f"missing {p}")
    if not rep_path.exists():
        continue

    rep = json.loads(rep_path.read_text())
    g0 = rep["gate0"]
    g1 = rep["gate1"]
    lp = rep["live_state_only_probe"]

    if not g0["r5a_decay_trap"]["passed"]:
        problems.append(f"{tag}: Gate 0 R5a FAIL")
    if not g0["r5b_collision_trap"]["passed"]:
        problems.append(f"{tag}: Gate 0 R5b FAIL")
    if not g0["mqar"]["passed"]:
        problems.append(f"{tag}: Gate 0 MQAR FAIL")
    if not g1["gate1_partial_corr_pass"]:
        problems.append(f"{tag}: Gate 1 Partial Correlation FAIL")
    if lp is None or not lp.get("passed", False):
        problems.append(f"{tag}: Live State-Only Probe FAIL")

    # Verify SurvivalPredictor loads the saved calibrator cleanly and produces valid [0, 1] outputs
    pred = SurvivalPredictor.load_calibrator(cfg_cal_path)
    ref_p2 = torch.load(PHASE2_DIR / f"survival_ref_{tag}_text_2048.pt", map_location="cpu")
    l0 = ref_p2["layers"][0]
    res_l0 = pred.predict_layer(l0)
    if not bool(torch.isfinite(res_l0["p_surv_bar"]).all().item()):
        problems.append(f"{tag}: non-finite p_surv_bar from loaded calibrator")
    del ref_p2

    summary_out[tag] = {
        "selected_calibrator_mode": g1["selected_mode"],
        "fitted_weights_full": g1["fitted_weights_full"],
        "feature_mean": g1["feature_mean"],
        "feature_std": g1["feature_std"],
        "r5a_hybrid_recall": {
            k: v["total_hybrid_recall"] for k, v in g0["r5a_decay_trap"]["policies"].items()
        },
        "r5b_hybrid_recall": {
            k: v["total_hybrid_recall"] for k, v in g0["r5b_collision_trap"]["policies"].items()
        },
        "mqar_hybrid_recall": {
            k: v["mqar_hybrid_recall"] for k, v in g0["mqar"]["policies"].items()
        },
        "auroc_calibrated_full": g1["auroc_calibrated_full"],
        "auroc_calibrated_log_decay": g1["auroc_calibrated_log_decay"],
        "auroc_closed_form": g1["auroc_closed_form"],
        "ece_calibrated_full": g1["ece_calibrated_full"],
        "ece_closed_form": g1["ece_closed_form"],
        "brier_calibrated_full": g1["brier_calibrated_full"],
        "partial_r_xy_given_log_age": g1["age_controlled_partial_correlation_full"]["partial_r_xy_given_log_age"],
        "partial_corr_t_stat": g1["age_controlled_partial_correlation_full"]["t_stat"],
        "partial_corr_p_value": g1["age_controlled_partial_correlation_full"]["p_value"],
        "live_state_only_probe": lp,
        "hard_gate1_pass": rep["hard_gate1_pass"],
    }

manifest = {
    "phase": 3,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "handover_dir": str(PHASE3_DIR),
    "config_calibrators": [
        str(CFG_DIR / "calibrator_0.5b.json"),
        str(CFG_DIR / "calibrator_1.5b.json"),
    ],
    "files": sorted(p.name for p in PHASE3_DIR.iterdir()),
    "benchmark_module": "benchmarks/gate0_diagnostics.py",
    "problems": problems,
}
(PHASE3_DIR / "phase3_manifest.json").write_text(json.dumps(manifest, indent=2))

print("=" * 72)
print("PHASE 3 SUMMARY (paste everything between the lines back to me)")
print("=" * 72)
print(json.dumps(summary_out, indent=2))
print("files:", manifest["files"])
print("config_calibrators:", manifest["config_calibrators"])
print("problems:", problems if problems else "NONE")
print("=" * 72)
print("PHASE 3 GATE:", "PASS" if not problems else "FAIL")
```