# %% [Cell 0.2]
# ==============================================================================
# CELLNAME: Cell 0.2 — Exact Per-Layer Memory Accounting & Ratio Verification
# Reason for Cell 0.2: Implements and verifies the exact mathematical formulas
# in Section 0.3 of the V13 Blueprint for Attention KV floats/bytes, Recurrent
# State S_P token-equivalent size, and standardized KV retention ratios rho_kv.
# ==============================================================================
from typing import Dict, Any, List
import yaml
from pathlib import Path

def compute_exact_memory_accounting(cfg: Dict[str, Any], prefix_len: int = 8192) -> Dict[str, Any]:
    """
    Enforces and verifies the exact architectural & memory accounting formulas
    from Master Plan V13 Section 0.3.
    """
    arch = cfg["architecture"]
    mem = cfg["memory_accounting"]
    evict = cfg["eviction"]

    L = int(arch["num_hidden_layers"])
    H_q = int(arch["num_attention_heads"])
    H_kv = int(arch["num_key_value_heads"])
    d_h = int(arch["head_dim"])
    H_ssm = int(arch["mamba_n_heads"])
    d_head_ssm = int(arch["mamba_d_head"])
    d_state = int(arch["mamba_d_state"])
    G_ssm = int(arch["mamba_n_groups"])
    d_conv = int(arch["mamba_d_conv"])

    # Formula 1: Attention KV Floats & Bytes per Token per Layer
    # Floats_{KV/token/layer} = 2 * H_kv * d_h = 2 * 2 * 64 = 256 floats (512 Bytes in FP16)
    kv_floats_per_tok_layer = 2 * H_kv * d_h
    kv_bytes_per_tok_layer_fp16 = kv_floats_per_tok_layer * 2

    # Formula 2: Recurrent State S_P Floats per Layer & Token-Equivalent Size
    # Floats_{S_P/layer} = H_ssm * d_head,ssm * d_state
    sp_floats_per_layer = H_ssm * d_head_ssm * d_state
    sp_token_equiv_per_layer = sp_floats_per_layer // kv_floats_per_tok_layer
    sp_bytes_per_layer_fp16 = sp_floats_per_layer * 2
    sp_bytes_per_layer_fp32 = sp_floats_per_layer * 4

    # Strict Contract Assertions against YAML specification
    assert kv_floats_per_tok_layer == mem["kv_floats_per_token_per_layer"], (
        f"KV float mismatch: {kv_floats_per_tok_layer} != {mem['kv_floats_per_token_per_layer']}"
    )
    assert kv_bytes_per_tok_layer_fp16 == mem["kv_bytes_per_token_per_layer_fp16"], (
        f"KV byte mismatch: {kv_bytes_per_tok_layer_fp16} != {mem['kv_bytes_per_token_per_layer_fp16']}"
    )
    assert sp_floats_per_layer == mem["ssm_state_floats_per_layer"], (
        f"S_P float mismatch: {sp_floats_per_layer} != {mem['ssm_state_floats_per_layer']}"
    )
    assert sp_token_equiv_per_layer == mem["ssm_state_kv_token_equiv_per_layer"], (
        f"S_P token-equivalent mismatch: {sp_token_equiv_per_layer} != {mem['ssm_state_kv_token_equiv_per_layer']}"
    )
    assert G_ssm == 1 and d_conv == 4, "Falcon-H1 requires G_ssm=1 and d_conv=4"
    assert H_q % H_kv == 0, "GQA requires H_q divisible by H_kv"

    # Formula 3: Retained KV Cache Ratio (rho_kv = K_total / (L * P))
    n_sink = int(evict["n_sink"])
    w_obs = int(evict["w_obs"])
    p_cand = max(0, prefix_len - w_obs - n_sink)

    budget_table: List[Dict[str, Any]] = []
    for rho_kv in evict["retained_kv_ratios"]:
        rho = float(rho_kv)
        k_total = int(round(rho * L * prefix_len))
        k_avg = k_total / L
        total_kv_mb_fp16 = (k_total * kv_bytes_per_tok_layer_fp16) / (1024 ** 2)
        budget_table.append({
            "rho_kv": rho,
            "K_total": k_total,
            "K_avg_per_layer": k_avg,
            "candidate_pool_per_layer": p_cand,
            "retained_kv_mb_fp16": round(total_kv_mb_fp16, 3),
        })

    full_kv_mb_fp16 = (L * prefix_len * kv_bytes_per_tok_layer_fp16) / (1024 ** 2)
    total_sp_mb_fp16 = (L * sp_bytes_per_layer_fp16) / (1024 ** 2)
    total_sp_mb_fp32 = (L * sp_bytes_per_layer_fp32) / (1024 ** 2)

    return {
        "model_id": cfg["model_id"],
        "L": L,
        "d_model": int(arch["hidden_size"]),
        "H_q": H_q,
        "H_kv": H_kv,
        "gqa_group_size": H_q // H_kv,
        "d_h": d_h,
        "H_ssm": H_ssm,
        "d_head_ssm": d_head_ssm,
        "d_state": d_state,
        "kv_floats_per_token_per_layer": kv_floats_per_tok_layer,
        "kv_bytes_per_token_per_layer_fp16": kv_bytes_per_tok_layer_fp16,
        "ssm_state_floats_per_layer": sp_floats_per_layer,
        "ssm_state_kv_token_equiv_per_layer": sp_token_equiv_per_layer,
        "prefix_len_evaluated": prefix_len,
        "full_kv_cache_mb_fp16": round(full_kv_mb_fp16, 3),
        "total_sp_state_mb_fp16": round(total_sp_mb_fp16, 3),
        "total_sp_state_mb_fp32": round(total_sp_mb_fp32, 3),
        "budget_schedule": budget_table,
    }

print("=" * 78)
print("[Phase 0 | Cell 0.2] Verifying Memory Accounting & Retention Ratios (P=8,192)")
print("=" * 78)

for yaml_name in ["falcon_h1_0.5b.yaml", "falcon_h1_1.5b_deep.yaml"]:
    cfg_data = yaml.safe_load((PROJECT_ROOT / "configs" / yaml_name).read_text(encoding="utf-8"))
    report = compute_exact_memory_accounting(cfg_data, prefix_len=8192)
    print(f"\nModel: {report['model_id']} (L={report['L']}, d_model={report['d_model']})")
    print(f"  - Attention GQA: H_q={report['H_q']}, H_kv={report['H_kv']} (group={report['gqa_group_size']}), d_h={report['d_h']}")
    print(f"  - Mamba-2 SSM  : H_ssm={report['H_ssm']}, d_head_ssm={report['d_head_ssm']}, d_state={report['d_state']}")
    print(f"  - KV / tok / L : {report['kv_floats_per_token_per_layer']} floats ({report['kv_bytes_per_token_per_layer_fp16']} B in FP16)")
    print(f"  - S_P / layer  : {report['ssm_state_floats_per_layer']:,} floats == {report['ssm_state_kv_token_equiv_per_layer']:,} KV token-equivalents")
    print(f"  - Full 8K KV   : {report['full_kv_cache_mb_fp16']:.2f} MB (FP16) | Total S_P: {report['total_sp_state_mb_fp16']:.2f} MB (FP16) / {report['total_sp_state_mb_fp32']:.2f} MB (FP32)")
    print("  - Standardized Retention Schedule (rho_kv):")
    for row in report["budget_table"] if "budget_table" in report else report["budget_schedule"]:
        print(
            f"      rho_kv={row['rho_kv']:0.2f} -> K_total={row['K_total']:7,d} tokens "
            f"(K_avg/layer={row['K_avg_per_layer']:6.1f}) | KV Mem={row['retained_kv_mb_fp16']:6.2f} MB"
        )

print("\n[Cell 0.2 Complete] All architectural & memory invariants verified.\n")

