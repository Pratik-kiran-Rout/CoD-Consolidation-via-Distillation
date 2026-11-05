# ==============================================================================
# MASTER PLAN V13 — PHASE 0: ENVIRONMENT SETUP, HARDWARE GUARDS & SCAFFOLDING
# Sequential Kaggle / Colab Execution Cells: 0.0, 0.1, 0.2, 0.3, 0.4
# ==============================================================================


# %% [Cell 0.0]
# ==============================================================================
# CELLNAME: Cell 0.0 — Environment Audit, Dependency Check & Directory Tree
# Reason for Cell 0.0: Audits Kaggle 2x T4 GPU hardware/VRAM, verifies/installs
# required packages (transformers >= 4.51.0 with FalconH1 support, accelerate,
# pyyaml, pytest), and creates the hybrid_state_eviction_v13 directory tree.
# ==============================================================================
import os
import sys
import subprocess
from pathlib import Path

def setup_phase0_environment(project_root: str = "hybrid_state_eviction_v13") -> Path:
    print("=" * 78)
    print("[Phase 0 | Cell 0.0] Auditing Environment & Creating Directory Scaffolding")
    print("=" * 78)

    # 1. Verify required packages and FalconH1 support in transformers
    try:
        import yaml
        import accelerate
        import transformers
        from transformers import FalconH1ForCausalLM
        from transformers.models.falcon_h1.modeling_falcon_h1 import (
            FalconH1Mixer,
            HybridMambaAttentionDynamicCache,
        )
        print(f"[OK] transformers {transformers.__version__} (FalconH1ForCausalLM & HybridMambaAttentionDynamicCache verified)")
    except ImportError as exc:
        print(f"[INFO] Missing or outdated dependency ({exc}). Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-U",
            "transformers>=4.51.3", "accelerate>=0.34.0", "pyyaml", "pytest"
        ])
        import transformers
        from transformers import FalconH1ForCausalLM
        print(f"[OK] Upgraded transformers to {transformers.__version__}")

    import torch
    print(f"[OK] PyTorch version: {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"[OK] Detected {num_gpus} CUDA device(s):")
        for idx in range(num_gpus):
            props = torch.cuda.get_device_properties(idx)
            vram_gb = props.total_memory / (1024 ** 3)
            print(
                f"     - GPU {idx}: {props.name} | SM {props.major}.{props.minor} | "
                f"VRAM: {vram_gb:.2f} GB"
            )
    else:
        print("[WARN] No CUDA GPU detected. Running in CPU verification mode.")

    # 2. Create the V13 repository directory structure
    root_path = Path(project_root).resolve()
    subdirs = [
        "configs",
        "hooks",
        "eviction",
        "consolidation",
        "benchmarks",
        "tests",
        "checkpoints",
    ]
    for subdir in subdirs:
        target = root_path / subdir
        target.mkdir(parents=True, exist_ok=True)
        print(f"[DIR] Created/Verified: {target}")

    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))

    print(f"[Cell 0.0 Complete] Project root initialized at: {root_path}\n")
    return root_path

PROJECT_ROOT = setup_phase0_environment("hybrid_state_eviction_v13")


# %% [Cell 0.1]
# ==============================================================================
# CELLNAME: Cell 0.1 — Model YAML Configurations & Package __init__.py Files
# Reason for Cell 0.1: Writes the exact architectural and memory YAML config
# files (configs/falcon_h1_0.5b.yaml and configs/falcon_h1_1.5b_deep.yaml) and
# initializes all five Python subpackages required by Master Plan V13.
# ==============================================================================
import yaml
from pathlib import Path

FALCON_H1_0_5B_YAML = """# Master Plan V13 — Falcon-H1-0.5B-Instruct Configuration (Fast Dev & Gate 1-2)
model_id: "tiiuae/Falcon-H1-0.5B-Instruct"
model_short_name: "falcon_h1_0.5b"

architecture:
  num_hidden_layers: 36          # L = 36
  hidden_size: 1024              # d_model = 1024
  num_attention_heads: 8         # H_q = 8
  num_key_value_heads: 2         # H_kv = 2
  head_dim: 64                   # d_h = 64
  mamba_n_heads: 24              # H_ssm = 24
  mamba_d_head: 64               # d_head,ssm = 64
  mamba_d_state: 128             # d_state = 128
  mamba_n_groups: 1              # G_ssm = 1
  mamba_d_conv: 4                # d_conv = 4

memory_accounting:
  kv_floats_per_token_per_layer: 256        # 2 * H_kv * d_h = 2 * 2 * 64
  kv_bytes_per_token_per_layer_fp16: 512    # 256 * 2 bytes
  ssm_state_floats_per_layer: 196608        # 24 * 64 * 128
  ssm_state_kv_token_equiv_per_layer: 768   # 196608 / 256
  approx_weights_vram_gb_fp16: 1.06

hardware_guard:
  weights_dtype: "float16"
  ssm_recurrence_dtype: "float32"
  mup_projection_dtype: "float32"
  survival_dtype: "float32"
  attn_implementation: "sdpa"
  device_map: "auto"

eviction:
  n_sink: 4
  w_obs: 64
  k_min: 32
  pool_kernel_size: 5
  retained_kv_ratios: [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
  streaming_chunk_size: 2048

consolidation:
  k_cons_options: [32, 64, 128]
  init_tilde_gamma: -2.1972245773362196     # sigmoid(-2.1972246) = 0.10
  init_tilde_lambda: 0.0                    # exp(0.0) = 1.0
  distill_steps: 300
  distill_lr: 0.03
  num_distill_scalars: 1728                 # 36 * 24 * 2
"""

FALCON_H1_1_5B_DEEP_YAML = """# Master Plan V13 — Falcon-H1-1.5B-Deep-Instruct Configuration (Primary Benchmark)
model_id: "tiiuae/Falcon-H1-1.5B-Deep-Instruct"
model_short_name: "falcon_h1_1.5b_deep"

architecture:
  num_hidden_layers: 72          # L = 72
  hidden_size: 1536              # d_model = 1536
  num_attention_heads: 12        # H_q = 12
  num_key_value_heads: 2         # H_kv = 2
  head_dim: 64                   # d_h = 64
  mamba_n_heads: 32              # H_ssm = 32
  mamba_d_head: 64               # d_head,ssm = 64
  mamba_d_state: 128             # d_state = 128
  mamba_n_groups: 1              # G_ssm = 1
  mamba_d_conv: 4                # d_conv = 4

memory_accounting:
  kv_floats_per_token_per_layer: 256        # 2 * H_kv * d_h = 2 * 2 * 64
  kv_bytes_per_token_per_layer_fp16: 512    # 256 * 2 bytes
  ssm_state_floats_per_layer: 262144        # 32 * 64 * 128
  ssm_state_kv_token_equiv_per_layer: 1024  # 262144 / 256
  approx_weights_vram_gb_fp16: 3.10

hardware_guard:
  weights_dtype: "float16"
  ssm_recurrence_dtype: "float32"
  mup_projection_dtype: "float32"
  survival_dtype: "float32"
  attn_implementation: "sdpa"
  device_map: "auto"

eviction:
  n_sink: 4
  w_obs: 64
  k_min: 32
  pool_kernel_size: 5
  retained_kv_ratios: [0.05, 0.10, 0.20, 0.35, 0.50, 1.00]
  streaming_chunk_size: 2048

consolidation:
  k_cons_options: [32, 64, 128]
  init_tilde_gamma: -2.1972245773362196     # sigmoid(-2.1972246) = 0.10
  init_tilde_lambda: 0.0                    # exp(0.0) = 1.0
  distill_steps: 300
  distill_lr: 0.03
  num_distill_scalars: 4608                 # 72 * 32 * 2
"""

def write_phase0_configs_and_packages(root_path: Path) -> None:
    print("=" * 78)
    print("[Phase 0 | Cell 0.1] Writing YAML Configurations & Package __init__.py Files")
    print("=" * 78)

    cfg_05b_path = root_path / "configs" / "falcon_h1_0.5b.yaml"
    cfg_15b_path = root_path / "configs" / "falcon_h1_1.5b_deep.yaml"
    cfg_05b_path.write_text(FALCON_H1_0_5B_YAML, encoding="utf-8")
    cfg_15b_path.write_text(FALCON_H1_1_5B_DEEP_YAML, encoding="utf-8")
    print(f"[FILE] Wrote {cfg_05b_path}")
    print(f"[FILE] Wrote {cfg_15b_path}")

    init_files = {
        "eviction/__init__.py": (
            '"""Phase 4: Token-Level SA-SnapKV, Risk-Adaptive Budgets, Streaming Prefill & Baselines."""\n'
        ),
        "consolidation/__init__.py": (
            '"""Phase 5: Batched Ridge State Consolidation & 300-Step Teacher KL Distillation."""\n'
        ),
        "benchmarks/__init__.py": (
            '"""Phase 3 & 6: Gate 0/1 Diagnostics, Prospective RULER, and SCBench/BABILong Suites."""\n'
        ),
        "tests/__init__.py": (
            '"""Phase 6: Mandatory 10-Test Verification Suite for Master Plan V13."""\n'
        ),
    }
    for rel_path, content in init_files.items():
        fpath = root_path / rel_path
        fpath.write_text(content, encoding="utf-8")
        print(f"[FILE] Wrote {fpath}")

    print("[Cell 0.1 Complete] Configs and base package initializers created.\n")

write_phase0_configs_and_packages(PROJECT_ROOT)


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


# %% [Cell 0.3]
# ==============================================================================
# CELLNAME: Cell 0.3 — T4 FP16/FP32 Split Execution Guard & Model Initializer
# Reason for Cell 0.3: Implements the T4 FP16/FP32 Split Execution Guard that
# loads Falcon-H1 weights in torch.float16 across 2x T4 while explicitly keeping
# A_log, D, dt_bias, muP projections, and Mamba-2 state recurrences in FP32, and
# exports these utilities via hooks/__init__.py for Phase 1 handover.
# ==============================================================================
HOOKS_INIT_CODE = '''"""
Phase 0 -> Phase 1 Handover Package:
Configuration Loader, Exact Memory Accounting, and T4 FP16/FP32 Split Execution Guard.
"""
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, FalconH1ForCausalLM
from transformers.models.falcon_h1.modeling_falcon_h1 import (
    FalconH1Mixer,
    HybridMambaAttentionDynamicCache,
)


def load_v13_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads and validates a Master Plan V13 YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def compute_exact_memory_accounting(cfg: Dict[str, Any], prefix_len: int = 8192) -> Dict[str, Any]:
    """Computes and validates exact per-layer KV and SSM state S_P memory accounting."""
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

    kv_floats = 2 * H_kv * d_h
    kv_bytes_fp16 = kv_floats * 2
    sp_floats = H_ssm * d_head_ssm * d_state
    sp_tok_equiv = sp_floats // kv_floats

    assert kv_floats == mem["kv_floats_per_token_per_layer"]
    assert kv_bytes_fp16 == mem["kv_bytes_per_token_per_layer_fp16"]
    assert sp_floats == mem["ssm_state_floats_per_layer"]
    assert sp_tok_equiv == mem["ssm_state_kv_token_equiv_per_layer"]

    schedule = []
    for rho in evict["retained_kv_ratios"]:
        rho_f = float(rho)
        k_total = int(round(rho_f * L * prefix_len))
        schedule.append({
            "rho_kv": rho_f,
            "K_total": k_total,
            "K_avg_per_layer": k_total / L,
            "retained_kv_mb_fp16": round((k_total * kv_bytes_fp16) / (1024 ** 2), 3),
        })

    return {
        "model_id": cfg["model_id"],
        "L": L,
        "H_q": H_q,
        "H_kv": H_kv,
        "d_h": d_h,
        "H_ssm": H_ssm,
        "d_head_ssm": d_head_ssm,
        "d_state": d_state,
        "kv_floats_per_token_per_layer": kv_floats,
        "kv_bytes_per_token_per_layer_fp16": kv_bytes_fp16,
        "ssm_state_floats_per_layer": sp_floats,
        "ssm_state_kv_token_equiv_per_layer": sp_tok_equiv,
        "budget_schedule": schedule,
    }


class T4MixedPrecisionGuard:
    """
    T4 FP16/FP32 Split Execution Guard for FalconH1ForCausalLM.
    Keeps all dense linear and conv1d weights in torch.float16 for Turing SM 7.5
    Tensor Core acceleration while forcing Mamba-2 selective scan recurrences,
    muP-scaled projections, A_log, D, dt_bias, and HybridMambaAttentionDynamicCache
    ssm_states into torch.float32.
    """

    def __init__(self, model: FalconH1ForCausalLM):
        self.model = model
        self.handles: List[Any] = []
        self.upgraded_layers: int = 0

    def apply(self) -> "T4MixedPrecisionGuard":
        # ==============================================================================
        # [MATH-UPGRADE | Phase 0 -> Phase 1]
        # Original Spec: Load FalconH1ForCausalLM in torch.float16 and cast A_log, D,
        #                and dt_bias to torch.float32.
        # Upgraded Spec: Explicitly cast (A_log, D, dt_bias, norm.weight) to FP32 AND
        #                install boundary dtype-isolation hooks around FalconH1Mixer,
        #                in_proj, conv1d, and out_proj, while promoting
        #                cache_params.ssm_states[l] and conv_states[l] to FP32.
        # Reason:        In HuggingFace transformers, FalconH1Mixer.torch_forward uses
        #                `hidden_states.dtype` for internal `.to(hidden_states.dtype)`
        #                casts and calls `torch.einsum` across (dt, B, C, x, L) before
        #                passing `scan_output` to `out_proj` (FP16). Running the mixer
        #                body in FP32 while boundary-casting `in_proj`, `conv1d`, and
        #                `out_proj` inputs to their FP16 weight dtypes eliminates
        #                Float/Half matmul & einsum crashes, prevents FP16 exp/muP
        #                overflow at P=8,192, and preserves exact FP32 ssm_states[l]
        #                required by Phase 1 Test 2 (< 1e-4 relative Frobenius error).
        # Contract Check: Decoder layer inputs/outputs and Attention KV cache remain
        #                 torch.float16; cache_params.ssm_states[l] is torch.float32
        #                 with intact shape [B, H_ssm, d_head_ssm, d_state].
        # ==============================================================================
        for layer in self.model.model.layers:
            mixer: FalconH1Mixer = layer.mamba
            self._guard_mixer(mixer)
            self.upgraded_layers += 1
        return self

    def _guard_mixer(self, mixer: FalconH1Mixer) -> None:
        # 1. Force sensitive SSM recurrence parameters into FP32
        for param_name in ("A_log", "D", "dt_bias"):
            if hasattr(mixer, param_name):
                param = getattr(mixer, param_name)
                if isinstance(param, nn.Parameter) and param.dtype != torch.float32:
                    param.data = param.data.to(torch.float32)

        if hasattr(mixer, "norm") and hasattr(mixer.norm, "weight"):
            if mixer.norm.weight.dtype != torch.float32:
                mixer.norm.weight.data = mixer.norm.weight.data.to(torch.float32)

        # 2. Pre-hook on FalconH1Mixer: promote input hidden_states and cache states to FP32
        def _mixer_pre_hook(module: FalconH1Mixer, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                hs = args[0]
                mixer._caller_input_dtype = hs.dtype
                if hs.dtype != torch.float32:
                    args = (hs.to(torch.float32),) + args[1:]
            elif "hidden_states" in kwargs and isinstance(kwargs["hidden_states"], torch.Tensor):
                hs = kwargs["hidden_states"]
                mixer._caller_input_dtype = hs.dtype
                if hs.dtype != torch.float32:
                    kwargs = dict(kwargs)
                    kwargs["hidden_states"] = hs.to(torch.float32)

            cache_params = kwargs.get("cache_params", None)
            if cache_params is None and len(args) > 1:
                cache_params = args[1]

            if cache_params is not None:
                l_idx = module.layer_idx
                if hasattr(cache_params, "ssm_states"):
                    if isinstance(cache_params.ssm_states, list):
                        if cache_params.ssm_states[l_idx].dtype != torch.float32:
                            cache_params.ssm_states[l_idx] = cache_params.ssm_states[l_idx].to(torch.float32)
                    elif isinstance(cache_params.ssm_states, torch.Tensor):
                        if cache_params.ssm_states.dtype != torch.float32:
                            cache_params.ssm_states = cache_params.ssm_states.to(torch.float32)
                if hasattr(cache_params, "conv_states"):
                    if isinstance(cache_params.conv_states, list):
                        if cache_params.conv_states[l_idx].dtype != torch.float32:
                            cache_params.conv_states[l_idx] = cache_params.conv_states[l_idx].to(torch.float32)
                    elif isinstance(cache_params.conv_states, torch.Tensor):
                        if cache_params.conv_states.dtype != torch.float32:
                            cache_params.conv_states = cache_params.conv_states.to(torch.float32)

            return args, kwargs

        # 3. Post-hook on FalconH1Mixer: restore output hidden_states to caller dtype (FP16)
        def _mixer_post_hook(module: FalconH1Mixer, args: Tuple[Any, ...], output: Any):
            target_dtype = getattr(module, "_caller_input_dtype", torch.float16)
            if isinstance(output, torch.Tensor) and output.dtype != target_dtype:
                return output.to(target_dtype)
            return output

        # 4. Boundary hooks for FP16 Tensor Core linear/conv submodules inside FP32 mixer body
        def _cast_input_to_weight_dtype(submodule: nn.Module, sub_args: Tuple[Any, ...]):
            if len(sub_args) > 0 and isinstance(sub_args[0], torch.Tensor):
                w_dtype = submodule.weight.dtype
                if sub_args[0].dtype != w_dtype:
                    return (sub_args[0].to(w_dtype),) + sub_args[1:]
            return sub_args

        def _cast_output_to_fp32(submodule: nn.Module, sub_args: Tuple[Any, ...], sub_out: Any):
            if isinstance(sub_out, torch.Tensor) and sub_out.dtype != torch.float32:
                return sub_out.to(torch.float32)
            return sub_out

        self.handles.append(mixer.register_forward_pre_hook(_mixer_pre_hook, with_kwargs=True))
        self.handles.append(mixer.register_forward_hook(_mixer_post_hook))

        self.handles.append(mixer.in_proj.register_forward_pre_hook(_cast_input_to_weight_dtype))
        self.handles.append(mixer.in_proj.register_forward_hook(_cast_output_to_fp32))

        self.handles.append(mixer.conv1d.register_forward_pre_hook(_cast_input_to_weight_dtype))
        self.handles.append(mixer.conv1d.register_forward_hook(_cast_output_to_fp32))

        self.handles.append(mixer.out_proj.register_forward_pre_hook(_cast_input_to_weight_dtype))
        self.handles.append(mixer.out_proj.register_forward_hook(_cast_output_to_fp32))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def initialize_hybrid_cache(
    model: FalconH1ForCausalLM,
    batch_size: int = 1,
) -> HybridMambaAttentionDynamicCache:
    """
    Initializes HybridMambaAttentionDynamicCache with FP16 Attention KV states
    and upgraded FP32 Mamba-2 ssm_states & conv_states across model devices.
    """
    cache = HybridMambaAttentionDynamicCache(
        model.config,
        batch_size=batch_size,
        dtype=torch.float16,
    )
    for l_idx in range(model.config.num_hidden_layers):
        layer_device = next(model.model.layers[l_idx].parameters()).device
        if isinstance(cache.ssm_states, list):
            cache.ssm_states[l_idx] = cache.ssm_states[l_idx].to(
                device=layer_device, dtype=torch.float32
            )
        if isinstance(cache.conv_states, list):
            cache.conv_states[l_idx] = cache.conv_states[l_idx].to(
                device=layer_device, dtype=torch.float32
            )
    return cache


def initialize_falcon_h1_for_t4(
    config_path: str | Path,
    load_pretrained: bool = True,
    device_map: Optional[str | Dict[str, Any]] = "auto",
) -> Tuple[FalconH1ForCausalLM, Optional[Any], Dict[str, Any], T4MixedPrecisionGuard]:
    """
    Loads Falcon-H1-0.5B-Instruct or Falcon-H1-1.5B-Deep-Instruct from YAML config
    in torch.float16 with the T4 FP16/FP32 Split Execution Guard active.
    """
    cfg = load_v13_config(config_path)
    model_id = cfg["model_id"]
    hw = cfg.get("hardware_guard", {})
    attn_impl = hw.get("attn_implementation", "sdpa")
    resolved_device_map = device_map if device_map is not None else hw.get("device_map", "auto")
    if not torch.cuda.is_available():
        resolved_device_map = None

    tokenizer = None
    if load_pretrained:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = FalconH1ForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=resolved_device_map,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
    else:
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        hf_config._attn_implementation = attn_impl
        model = FalconH1ForCausalLM(hf_config).to(dtype=torch.float16)
        if torch.cuda.is_available():
            model = model.to("cuda:0")

    model.eval()
    guard = T4MixedPrecisionGuard(model).apply()
    return model, tokenizer, cfg, guard


__all__ = [
    "load_v13_config",
    "compute_exact_memory_accounting",
    "T4MixedPrecisionGuard",
    "initialize_hybrid_cache",
    "initialize_falcon_h1_for_t4",
]
'''

print("=" * 78)
print("[Phase 0 | Cell 0.3] Writing T4 FP16/FP32 Split Execution Guard to hooks/__init__.py")
print("=" * 78)

hooks_init_path = PROJECT_ROOT / "hooks" / "__init__.py"
hooks_init_path.write_text(HOOKS_INIT_CODE, encoding="utf-8")
print(f"[FILE] Wrote {hooks_init_path}")

# Verify import of hooks package
import importlib
import hooks
importlib.reload(hooks)
print(
    "[OK] Exported symbols in `hooks`:",
    [name for name in hooks.__all__]
)
print("[Cell 0.3 Complete] T4MixedPrecisionGuard & initializer ready for Phase 0 Exit Gate.\n")


# %% [Cell 0.4]
# ==============================================================================
# CELLNAME: Cell 0.4 — Phase 0 Exit Gate (P=8,192 Baseline Check & Handover Audit)
# Reason for Cell 0.4: Executes the Phase 0 Exit Gate from Section 0.5 of the
# V13 Blueprint, verifying that both Falcon-H1-0.5B-Instruct and
# Falcon-H1-1.5B-Deep-Instruct load in FP16 with the FP32 guard and complete a
# P=8,192 prefill without OOM, NaN, or Inf (Test 9 baseline check).
# ==============================================================================
import gc
import time
import torch
from hooks import (
    initialize_falcon_h1_for_t4,
    initialize_hybrid_cache,
)

def run_phase0_exit_gate(
    root_path: Path,
    seq_len: int = 8192,
    load_pretrained: bool = True,
) -> Dict[str, Any]:
    print("=" * 78)
    print(f"[Phase 0 | Cell 0.4] Running Phase 0 Exit Gate at P = {seq_len:,} tokens")
    print("=" * 78)

    configs_to_verify = [
        root_path / "configs" / "falcon_h1_0.5b.yaml",
        root_path / "configs" / "falcon_h1_1.5b_deep.yaml",
    ]
    exit_gate_results: Dict[str, Any] = {}

    for cfg_path in configs_to_verify:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        print(f"\n>>> Initializing from {cfg_path.name} (load_pretrained={load_pretrained})...")
        model, tokenizer, cfg, guard = initialize_falcon_h1_for_t4(
            cfg_path,
            load_pretrained=load_pretrained,
        )
        arch = cfg["architecture"]
        L = int(arch["num_hidden_layers"])
        H_kv = int(arch["num_key_value_heads"])
        d_h = int(arch["head_dim"])
        H_ssm = int(arch["mamba_n_heads"])
        d_head_ssm = int(arch["mamba_d_head"])
        d_state = int(arch["mamba_d_state"])

        # Verify model config matches YAML constants
        assert model.config.num_hidden_layers == L
        assert model.config.num_attention_heads == int(arch["num_attention_heads"])
        assert model.config.num_key_value_heads == H_kv
        assert model.config.mamba_n_heads == H_ssm
        assert model.config.mamba_d_head == d_head_ssm
        assert model.config.mamba_d_state == d_state

        # Verify FP32 guard on SSM parameters and FP16 on linear weights
        sample_mixer = model.model.layers[0].mamba
        assert sample_mixer.in_proj.weight.dtype == torch.float16, "in_proj must stay FP16"
        assert sample_mixer.out_proj.weight.dtype == torch.float16, "out_proj must stay FP16"
        assert sample_mixer.A_log.dtype == torch.float32, "A_log must be guarded in FP32"
        assert sample_mixer.D.dtype == torch.float32, "D must be guarded in FP32"
        assert sample_mixer.dt_bias.dtype == torch.float32, "dt_bias must be guarded in FP32"

        # Initialize HybridMambaAttentionDynamicCache & synthetic P=8,192 input
        cache = initialize_hybrid_cache(model, batch_size=1)
        first_device = next(model.parameters()).device
        torch.manual_seed(13)
        input_ids = torch.randint(
            low=10,
            high=1000,
            size=(1, seq_len),
            device=first_device,
            dtype=torch.long,
        )

        # Run P=8,192 prefill under inference_mode (logits_to_keep=1 avoids unnecessary 8K vocab projection)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )

        logits = outputs.logits
        returned_cache = outputs.past_key_values

        # Check 1: Zero NaN or Inf in output logits
        logits_finite = bool(torch.isfinite(logits).all().item())
        assert logits_finite, f"Non-finite logits detected in {cfg['model_id']} at P={seq_len}"

        # Check 2: Verify HybridMambaAttentionDynamicCache KV & SSM states across all L layers
        for l_idx in range(L):
            ssm_state = returned_cache.ssm_states[l_idx]
            assert ssm_state.dtype == torch.float32, (
                f"Layer {l_idx} ssm_state dtype is {ssm_state.dtype}, expected torch.float32"
            )
            assert tuple(ssm_state.shape) == (1, H_ssm, d_head_ssm, d_state), (
                f"Layer {l_idx} unexpected ssm_state shape: {tuple(ssm_state.shape)}"
            )
            assert bool(torch.isfinite(ssm_state).all().item()), (
                f"Layer {l_idx} ssm_state contains NaN/Inf at P={seq_len}"
            )

            k_cache = returned_cache.key_cache[l_idx]
            v_cache = returned_cache.value_cache[l_idx]
            assert k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16
            assert tuple(k_cache.shape) == (1, H_kv, seq_len, d_h)
            assert bool(torch.isfinite(k_cache).all().item()) and bool(torch.isfinite(v_cache).all().item())

        elapsed = time.perf_counter() - t0
        gpu_peak_gb = []
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                peak_gb = torch.cuda.max_memory_allocated(idx) / (1024 ** 3)
                gpu_peak_gb.append(round(peak_gb, 3))

        exit_gate_results[cfg["model_short_name"]] = {
            "model_id": cfg["model_id"],
            "layers_guarded": guard.upgraded_layers,
            "seq_len": seq_len,
            "logits_shape": tuple(logits.shape),
            "logits_finite": logits_finite,
            "ssm_state_shape_per_layer": (1, H_ssm, d_head_ssm, d_state),
            "ssm_state_dtype": "torch.float32",
            "kv_cache_shape_per_layer": (1, H_kv, seq_len, d_h),
            "kv_cache_dtype": "torch.float16",
            "gpu_peak_vram_gb": gpu_peak_gb,
            "elapsed_sec": round(elapsed, 2),
        }

        print(f"[PASS] {cfg['model_id']} passed P={seq_len:,} Exit Gate in {elapsed:.2f}s:")
        print(f"       - Guarded Layers : {guard.upgraded_layers}/{L}")
        print(f"       - Logits         : shape={tuple(logits.shape)}, finite={logits_finite}")
        print(f"       - SSM State S_P  : shape=(1, {H_ssm}, {d_head_ssm}, {d_state}), dtype=torch.float32 (finite=True)")
        print(f"       - KV Cache       : shape=(1, {H_kv}, {seq_len}, {d_h}), dtype=torch.float16 (finite=True)")
        print(f"       - Peak VRAM (GB) : {gpu_peak_gb}")

        # Clean up model before loading the next one
        guard.remove()
        del outputs, returned_cache, cache, input_ids, model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 78)
    print("[PHASE 0 EXIT GATE: 100% PASSED] Handover artifacts verified for Phase 1.")
    print("=" * 78)
    return exit_gate_results

# Execute Phase 0 Exit Gate at P = 8,192
PHASE0_RESULTS = run_phase0_exit_gate(PROJECT_ROOT, seq_len=8192, load_pretrained=True)