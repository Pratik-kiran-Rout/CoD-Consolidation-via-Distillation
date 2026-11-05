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

