# Master Plan V13 — Falcon-H1-1.5B-Deep-Instruct Configuration (Primary Benchmark)
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