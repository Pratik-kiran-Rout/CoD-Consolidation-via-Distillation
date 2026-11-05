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

