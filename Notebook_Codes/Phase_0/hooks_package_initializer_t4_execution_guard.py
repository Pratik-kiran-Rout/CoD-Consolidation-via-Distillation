"""
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