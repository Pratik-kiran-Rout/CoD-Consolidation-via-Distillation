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