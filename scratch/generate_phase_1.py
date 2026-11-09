import json
import os
import shutil
from pathlib import Path

# Paths
ROOT = Path(".").resolve()
NB_DIR = ROOT / "Notebook_Codes" / "phase_1"
ROOT_PHASE1 = ROOT / "phase_1"
NB_DIR.mkdir(parents=True, exist_ok=True)
ROOT_PHASE1.mkdir(parents=True, exist_ok=True)

with open('scratch/user_input_194.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 1.0
idx_10 = text.find('#Cell 1.0')
idx_code_end_10 = text.find('print(f"[Phase 1.0] handover dir: {PHASE1_DIR}")') + len('print(f"[Phase 1.0] handover dir: {PHASE1_DIR}")')
c10 = text[idx_10:idx_code_end_10].strip()

# 1.1
c11 = '''# ==============================================================================
# Cell 1.1 - Patch Phase 0 Loader: Optional Single-GPU `device_map` | Phase 1 (edit of Phase 0 `hooks/phase0_core.py`)
# Idempotent. Adds `device_map=None` argument to `load_falcon_h1_for_t4` (default behaviour unchanged).
# ==============================================================================
from pathlib import Path
PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
p = PROJECT_ROOT / "hooks" / "phase0_core.py"
txt = p.read_text()

if "device_map: Optional[Any] = None" in txt:
    print("[Phase 1.1] phase0_core.py already patched.")
else:
    old_sig = ("def load_falcon_h1_for_t4(\\n    yaml_path: Union[str, Path],\\n    load_tokenizer: bool = True,\\n)")
    new_sig = ("def load_falcon_h1_for_t4(\\n    yaml_path: Union[str, Path],\\n    load_tokenizer: bool = True,\\n"
               "    device_map: Optional[Any] = None,\\n)")
    old_dm = '        device_map=cfg["precision_guard"].get("device_map", "auto"),'
    new_dm = ('        device_map=(device_map if device_map is not None '
              'else cfg["precision_guard"].get("device_map", "auto")),')
    assert txt.count(old_sig) == 1 and txt.count(old_dm) == 1, "phase0_core.py differs from expected; paste it to me."
    p.write_text(txt.replace(old_sig, new_sig).replace(old_dm, new_dm))
    print("[Phase 1.1] Patched phase0_core.py (device_map override).")
'''.strip()

# 1.2
idx_12 = text.find('#Cell 1.2')
idx_code_end_12 = text.find('print("[Phase 1.2] wrote hooks/falcon_h1_state_hook.py")') + len('print("[Phase 1.2] wrote hooks/falcon_h1_state_hook.py")')
c12 = text[idx_12:idx_code_end_12].strip()

# 1.3
idx_13 = text.find('#Cell 1.3')
idx_code_end_13 = text.find('print("[Phase 1.3] wrote hooks/phase1_checks.py")') + len('print("[Phase 1.3] wrote hooks/phase1_checks.py")')
c13 = text[idx_13:idx_code_end_13].strip()

# 1.4
idx_14 = text.find('#Cell 1.4')
idx_code_end_14 = text.find('print("[Phase 1.4] imports OK")') + len('print("[Phase 1.4] imports OK")')
c14 = text[idx_14:idx_code_end_14].strip()

# 1.5
idx_15 = text.find('#Cell 1.5')
idx_code_end_15 = text.find('raise SystemExit("Smoke failed -> send me this cell\'s full output (do NOT run 1.6).")') + len('raise SystemExit("Smoke failed -> send me this cell\'s full output (do NOT run 1.6).")')
c15 = text[idx_15:idx_code_end_15].strip()

# 1.6
idx_16 = text.find('#Cell 1.6')
idx_code_end_16 = text.find('print("\\n[Phase 1.6] done. 0.5B pass:", rep_05b["phase1_pass"], "| 1.5B pass:", rep_15b["phase1_pass"])') + len('print("\\n[Phase 1.6] done. 0.5B pass:", rep_05b["phase1_pass"], "| 1.5B pass:", rep_15b["phase1_pass"])')
c16 = text[idx_16:idx_code_end_16].strip()

# 1.7
idx_17 = text.find('#Cell 1.7')
idx_code_end_17 = text.find('print("PHASE 1 GATE:", "PASS" if not problems else "FAIL")') + len('print("PHASE 1 GATE:", "PASS" if not problems else "FAIL")')
c17 = text[idx_17:idx_code_end_17].strip()

# Outputs
out10 = """python 3.10.12 | torch 2.4.0 | transformers 4.44.2
cuda available: True | gpus: ['Tesla T4', 'Tesla T4']
mamba_ssm: not installed (expected; reference torch path)
causal_conv1d: not installed (expected; reference torch path)
[Phase 1.0] handover dir: /kaggle/working/hybrid_state_eviction_v13/checkpoints/phase1
"""

out11 = """[Phase 1.1] Patched phase0_core.py (device_map override).
"""

out12 = """[Phase 1.2] wrote hooks/falcon_h1_state_hook.py
"""

out13 = """[Phase 1.3] wrote hooks/phase1_checks.py
"""

out14 = """[Phase 1.4] imports OK
"""

out15 = """tokenizer_config.json: 100%
 99.9k/99.9k [00:00<00:00, 10.5MB/s]
tokenizer.json: 100%
 2.35M/2.35M [00:00<00:00, 47.2MB/s]
special_tokens_map.json: 100%
 7.69k/7.69k [00:00<00:00, 860kB/s]
chat_template.jinja: 100%
 1.15k/1.15k [00:00<00:00, 133kB/s]
Loading weights: 100%
 579/579 [00:00<00:00, 1188.50it/s]
--- environment facts the hook relies on ---
mup_vector dtype      : torch.float16 | ssm_multipliers: [0.3535533905932738, 0.25, 0.3535533905932738, 0.5, 0.3535533905932738] | ssm_in_multiplier: 1.25
time_step_limit       : (0.0, inf) | n_groups: 1 | conv_dim: 1792
record shapes (L0)    : b_hat (1, 512, 128) u (1, 512, 24, 64) w (1, 512, 24) alpha (1, 512, 24) S_P (1, 24, 128, 64)
records size          : 158.1 MB | capture 0.24s
hook passive (max |logit diff| with vs without hook): 0.000e+00
--- summary ---
{
  "n_layers": 36,
  "test2_decomp_max": 2.874617742270267e-06,
  "test2_decomp_median": 6.24346967015299e-07,
  "test2_step_max": 6.066701415961222e-06,
  "test1_scaled_err_max": 5.812816271700333e-07,
  "all_finite": true,
  "ranges_ok": true,
  "verify_seconds": 0.93,
  "test2_pass": true,
  "test1_pass": true,
  "stability_pass": true,
  "phase1_pass": true
}
worst 5 layers (decomp vs cache): [(30, '2.87e-06', '2.94e-06'), (29, '1.97e-06', '2.27e-06'), (33, '1.51e-06', '1.53e-06'), (4, '1.42e-06', '1.93e-06'), (9, '1.38e-06', '1.55e-06')]
test1 samples: [(0, {'max_abs_err': 9.54e-06, 'scale': 24.66796303, 'scaled_err': 3.9e-07, 'cache_readout_rel_err': 1.66e-06}), (18, {'max_abs_err': 9.155e-05, 'scale': 157.50151062, 'scaled_err': 5.8e-07, 'cache_readout_rel_err': 1.45e-06}), (35, {'max_abs_err': 9.155e-05, 'scale': 271.79519653, 'scaled_err': 3.4e-07, 'cache_readout_rel_err': 1.07e-06})]

SMOKE PASS
"""

out16 = """Loading weights: 100%
 579/579 [00:00<00:00, 1172.79it/s]
[1.6] 05b/text_8192: P=8192 pass=True | test2 decomp max=2.62e-06 step max=5.30e-05 | test1 6.31e-07 | finite=True | cap 8.4s verify 18.37s | peak VRAM 6.52 GB | records 2.11 GB
[1.6] 05b/text_2048: P=2048 pass=True | test2 decomp max=2.80e-06 step max=2.00e-05 | test1 6.55e-07 | finite=True | cap 1.3s verify 4.18s | peak VRAM 1.50 GB | records 0.55 GB
[1.6] 05b/random_2048: P=2048 pass=True | test2 decomp max=1.77e-06 step max=1.15e-05 | test1 5.31e-07 | finite=True | cap 0.9s verify 4.14s | peak VRAM 1.50 GB | records 0.55 GB
tokenizer_config.json: 100%
 99.9k/99.9k [00:00<00:00, 9.13MB/s]
tokenizer.json: 100%
 5.06M/5.06M [00:00<00:00, 69.5MB/s]
special_tokens_map.json: 100%
 7.69k/7.69k [00:00<00:00, 771kB/s]
chat_template.jinja: 100%
 1.15k/1.15k [00:00<00:00, 132kB/s]
Loading weights: 100%
 1125/1125 [00:01<00:00, 880.77it/s]
[1.6] 15b/text_8192: P=8192 pass=True | test2 decomp max=2.58e-06 step max=9.28e-05 | test1 1.38e-06 | finite=True | cap 17.2s verify 34.44s | peak VRAM 7.95 GB | records 4.19 GB
[1.6] 15b/text_2048: P=2048 pass=True | test2 decomp max=2.36e-06 step max=2.78e-05 | test1 1.04e-06 | finite=True | cap 2.5s verify 7.99s | peak VRAM 3.82 GB | records 1.13 GB
[1.6] 15b/random_2048: P=2048 pass=True | test2 decomp max=2.80e-06 step max=2.61e-05 | test1 1.21e-06 | finite=True | cap 2.3s verify 7.95s | peak VRAM 3.82 GB | records 1.13 GB

[Phase 1.6] done. 0.5B pass: True | 1.5B pass: True
"""

out17 = """======================================================================
PHASE 1 SUMMARY (paste everything between the lines back to me)
======================================================================
{
 "05b": {
  "text_8192": {
   "P": 8192,
   "n_layers": 36,
   "test2_decomp_max": 2.6244198196263277e-06,
   "test2_step_max": 5.2973746073328696e-05,
   "test1_scaled_err_max": 6.305927847344264e-07,
   "all_finite": true,
   "ranges_ok": true,
   "phase1_pass": true,
   "capture_seconds": 8.39,
   "records_gb": 2.106,
   "peak_vram_gb_cuda0": 6.52
  },
  "text_2048": {
   "P": 2048,
   "n_layers": 36,
   "test2_decomp_max": 2.8004889443809444e-06,
   "test2_step_max": 2.0021260287095426e-05,
   "test1_scaled_err_max": 6.546528738844567e-07,
   "all_finite": true,
   "ranges_ok": true,
   "phase1_pass": true,
   "capture_seconds": 1.3,
   "records_gb": 0.548,
   "peak_vram_gb_cuda0": 1.497
  },
  "random_2048": {
   "P": 2048,
   "n_layers": 36,
   "test2_decomp_max": 1.7690095758538785e-06,
   "test2_step_max": 1.1465820812928693e-05,
   "test1_scaled_err_max": 5.31049226882604e-07,
   "all_finite": true,
   "ranges_ok": true,
   "phase1_pass": true,
   "capture_seconds": 0.92,
   "records_gb": 0.548,
   "peak_vram_gb_cuda0": 1.497
  }
 },
 "15b": {
  "text_8192": {
   "P": 8192,
   "n_layers": 66,
   "test2_decomp_max": 2.5804735936523784e-06,
   "test2_step_max": 9.275309341839076e-05,
   "test1_scaled_err_max": 1.3811595399357572e-06,
   "all_finite": true,
   "ranges_ok": true,
   "phase1_pass": true,
   "capture_seconds": 17.23,
   "records_gb": 4.189,
   "peak_vram_gb_cuda0": 7.952
  },
  "text_2048": {
   "P": 2048,
   "n_layers": 66,
   "test2_decomp_max": 2.3600371643430305e-06,
   "test2_step_max": 2.784875741470408e-05,
   "test1_scaled_err_max": 1.035842748712318e-06,
   "all_finite": true,
   "ranges_ok": true,
   "phase1_pass": true,
   "capture_seconds": 2.51,
   "records_gb": 1.125,
   "peak_vram_gb_cuda0": 3.817
  },
  "random_2048": {
   "P": 2048,
   "n_layers": 66,
   "test2_decomp_max": 2.7973443524893727e-06,
   "test2_step_max": 2.612186549065055e-05,
   "test1_scaled_err_max": 1.208491030622731e-06,
   "all_finite": true,
   "ranges_ok": true,
   "phase1_pass": true,
   "capture_seconds": 2.31,
   "records_gb": 1.125,
   "peak_vram_gb_cuda0": 3.817
  }
 }
}
problems: NONE
======================================================================
PHASE 1 GATE: PASS
"""

cells_data = [
    ("Cell 1.0 - Setup, Paths & Environment Report | Phase 1: State Hook & Canonicalization", c10, out10),
    ("Cell 1.1 - Patch Phase 0 Loader: Optional Single-GPU `device_map` | Phase 1 (edit of Phase 0 `hooks/phase0_core.py`)", c11, out11),
    ("Cell 1.2 - Write hooks/falcon_h1_state_hook.py | Phase 1: State Hook & Canonicalization", c12, out12),
    ("Cell 1.3 - Write hooks/phase1_checks.py (Tests 1, 2, Stability) | Phase 1: State Hook & Canonicalization", c13, out13),
    ("Cell 1.4 - Update hooks/__init__.py, Purge Stale Imports, Import Check | Phase 1: State Hook & Canonicalization", c14, out14),
    ("Cell 1.5 - SMOKE TEST (0.5B, P=512, all layers) | Phase 1: State Hook & Canonicalization", c15, out15),
    ("Cell 1.6 - FULL RUN: P=8192 Capture + Verification (0.5B then 1.5B-Deep) + Reference Captures | Phase 1: State Hook & Canonicalization", c16, out16),
    ("Cell 1.7 - PHASE 1 CHECK CELL (validates outputs, writes manifest, prints what to send back) | Phase 1: State Hook & Canonicalization", c17, out17),
]

# 1. Generate phase_1.py
phase_1_py_parts = []
for title, code, _ in cells_data:
    phase_1_py_parts.append(f"# %% [code]\n# ==============================================================================\n# {title}\n# ==============================================================================\n{code}\n")
phase_1_py_content = "\n".join(phase_1_py_parts)

(NB_DIR / "phase_1.py").write_text(phase_1_py_content, encoding="utf-8")
print(f"Wrote {NB_DIR / 'phase_1.py'}")

# 2. Generate build_phase_1_nb.py
build_script_content = '''import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent / "phase_1.ipynb"

intro_markdown = [
    "# Phase 1: Exact Post-Conv / Post-$\\\\mu$P `FalconH1Mixer` State Hook & Canonicalization\\n",
    "\\n",
    "This notebook contains the complete Phase 1 implementation and verified outputs for the CoD (`hybrid_state_eviction_v13`) pipeline:\\n",
    "- **Cell 1.0:** Setup, directory paths, environment report, and module check\\n",
    "- **Cell 1.1:** Patch Phase 0 loader for optional single-GPU `device_map` override\\n",
    "- **Cell 1.2:** Implement `hooks/falcon_h1_state_hook.py` with exact mixer front-end replay and unit-key canonicalization\\n",
    "- **Cell 1.3:** Implement `hooks/phase1_checks.py` containing dual verification (Test 1: Readout identity, Test 2: State reconstruction & stepwise recurrence) and stability assertions\\n",
    "- **Cell 1.4:** Update `hooks/__init__.py`, purge stale imports, and verify module signatures\\n",
    "- **Cell 1.5:** Fast smoke test on `Falcon-H1-0.5B` ($P=512$) verifying zero passive interference on logits\\n",
    "- **Cell 1.6:** Full run with $P=8,192$ and $P=2,048$ captures on both `0.5B` and `1.5B-Deep` models\\n",
    "- **Cell 1.7:** Phase 1 gate audit, summary JSON verification, and handover manifest generation (`checkpoints/phase1/phase1_manifest.json`)"
]

cells_data = ''' + repr([(t, c, o) for t, c, o in cells_data]) + '''

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": intro_markdown
    }
]

for idx, (title, code, output_text) in enumerate(cells_data, start=1):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"### {title}"]
    })
    
    code_lines = [line + "\\n" for line in code.splitlines()]
    if code_lines:
        code_lines[-1] = code_lines[-1].rstrip("\\n")
        
    outputs = []
    if output_text.strip():
        out_lines = [line + "\\n" for line in output_text.splitlines()]
        if out_lines:
            out_lines[-1] = out_lines[-1].rstrip("\\n")
        outputs.append({
            "name": "stdout",
            "output_type": "stream",
            "text": out_lines
        })
        
    cells.append({
        "cell_type": "code",
        "execution_count": idx,
        "metadata": {},
        "outputs": outputs,
        "source": code_lines
    })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

nb_path.parent.mkdir(parents=True, exist_ok=True)
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated {nb_path}")
'''

(NB_DIR / "build_phase_1_nb.py").write_text(build_script_content, encoding="utf-8")
print(f"Wrote {NB_DIR / 'build_phase_1_nb.py'}")

# 3. Generate README1.md
readme_content = f"""# Phase 1: Exact Post-Conv / Post-$\\mu$P `FalconH1Mixer` State Hook & Canonicalization

This directory contains the complete implementation, build generator, interactive Jupyter Notebook, and execution verification outputs for **Phase 1** of the **CoD — Consolidation via Distillation** (`hybrid_state_eviction_v13`) pipeline on Kaggle Free Tier ($2 \\times \\text{{NVIDIA T4 16GB GPUs}}$).

---

## Handover Mathematical Contract to Phase 2 (`hooks/survival_metrics.py`)

Per mixer layer $l \\in \\{{1, \\dots, L\\}}$ (dict key = `layer_idx`), Phase 1 provides:
- **`b_hat`** $[B, P, d_{{\\text{{state}}}}]$: Unit-norm write key (where $n_{{\\text{{groups}}}} = 1$, $\\hat{{b}}_t = \\frac{{B_t}}{{\\|B_t\\| + \\epsilon}}$).
- **`u`** $[B, P, H_{{\\text{{ssm}}}}, d_{{\\text{{head,ssm}}}}]$: Effective value vector ($u_{{t,m}} = \\Delta_{{t,m}} (\\|B_t\\| + \\epsilon) x_{{t,m}}$).
- **`w`** $[B, P, H_{{\\text{{ssm}}}}]$: Write energy ($\\|u_{{t,m}}\\|_2$).
- **`log_alpha`** $[B, P, H_{{\\text{{ssm}}}}]$: Reverse cumulative decay $\\ell_{{t,m}}(P) = \\sum_{{\\tau=t+1}}^{{P-1}} \\log a_{{\\tau,m}} \\le 0$ computed via FP64 exclusive reverse cumsum.
- **`alpha`** $[B, P, H_{{\\text{{ssm}}}}]$: Survival decay $\\alpha_{{t,m}}(P) = \\exp(\\ell_{{t,m}}(P)) \\in [0, 1]$.
- **`S_P`** $[B, H_{{\\text{{ssm}}}}, d_{{\\text{{state}}}}, d_{{\\text{{head,ssm}}}}]$: Terminal prefill recurrent state extracted directly from the model cache in canonical layout (FP32).
- **Additive Extras:** `delta` $[B, P, H_{{\\text{{ssm}}}}]$, `b_norm` $[B, P]$, `S_P_cache_layout` (str).

Invariant guaranteed:
$$\\hat{{b}}_t^\\top u_{{t,m}} \\equiv B_t^\\top (\\Delta_{{t,m}} x_{{t,m}})$$

---

## B. What I Changed vs the Plan, Why, and Expected Gain

| # | Change | Why Better | Expected Gain |
|---|---|---|---|
| **1** | **$\\mu$P is applied BEFORE conv, not after.** Real `FalconH1Mixer`: `proj = in_proj(ssm_in_multiplier·h) * mup_vector` $\\to$ `conv1d` $\\to$ `SiLU`. Plan Step 1 says `conv` $\\to$ `SiLU` $\\to$ `multipliers`. | $\\text{{SiLU}}(\\text{{conv}}(\\mu z)) \\ne \\mu \\cdot \\text{{SiLU}}(\\text{{conv}}(z))$. Plan order gives mathematically wrong $x, B, \\Delta$. | $S_P$ identity matches at $\\sim 10^{{-6}}$ instead of an $\\mathcal{{O}}(1)$ structural divergence. Marked `[MATH-UPGRADE \\| Phase 1 -> Phase 2]`. |
| **2** | Front-end replayed op-for-op with the model's own dtypes (FP16 `in_proj`, $\\mu$P dtype, conv in weight dtype, `dt` softplus/clamp). | Tensors are identical to what `mamba2_chunk_scan` consumes. | Test 2 error $\\approx$ FP32 rounding ($10^{{-6}} \\dots 10^{{-5}}$), well under $10^{{-4}}$ tolerance threshold. |
| **3** | FP64 exclusive reverse cumsum for $\\ell_t(P)$. | FP32 $c_{{P-1}} - c_t$ suffers from catastrophic cancellation, losing $\\sim 6 \\times 10^{{-8}} \\cdot |c|$ absolute ($\\alpha$ error $\\sim 10^{{-4}}$ when $|c| \\sim 10^3$). | $\\alpha$ relative error reduced from $\\sim 10^{{-4}}$ to $< 10^{{-9}}$. |
| **4** | Unit canonicalization: $u = \\Delta(\\|B\\| + \\epsilon)x$, $\\hat{{b}} = B / (\\|B\\| + \\epsilon)$. | $\\hat{{b}}^\\top u = B^\\top (\\Delta x)$ holds identically; eliminates $\\mathcal{{O}}(\\epsilon)$ numerical gap. | Exact mathematical identity preservation. |
| **5** | Two independent checks vs real cache state: closed-form state decomposition AND stepwise FP32 recurrence built from $\\Delta$ and $A_{{\\log}}$. | Catches hook bugs or indexing slips that either check alone could disguise. | Test 2 gate is fully robust and mathematically rigorous. |
| **6** | Scale-aware Test 1 tolerance: $\\text{{max\\_abs}} \\le 10^{{-5}} \\cdot \\max(1, \\max|\\text{{readout}}|)$. | Absolute $10^{{-5}}$ in FP32 is below machine precision when readouts are $\\mathcal{{O}}(10^2)$. | Zero false rejections; enforces tightest valid bound. |
| **7** | Handed-over $S_P$ is the true cache state in canonical layout; $S_P^{{\\text{{recon}}}}$ not stored. | Ground truth for Phase 2 readouts. | Phase 2 uses exact model state without reconstruction drift. |
| **8** | `store_device` option (`"cpu"` default in full run). | $1.5\\text{{B-Deep}}$ $P=8,192$ records consume $\\approx 4.19\\text{{ GB}}$. | Frees GPU VRAM completely for model parameters and scan execution. |
| **9** | Single-GPU loader patch (`device_map={{\\"\\": 0}}`). | Eliminates cross-device tensor transfers inside hooks; frees GPU 1 entirely. | Simpler, faster, Phase 5 teacher-distillation ready. |
| **10** | Hook refuses chunked prefill (raises on non-empty cache). | Chunked prefill requires conv/recurrent carry; silent misuse would corrupt $\\alpha$. | Completely safe by design and construction. |

---

## Files Inventory

- **`phase_1.ipynb`**: Complete Jupyter Notebook containing all 8 Phase 1 cells along with their exact saved outputs.
- **`phase_1.py`**: Standalone Python script executable end-to-end.
- **`build_phase_1_nb.py`**: Programmatic generator script that constructs `phase_1.ipynb`.
- **`README1.md`**: Complete documentation with all cell codes, modifications, and captured outputs.

---

## Cells and Outputs

### Cell 1.0 - Setup, Paths & Environment Report | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c10}
```

**Output:**
```
{out10.strip()}
```

---

### Cell 1.1 - Patch Phase 0 Loader: Optional Single-GPU `device_map` | Phase 1 (edit of Phase 0 `hooks/phase0_core.py`)

**Code:**
```python
{c11}
```

**Output:**
```
{out11.strip()}
```

---

### Cell 1.2 - Write `hooks/falcon_h1_state_hook.py` | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c12}
```

**Output:**
```
{out12.strip()}
```

---

### Cell 1.3 - Write `hooks/phase1_checks.py` (Tests 1, 2, Stability) | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c13}
```

**Output:**
```
{out13.strip()}
```

---

### Cell 1.4 - Update `hooks/__init__.py`, Purge Stale Imports, Import Check | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c14}
```

**Output:**
```
{out14.strip()}
```

---

### Cell 1.5 - SMOKE TEST (0.5B, P=512, all layers) | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c15}
```

**Output:**
```
{out15.strip()}
```

---

### Cell 1.6 - FULL RUN: P=8192 Capture + Verification (0.5B then 1.5B-Deep) + Reference Captures | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c16}
```

**Output:**
```
{out16.strip()}
```

---

### Cell 1.7 - PHASE 1 CHECK CELL (validates outputs, writes manifest, prints what to send back) | Phase 1: State Hook & Canonicalization

**Code:**
```python
{c17}
```

**Output:**
```
{out17.strip()}
```

---

## Phase 1 Gate Audit Verdict

```
======================================================================
PHASE 1 SUMMARY
======================================================================
{json.dumps(json.loads(out17.split("PHASE 1 SUMMARY (paste everything between the lines back to me)\n======================================================================\n")[1].split("\nproblems:")[0]), indent=1)}
problems: NONE
======================================================================
PHASE 1 GATE: PASS
```

- **Falcon-H1-0.5B-Instruct ($P=8,192$):** Test 2 Decomp Max $= 2.62 \\times 10^{{-6}}$, Step Max $= 5.30 \\times 10^{{-5}}$, Test 1 Scaled Err $= 6.31 \\times 10^{{-7}}$, Peak VRAM $= 6.52\\text{{ GB}}$, Capture Time $= 8.39\\text{{ s}}$.
- **Falcon-H1-1.5B-Deep-Instruct ($P=8,192$):** Test 2 Decomp Max $= 2.58 \\times 10^{{-6}}$, Step Max $= 9.28 \\times 10^{{-5}}$, Test 1 Scaled Err $= 1.38 \\times 10^{{-6}}$, Peak VRAM $= 7.95\\text{{ GB}}$, Capture Time $= 17.23\\text{{ s}}$.
- **Passive Logit Difference:** $0.000\\text{{e}}+00$ (zero model interference).
"""

(NB_DIR / "README1.md").write_text(readme_content, encoding="utf-8")
print(f"Wrote {NB_DIR / 'README1.md'}")

# Also mirror to ROOT_PHASE1
shutil.copy(NB_DIR / "phase_1.py", ROOT_PHASE1 / "phase_1.py")
shutil.copy(NB_DIR / "build_phase_1_nb.py", ROOT_PHASE1 / "build_phase_1_nb.py")
shutil.copy(NB_DIR / "README1.md", ROOT_PHASE1 / "README1.md")
print(f"Mirrored files to {ROOT_PHASE1}")

