import json
from pathlib import Path

repo_root = Path(r"A:\OneDrive\Desktop\Folders\Project\Research Paper\C-o-D\CoD-Consolidation-via-Distillation")

new_code = (
    "#Cell 2.3 - Update hooks/__init__.py, Purge Stale Imports & Verify Signatures | Phase 2: Realized State-Readout Decomposition & Calibrated Survival Predictor\n"
    "import sys, importlib\n"
    "from pathlib import Path\n"
    "\n"
    'PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")\n'
    "\n"
    '# Write updated hooks package exports for Phase 0, 1, and 2\n'
    '(PROJECT_ROOT / "hooks" / "__init__.py").write_text(\n'
    '    \'"""Hooks package: Phase 0 T4 guards + Phase 1 state hook + Phase 2 survival metrics."""\\n\'\n'
    '    "from .phase0_core import (\\n"\n'
    '    "    load_model_config, sync_arch_config_with_hf_config, verify_memory_accounting,\\n"\n'
    '    "    compute_retained_kv_ratio, apply_t4_fp32_ssm_guard, load_falcon_h1_for_t4,\\n"\n'
    '    "    ensure_unified_cache_views, force_ssm_states_fp32,\\n"\n'
    '    ")\\n"\n'
    '    "from .falcon_h1_state_hook import FalconH1StateHook, canonical_state\\n"\n'
    '    "from .phase1_checks import (\\n"\n'
    '    "    verify_capture, make_input_ids, reconstruct_terminal_state,\\n"\n'
    '    "    stepwise_recurrence, readout_identity_error,\\n"\n'
    '    ")\\n"\n'
    '    "from .survival_metrics import (\\n"\n'
    '    "    SurvivalPredictor, compute_readout_decomposition, compute_layer_survival_features,\\n"\n'
    '    ")\\n"\n'
    '    "from .phase2_checks import (\\n"\n'
    '    "    test_03_sir_single_write_limit, test_04_r5b_equal_age_interference,\\n"\n'
    '    "    audit_model_survival_profile,\\n"\n'
    '    ")\\n"\n'
    ")\n"
    "\n"
    'for m in [m for m in sys.modules if m == "hooks" or m.startswith("hooks.")]:\n'
    "    del sys.modules[m]\n"
    "importlib.invalidate_caches()\n"
    "\n"
    "from hooks import (\n"
    "    SurvivalPredictor,\n"
    "    compute_readout_decomposition,\n"
    "    compute_layer_survival_features,\n"
    "    test_03_sir_single_write_limit,\n"
    "    test_04_r5b_equal_age_interference,\n"
    "    audit_model_survival_profile,\n"
    ")\n"
    "\n"
    'print("[Phase 2.3] Reloaded updated hooks package and verified all Phase 2 exports.")\n'
)

# Convert string to list of lines with \n
new_source_lines = [line + "\n" for line in new_code.splitlines()]

# 1. Update phase_2.ipynb
p2_nb_path = repo_root / "Notebook_Codes" / "phase_2" / "phase_2.ipynb"
with open(p2_nb_path, "r", encoding="utf-8") as f:
    p2_nb = json.load(f)

for c in p2_nb["cells"]:
    src = "".join(c.get("source", []))
    if "Cell 2.3" in src and c["cell_type"] == "code":
        c["source"] = new_source_lines
        print("Updated Cell 2.3 in phase_2.ipynb")

with open(p2_nb_path, "w", encoding="utf-8") as f:
    json.dump(p2_nb, f, indent=2)

# 2. Update phase_2.py
p2_py_path = repo_root / "Notebook_Codes" / "phase_2" / "phase_2.py"
p2_py_text = p2_py_path.read_text(encoding="utf-8")

old_code_py = (
    "#Cell 2.3 - Update hooks/__init__.py, Purge Stale Imports & Verify Signatures | Phase 2: Realized State-Readout Decomposition & Calibrated Survival Predictor\n"
    "import sys, importlib\n"
    "from pathlib import Path\n"
    "\n"
    'PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")\n'
    "\n"
    'for m in [m for m in sys.modules if m == "hooks" or m.startswith("hooks.")]:\n'
    "    del sys.modules[m]\n"
    "importlib.invalidate_caches()\n"
    "\n"
    "from hooks import (\n"
    "    SurvivalPredictor,\n"
    "    compute_readout_decomposition,\n"
    "    compute_layer_survival_features,\n"
    "    test_03_sir_single_write_limit,\n"
    "    test_04_r5b_equal_age_interference,\n"
    "    audit_model_survival_profile,\n"
    ")\n"
    "\n"
    'print("[Phase 2.3] Reloaded updated hooks package and verified all Phase 2 exports.")'
)

if old_code_py in p2_py_text:
    p2_py_text = p2_py_text.replace(old_code_py, new_code.strip())
    p2_py_path.write_text(p2_py_text, encoding="utf-8")
    print("Updated Cell 2.3 in phase_2.py")
else:
    print("Warning: old_code_py not found in phase_2.py")

# 3. Update build_phase_2_nb.py
b2_py_path = repo_root / "Notebook_Codes" / "phase_2" / "build_phase_2_nb.py"
b2_text = b2_py_path.read_text(encoding="utf-8")
if old_code_py in b2_text:
    b2_text = b2_text.replace(old_code_py, new_code.strip())
    b2_py_path.write_text(b2_text, encoding="utf-8")
    print("Updated Cell 2.3 in build_phase_2_nb.py")
else:
    print("Warning: old_code_py not found in build_phase_2_nb.py")

# 4. Regenerate master files: all_phases_code_and_output.ipynb and ALL_PHASES_CODE_AND_OUTPUT.md
import subprocess
res = subprocess.run(["python", str(repo_root / "scratch" / "generate_all_phases_file.py")], capture_output=True, text=True)
print("generate_all_phases_file.py output:\n", res.stdout)
