import ast

with open('scratch/user_input_phase2.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Cell 2.0
c20 = '''import os, sys, json, gc, time, platform
from pathlib import Path
import torch
import transformers

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
CFG_DIR = PROJECT_ROOT / "configs"
PHASE1_DIR = PROJECT_ROOT / "checkpoints" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "checkpoints" / "phase2"   # FIXED handover path for Phase 3 & Phase 4
PHASE2_DIR.mkdir(parents=True, exist_ok=True)
FORCE_RERUN = False

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Verify Phase 0 and Phase 1 modules and handover artifacts exist
assert (PROJECT_ROOT / "hooks" / "phase0_core.py").exists(), "Missing hooks/phase0_core.py (run Phase 0 first)."
assert (PROJECT_ROOT / "hooks" / "falcon_h1_state_hook.py").exists(), "Missing hooks/falcon_h1_state_hook.py (run Phase 1 first)."
assert (PHASE1_DIR / "phase1_manifest.json").exists(), "Missing checkpoints/phase1/phase1_manifest.json (run Phase 1 first)."

p1_manifest = json.loads((PHASE1_DIR / "phase1_manifest.json").read_text())
assert not p1_manifest.get("problems"), f"Phase 1 manifest reported problems: {p1_manifest.get('problems')}"

for tag in ("05b", "15b"):
    ref_pt = PHASE1_DIR / f"ref_capture_{tag}_text_2048.pt"
    assert ref_pt.exists(), f"Missing Phase 1 reference capture: {ref_pt}"

print(f"python {platform.python_version()} | torch {torch.__version__} | transformers {transformers.__version__}")
print(f"cuda available: {torch.cuda.is_available()} | gpus: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
print(f"[Phase 2.0] Verified Phase 1 handover at: {PHASE1_DIR}")
print(f"[Phase 2.0] Initialized Phase 2 handover directory at: {PHASE2_DIR}")
'''

# Cell 2.1
idx_21 = text.find('#Cell 2.1')
idx_22 = text.find('#Cell 2.2')
c21 = text[idx_21:idx_22].strip()

# Cell 2.2
idx_23 = text.find('#Cell 2.3')
c22 = text[idx_22:idx_23].strip()

# Cell 2.3
idx_24 = text.find('#Cell 2.4')
c23 = text[idx_23:idx_24].strip()

# Cell 2.4
idx_25 = text.find('#Cell 2.5')
c24 = text[idx_24:idx_25].strip()

# Cell 2.5
idx_26 = text.find('#Cell 2.6')
c25 = text[idx_25:idx_26].strip()

# Cell 2.6
idx_out = text.find('AND THERE ARE THE OUTPUTSB')
c26 = text[idx_26:idx_out].strip()

cells = {
    '2.0': c20,
    '2.1': c21,
    '2.2': c22,
    '2.3': c23,
    '2.4': c24,
    '2.5': c25,
    '2.6': c26
}

for name, code in cells.items():
    print(f"Parsing Cell {name}...")
    ast.parse(code)
    print(f"Cell {name} parsed successfully ({len(code.splitlines())} lines)!")

print("\nALL 7 PHASE 2 CELLS PARSED PERFECTLY!")

# Let's inspect what is after AND THERE ARE THE OUTPUTSB
print("\n=== Post code text length ===", len(text[idx_out:]))
print(text[idx_out:idx_out+500])
