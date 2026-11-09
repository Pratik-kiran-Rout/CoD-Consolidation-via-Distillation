import ast

with open('scratch/user_input_194.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 1.0
idx_10 = text.find('#Cell 1.0')
idx_code_end_10 = text.find('print(f"[Phase 1.0] handover dir: {PHASE1_DIR}")') + len('print(f"[Phase 1.0] handover dir: {PHASE1_DIR}")')
c10 = text[idx_10:idx_code_end_10].strip()
print('Parsing 1.0...')
ast.parse(c10)
print('1.0 OK')

# 1.1
c11 = '''from pathlib import Path
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
'''
print('Parsing 1.1...')
ast.parse(c11)
print('1.1 OK')

# 1.2
idx_12 = text.find('#Cell 1.2')
idx_code_end_12 = text.find('print("[Phase 1.2] wrote hooks/falcon_h1_state_hook.py")') + len('print("[Phase 1.2] wrote hooks/falcon_h1_state_hook.py")')
c12 = text[idx_12:idx_code_end_12].strip()
print('Parsing 1.2...')
ast.parse(c12)
print('1.2 OK')

# 1.3
idx_13 = text.find('#Cell 1.3')
idx_code_end_13 = text.find('print("[Phase 1.3] wrote hooks/phase1_checks.py")') + len('print("[Phase 1.3] wrote hooks/phase1_checks.py")')
c13 = text[idx_13:idx_code_end_13].strip()
print('Parsing 1.3...')
ast.parse(c13)
print('1.3 OK')

# 1.4
idx_14 = text.find('#Cell 1.4')
idx_code_end_14 = text.find('print("[Phase 1.4] imports OK")') + len('print("[Phase 1.4] imports OK")')
c14 = text[idx_14:idx_code_end_14].strip()
print('Parsing 1.4...')
ast.parse(c14)
print('1.4 OK')

# 1.5
idx_15 = text.find('#Cell 1.5')
idx_code_end_15 = text.find('raise SystemExit("Smoke failed -> send me this cell\'s full output (do NOT run 1.6).")') + len('raise SystemExit("Smoke failed -> send me this cell\'s full output (do NOT run 1.6).")')
c15 = text[idx_15:idx_code_end_15].strip()
print('Parsing 1.5...')
ast.parse(c15)
print('1.5 OK')

# 1.6
idx_16 = text.find('#Cell 1.6')
idx_code_end_16 = text.find('print("\\n[Phase 1.6] done. 0.5B pass:", rep_05b["phase1_pass"], "| 1.5B pass:", rep_15b["phase1_pass"])') + len('print("\\n[Phase 1.6] done. 0.5B pass:", rep_05b["phase1_pass"], "| 1.5B pass:", rep_15b["phase1_pass"])')
c16 = text[idx_16:idx_code_end_16].strip()
print('Parsing 1.6...')
ast.parse(c16)
print('1.6 OK')

# 1.7
idx_17 = text.find('#Cell 1.7')
idx_code_end_17 = text.find('print("PHASE 1 GATE:", "PASS" if not problems else "FAIL")') + len('print("PHASE 1 GATE:", "PASS" if not problems else "FAIL")')
c17 = text[idx_17:idx_code_end_17].strip()
print('Parsing 1.7...')
ast.parse(c17)
print('1.7 OK')

print('ALL CELLS PARSED PERFECTLY!')
