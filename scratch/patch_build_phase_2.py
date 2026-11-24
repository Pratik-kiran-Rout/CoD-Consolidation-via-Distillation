from pathlib import Path

b2_path = Path("Notebook_Codes/phase_2/build_phase_2_nb.py")
text = b2_path.read_text(encoding="utf-8")

from scratch.apply_cell_2_3_fix import new_code

start_marker = "('Cell 2.3 - Update hooks/__init__.py, Purge Stale Imports & Verify Signatures | Phase 2: Realized State-Readout Decomposition & Calibrated Survival Predictor', "
end_marker = "[Phase 2.3] Reloaded updated hooks package and verified all Phase 2 exports.\\n'),"

idx_start = text.find(start_marker)
if idx_start != -1:
    idx_end = text.find(end_marker, idx_start)
    if idx_end != -1:
        full_end = idx_end + len(end_marker)
        
        # Format replacement tuple
        # The code string is stored as a python string literal in cells_data
        encoded_code = repr(new_code.strip() + "\n")
        replacement = f"{start_marker}{encoded_code}),"
        
        new_text = text[:idx_start] + replacement + text[full_end:]
        b2_path.write_text(new_text, encoding="utf-8")
        print("Updated build_phase_2_nb.py successfully!")
    else:
        print("Could not find end marker")
else:
    print("Could not find start marker")
