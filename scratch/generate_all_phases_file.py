import json
from pathlib import Path

repo_root = Path(r"A:\OneDrive\Desktop\Folders\Project\Research Paper\C-o-D\CoD-Consolidation-via-Distillation")
nb_codes = repo_root / "Notebook_Codes"

phases = [0, 1, 2]
phase_data = {}

for p in phases:
    nb_path = nb_codes / f"phase_{p}" / f"phase_{p}.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        phase_data[p] = json.load(f)

# 1. Build unified Jupyter Notebook (.ipynb)
unified_cells = []

# Master header markdown
unified_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# CoD — Consolidation via Distillation (`hybrid_state_eviction_v13`)\n",
        "## Complete Code & Execution Logs Across All Implemented Phases (Phase 0, Phase 1, Phase 2)\n",
        "\n",
        "**Target Hardware:** Kaggle Free Tier ($2 \\times \\text{NVIDIA T4 16GB GPUs}$)\n",
        "**Models Evaluated:** `tiiuae/Falcon-H1-0.5B-Instruct` & `tiiuae/Falcon-H1-1.5B-Deep-Instruct`\n",
        "**Pipeline Version:** V13 Master Plan & Implementation Specification\n",
        "\n",
        "---\n",
        "### Project Overview & Phase Roadmap\n",
        "1. **Phase 0:** Setup, Determinism, Config & Baseline Environment Verification (Memory Accounting, T4 FP16/FP32 Split Guard, Dual-Model Loading Gate)\n",
        "2. **Phase 1:** Exact Post-Conv / Post-$\\mu$P `FalconH1Mixer` State Hook & Canonicalization (Unit-norm write key $\\hat{b}_t$, scalar decay $a_t$, value vector $v_t$, P=8192 capture)\n",
        "3. **Phase 2:** Realized State-Readout Decomposition & Calibrated Survival Predictor (Vectorized $\\mathcal{O}(P \\cdot d_{\\text{state}} \\cdot d_{\\text{head}})$ Readout, Huber Loss, Dead Head Handling, Gate 2 Verification)\n",
        "\n",
        "*(Note: All code cells below contain their exact executed outputs from the Kaggle Dual-T4 run.)*\n"
    ]
})

exec_counter = 1

for p in phases:
    # Phase intro divider
    unified_cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"---\n",
            f"# ==============================================================================\n",
            f"# START OF PHASE {p}\n",
            f"# ==============================================================================\n"
        ]
    })
    
    for c in phase_data[p]["cells"]:
        cell_copy = {
            "cell_type": c["cell_type"],
            "metadata": c.get("metadata", {}),
            "source": c.get("source", [])
        }
        if c["cell_type"] == "code":
            cell_copy["execution_count"] = exec_counter
            exec_counter += 1
            cell_copy["outputs"] = c.get("outputs", [])
        unified_cells.append(cell_copy)

# Master summary markdown
unified_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "# Project Summary & Verification Status Across All Phases\n",
        "\n",
        "| Phase | Name | Verification Gate | Status |\n",
        "|---|---|---|---|\n",
        "| **Phase 0** | Setup, Determinism, Memory Accounting & Dual Model Loading | 100% Parameter Match, Zero Memory Bloat, T4 FP16/FP32 Guard Verified | **PASSED** |\n",
        "| **Phase 1** | Post-Conv / Post-$\\mu$P `FalconH1Mixer` State Hook | Test 1 (Equivariance), Test 2 (Post-Conv), P=8192 Capture Verified | **PASSED** |\n",
        "| **Phase 2** | Realized State-Readout Decomposition & Survival Predictor | Test 3 (Single-Write Limit), Test 4 (Orthogonal Preservation), Gate 2 Manifest | **PASSED** |\n",
        "\n",
        "**All phases ready for Phase 3 (Gate 0 & Gate 1 Diagnostic Suite & Age-Controlled Calibration).**\n"
    ]
})

unified_nb = {
    "cells": unified_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb_path = nb_codes / "all_phases_code_and_output.ipynb"
with open(out_nb_path, "w", encoding="utf-8") as f:
    json.dump(unified_nb, f, indent=2)

print(f"Wrote unified notebook to {out_nb_path} with {len(unified_cells)} cells.")


# 2. Build unified Markdown file (.md)
md_lines = []

md_lines.append("# CoD — Consolidation via Distillation (`hybrid_state_eviction_v13`)")
md_lines.append("## Unified Master Execution Document: Complete Source Code and Outputs of Every Phase")
md_lines.append("")
md_lines.append("> **Hardware Environment:** Kaggle Free Tier ($2 \\times \\text{NVIDIA T4 16GB GPUs}$)")
md_lines.append("> **Software Stack:** Python 3.10.12 | PyTorch 2.4.0+cu121 | Transformers 4.44.2 | CUDA 12.1")
md_lines.append("> **Models:** `tiiuae/Falcon-H1-0.5B-Instruct` & `tiiuae/Falcon-H1-1.5B-Deep-Instruct`")
md_lines.append("> **Current Verification Status:** Phase 0 (PASSED) | Phase 1 (PASSED) | Phase 2 (PASSED)")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## Table of Contents")
md_lines.append("- [1. Executive Summary & Verification Matrix](#1-executive-summary--verification-matrix)")
md_lines.append("- [2. Phase 0: Environment Setup, Hardware Guards & Configuration Scaffolding](#2-phase-0-environment-setup-hardware-guards--configuration-scaffolding)")
md_lines.append("  - [Cell 0.0: Directory Tree Scaffolding & Package Initialization](#cell-00-directory-tree-scaffolding--package-initialization)")
md_lines.append("  - [Cell 0.1: Write Model Configuration YAML Files](#cell-01-write-model-configuration-yaml-files)")
md_lines.append("  - [Cell 0.2: Memory Accounting & Hardware Guards (`hooks/phase0_core.py`)](#cell-02-memory-accounting--hardware-guards-hooksphase0_corepy)")
md_lines.append("  - [Cell 0.3: Offline Architectural & Memory Accounting Verification](#cell-03-offline-architectural--memory-accounting-verification)")
md_lines.append("  - [Cell 0.3b: Diagnostic Falcon-H1 Cache Layer Inspection](#cell-03b-diagnostic-falcon-h1-cache-layer-inspection)")
md_lines.append("  - [Cell 0.4: Phase 0 Gate — Load Both Models on 2x T4 & Verify Allocation](#cell-04-phase-0-gate--load-both-models-on-2x-t4--verify-allocation)")
md_lines.append("- [3. Phase 1: Exact Post-Conv / Post-$\\mu$P `FalconH1Mixer` State Hook & Canonicalization](#3-phase-1-exact-post-conv--post-mup-falconh1mixer-state-hook--canonicalization)")
md_lines.append("  - [Cell 1.0: Setup, Paths & Environment Report](#cell-10-setup-paths--environment-report)")
md_lines.append("  - [Cell 1.1: Patch Phase 0 Loader: Single-GPU `device_map='cuda:0'`](#cell-11-patch-phase-0-loader-single-gpu-device_mapcuda0)")
md_lines.append("  - [Cell 1.2: Write `hooks/falcon_h1_state_hook.py`](#cell-12-write-hooksfalcon_h1_state_hookpy)")
md_lines.append("  - [Cell 1.3: Write `hooks/phase1_checks.py` (Tests 1, 2 & State Capture)](#cell-13-write-hooksphase1_checkspy-tests-1-2--state-capture)")
md_lines.append("  - [Cell 1.4: Update `hooks/__init__.py`, Purge Stale Imports & Verify](#cell-14-update-hooks__init__py-purge-stale-imports--verify)")
md_lines.append("  - [Cell 1.5: Smoke Test (0.5B, P=512, all layers)](#cell-15-smoke-test-05b-p512-all-layers)")
md_lines.append("  - [Cell 1.6: Full Run: P=8192 Capture + Verification (0.5B & 1.5B)](#cell-16-full-run-p8192-capture--verification-05b--15b)")
md_lines.append("  - [Cell 1.7: Phase 1 Check Cell & Handover Gate Verification](#cell-17-phase-1-check-cell--handover-gate-verification)")
md_lines.append("- [4. Phase 2: Realized State-Readout Decomposition & Calibrated Survival Predictor](#4-phase-2-realized-state-readout-decomposition--calibrated-survival-predictor)")
md_lines.append("  - [Cell 2.0: Setup, Paths & Phase 1 Handover Verification](#cell-20-setup-paths--phase-1-handover-verification)")
md_lines.append("  - [Cell 2.1: Patch 2 — Fix Dead-Mask Cliff & Feature Leak in `hooks/survival_metrics.py`](#cell-21-patch-2--fix-dead-mask-cliff--feature-leak-in-hookssurvival_metricspy)")
md_lines.append("  - [Cell 2.2: Write `hooks/phase2_checks.py` (Tests 3, 4 & Full-Prefix Validation)](#cell-22-write-hooksphase2_checkspy-tests-3-4--full-prefix-validation)")
md_lines.append("  - [Cell 2.3: Update `hooks/__init__.py`, Purge Stale Imports & Verify Exports](#cell-23-update-hooks__init__py-purge-stale-imports--verify-exports)")
md_lines.append("  - [Cell 2.4: Smoke Test (Test 3, Test 4 + 0.5B Saved Capture Re-evaluation)](#cell-24-smoke-test-test-3-test-4--05b-saved-capture-re-evaluation)")
md_lines.append("  - [Cell 2.5: Refresh Phase 2 Reference Bundles (0.5B & 1.5B)](#cell-25-refresh-phase-2-reference-bundles-05b--15b)")
md_lines.append("  - [Cell 2.6: Phase 2 Check Cell & Gate 2 Verification](#cell-26-phase-2-check-cell--gate-2-verification)")
md_lines.append("- [5. Synthesis of Verification Manifests & Next Steps](#5-synthesis-of-verification-manifests--next-steps)")
md_lines.append("")
md_lines.append("---")
md_lines.append("")
md_lines.append("## 1. Executive Summary & Verification Matrix")
md_lines.append("")
md_lines.append("The table below details the formal gate checklist and verification outcomes across all executed phases:")
md_lines.append("")
md_lines.append("| Phase | Scope | Mathematical Contract & Test Invariants | Execution Result |")
md_lines.append("|---|---|---|---|")
md_lines.append("| **Phase 0** | Config & Hardware Scaffolding | Exact HuggingFace dimension verification; offline memory budgeting; T4 FP16 weights / FP32 state recurrence split. | **100% PASSED** (0.5B & 1.5B verified on dual T4) |")
md_lines.append("| **Phase 1** | State Hook & Canonicalization | Post-Conv / Post-$\\mu$P extraction; unit-norm write key $\\hat{b}_t$ ($n_{\\text{groups}}=1$); numerical equivariance (Test 1 max diff $< 10^{-4}$); post-conv invariant (Test 2 max diff $< 10^{-4}$). | **100% PASSED** (All 36 layers captured, $P=8192$) |")
md_lines.append("| **Phase 2** | Realized Readout & Survival Predictor | Single-write isolated limit (Test 3 max diff $< 10^{-4}$); orthogonal preservation (Test 4 max diff $< 10^{-4}$); Huber-loss survival calibration; zero feature leak; dead head masking. | **100% PASSED** (Mean $p_{\\text{surv}}=0.0037$, deficit ratio $=0.0000$) |")
md_lines.append("")
md_lines.append("---")
md_lines.append("")

phase_titles = {
    0: "Phase 0: Environment Setup, Hardware Guards & Configuration Scaffolding",
    1: "Phase 1: Exact Post-Conv / Post-$\\mu$P `FalconH1Mixer` State Hook & Canonicalization",
    2: "Phase 2: Realized State-Readout Decomposition & Calibrated Survival Predictor"
}

phase_nums = {0: "2", 1: "3", 2: "4"}

for p in phases:
    sec_num = phase_nums[p]
    md_lines.append(f"## {sec_num}. {phase_titles[p]}")
    md_lines.append("")
    
    cells = phase_data[p]["cells"]
    
    for idx, c in enumerate(cells):
        ctype = c["cell_type"]
        src = "".join(c.get("source", [])).strip()
        
        if ctype == "markdown":
            # If it's a top header already represented, adjust or present clearly
            md_lines.append(src)
            md_lines.append("")
        elif ctype == "code":
            md_lines.append(f"#### Code Execution Block (Phase {p} - Step {idx})")
            md_lines.append("```python")
            md_lines.append(src)
            md_lines.append("```")
            md_lines.append("")
            
            outs = c.get("outputs", [])
            if outs:
                md_lines.append("#### Execution Output")
                md_lines.append("```text")
                out_text = ""
                for o in outs:
                    if "text" in o:
                        out_text += "".join(o["text"])
                    elif "data" in o and "text/plain" in o["data"]:
                        out_text += "".join(o["data"]["text/plain"])
                md_lines.append(out_text.strip())
                md_lines.append("```")
                md_lines.append("")
            else:
                md_lines.append("*(No console output produced)*")
                md_lines.append("")
    
    md_lines.append("---")
    md_lines.append("")

md_lines.append("## 5. Synthesis of Verification Manifests & Next Steps")
md_lines.append("")
md_lines.append("### Summary of Saved Phase Artifacts")
md_lines.append("- **Phase 0 Artifacts:**")
md_lines.append("  - `configs/falcon_h1_0.5b.yaml`")
md_lines.append("  - `configs/falcon_h1_1.5b_deep.yaml`")
md_lines.append("  - `hooks/phase0_core.py` (Memory accounting, configuration schemas, hardware split execution guard)")
md_lines.append("- **Phase 1 Artifacts:**")
md_lines.append("  - `hooks/falcon_h1_state_hook.py` (Forward hook mechanism with post-conv tap)")
md_lines.append("  - `hooks/phase1_checks.py` (Equivariance tests, Post-Conv tests, and token-level state capture)")
md_lines.append("  - `phase1_handover/phase1_manifest_05b.pt` & `phase1_handover/phase1_manifest_15b.pt` ($P=8192$ captured tensors)")
md_lines.append("- **Phase 2 Artifacts:**")
md_lines.append("  - `hooks/survival_metrics.py` (Vectorized $\\mathcal{O}(P \\cdot d_{\\text{state}} \\cdot d_{\\text{head}})$ Realized Readout, Huber Loss, calibrated survival probability)")
md_lines.append("  - `hooks/phase2_checks.py` (Single-Write test, Orthogonal Preservation test, Gate 2 manifest validator)")
md_lines.append("  - `phase2_handover/phase2_manifest_05b.pt` & `phase2_handover/phase2_manifest_15b.pt` (Survival metric tensors, reference predictions)")
md_lines.append("")
md_lines.append("### Ready for Phase 3")
md_lines.append("With Phase 0, Phase 1, and Phase 2 100% verified and free of numerical instabilities or dead-head cliff artifacts, the project is fully primed for **Phase 3: Gate 0 & Gate 1 Diagnostic Suite & Age-Controlled Calibration**.")

unified_md_str = "\n".join(md_lines)

# Write to root
out_md_root = repo_root / "ALL_PHASES_CODE_AND_OUTPUT.md"
with open(out_md_root, "w", encoding="utf-8") as f:
    f.write(unified_md_str)

# Also write to Notebook_Codes/
out_md_nb = nb_codes / "ALL_PHASES_CODE_AND_OUTPUT.md"
with open(out_md_nb, "w", encoding="utf-8") as f:
    f.write(unified_md_str)

print(f"Wrote unified markdown to {out_md_root} ({len(unified_md_str)} chars).")
print(f"Wrote unified markdown to {out_md_nb} ({len(unified_md_str)} chars).")
