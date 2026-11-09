import ast

code_25 = '''import gc, json, torch
from pathlib import Path
from hooks import (
    SurvivalPredictor,
    test_03_sir_single_write_limit,
    test_04_r5b_equal_age_interference,
    audit_model_survival_profile,
)

PROJECT_ROOT = Path("/kaggle/working/hybrid_state_eviction_v13")
PHASE1_DIR = PROJECT_ROOT / "checkpoints" / "phase1"
PHASE2_DIR = PROJECT_ROOT / "checkpoints" / "phase2"

for tag, d_state in (("05b", 128), ("15b", 256)):
    predictor = SurvivalPredictor(mode="closed_form")
    t3 = test_03_sir_single_write_limit(P=256, H_ssm=24, d_state=d_state, d_head=64, device="cuda:0")

    ref_2048_blob = torch.load(PHASE1_DIR / f"ref_capture_{tag}_text_2048.pt", map_location="cpu")
    records_2048 = ref_2048_blob["layers"]
    L = len(records_2048)

    t4 = test_04_r5b_equal_age_interference(
        P=256, H_ssm=24, d_state=d_state, d_head=64, n_pairs=16, target_collision_cos=0.80,
        device="cuda:0", real_layer_record=records_2048[L // 2],
    )
    audit_2048, surv_2048 = audit_model_survival_profile(
        records_2048, predictor=predictor, compute_device="cuda:0", store_device="cpu"
    )

    compact_layers_2048 = {
        int(l_idx): {
            "log_sir": l_dict["log_sir"].cpu(),
            "cosine": l_dict["cosine"].cpu(),
            "log_alpha": l_dict["log_alpha"].cpu(),
            "w": l_dict["w"].cpu(),
            "p_surv_head": l_dict["p_surv_head"].cpu(),
            "p_surv_bar": l_dict["p_surv_bar"].cpu(),
            "deficit_bar": l_dict["deficit_bar"].cpu(),
            "alpha_bar": l_dict["alpha_bar"].cpu(),
        }
        for l_idx, l_dict in surv_2048["layers"].items()
    }
    torch.save({
        "meta": {
            "tag": tag,
            "model": ref_2048_blob["meta"]["model"],
            "L": L,
            "P": 2048,
            "d_state": d_state,
            "calibrator_config": predictor.to_config_dict(),
        },
        "input_ids": ref_2048_blob["input_ids"].cpu(),
        "p_surv_bar_all": surv_2048["p_surv_bar_all"].cpu(),
        "deficit_bar_all": surv_2048["deficit_bar_all"].cpu(),
        "alpha_bar_all": surv_2048["alpha_bar_all"].cpu(),
        "layers": compact_layers_2048,
    }, PHASE2_DIR / f"survival_ref_{tag}_text_2048.pt")

    s = audit_2048["summary"]
    print(f"[Patched Phase 2 | {tag}] test3={t3['passed']} | test4={t4['passed']} | "
          f"mean_p_surv={s['mean_candidate_p_surv_bar']:.4f} | "
          f"deficit_range=[{s['min_layer_deficit_bar']:.4f}, {s['max_layer_deficit_bar']:.4f}] | "
          f"within_age_log_sir_std=[{s['min_within_age_log_sir_std']:.3f}, {s['mean_within_age_log_sir_std']:.3f}]")

    # Update saved report_phase2_{tag}.json with refreshed P=2048 metrics
    rep_path = PHASE2_DIR / f"report_phase2_{tag}.json"
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())
        rep["test3"] = t3
        rep["test4"] = t4
        rep["runs"]["text_2048"] = audit_2048
        rep_path.write_text(json.dumps(rep, indent=2))

    del ref_2048_blob, records_2048, surv_2048, compact_layers_2048
    gc.collect()
    torch.cuda.empty_cache()
'''

print("Checking code_25 with ast.parse...")
ast.parse(code_25)
print("code_25: OK")
