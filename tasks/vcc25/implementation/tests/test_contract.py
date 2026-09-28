from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_files_and_cli_contract() -> None:
    required = ["README.md", "IMPLEMENTATION_PLAN.md", "ASSUMPTIONS_AND_DEVIATIONS.md",
                "requirements.txt", "train.py", "scripts/run_smoke.sh", "scripts/run_full.sh"]
    assert all((ROOT / name).is_file() for name in required)
    source = (ROOT / "train.py").read_text()
    for flag in ["--data-root", "--dataset", "--device", "--seed", "--train-limit",
                 "--eval-limit", "--max-steps", "--method", "--metrics-out",
                 "--use-guide-prior", "--guide-shrinkage-type"]:
        assert flag in source
    assert "global_mean_baseline_metrics" in source


def test_worker_payload_does_not_submit_or_proxy() -> None:
    smoke = (ROOT / "scripts/run_smoke.sh").read_text()
    assert "rjob submit" not in smoke
    assert "proxyon" not in smoke and "proxy_on" not in smoke


def test_no_forbidden_domain_terms() -> None:
    sources = "\n".join(p.read_text() for p in ROOT.glob("*.py"))
    for forbidden in ["CIFAR", "BADE", "AFR", "attention budget"]:
        assert forbidden not in sources


def test_rndp_contract_and_activation_diagnostics() -> None:
    source = (ROOT / "crpm/rndp.py").read_text()
    launcher = (ROOT / "scripts/run_rndp.sh").read_text()
    for capability in ["build_coexpression_graph", "pagerank", "calibration",
                       "sparse_residual", "degree_matched_random_graph"]:
        assert capability in source
    for diagnostic in ["pagerank_diffusion_nonzero",
                       "calibration_weight_norm_positive",
                       "sparse_correction_nonzero_count",
                       "random_graph_control_pearson_delta"]:
        assert diagnostic in source
    assert "rjob submit" not in launcher
    assert "proxyon" not in launcher and "proxy_on" not in launcher
    assert "candidate_rndp.json" in launcher


def test_rndp_torch_pagerank_matches_numpy() -> None:
    import numpy as np
    import torch
    from crpm.rndp import pagerank, pagerank_torch
    indices=np.asarray([[1],[2],[0]],dtype=np.int32); weights=np.ones_like(indices,dtype=np.float32)
    expected=pagerank(np.asarray([0,2]),indices,weights,iterations=5)
    actual=pagerank_torch(np.asarray([0,2]),indices,weights,iterations=5,device="cpu")
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6)


def test_rndp_official_runner_enforces_full_gene_and_random_control_gates() -> None:
    source=(ROOT/"official_h1_rndp_validation.py").read_text()
    for value in ["full_gene_target_coverage","structured_graph_beats_random_control","beats_validation_promotion_reference","test_expression_read"]:
        assert value in source


def test_cc_gat_contract_and_controls() -> None:
    source = (ROOT / "crpm/cc_gat.py").read_text()
    launcher = (ROOT / "scripts/run_cc_gat.sh").read_text()
    for capability in ["construct_context_vectors", "context_conditioned_propagation",
                       "pre_decoder_de_overlap_at_100", "shuffled_context_metrics"]:
        assert capability in source
    for diagnostic in ["context_vector_norm_positive",
                       "attention_conditioning_events_nonzero",
                       "propagated_signal_l2_variance_across_targets",
                       "shuffled_context_performance_delta"]:
        assert diagnostic in source
    assert "rjob submit" not in launcher
    assert "proxyon" not in launcher and "proxy_on" not in launcher
    assert "candidate_cc_gat.json" in launcher


def test_cc_gat_official_validation_gates() -> None:
    source=(ROOT/"official_h1_cc_gat_validation.py").read_text()
    for value in ["full_gene_target_coverage","context_conditioning_beats_shuffle","beats_promotion_reference","test_expression_read"]:
        assert value in source


def test_official_go_representation_and_promotion_gate() -> None:
    import json

    root = ROOT.parents[2]
    manifest = json.loads((root / "datasets/vcc25/annotations/official-h1-go-bp-v1/manifest.json").read_text())
    assert manifest["test_expression_used"] is False
    assert {name: row["targets"] for name, row in manifest["coverage"].items()} == {"train": 150, "validation": 50, "test": 100}
    assert min(row["coverage"] for row in manifest["coverage"].values()) >= 0.95
    source = (ROOT / "official_h1_candidate_h_research.py").read_text()
    for requirement in ["go_pca", "go_representation_gate", "delta_correlation_gain_vs_best_non_go", "minimum_go_coverage"]:
        assert requirement in source
    for requirement in ["pathway_neighbors", "pathway_alpha", "joint_pathway_conditioning", "pathway_shuffled_control", "gain_vs_embedding_only", "gain_vs_shuffled_control"]:
        assert requirement in source


def test_cgbm_contract_and_activation_diagnostics() -> None:
    source = (ROOT / "crpm/cgbm.py").read_text()
    launcher = (ROOT / "scripts/run_cgbm.sh").read_text()
    for capability in ["de_rank_order_pretext_loss", "scm_do_operator_propagation",
                       "PerGeneIsotonicCalibrator", "cross_fit_isotonic",
                       "degree_matched_random_graph",
                       "build_go_bp_neighborhoods"]:
        assert capability in source
    for diagnostic in ["pretext_loss_nonzero", "gat_attention_events_nonzero",
                       "mediation_effect_nonzero", "isotonic_calibration_fitted",
                       "random_control_delta_correlation_reported"]:
        assert diagnostic in source
    assert "rjob submit" not in launcher
    assert "proxyon" not in launcher and "proxy_on" not in launcher
    assert "candidate_cgbm.json" in launcher
    assert "official-h1-go-bp-v1" in launcher


def test_candidate_i_pathway_basis_has_new_fold_and_negative_control_gates() -> None:
    source = (ROOT / "official_h1_candidate_i_pathway_basis.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_i_validation.sh").read_text()
    for requirement in [
        "go_bp_ic_features", "fit_pathway_basis", "de_strength",
        "confidence_power", "fold_salt", "shuffled_control",
        "gain_vs_shuffled", "promotion_reference",
    ]:
        assert requirement in source
    assert "test expression" not in source.lower()
    assert "rjob submit" not in launcher


def test_replogle_download_is_pinned_and_atomic() -> None:
    source = (ROOT / "scripts/download_replogle_k562.sh").read_text()
    for value in [
        "7416068", "1546729675", "d8cba17576d1a8afc0f7d71b79cad0f7",
        ".partial", "sha256sum",
    ]:
        assert value in source
    assert "rjob submit" not in source


def test_context_corpus_downloads_are_pinned_and_atomic() -> None:
    source = (ROOT / "scripts/download_scperturb_context_corpora.sh").read_text()
    for value in ["10044268", "1236886900", "cc7f1ec50aeb3a3e1b4a6cfa713d80fa",
                  "350848489", "8279484264b513fd0419dacdc639ecef", ".partial"]:
        assert value in source


def test_candidate_m_keeps_six_direction_gate_leakage_safe() -> None:
    source = (ROOT / "official_h1_candidate_m_six_direction_gate.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_m_validation.sh").read_text()
    assert '"test_expression_used": False' in source
    assert '"official_test_prediction_generated": False' in source
    assert "shuffled_feature_control" in source
    assert "shuffled_context_control" in source
    assert "hierarchical_confidence_fallback" in source
    assert "candidate-m-six-direction-folds-20260913" in launcher


def test_replogle_basis_excludes_official_validation_and_test_targets() -> None:
    source = (ROOT / "prepare_replogle_basis.py").read_text()
    for value in [
        "SOURCE_SHA256", "forbidden = validation | test", "test_expression_used",
        "excluded_validation_targets", "excluded_test_targets", "artifact_sha256",
    ]:
        assert value in source


def test_candidate_j_external_mixture_and_controls() -> None:
    source = (ROOT / "official_h1_candidate_j_external_basis.py").read_text()
    for value in ["external_model", "pairwise_rank_loss", "confidence_power", "shuffled_external", "promotion_reference"]:
        assert value in source
    assert "test_expression_used\":False" in source.replace(" ", "")


def test_candidate_k_train_only_context_alignment_and_controls() -> None:
    source = (ROOT / "official_h1_candidate_k_context_alignment.py").read_text()
    for value in ["context_aligned_model", "train_only_anchor_counts", "minimum_anchors",
                  "shuffled_feature_control", "shuffled_context_control", "promotion_reference"]:
        assert value in source
    assert "test_expression_used\":False" in source.replace(" ", "")


def test_candidate_l_nonlinear_encoder_is_fixed_and_controlled() -> None:
    source = (ROOT / "official_h1_candidate_l_nonlinear_response_encoder.py").read_text()
    for value in ["random_fourier", "encoder_width", "encoder_seed", "train_only_anchor_counts",
                  "shuffled_feature_control", "shuffled_context_control", "promotion_reference"]:
        assert value in source
    assert "for er in" not in source


def test_multicontext_preparation_is_hash_pinned_and_exclusion_safe() -> None:
    source = (ROOT / "prepare_multicontext_corpus.py").read_text()
    for value in [
        'actual_hash != spec["sha256"]', "forbidden = validation | test",
        'spec.get("control_column")', 'spec.get("source_gene_column")',
        "source_sha256s=np.asarray", "targets_retained",
        "excluded_validation_targets", "excluded_test_targets",
        '"test_expression_used": False', "contexts=np.asarray(contexts)",
        'spec.get("official_targets_only", False)', "official_target_overlap",
        "train/pert_counts_Training.csv",
    ]:
        assert value in source


def test_candidate_n_uses_nested_multifold_and_negative_controls() -> None:
    source = (ROOT / "official_h1_candidate_n_multicontext.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_n_validation.sh").read_text()
    for value in [
        "fit_context_model", "nested_select", 'row["fold"] != fold',
        "shuffled_target_control", "shuffled_context_control",
        "promotion_reference", "at least two external contexts",
        "row_weights", "np.median(norms[norms > 0])",
        "candidate_n_decomposition_cache", "candidate_n_fit_cache",
        "_randomized_vt", "power_iterations=2",
        '"test_expression_used": False', '"official_test_prediction_generated": False',
    ]:
        assert value in source
    assert "rjob submit" not in launcher


def test_target_complete_policy_is_additive_and_blocks_h1_expression() -> None:
    source = (ROOT / "prepare_multicontext_corpus.py").read_text()
    assert 'choices=("exclusion_safe", "vcc_official")' in source
    assert 'forbidden = validation | test' in source
    assert 'forbidden = set()' in source
    for value in ["same_target_external_context_allowed", "h1_validation_expression_used", "h1_test_expression_used"]:
        assert value in source


def test_candidate_o_uses_target_matched_transport_and_nested_controls() -> None:
    source = (ROOT / "official_h1_candidate_o_target_complete.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_o_validation.sh").read_text()
    for value in ["_fit_transport", "_target_priors", "same_target_external_context_allowed",
                  'row["fold"] != fold', "shuffled_target_control", "shuffled_context_control",
                  "h1_validation_expression_used_for_training", "h1_test_expression_used"]:
        assert value in source
    assert "rjob submit" not in launcher


def test_candidate_p_uses_pinned_genomewide_signatures_and_existing_gate() -> None:
    preparation = (ROOT / "prepare_replogle_harmonizome.py").read_text()
    candidate = (ROOT / "official_h1_candidate_p_genomewide.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_p_validation.sh").read_text()
    for value in [
        "matrix_sha256", "official_target_coverage", "source_sha256",
        '"h1_validation_expression_used": False', '"h1_test_expression_used": False',
        "perturbation_target", "vcc_official_external_only",
    ]:
        assert value in preparation
    assert "official_h1_candidate_o_target_complete" in candidate
    assert "candidate-p-genomewide-signature-folds-20260914" in launcher
    assert "rjob submit" not in launcher


def test_candidate_q_consumes_quantitative_gwps_with_existing_gate() -> None:
    candidate = (ROOT / "official_h1_candidate_q_quantitative_gwps.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_q_validation.sh").read_text()
    worker = (ROOT / "scripts/run_candidate_q_prepare_and_validate.sh").read_text()
    config = (ROOT / "configs/replogle_gwps_quantitative_target_complete_20260914.json").read_text()
    assert "official_h1_candidate_o_target_complete" in candidate
    assert "quantitative-gwps-transport-v1" in candidate
    assert "candidate-q-quantitative-gwps-folds-20260914" in launcher
    assert "EXTERNAL_SHA256" in launcher
    assert "rjob submit" not in launcher
    assert "prepare_multicontext_corpus.py" in worker
    assert "--information-policy vcc_official" in worker
    assert "official_targets_only" in config
    assert "d3269d9c863d96555eba768b42ab330cc1b57493cd33b4ac8fe7fbab8278f104" in config


def test_candidate_r_decomposes_template_and_residual_with_controls() -> None:
    candidate = (ROOT / "official_h1_candidate_r_decomposed_gwps.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_r_validation.sh").read_text()
    for value in [
        "external_template", "h1_template", "external_residual", "h1_residual",
        'control == "context"', 'control == "targets"', "candidate_r_transport_cache",
    ]:
        assert value in candidate
    assert "candidate-r-decomposed-gwps-folds-20260914" in launcher
    assert "rjob submit" not in launcher


def test_candidate_r_official_inference_is_promoted_and_bounded() -> None:
    candidate = (ROOT / "candidate_r_promoted_transport_model.py").read_text()
    launcher = (ROOT / "scripts/run_official_h1_candidate_r.sh").read_text()
    for value in ['CANDIDATE_ID = "candidate_r_promoted_decomposed_gwps"',
                  "TRANSPORT_RIDGE = 0.1", "EXTERNAL_WEIGHT = 0.5",
                  "uses_test_expression_for_training_or_selection", "fit_transport", "np.generic"]:
        assert value in candidate
    assert "--profile full" in launcher
    assert "--skip-metrics" in launcher
    assert "cell_eval_target_metrics" in launcher
    assert "float(counts[k])==100" in launcher
    assert "py312_infer_env/lib/python3.12/site-packages" in launcher
    assert 'if [ ! -e "$PRED" ]' in launcher
    assert "NUMBA_CACHE_DIR" in launcher
    three_seed = (ROOT / "scripts/run_official_h1_candidate_r_3seed.sh").read_text()
    assert "20260907 20260908 20260909" in three_seed
    assert "official_h1_seed_ensemble.py" in three_seed
    ensemble_only = (ROOT / "scripts/run_candidate_r_ensemble_only.sh").read_text()
    assert "seed_*/official_h1_prediction.h5ad" in ensemble_only
    assert "--skip-metrics" in ensemble_only

