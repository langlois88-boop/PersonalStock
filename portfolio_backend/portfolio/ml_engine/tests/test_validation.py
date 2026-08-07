from portfolio.ml_engine.training.validation import is_majority_only_predictor, validate_gate


def test_majority_only_model_fails_gate_on_balanced_metrics() -> None:
    """A model that always predicts the majority class scores exactly 0.5
    balanced accuracy and 0.0 F1, no matter how skewed the label split is —
    that's what makes those metrics (rather than raw accuracy) the right
    ones to gate on, and this must keep failing the default gate."""
    cv_mean_majority_only = 0.5
    wf_f1_majority_only = 0.0
    gate = validate_gate(cv_mean_majority_only, wf_f1_majority_only)
    assert gate.passed is False


def test_raw_accuracy_would_have_let_a_skewed_majority_model_through() -> None:
    """Sanity check documenting the bug this gate change closes: on a 90/10
    label split, "always predict majority" scores 0.90 *raw* accuracy —
    comfortably above the 0.55 threshold — which is exactly why cv_mean must
    not be raw accuracy."""
    accuracy_of_majority_only_on_90_10_split = 0.90
    gate = validate_gate(accuracy_of_majority_only_on_90_10_split, wf_f1=0.0)
    # With balanced accuracy this same model scores 0.5 and fails (see test
    # above); the raw-accuracy number alone would have passed the cv_mean
    # check, showing why wf_f1 (already F1-based) had to backstop it before,
    # and why cv_mean itself now needs to be imbalance-robust too.
    assert gate.passed is False  # wf_f1=0.0 still catches it via the F1 leg
    assert gate.reason is not None and "Walk-forward F1" in gate.reason


def test_is_majority_only_predictor() -> None:
    y_true = [0] * 90 + [1] * 10
    always_majority = [0] * 100
    assert is_majority_only_predictor(y_true, always_majority) is True

    discriminating = [0] * 85 + [1] * 15  # predicts some 1s too
    assert is_majority_only_predictor(y_true, discriminating) is False

    single_class_labels = [0] * 100  # dataset itself has only one class
    assert is_majority_only_predictor(single_class_labels, always_majority) is False
