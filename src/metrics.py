"""
Evaluation metrics for FedActuary.

Rewritten after checking the base paper's actual training and evaluation
code (github.com/actuari/IFoA-FL-WP), not assumed. Two things this
corrects from the first version:

1. The model's prediction is compared DIRECTLY against y = ClaimNb (the
   raw count) -- verified from split_data(): `y = df_encoded.iloc[:, 0]`,
   the first column after dropping IDpol, i.e. ClaimNb itself. The
   training loss is `PoissonNLLLoss(log_input=False, full=True)(pred, y)`
   with no exposure multiplication anywhere near it. Exposure earns its
   effect only by being one of the 39 input features -- there is no
   classical offset step. The first version of this file assumed the
   model output was a frequency needing `* exposure` before comparison;
   that was wrong for this specific architecture.
2. %PDE is computed with sklearn's own `d2_tweedie_score(power=1)`,
   weighted by `sample_weight=exposure` -- verified directly from
   skorch_tuning.py, which calls this exact function this exact way.
   Using sklearn's implementation directly removes any risk of a
   hand-rolled formula subtly disagreeing with theirs.
"""
import numpy as np
from sklearn.metrics import d2_tweedie_score


def percentage_deviance_explained(y_true, y_pred, exposure):
    """%PDE, exposure-weighted, matching the base paper's own evaluation
    call exactly: d2_tweedie_score(y, pred, sample_weight=exposure,
    power=1) * 100. y_pred is the model's raw output -- a predicted
    COUNT, comparable directly to y_true = ClaimNb. Higher is better;
    100 is a perfect fit, 0 matches an exposure-weighted intercept-only
    (null) model, negative is worse than the null."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), 1e-9, None)
    exposure = np.asarray(exposure, dtype=np.float64)
    return float(d2_tweedie_score(y_true, y_pred, sample_weight=exposure, power=1)) * 100


def gini_coefficient(y_true, y_pred, exposure):
    """Actuarial (ordered-Lorenz) Gini: ranking ability, independent of
    %PDE. Since y_pred is a predicted COUNT (not a rate), ranking by raw
    y_pred would systematically favor long-exposure policies regardless
    of true risk -- a policy insured all year predicts a bigger count
    than an identical-risk policy insured for a month. Divide back out to
    an implied frequency (y_pred / exposure) before ordering, so the
    ranking reflects risk, not exposure length."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)

    implied_freq = y_pred / np.clip(exposure, 1e-9, None)
    order = np.argsort(implied_freq)
    y_sorted = y_true[order]
    e_sorted = exposure[order]

    cum_exposure = np.concatenate([[0.0], np.cumsum(e_sorted) / e_sorted.sum()])
    cum_claims = np.concatenate([[0.0], np.cumsum(y_sorted) / y_sorted.sum()])

    area_under_curve = np.trapezoid(cum_claims, cum_exposure)
    return 1 - 2 * area_under_curve


if __name__ == "__main__":
    # Sanity checks against the CORRECTED mental model: the "model" here
    # predicts a count directly, matching how the real training loop
    # actually works, not a frequency needing exposure multiplication.
    rng = np.random.default_rng(0)
    n = 20_000
    exposure = rng.uniform(0.1, 1.0, n).astype(np.float64)
    true_freq = rng.gamma(shape=2.0, scale=0.03, size=n)
    y = rng.poisson(true_freq * exposure).astype(np.float64)

    # An oracle predicting the true expected COUNT (freq * exposure) --
    # this is the correct comparison now, not freq alone.
    true_count = true_freq * exposure
    pde_oracle = percentage_deviance_explained(y, true_count, exposure)
    gini_oracle = gini_coefficient(y, true_count, exposure)

    # True null, matching d2_tweedie_score's OWN internal baseline: a
    # single constant (the exposure-weighted mean of y) for every policy.
    # This is NOT the same as lambda_bar * exposure below -- that varies
    # per row and is already mildly informative, so it should NOT score
    # ~0 against sklearn's flatter internal null. Worth knowing, not a bug.
    flat_null = np.full(n, np.average(y, weights=exposure))
    pde_flat_null = percentage_deviance_explained(y, flat_null, exposure)
    gini_flat_null = gini_coefficient(y, flat_null, exposure)

    lambda_bar = y.sum() / exposure.sum()
    exposure_scaled_null = lambda_bar * exposure
    pde_scaled_null = percentage_deviance_explained(y, exposure_scaled_null, exposure)

    print(f"oracle (true count)      : %PDE={pde_oracle:6.2f}   Gini={gini_oracle:5.3f}  (expect: clears 5.57, clearly positive)")
    print(f"flat constant null       : %PDE={pde_flat_null:6.2f}   Gini={gini_flat_null:5.3f}  (expect: ~0, ~0 -- matches sklearn's own null)")
    print(f"exposure-scaled 'null'   : %PDE={pde_scaled_null:6.2f}   (expect: small positive -- already beats a flat constant)")

    assert pde_oracle > 5, "oracle should clear the paper's real Global benchmark (5.57%, PRD Section 2)"
    assert abs(pde_flat_null) < 1.0, "sklearn's own flat null must score ~0 against itself"
    assert gini_oracle > 0.15, "oracle should show real ranking ability"
    assert abs(gini_flat_null) < 0.05, "flat null should show ~no ranking ability"
    print("all sanity checks passed")
