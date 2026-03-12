# insurance-autodml

Automatic Debiased ML via Riesz Representers for continuous treatment causal inference in UK personal lines insurance pricing.

## The problem

You want to know: "If I increase this policyholder's premium by £20, by how much does their claim probability change?" Or: "What happens to average claims if I raise all renewals 5%?"

These are causal questions. OLS on observed premiums and claims is biased because premiums are set by an underwriting model that already incorporates risk — high-risk policyholders are charged more, so premium and claims are positively correlated through confounding, not causal structure.

Standard Double ML handles this for discrete or well-behaved treatments. But in UK motor/home insurance, the treatment (premium) is continuous, and the standard approach requires estimating the generalised propensity score (GPS) — the conditional density p(D|X). This is numerically unstable when:

- Renewal rates vary from 80% at low premiums to 20% at high premiums (selection creates heavy tails)
- Premium distributions are multimodal (tiered pricing bands)
- High-premium policyholders are sparse but influential

The Riesz representer approach (Chernozhukov et al. 2022) bypasses the GPS entirely. It directly estimates the reweighting functional via a minimax regression, which is stable even at the extremes of the treatment distribution.

## What this library estimates

**Average Marginal Effect (AME)**: E[dE[Y|D,X]/dD] — the average derivative of the outcome with respect to premium. This is your price elasticity.

**Dose-response curve**: E[Y(d)] for a grid of premium values. Answers "what would average claims be if everyone paid £d?"

**Policy shift effect**: E[Y(D*(1+delta))] - E[Y]. Answers "what if we raised all premiums 5%?"

**Selection-corrected elasticity**: All of the above, but corrected for the renewal selection bias problem — claims are only observed for policies that renew.

## Installation

```bash
pip install insurance-autodml
```

For CatBoost nuisance models:

```bash
pip install "insurance-autodml[catboost]"
```

For HTML reports:

```bash
pip install "insurance-autodml[reports]"
```

## Quick start

```python
from insurance_autodml import PremiumElasticity, SyntheticContinuousDGP

# Generate synthetic data (or use your own)
dgp = SyntheticContinuousDGP(n=5000, outcome_family="gaussian", random_state=42)
X, D, Y, _ = dgp.generate()

# Fit the AME estimator
model = PremiumElasticity(
    outcome_family="gaussian",
    n_folds=5,
    random_state=0,
)
model.fit(X, D, Y)
result = model.estimate()

print(result.summary())
# estimate=-0.0021  se=0.0003  95% CI=[-0.0027, -0.0015]  p=0.0000***

# True AME for comparison
print(f"True AME: {dgp.true_ame_:.4f}")
```

## Price elasticity with exposure (motor claims)

```python
from insurance_autodml import PremiumElasticity

# D: annual premium (£), Y: claim count, exposure: years at risk
model = PremiumElasticity(
    outcome_family="poisson",
    n_folds=5,
)
model.fit(X, D, Y_claims, exposure=years_at_risk)
result = model.estimate()
# Interpretation: change in claim RATE per £1 premium increase
```

## Dose-response curve

```python
from insurance_autodml import DoseResponseCurve
import numpy as np

model = DoseResponseCurve(outcome_family="gaussian", n_folds=5)
model.fit(X, D, Y)

d_grid = np.linspace(200, 700, 50)
result = model.predict(d_grid)

# Plot
model.plot(d_grid=d_grid, xlabel="Annual Premium (£)", ylabel="Claim Rate")
```

## Policy shift

```python
from insurance_autodml import PolicyShiftEffect

model = PolicyShiftEffect(outcome_family="gaussian", n_folds=5)
model.fit(X, D, Y)

# What happens if all premiums increase 5%?
result = model.estimate(delta=0.05)
print(result.summary())

# Full curve of effects
effects = model.estimate_curve(np.linspace(-0.10, 0.10, 21))
```

## Handling renewal selection bias

```python
from insurance_autodml import SelectionCorrectedElasticity

# S: renewal indicator (1=renewed, 0=lapsed)
# Y: claims (observed only for renewals; set to 0 or NaN for lapses)
model = SelectionCorrectedElasticity(
    outcome_family="gaussian",
    n_folds=5,
)
model.fit(X, D, Y_observed, S=renewal_indicator)
result = model.estimate()

# Sensitivity analysis: how robust is this to unobserved selection confounding?
bounds = model.sensitivity_bounds(gamma_grid=np.array([1.0, 1.5, 2.0, 3.0]))
for gamma, b in bounds.items():
    print(f"Gamma={gamma}: AME in [{b['lower']:.4f}, {b['upper']:.4f}]")
```

## Segment-level effects

```python
# No refitting required — segments computed from EIF scores
age_bands = pd.cut(age_feature, bins=[17, 25, 35, 50, 65, 100], labels=["17-25", "26-35", "36-50", "51-65", "66+"])
segment_results = model.effect_by_segment(age_bands)

for sr in segment_results:
    print(f"{sr.segment_name}: {sr.result.summary()}")
```

## FCA evidence report

```python
from insurance_autodml import ElasticityReport

report = ElasticityReport(
    estimator=model,
    segment_results=segment_results,
    sensitivity_bounds=bounds,
    analyst="Pricing Team",
)
report.to_html("elasticity_report.html")
report.to_json("elasticity_report.json")
```

## Design choices

**Why not GPS-based double ML?** The GPS (p(D|X)) requires density estimation in high dimensions. In renewal portfolios, the treatment density has long tails and selection-induced gaps. The Riesz minimax regression is a regression problem — more stable, standard ML machinery applies directly.

**Why ForestRiesz over genriesz?** We implement our own forest-based Riesz regressor rather than depending on genriesz (which requires JAX). The scikit-learn RandomForest is sufficient for the derivative estimation task and avoids GPU/JAX dependency issues in production insurance environments.

**Why 5-fold cross-fitting?** Standard in the DML literature. 3 folds for n < 2000; 5 folds is the default sweet spot. More folds give smaller bias but higher variance in the nuisance estimates.

**Outcome families**: The library uses GradientBoostingRegressor for all families by default (transforming Y for Poisson/Gamma to ensure positivity). CatBoost's native Poisson loss is available via the `catboost` extra and gives better calibration for claim count models.

## References

- Chernozhukov et al. (2022). Automatic Debiased Machine Learning of Causal and Structural Effects. *Econometrica* 90(3):967-1027.
- Colangelo & Lee (2020). Double Debiased Machine Learning Nonparametric Inference with Continuous Treatments. arXiv:2004.03036.
- Hirshberg & Wager (2021). Augmented minimax linear estimation. *Annals of Statistics* 49(6):3206-3227.
- arXiv:2601.08643. Automatic debiased machine learning and sensitivity analysis for sample selection models.

## Related libraries

- [insurance-causal](https://github.com/burning-cost/insurance-causal) — binary treatment effects via DoubleML
- [insurance-elasticity](https://github.com/burning-cost/insurance-elasticity) — GLM-based elasticity without causal identification

---

Built by [Burning Cost](https://github.com/burning-cost) — insurance pricing tools for practitioners.
