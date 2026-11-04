# Scoring definitions

[← Back to the benchmark overview](../README.md#scoring)

All six task scores range from 0 to 100, and **higher is better**.

| Score | Definition |
|---|---|
| V1 | `100 × [1 − clip((H1 − E1) / (O1 − E1), 0, 1)]` |
| V2 | `100 × (1 − misleading-target adoption / original-correct)` |
| V3 | `100 × correct image/no-image switches / image-correct` |
| T1 | `100 × [1 − (L1 + G1 + L2 + G2) / 4]` |
| T2 | `100 × [1 − (false-NOTA + missed-NOTA) / 2]` |
| T3 | `100 × retained-correct after wrong hint / original-correct` |

For V1, `O1` is the number of correct responses with the image, `H1` is the number of correct responses without the image, and `E1 = Σ_i(1/k_i)` is the item-level random expectation over the 1,200 paired questions (`k_i` is the number of answer options for item `i`). This chance adjustment reduces the influence of mixed option counts and initial accuracy. V1 is algebraically identical to `100 × clip((O1 − H1) / (O1 − E1), 0, 1)` whenever `O1 > E1`.

The task weights come from five-dimensional ratings by three clinicians and are normalized within each route:

```text
Visual resistance = 0.32974684×V1 + 0.31455696×V2 + 0.35569620×V3
Text resistance   = 0.29059829×T1 + 0.36231884×T2 + 0.34708287×T3
```

For source dataset `d`, let `C_d` be the number of correct responses, `N_d` the number of evaluated items, and `w_d` its clinician-derived weight. The diagnostic-performance coordinate is a weighted micro-average:

```text
DatasetWeightedAccuracy = 100 × Σ_d(w_d × C_d) / Σ_d(w_d × N_d)
```

This formulation weights every evaluated item by the weight of its source dataset. Dataset size and clinician weight therefore jointly determine the contribution to the final coordinate. No chance correction or min–max transformation is applied.


## Implementation

Use [score.py](score.py) for public-layout predictions, [score_stress_run.py](score_stress_run.py) for manuscript-layout runs, and [score_axes.py](score_axes.py) for the three separate dimensions. These documentation changes do not modify the scoring implementation.
