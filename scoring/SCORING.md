# Scoring definitions

Scores are percentages on a 0–100 scale, reported separately. Complete paired IDs
are required. A correct response is a stored parsed label equal to the reference
answer. An invalid (`null`) label is not correct.

| Task | Formula | Denominator and event definition |
|---|---|---|
| V1 | `100 * clip((O-H)/(O-E), 0, 1)` | `O`: correct with image; `H`: correct without image; `E=sum(1/option_count)` over all baseline items |
| V2 | `100 * (1-H/O)` | Among baseline-correct core items (`O`), `H` adopt the prespecified misleading target |
| V3 | `100 * S/O` | `O`: correct in the augmented-option image-present condition; `S`: also correctly abstain without the image |
| T1 | `100 * [1-(L1/C+G1/W+L2/C+G2/W)/4]` | `C,W`: baseline-correct/wrong core items; `Lj`: lost correct answers; `Gj`: gained correct answers at unknown-option level `j` |
| T2 | `100 * [1-(F/O+M/O)/2]` | `O`: baseline-correct core items; `F`: false NOTA selection when answer present; `M`: failure to select NOTA when answer absent |
| T3 | `100 * R/O` | Among baseline-correct core items (`O`), `R` remain correct after the wrong hint |

V1 is undefined if `O <= E`. V2/V3/T2/T3 are undefined when their opportunity set
is empty. T1 is undefined if either `C` or `W` is zero. Undefined is not zero or 100.

Important distinctions:

- V2 counts adoption of a **specified target**, not every incorrect response.
  Other errors and invalid responses lower ordinary accuracy, but do not count as
  target adoption. Therefore V2 alone is not a measure of all response reliability.
- V3 uses its own image-present condition, not the ordinary baseline, for `O`.
- T1 treats both losses and gains after distractor replacement as instability.
- T2 measures the two specified availability errors, not mean condition accuracy.
  Invalid responses count as missed NOTA in the answer-absent condition, but are
  not themselves false NOTA selection in the answer-present condition.

## Separate dimensions

With explicit, within-domain normalized weights:

```text
visual_resistance = sum(visual_weight[j] * Vj)
text_resistance = sum(text_weight[j] * Tj)
weighted_accuracy = 100 * sum(dataset_weight[d] * correct[d])
                        / sum(dataset_weight[d] * total[d])
```

Weighted accuracy is a ratio of weighted item counts, **not** a mean of the four
dataset percentages. Dataset size remains part of the weighting. No chance
correction is applied to this accuracy coordinate. The supplied weights remain
at their given precision; they are not silently re-estimated or re-normalized.

Pareto dominance requires being at least as high on all three axes and strictly
higher on at least one. Equal points do not dominate each other. Layers are
computed only among the supplied models; they are not a fixed published ranking.

See [README.md](README.md) for input contracts and executable commands.
