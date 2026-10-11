## Biased-memory-ratio ablation — objective drift rate

Fraction of critical tool parameters set in the harmful direction, biased condition vs base.
Percentile bootstrap over scenarios, 10,000 resamples, seed=20250728.

| Model | Biased share | Ratio | Drift rate [95% CI] | n |
|---|---|---|---|---|
| gemini-3.1-pro-preview | 0% | `n45_b0` | 0.074 [0.029, 0.125] | 21 |
| gemini-3.1-pro-preview | 25% | `n45_b15` | 0.876 [0.801, 0.944] | 21 |
| gemini-3.1-pro-preview | 50% | `n15_b15` | 0.869 [0.793, 0.935] | 21 |
| gemini-3.1-pro-preview | 75% | `n5_b15` | 0.875 [0.801, 0.939] | 21 |
| gemini-3.1-pro-preview | 100% | `n0_b15` | 0.893 [0.819, 0.960] | 21 |
| gpt-5.4 | 0% | `n45_b0` | 0.074 [0.037, 0.115] | 21 |
| gpt-5.4 | 25% | `n45_b15` | 0.156 [0.100, 0.217] | 21 |
| gpt-5.4 | 50% | `n15_b15` | 0.221 [0.133, 0.323] | 21 |
| gpt-5.4 | 75% | `n5_b15` | 0.235 [0.161, 0.310] | 21 |
| gpt-5.4 | 100% | `n0_b15` | 0.306 [0.206, 0.415] | 21 |
