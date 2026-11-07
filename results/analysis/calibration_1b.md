# Stage 1b step 1: calibration arms (out-of-fold, calibration+validation only)

2200 claims, 1649 documents, 5-fold document-grouped CV, 1000-iteration cluster bootstrap. PRIMARY: Brier. SECONDARY: macro per-dataset ECE.

| arm | type | Brier [95% CI] | ECE [95% CI] | AUROC | macro-ECE | datasets with ECE < raw | paired dBrier vs global isotonic (p) |
|---|---|---|---|---|---|---|---|
| CONTROL: dataset base rate only (ignores Jev) | control | 0.179 [0.169, 0.189] | 0.035 [0.026, 0.056] | 0.775 | 0.042 | 10/11 | -0.0329 (1.000) |
| raw (uncalibrated) | - | 0.158 [0.147, 0.169] | 0.083 [0.073, 0.104] | 0.865 | 0.129 | 0/11 | -0.0116 (1.000) |
| global isotonic  [current] | agnostic | 0.146 [0.137, 0.155] | 0.025 [0.017, 0.043] | 0.857 | 0.119 | 6/11 | - |
| global temperature | agnostic | 0.153 [0.144, 0.162] | 0.083 [0.067, 0.100] | 0.865 | 0.139 | 4/11 | -0.0067 (0.999) |
| logistic on logit(p)+claim/doc length+BM25 | agnostic | 0.144 [0.134, 0.153] | 0.017 [0.017, 0.039] | 0.865 | 0.113 | 7/11 | +0.0023 (0.012) |
| logistic on logit(p)+dataset intercept | domain-aware | 0.128 [0.119, 0.137] | 0.016 [0.015, 0.038] | 0.889 | 0.075 | 9/11 | +0.0178 (0.001) |
| per-dataset isotonic | domain-aware | 0.131 [0.122, 0.141] | 0.031 [0.027, 0.052] | 0.883 | 0.073 | 10/11 | +0.0153 (0.001) |

dBrier = (Brier of global isotonic) - (Brier of the arm): positive means the arm is better; p is one-sided, UNADJUSTED for the 5 comparisons against global isotonic (Bonferroni: multiply p by 5).
