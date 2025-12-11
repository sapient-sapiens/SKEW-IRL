# SKEW IRL Experiments

Maximum Entropy IRL pipeline around daily SKEW, VIX, SPX, and credit-spread data. 

- Because SKEW gets inflated overtime, we take its Z score with respect to its recent values. 

- We don't use a complicated deep nn because not enough data - Just linear model to find which factors are most important. 


## Latest Experiment

| Item | Value |
| --- | --- |
| Dataset size | 9,026 trading days |
| Expert feature expectations `μ_E` | `[-2.085, 2.568, 3.66e-3, 1.798, -2.466]` |
| Learned weights `θ` | `[1.555, -3.226, -0.779, -2.857, 3.104]` |
| Final gradient norm | `4.50` |
| Log-likelihood ratio vs. uniform policy | `-1.1031e5` |