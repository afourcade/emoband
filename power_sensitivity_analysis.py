"""
Power sensitivity analysis for the pooled dataset of the two studies. 
This analysis estimates the minimum detectable effect size (Cohen's f^2) for a single predictor in a linear multiple regression model, given the sample size and number of predictors. 
The results can be used to interpret the findings of the regression analyses in the context of statistical power and to understand the smallest effect size that the study is capable of detecting with a given level of confidence.

Author: Aleksandra Piejka, Antonin Fourcade (piejka[at]cbs.mpg.de, antonin.fourcade[at]maxplanckschools.de)
Last version: 15.06.2026
"""

#%% IMPORTS AND SETTINGS
from scipy.stats import ncf, f as fdist
from scipy.optimize import brentq
import math

N = 176 # sample size of pooled dataset from the two studies

#%% FUNCTIONS
def min_f2(u, k, power=0.80, alpha=0.05):
    """
    Minimum detectable Cohen's f^2 for a linear multiple regression.
    u     = numerator df (number of predictors tested; 1 for a single predictor)
    k     = total number of predictors in the model
    power = desired statistical power (1 - beta)
    alpha = significance level
    G*Power convention: noncentrality lambda = f^2 * (u + v + 1) = f^2 * N
    """
    v = N - k - 1                       # denominator (residual) df
    Fcrit = fdist.ppf(1 - alpha, u, v)  # critical F
    def power_at(f2):
        lam = f2 * N
        return 1 - ncf.cdf(Fcrit, u, v, lam)
    return brentq(lambda f2: power_at(f2) - power, 1e-6, 5)

#%% ANALYSIS
# Single-predictor test (u=1) in the depression model (k=7) and loneliness model (k=8)
for power in (0.80, 0.90, 0.95):
    for k in (7, 8):
        f2 = min_f2(u=1, k=k, power=power)
        partial_r = math.sqrt(f2 / (1 + f2))
        print(f"power={power:.2f} | {k} predictors | single-predictor "
              f"min f2={f2:.3f} | partial r≈{partial_r:.2f}")