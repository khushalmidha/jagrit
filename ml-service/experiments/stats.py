import numpy as np
from scipy.stats import norm
import math

def two_proportion_z_test(clicks_A, matches_A, clicks_B, matches_B):
    """
    Performs a two-proportion z-test.
    """
    if (matches_A + matches_B) == 0:
        return {'ctr_A': 0, 'ctr_B': 0, 'lift': 0, 'z_stat': 0, 'p_value': 1.0, 'ci_lower': 0, 'ci_upper': 0}

    p_A = clicks_A / matches_A if matches_A > 0 else 0
    p_B = clicks_B / matches_B if matches_B > 0 else 0
    
    p_pool = (clicks_A + clicks_B) / (matches_A + matches_B)
    
    se = np.sqrt(p_pool * (1 - p_pool) * (1/matches_A + 1/matches_B))
    
    z_stat = (p_B - p_A) / se if se > 0 else 0
    
    # Two-tailed p-value
    p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    
    # 95% Confidence Interval for the difference (p_B - p_A)
    ci_se = np.sqrt(p_A*(1-p_A)/matches_A + p_B*(1-p_B)/matches_B)
    margin = 1.96 * ci_se
    diff = p_B - p_A
    ci_lower = diff - margin
    ci_upper = diff + margin
    
    return {
        'ctr_A': p_A,
        'ctr_B': p_B,
        'lift': (p_B - p_A) / p_A if p_A > 0 else 0,
        'z_stat': z_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def required_sample_size(baseline_ctr, mde, alpha=0.05, power=0.80):
    """
    Calculates the required sample size per variant for a given Minimum Detectable Effect (MDE).
    alpha: significance level (0.05)
    power: 1 - beta (0.80)
    """
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    p1 = baseline_ctr
    p2 = baseline_ctr * (1 + mde)
    
    p_pool = (p1 + p2) / 2
    
    n = ( (z_alpha * np.sqrt(2 * p_pool * (1 - p_pool)) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2 ) / ((p2 - p1)**2)
    
    return math.ceil(n)
