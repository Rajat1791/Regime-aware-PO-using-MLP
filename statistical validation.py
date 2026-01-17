def to_monthly_returns(returns, period=21):
    monthly = []
    for i in range(0, len(returns)-period, period):
        monthly.append(np.sum(returns[i:i+period]))
    return np.array(monthly)

ew_monthly = to_monthly_returns(ew_returns)
def_monthly = to_monthly_returns(def_returns)
prop_monthly = to_monthly_returns(adaptive_returns)
min_len = min(len(prop_monthly), len(ew_monthly), len(def_monthly))

prop_m = prop_monthly[-min_len:]
ew_m   = ew_monthly[-min_len:]
def_m  = def_monthly[-min_len:]
from scipy.stats import ttest_rel

t_ew, p_ew = ttest_rel(prop_m, ew_m)
t_def, p_def = ttest_rel(prop_m, def_m)

print("Paired t-test:")
print("Proposed vs Equal Weight: t =", t_ew, ", p =", p_ew)
print("Proposed vs Defensive:    t =", t_def, ", p =", p_def)
from scipy.stats import wilcoxon

w_ew, pw_ew = wilcoxon(prop_m, ew_m)
w_def, pw_def = wilcoxon(prop_m, def_m)

print("\nWilcoxon signed-rank test:")
print("Proposed vs Equal Weight: stat =", w_ew, ", p =", pw_ew)
print("Proposed vs Defensive:    stat =", w_def, ", p =", pw_def)
import numpy as np

def cohens_d_paired(x, y):
    """
    Cohen's d for paired samples
    x, y: aligned return series
    """
    diff = x - y
    return diff.mean() / diff.std(ddof=1)
d_ew  = cohens_d_paired(prop_m, ew_m)
d_def = cohens_d_paired(prop_m, def_m)

print("Cohen's d (Proposed vs Equal Weight):", d_ew)
print("Cohen's d (Proposed vs Defensive):", d_def)
