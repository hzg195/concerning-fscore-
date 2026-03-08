import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm   # 新增：用于 Newey-West 调整


fs = pd.read_csv(r'C:\Users\韩治赓\Desktop\fscore.csv',     index_col=0, parse_dates=True)   # F-scores
rv = pd.read_csv(r'C:\Users\韩治赓\Desktop\rev.csv',        index_col=0, parse_dates=True)   # 1-month returns



common_stocks = fs.columns.intersection(rv.columns)
fs = fs[common_stocks]
rv = rv[common_stocks]

print(f"Universe : {len(common_stocks):,} stocks  |  {len(rv)} month-ends "
      f"({rv.index[0].date()} -> {rv.index[-1].date()})")


fs_filled = fs.ffill().shift(1)           
print(fs_filled)
def fscore_group(x):
    """Map raw F-score to Low / Mid / High category."""
    if pd.isna(x):  return np.nan
    if x <= 3:      return 'Low'         # 0-3
    if x <= 6:      return 'Mid'         # 4-6
    return 'High'                        # 7-9

fs_cat = fs_filled.apply(lambda col: col.map(fscore_group))


dates = rv.index
records = []

for i in range(1, len(dates)):
    holding_date   = dates[i]       
    formation_date = dates[i - 1]   

    ret_form = rv.loc[formation_date]           
    ret_hold = rv.loc[holding_date]             
    fsc      = fs_cat.loc[holding_date]         


    valid    = ret_form.notna() & ret_hold.notna() & fsc.notna()
    ret_form = ret_form[valid]
    ret_hold = ret_hold[valid]
    fsc      = fsc[valid]

    if len(ret_form) < 25:        
        continue


    quintile = pd.qcut(ret_form, 5,
                       labels=['Loser', 'Q2', 'Q3', 'Q4', 'Winner'])

    row = {'date': holding_date}
    for ret_grp in ['Loser', 'Q2', 'Q3', 'Q4', 'Winner']:
        for fs_grp in ['Low', 'Mid', 'High']:
            mask = (quintile == ret_grp) & (fsc == fs_grp)
            key  = f"{ret_grp}_{fs_grp}"
            row[key]       = ret_hold[mask].mean() if mask.sum() > 0 else np.nan
            row[f"n_{key}"] = int(mask.sum())

 
    long_ret  = row.get('Loser_High', np.nan)
    short_ret = row.get('Winner_Low', np.nan)

    if pd.notna(long_ret) and pd.notna(short_ret):
        row['far']     = long_ret - short_ret
        row['long']    = long_ret
        row['short']   = short_ret

    records.append(row)

df = pd.DataFrame(records).set_index('date')


far_series = df['far'].dropna()
mean_far   = far_series.mean()


t_stat, p_val = stats.ttest_1samp(far_series, popmean=0)

n = len(far_series)
X = sm.add_constant(np.ones(n))               
model = sm.OLS(far_series, X)
results = model.fit(cov_type='HAC', cov_kwds={'maxlags': None})  
t_newey = results.tvalues[0]
p_newey = results.pvalues[0]

print("\n" + "=" * 55)
print("  RESULTS — Fundamental-Anchored Reversal (FAR)")
print("=" * 55)
print(f"  Number of months              : {len(far_series)}")
print(f"  Avg monthly FAR  return       : {mean_far*100:.4f}%")
print(f"  Std dev of FAR returns        : {far_series.std()*100:.4f}%")
print(f"  Simple t-statistic            : {t_stat:.3f}  (p={p_val:.3f})")
print(f"  Newey-West t-statistic        : {t_newey:.3f}  (p={p_newey:.3f})")
print("=" * 55)
print(f"\n  Avg stocks/month — long  leg  : {df['n_Loser_High'].mean():.1f}")
print(f"  Avg stocks/month — short leg  : {df['n_Winner_Low'].mean():.1f}")
