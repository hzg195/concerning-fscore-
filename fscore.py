import pandas as pd
import numpy as np
from scipy import stats
fscore = pd.read_csv(r'C:\Users\韩治赓\Desktop\fscore.csv', index_col=0, parse_dates=True)
returns = pd.read_csv(r'C:\Users\韩治赓\Desktop\stock_returns.csv', index_col=0, parse_dates=True)
common_stocks = fscore.columns.intersection(returns.columns)#选择两者都有的股票，避免后续对齐时出现问题
print(common_stocks)
fscore = fscore[common_stocks]#直接修改fscore和returns，保留共同的股票列
print(fscore)
returns = returns[common_stocks]
fscore_shifted = fscore.shift(1)#将F‑Score数据向下移动一个月，以确保使用的是上个月末的F‑Score来预测下个月的收益率(ai帮助下注意到这一点)


common_dates = fscore_shifted.index.intersection(returns.index)#找到两者都有的日期，确保后续计算时数据对齐
print(common_dates)
print(fscore_shifted.loc[common_dates])
fscore_aligned = fscore_shifted.loc[common_dates]#只选出共同日期的数据，确保后续计算时数据对齐
returns_aligned = returns.loc[common_dates]
print(fscore_aligned)
print(returns_aligned)
factor_returns = []

for date in common_dates:
    scores = fscore_aligned.loc[date]#获取当月的F‑Score数据
    rets = returns_aligned.loc[date]#获取当月的收益率数据
    long_stocks = scores[scores >= 7].index#选择F‑Score大于等于7的股票作为多头组合
    short_stocks = scores[scores <= 3].index#选择F‑Score小于等于3的股票作为空头组合

    if len(long_stocks) > 0 and len(short_stocks) > 0:#确保多头和空头组合都有股票，避免计算时出现除以零的情况
        long_ret = rets[long_stocks].mean()#取平均值
        short_ret = rets[short_stocks].mean()
        factor_ret = long_ret - short_ret
        factor_returns.append(factor_ret)
    else:
        pass
factor_returns = pd.Series(factor_returns, index=common_dates[:len(factor_returns)])
mean_ret = factor_returns.mean()
std_ret = factor_returns.std()
n = len(factor_returns)

t_stat = mean_ret / (std_ret / np.sqrt(n))#t值检验
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))#p值检验

print(f"Number of months: {n}")#这里只有291个月是因为一个月的fscore对应下个月的return
print(f"Average factor return: {mean_ret:.3f}")
print(f"Standard deviation: {std_ret:.3f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")
print(f"Factor returns:\n{factor_returns}")