import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_excel(r'path', header=None)

station_names = df.iloc[0, :].tolist()

results = []

for i, station in enumerate(station_names):
    period1 = df.iloc[1:12, i].dropna().astype(float).values
    period2 = df.iloc[13:26, i].dropna().astype(float).values

    if len(period1) > 1 and len(period2) > 1:

        U1, mann_p_value = stats.mannwhitneyu(period1, period2, alternative='two-sided')
        U2 = len(period1) * len(period2) - U1  # 另一个 U 值
        U = max(U1, U2)


        n1, n2 = len(period1), len(period2)
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        Z = (U - mu_U) / sigma_U if sigma_U != 0 else np.nan


        corrected_mann_p_value = 2 * (1 - stats.norm.cdf(abs(Z)))
    else:
        Z, corrected_mann_p_value = np.nan, np.nan


    results.append([station, Z, corrected_mann_p_value])


results_df = pd.DataFrame(results, columns=['水文站', 'Z值', 'Mann-Whitney p值'])


output_path = r'outpath'
results_df.to_excel(output_path, index=False, header=True)


