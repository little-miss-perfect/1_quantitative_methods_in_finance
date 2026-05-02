import pandas as pd
from scipy import stats

def compute_moving_volatility_var(
    returns: pd.Series,
    significance_levels: list[float],
    window: int = 252,
) -> pd.DataFrame:
    """
    Computes moving-volatility VaR using the project formula:

        VaR_{1-alpha} = q_alpha * sigma_t^{252}

    where:
        q_alpha is the normal alpha-quantile,
        alpha is 0.05 or 0.01,
        sigma_t^{252} is the rolling standard deviation of the previous
        252 returns.

    Since q_alpha is negative for alpha = 0.05 and 0.01, the result is
    a negative return threshold.
    """
    clean_returns = returns.dropna()

    rows = []

    for i in range(window, len(clean_returns)):
        estimation_window = clean_returns.iloc[i - window:i]
        realized_return = clean_returns.iloc[i]
        date = clean_returns.index[i]

        rolling_sigma = estimation_window.std(ddof=1)

        row = {
            "Date": date,
            "Return": realized_return,
            "Rolling volatility": rolling_sigma,
        }

        for significance_level in significance_levels:
            confidence_level = 1.0 - significance_level
            confidence_label = int(confidence_level * 100)

            q_alpha = stats.norm.ppf(significance_level)
            var_threshold = q_alpha * rolling_sigma

            row[f"Moving volatility VaR {confidence_label}%"] = var_threshold

        rows.append(row)

    moving_vol_df = pd.DataFrame(rows)
    moving_vol_df = moving_vol_df.set_index("Date")

    return moving_vol_df
