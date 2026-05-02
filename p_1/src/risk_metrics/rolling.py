import pandas as pd

from src.risk_metrics.var_es import historical_var_es, normal_var_es

def compute_rolling_var_es(
    returns: pd.Series,
    confidence_levels: list[float],
    window: int = 252,
) -> pd.DataFrame:
    """
    Computes rolling-window VaR and ES.

    For each date t, we use the previous `window` returns to estimate
    the VaR and ES thresholds for the next observed return.

    This follows the project logic:

        use r_1, ..., r_252 to forecast r_253
        use r_2, ..., r_253 to forecast r_254
        etc.
    """
    clean_returns = returns.dropna()

    rows = []

    for i in range(window, len(clean_returns)):
        estimation_window = clean_returns.iloc[i - window:i]
        realized_return = clean_returns.iloc[i]
        date = clean_returns.index[i]

        row = {
            "Date": date,
            "Return": realized_return,
        }

        for alpha in confidence_levels:
            hist_var, hist_es = historical_var_es(
                estimation_window,
                alpha,
            )
            norm_var, norm_es = normal_var_es(
                estimation_window,
                alpha,
            )

            alpha_label = int(alpha * 100)

            row[f"Historical VaR {alpha_label}%"] = hist_var
            row[f"Historical ES {alpha_label}%"] = hist_es
            row[f"Normal VaR {alpha_label}%"] = norm_var
            row[f"Normal ES {alpha_label}%"] = norm_es

        rows.append(row)

    rolling_df = pd.DataFrame(rows)
    rolling_df = rolling_df.set_index("Date")

    return rolling_df
