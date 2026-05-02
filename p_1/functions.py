import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def download_prices(ticker: str, start_date: str, end_date: str) -> pd.Series:
    """
    Downloads adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker, for example:
        - AAPL
        - MSFT
        - ^GSPC
        - BTC-USD
        - GC=F

    start_date:
        Initial date in YYYY-MM-DD format.

    end_date:
        Final date in YYYY-MM-DD format.

    Returns
    -------
    prices:
        A pandas Series with daily closing prices.
    """
    data = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        raise ValueError(
            "No data was downloaded. Check the ticker symbol or the selected dates."
        )

    # yfinance usually returns a simple DataFrame for one ticker.
    # However, depending on the version, it may return MultiIndex columns.
    if isinstance(data.columns, pd.MultiIndex):
        if ("Close", ticker) in data.columns:
            prices = data[("Close", ticker)]
        elif ("Adj Close", ticker) in data.columns:
            prices = data[("Adj Close", ticker)]
        else:
            # Fallback: try to get the first close-like column
            close_columns = [
                col for col in data.columns
                if "Close" in str(col)
            ]
            if not close_columns:
                raise ValueError("Could not identify a closing price column.")
            prices = data[close_columns[0]]
    else:
        if "Close" in data.columns:
            prices = data["Close"]
        elif "Adj Close" in data.columns:
            prices = data["Adj Close"]
        else:
            raise ValueError("Could not identify a closing price column.")

    prices = prices.dropna()
    prices.name = ticker

    return prices


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Computes daily logarithmic returns:

        r_t = log(S_t) - log(S_{t-1})

    Parameters
    ----------
    prices:
        Price series.

    Returns
    -------
    returns:
        Daily log returns.
    """
    returns = np.log(prices).diff().dropna()
    returns.name = "Log returns"

    return returns


def compute_return_statistics(returns: pd.Series) -> pd.DataFrame:
    """
    Computes mean, skewness, and excess kurtosis.

    The project asks for:
    - mean
    - skewness
    - excess kurtosis
    """
    clean_returns = returns.dropna()

    statistics_dict = {
        "Media": clean_returns.mean(),
        "Desviación estándar": clean_returns.std(ddof=1),
        "Sesgo": stats.skew(clean_returns, bias=False),
        "Exceso de curtosis": stats.kurtosis(
            clean_returns,
            fisher=True,
            bias=False,
        ),
        "Número de observaciones": len(clean_returns),
    }

    return pd.DataFrame.from_dict(
        statistics_dict,
        orient="index",
        columns=["Valor"],
    )


def historical_var_es(
    returns: pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Historical VaR and ES using return thresholds.

    Since violations are defined as:

        realized_return < VaR_threshold

    we work directly with lower-tail return quantiles.

    For confidence level alpha:
        tail probability = 1 - alpha

    Example:
        alpha = 0.95 -> lower 5% quantile.
    """
    clean_returns = returns.dropna()
    tail_probability = 1.0 - confidence_level

    var_threshold = clean_returns.quantile(tail_probability)
    es_threshold = clean_returns[clean_returns <= var_threshold].mean()

    return float(var_threshold), float(es_threshold)


def normal_var_es(
    returns: pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Parametric normal VaR and ES using return thresholds.

    If R ~ N(mu, sigma^2), then the lower-tail VaR threshold is:

        VaR_alpha = mu + sigma * Phi^{-1}(1 - alpha)

    The lower-tail ES threshold is:

        ES_alpha = mu - sigma * phi(z) / p

    where:
        p = 1 - alpha
        z = Phi^{-1}(p)
    """
    clean_returns = returns.dropna()
    mu = clean_returns.mean()
    sigma = clean_returns.std(ddof=1)

    tail_probability = 1.0 - confidence_level
    z = stats.norm.ppf(tail_probability)

    var_threshold = mu + sigma * z
    es_threshold = mu - sigma * stats.norm.pdf(z) / tail_probability

    return float(var_threshold), float(es_threshold)


def student_t_var_es(
    returns: pd.Series,
    confidence_level: float,
) -> tuple[float, float]:
    """
    Parametric Student-t VaR and ES using return thresholds.

    We fit a Student-t distribution to the observed returns:

        R ~ t(df, loc, scale)

    Then we compute the lower-tail VaR and ES.
    """
    clean_returns = returns.dropna()
    tail_probability = 1.0 - confidence_level

    df, loc, scale = stats.t.fit(clean_returns)

    if df <= 1:
        return np.nan, np.nan

    q = stats.t.ppf(tail_probability, df)

    var_threshold = loc + scale * q

    # Lower-tail conditional expectation of a standardized t variable.
    standardized_es = -(
        (df + q**2) / (df - 1)
    ) * stats.t.pdf(q, df) / tail_probability

    es_threshold = loc + scale * standardized_es

    return float(var_threshold), float(es_threshold)


def monte_carlo_var_es(
    returns: pd.Series,
    confidence_level: float,
    n_simulations: int = 100_000,
    random_seed: int = 123,
) -> tuple[float, float]:
    """
    Monte Carlo VaR and ES.

    We fit a normal distribution to the historical returns and simulate
    artificial one-day returns. Then we estimate VaR and ES from the
    simulated lower tail.
    """
    clean_returns = returns.dropna()
    mu = clean_returns.mean()
    sigma = clean_returns.std(ddof=1)

    rng = np.random.default_rng(random_seed)

    simulated_returns = rng.normal(
        loc=mu,
        scale=sigma,
        size=n_simulations,
    )

    tail_probability = 1.0 - confidence_level

    var_threshold = np.quantile(simulated_returns, tail_probability)
    es_threshold = simulated_returns[
        simulated_returns <= var_threshold
    ].mean()

    return float(var_threshold), float(es_threshold)


def compute_full_sample_var_es(
    returns: pd.Series,
    confidence_levels: list[float],
    n_simulations: int = 100_000,
) -> pd.DataFrame:
    """
    Computes full-sample VaR and ES using the four methods requested
    in the project:

    1. Parametric normal
    2. Parametric Student-t
    3. Historical
    4. Monte Carlo
    """
    rows = []

    methods = {
        "Normal paramétrico": normal_var_es,
        "Student-t paramétrico": student_t_var_es,
        "Histórico": historical_var_es,
        "Monte Carlo": lambda x, alpha: monte_carlo_var_es(
            x,
            alpha,
            n_simulations=n_simulations,
        ),
    }

    for method_name, method_function in methods.items():
        for alpha in confidence_levels:
            var_threshold, es_threshold = method_function(returns, alpha)

            rows.append(
                {
                    "Método": method_name,
                    "Nivel de confianza": alpha,
                    "VaR": var_threshold,
                    "ES": es_threshold,
                }
            )

    return pd.DataFrame(rows)


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


def compute_violations(
    rolling_results: pd.DataFrame,
    confidence_levels: list[float],
    methods: list[str],
    measures: list[str],
) -> pd.DataFrame:
    """
    Counts violations.

    A violation occurs when:

        realized_return < risk_threshold

    where risk_threshold is VaR or ES estimated using the previous
    rolling window.
    """
    rows = []
    total_observations = len(rolling_results)

    for method in methods:
        for measure in measures:
            for alpha in confidence_levels:
                alpha_label = int(alpha * 100)
                column_name = f"{method} {measure} {alpha_label}%"

                if column_name not in rolling_results.columns:
                    continue

                violation_series = (
                    rolling_results["Return"] < rolling_results[column_name]
                )

                number_violations = int(violation_series.sum())
                percentage_violations = (
                    number_violations / total_observations
                ) * 100

                rows.append(
                    {
                        "Método": method,
                        "Medida": measure,
                        "Nivel de confianza": alpha,
                        "Número de violaciones": number_violations,
                        "Porcentaje de violaciones": percentage_violations,
                    }
                )

    return pd.DataFrame(rows)


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


def compute_moving_volatility_violations(
    moving_vol_results: pd.DataFrame,
    significance_levels: list[float],
) -> pd.DataFrame:
    """
    Counts violations for moving-volatility VaR.
    """
    rows = []
    total_observations = len(moving_vol_results)

    for significance_level in significance_levels:
        confidence_level = 1.0 - significance_level
        confidence_label = int(confidence_level * 100)

        column_name = f"Moving volatility VaR {confidence_label}%"

        violation_series = (
            moving_vol_results["Return"] < moving_vol_results[column_name]
        )

        number_violations = int(violation_series.sum())
        percentage_violations = (
            number_violations / total_observations
        ) * 100

        rows.append(
            {
                "Método": "Volatilidad móvil normal",
                "Medida": "VaR",
                "Nivel de confianza": confidence_level,
                "Número de violaciones": number_violations,
                "Porcentaje de violaciones": percentage_violations,
            }
        )

    return pd.DataFrame(rows)