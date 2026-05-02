import numpy as np
import pandas as pd
from scipy import stats

from src.simulations.monte_carlo import monte_carlo_var_es

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
