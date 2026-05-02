import numpy as np
import pandas as pd

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
