import numpy as np
import pandas as pd

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
