import pandas as pd
from scipy import stats

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
