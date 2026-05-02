import pandas as pd

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
