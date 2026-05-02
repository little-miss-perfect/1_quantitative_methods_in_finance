import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from functions import (
    download_prices,
    compute_log_returns,
    compute_return_statistics,
    compute_full_sample_var_es,
    compute_rolling_var_es,
    compute_violations,
    compute_moving_volatility_var,
    compute_moving_volatility_violations,
)

st.set_page_config(
    page_title="Proyecto I - Métodos Cuantitativos en Finanzas",
    layout="wide",
)


@st.cache_data
def cached_download_prices(ticker: str, start_date: str, end_date: str):
    return download_prices(ticker, start_date, end_date)


def make_price_plot(prices: pd.Series, ticker: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name=f"Precio de cierre - {ticker}",
        )
    )

    fig.update_layout(
        title=f"Serie de precios: {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Precio",
        hovermode="x unified",
    )

    return fig


def make_returns_plot(returns: pd.Series, ticker: str):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=returns.index,
            y=returns.values,
            mode="lines",
            name="Rendimientos logarítmicos",
        )
    )

    fig.update_layout(
        title=f"Rendimientos logarítmicos diarios: {ticker}",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento logarítmico",
        hovermode="x unified",
    )

    return fig


def make_rolling_risk_plot(
        rolling_results: pd.DataFrame,
        method: str,
        title: str,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=rolling_results.index,
            y=rolling_results["Return"],
            mode="lines",
            name="Rendimiento observado",
            line=dict(width=1),
        )
    )

    columns_to_plot = [
        f"{method} VaR 95%",
        f"{method} ES 95%",
        f"{method} VaR 99%",
        f"{method} ES 99%",
    ]

    for column in columns_to_plot:
        fig.add_trace(
            go.Scatter(
                x=rolling_results.index,
                y=rolling_results[column],
                mode="lines",
                name=column,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Fecha",
        yaxis_title="Rendimiento / umbral de riesgo",
        hovermode="x unified",
    )

    return fig


def make_moving_volatility_plot(moving_vol_results: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=moving_vol_results.index,
            y=moving_vol_results["Return"],
            mode="lines",
            name="Rendimiento observado",
            line=dict(width=1),
        )
    )

    for column in [
        "Moving volatility VaR 95%",
        "Moving volatility VaR 99%",
    ]:
        fig.add_trace(
            go.Scatter(
                x=moving_vol_results.index,
                y=moving_vol_results[column],
                mode="lines",
                name=column,
            )
        )

    fig.update_layout(
        title="VaR con volatilidad móvil",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento / VaR",
        hovermode="x unified",
    )

    return fig


st.title("Proyecto I - Métodos Cuantitativos en Finanzas")
st.subheader("VaR y ES para un activo financiero")

st.write(
    """
    Esta aplicación descarga información financiera desde Yahoo Finance,
    calcula rendimientos logarítmicos diarios, estima VaR y ES mediante
    diferentes métodos, realiza estimaciones con rolling window y cuenta
    violaciones.
    """
)

st.sidebar.header("Parámetros del análisis")

ticker = st.sidebar.text_input(
    "Ticker de Yahoo Finance",
    value="AAPL",
    help="Ejemplos: AAPL, MSFT, ^GSPC, BTC-USD, GC=F",
)

start_date = st.sidebar.date_input(
    "Fecha inicial",
    value=pd.to_datetime("2010-01-01"),
)

end_date = st.sidebar.date_input(
    "Fecha final",
    value=pd.Timestamp.today(),
)

n_simulations = st.sidebar.number_input(
    "Número de simulaciones Monte Carlo",
    min_value=10_000,
    max_value=1_000_000,
    value=100_000,
    step=10_000,
)

confidence_levels = [0.95, 0.975, 0.99]
rolling_confidence_levels = [0.95, 0.99]
significance_levels = [0.05, 0.01]
window = 252

st.sidebar.markdown("---")
run_analysis = st.sidebar.button("Ejecutar análisis")

if run_analysis:
    try:
        prices = cached_download_prices(
            ticker=ticker,
            start_date=str(start_date),
            end_date=str(end_date),
        )

        returns = compute_log_returns(prices)

        if len(returns) <= window:
            st.error(
                "No hay suficientes observaciones para una rolling window "
                "de 252 rendimientos. Selecciona un rango de fechas más amplio."
            )
            st.stop()

        st.success("Datos descargados correctamente desde Yahoo Finance.")

        # ------------------------------------------------------------
        # (a) Data download and asset description
        # ------------------------------------------------------------
        st.header("(a) Descarga de datos financieros")

        st.write(
            f"""
            El activo seleccionado es **{ticker}**. La información se descarga
            automáticamente desde Yahoo Finance para el periodo:
            **{start_date}** a **{end_date}**.
            """
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Primer precio disponible", f"{prices.iloc[0]:.4f}")
        col2.metric("Último precio disponible", f"{prices.iloc[-1]:.4f}")
        col3.metric("Número de precios", f"{len(prices)}")

        st.plotly_chart(
            make_price_plot(prices, ticker),
            width="stretch",
        )

        with st.expander("Ver datos de precios"):
            st.dataframe(prices.to_frame(name="Precio de cierre"))

        # ------------------------------------------------------------
        # (b) Daily returns and descriptive statistics
        # ------------------------------------------------------------
        st.header("(b) Rendimientos diarios y estadísticos descriptivos")

        st.write(
            """
            Los rendimientos diarios se calculan como rendimientos logarítmicos:

            r_t = log(S_t) - log(S_{t-1}).
            """
        )

        st.plotly_chart(
            make_returns_plot(returns, ticker),
            width="stretch",
        )

        statistics_table = compute_return_statistics(returns)

        st.subheader("Media, sesgo y exceso de curtosis")

        st.dataframe(
            statistics_table.round(8),
            width="stretch",
        )

        with st.expander("Ver rendimientos diarios"):
            st.dataframe(returns.to_frame(name="Rendimiento logarítmico"))

        # ------------------------------------------------------------
        # (c) Full-sample VaR and ES
        # ------------------------------------------------------------
        st.header("(c) VaR y ES para la serie completa")

        st.write(
            """
            En esta sección se calcula VaR y ES para la serie completa de
            rendimientos usando cuatro métodos:

            1. Normal paramétrico.
            2. Student-t paramétrico.
            3. Histórico.
            4. Monte Carlo.

            Los valores se reportan como umbrales de rendimiento. Por eso,
            usualmente aparecen como números negativos: una violación ocurre
            cuando el rendimiento observado cae por debajo del umbral.
            """
        )

        full_sample_table = compute_full_sample_var_es(
            returns=returns,
            confidence_levels=confidence_levels,
            n_simulations=int(n_simulations),
        )

        display_full_sample_table = full_sample_table.copy()
        display_full_sample_table["Nivel de confianza"] = display_full_sample_table[
            "Nivel de confianza"
        ].round(3)
        display_full_sample_table["VaR"] = display_full_sample_table["VaR"].round(8)
        display_full_sample_table["ES"] = display_full_sample_table["ES"].round(8)

        st.dataframe(
            display_full_sample_table,
            width="stretch",
            hide_index=True,
        )

        # ------------------------------------------------------------
        # (d) Rolling-window VaR and ES
        # ------------------------------------------------------------
        st.header("(d) Rolling window VaR y ES")

        st.write(
            """
            Se usa una rolling window de 252 rendimientos. Es decir, los primeros
            252 rendimientos se usan para estimar el VaR y ES del siguiente día.
            Luego la ventana se mueve un día hacia adelante y se repite el
            procedimiento.
            """
        )

        rolling_results = compute_rolling_var_es(
            returns=returns,
            confidence_levels=rolling_confidence_levels,
            window=window,
        )

        tab_hist, tab_norm = st.tabs(
            ["Método histórico", "Método normal paramétrico"]
        )

        with tab_hist:
            st.plotly_chart(
                make_rolling_risk_plot(
                    rolling_results=rolling_results,
                    method="Historical",
                    title="Rolling window: VaR y ES históricos",
                ),
                width="stretch",
            )

        with tab_norm:
            st.plotly_chart(
                make_rolling_risk_plot(
                    rolling_results=rolling_results,
                    method="Normal",
                    title="Rolling window: VaR y ES normal paramétrico",
                ),
                width="stretch",
            )

        with st.expander("Ver resultados rolling window"):
            st.dataframe(rolling_results)

        # ------------------------------------------------------------
        # (e) Violations
        # ------------------------------------------------------------
        st.header("(e) Violaciones de VaR y ES")

        st.write(
            """
            Una violación ocurre cuando el rendimiento observado es menor que
            el VaR o ES estimado con la ventana anterior.
            """
        )

        violations_table = compute_violations(
            rolling_results=rolling_results,
            confidence_levels=rolling_confidence_levels,
            methods=["Historical", "Normal"],
            measures=["VaR", "ES"],
        )

        st.dataframe(
            violations_table.round(
                {
                    "Nivel de confianza": 2,
                    "Porcentaje de violaciones": 4,
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.info(
            """
            De acuerdo con la instrucción del proyecto, una buena estimación
            debería generar un porcentaje de violaciones menor al 2.5%.
            """
        )

        # ------------------------------------------------------------
        # (f) Moving-volatility VaR
        # ------------------------------------------------------------
        st.header("(f) VaR con volatilidad móvil")

        st.write(
            """
            En esta sección se calcula VaR usando volatilidad móvil y
            distribución normal. La volatilidad se estima como la desviación
            estándar de una ventana de 252 rendimientos.
            """
        )

        moving_vol_results = compute_moving_volatility_var(
            returns=returns,
            significance_levels=significance_levels,
            window=window,
        )

        st.plotly_chart(
            make_moving_volatility_plot(moving_vol_results),
            width="stretch",
        )

        moving_vol_violations = compute_moving_volatility_violations(
            moving_vol_results=moving_vol_results,
            significance_levels=significance_levels,
        )

        st.subheader("Violaciones con volatilidad móvil")

        st.dataframe(
            moving_vol_violations.round(
                {
                    "Nivel de confianza": 2,
                    "Porcentaje de violaciones": 4,
                }
            ),
            width="stretch",
            hide_index=True,
        )

        with st.expander("Ver resultados de volatilidad móvil"):
            st.dataframe(moving_vol_results)

        # ------------------------------------------------------------
        # Final interpretation
        # ------------------------------------------------------------
        st.header("Conclusiones")

        st.write(
            """
            El análisis permite comparar diferentes métodos de estimación de riesgo.
            En general, las diferencias entre VaR, ES, métodos históricos, métodos
            paramétricos y volatilidad móvil reflejan que la estimación del riesgo
            depende fuertemente de la distribución asumida y de la ventana temporal
            utilizada.

            Las tablas de violaciones permiten evaluar empíricamente si los umbrales
            de riesgo fueron suficientemente conservadores para el activo seleccionado.
            Una estimación con menos violaciones suele interpretarse como más
            conservadora, aunque también puede producir umbrales demasiado extremos.
            """
        )

    except Exception as error:
        st.error("Ocurrió un error durante la ejecución del análisis.")
        st.exception(error)

else:
    st.info(
        """
        Selecciona los parámetros en la barra lateral y presiona
        **Ejecutar análisis** para comenzar.
        """
    )