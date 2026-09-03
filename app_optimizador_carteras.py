# -*- coding: utf-8 -*-
"""
=====================================================================
 OPTIMIZADOR DE CARTERAS — Ágora Asset Management
=====================================================================
App web interactiva (Streamlit) para calcular la cartera de mínima
volatilidad y de máximo Sharpe (modelo de Markowitz) a partir de una
lista de tickers, sin necesidad de saber Python.

Uso local:
    streamlit run app_optimizador_carteras.py

Deploy (gratis): subir este archivo + requirements_app.txt a un repo de
GitHub y conectarlo en https://share.streamlit.io -> queda con una URL
pública protegida por la contraseña que configures en "Secrets".
"""

# %% -------------------- IMPORTS --------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf
from scipy.optimize import minimize

# %% -------------------- CONFIGURACIÓN DE PÁGINA --------------------
st.set_page_config(page_title="Optimizador de Carteras · Ágora", layout="wide", page_icon="📈")

TICKERS_DEFAULT = "AMZN, BRK-B, GOOGL, JPM, MCD, MELI, META, MSFT, NU, QQQ, SPY"
COLOR_PRIMARIO = "#0f1c3f"   # navy Ágora
COLOR_ACENTO = "#c9a84c"     # dorado Ágora


# =====================================================================
#  NÚCLEO FINANCIERO (funciones puras, sin dependencia de Streamlit —
#  así se pueden testear por separado)
# =====================================================================
def descargar_precios(tickers, years):
    """Descarga precios de cierre ajustado desde Yahoo! Finance para la
    ventana de `years` años. Devuelve (precios, tickers_descartados)."""
    start = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    data = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0] if len(tickers) == 1 else "ACTIVO")

    tickers_ok = [t for t in tickers if t in data.columns and data[t].notna().sum() > 30]
    tickers_descartados = [t for t in tickers if t not in tickers_ok]

    data = data[tickers_ok].dropna(how="all").ffill().dropna()
    return data, tickers_descartados


def calcular_cagr_por_activo(prices, trading_days=252):
    """CAGR de cada activo sobre todo el período descargado."""
    n_periodos = len(prices) / trading_days
    return (prices.iloc[-1] / prices.iloc[0]) ** (1 / n_periodos) - 1


def _bounds(n, min_weight):
    return tuple((min_weight, 1.0) for _ in range(n))


def cartera_minima_varianza(cov, min_weight=0.0):
    n = len(cov)
    if min_weight * n >= 1:
        min_weight = 0.0
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.repeat(1 / n, n)
    res = minimize(lambda w: np.sqrt(w @ cov.values @ w), x0, method="SLSQP",
                    bounds=_bounds(n, min_weight), constraints=cons)
    return pd.Series(res.x, index=cov.columns), res.success


def cartera_maximo_sharpe(mu, cov, rf, min_weight=0.0):
    n = len(cov)
    if min_weight * n >= 1:
        min_weight = 0.0

    def neg_sharpe(w):
        ret = w @ mu.values
        vol = np.sqrt(w @ cov.values @ w)
        return -(ret - rf) / vol

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    x0 = np.repeat(1 / n, n)
    res = minimize(neg_sharpe, x0, method="SLSQP", bounds=_bounds(n, min_weight), constraints=cons)
    return pd.Series(res.x, index=cov.columns), res.success


def cartera_retorno_objetivo(mu, cov, target, min_weight=0.0):
    n = len(cov)
    if min_weight * n >= 1:
        min_weight = 0.0
    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: w @ mu.values - target},
    ]
    x0 = np.repeat(1 / n, n)
    res = minimize(lambda w: np.sqrt(w @ cov.values @ w), x0, method="SLSQP",
                    bounds=_bounds(n, min_weight), constraints=cons)
    if not res.success:
        return None, False
    return pd.Series(res.x, index=cov.columns), True


def performance(w, mu, cov, rf):
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = (ret - rf) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def frontera_eficiente(mu, cov, min_weight=0.0, n_puntos=40):
    """Puntos (vol, ret) de la frontera eficiente long-only."""
    n = len(cov)
    if min_weight * n >= 1:
        min_weight = 0.0
    rets_objetivo = np.linspace(mu.min(), mu.max(), n_puntos)
    vols = []
    for t in rets_objetivo:
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=t: w @ mu.values - t},
        ]
        x0 = np.repeat(1 / n, n)
        res = minimize(lambda w: np.sqrt(w @ cov.values @ w), x0, method="SLSQP",
                        bounds=_bounds(n, min_weight), constraints=cons)
        vols.append(res.fun if res.success else np.nan)
    return np.array(vols), rets_objetivo


def calcular_drawdown(retornos_diarios):
    """Serie de drawdown (caída desde el máximo acumulado) y su mínimo
    (máximo drawdown) para una serie de retornos diarios."""
    cum = (1 + retornos_diarios).cumprod()
    dd = cum / cum.cummax() - 1
    return dd, dd.min()


def cartera_equally_weighted(cov):
    """Cartera 'estándar' de referencia: igual ponderación (1/N) entre
    todos los activos, sin ninguna optimización. Sirve como benchmark
    para ver cuánto realmente aporta el modelo de Markowitz por sobre
    la alternativa más simple posible."""
    n = len(cov)
    return pd.Series(np.repeat(1 / n, n), index=cov.columns)


def portafolios_aleatorios(mu, cov, rf, min_weight=0.0, n=4000, seed=42):
    """Nube de portfolios aleatorios (long-only, respetando el piso
    mínimo) para visualizar el espacio riesgo-retorno."""
    rng = np.random.default_rng(seed)
    n_activos = len(cov)
    piso = min_weight if min_weight * n_activos < 1 else 0.0
    libre = 1 - piso * n_activos

    pesos_base = rng.dirichlet(np.ones(n_activos), size=n)
    pesos = piso + libre * pesos_base

    rets = pesos @ mu.values
    vols = np.sqrt(np.einsum("ij,jk,ik->i", pesos, cov.values, pesos))
    sharpes = (rets - rf) / vols
    return pd.DataFrame({"vol": vols, "ret": rets, "sharpe": sharpes})


# =====================================================================
#  AUTENTICACIÓN SIMPLE (uso interno del equipo)
# =====================================================================
def chequear_password():
    """Gate simple por contraseña compartida. La contraseña se define en
    Secrets de Streamlit (APP_PASSWORD) -nunca hardcodeada en el código-.
    Alcanza para uso interno del equipo; no es un sistema de usuarios."""
    if st.session_state.get("autenticado"):
        return True

    st.title("🔒 Optimizador de Carteras · Ágora")
    pwd = st.text_input("Contraseña de acceso", type="password")
    if st.button("Ingresar"):
        clave_correcta = st.secrets.get("APP_PASSWORD", None)
        if clave_correcta is None:
            st.error("No hay APP_PASSWORD configurada en Secrets. Avisá al administrador.")
        elif pwd == clave_correcta:
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    return False


# =====================================================================
#  INTERFAZ (Streamlit)
# =====================================================================
def main():
    if not chequear_password():
        st.stop()

    st.title("📈 Optimizador de Carteras — Ágora")
    st.caption("Modelo de Markowitz: cargá tickers y obtené la cartera de mínima volatilidad "
               "y de máximo Sharpe, con sus gráficos y métricas.")

    col_form, col_resumen = st.columns([2, 1])

    with col_form:
        tickers_raw = st.text_area("Tickers (separados por coma)", TICKERS_DEFAULT, height=90)
        c1, c2 = st.columns(2)
        with c1:
            years = st.number_input("Años de historia", min_value=1, max_value=20, value=5)
        with c2:
            rf_pct = st.number_input("Tasa libre de riesgo anual (%)", min_value=0.0,
                                      max_value=20.0, value=4.0, step=0.5)
        c3, c4 = st.columns(2)
        with c3:
            target_pct_raw = st.text_input("Retorno objetivo anual (%) — opcional", "")
        with c4:
            min_weight_pct = st.number_input("Peso mínimo por activo (%)", min_value=0.0,
                                              max_value=20.0, value=0.0, step=1.0)
        calcular = st.button("Calcular optimización ▶", type="primary", use_container_width=True)

    if not calcular and "resultado" not in st.session_state:
        st.info("Completá los parámetros y presioná **Calcular optimización** para empezar.")
        return

    if calcular:
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if len(tickers) < 2:
            st.error("Cargá al menos 2 tickers.")
            return

        rf = rf_pct / 100
        min_weight = min_weight_pct / 100
        target = None
        if target_pct_raw.strip():
            try:
                target = float(target_pct_raw.replace(",", ".")) / 100
            except ValueError:
                st.warning("El retorno objetivo no es un número válido, se ignora.")

        with st.spinner("Descargando precios y optimizando..."):
            prices, descartados = descargar_precios(tickers, years)

            if prices.shape[1] < 2:
                st.error("No se pudo descargar suficiente información para al menos 2 tickers. "
                         "Revisá que los símbolos sean correctos.")
                return

            returns = prices.pct_change().dropna()
            mu = returns.mean() * 252
            cov = returns.cov() * 252
            corr = returns.corr()
            cagr = calcular_cagr_por_activo(prices)

            w_minvar, ok_minvar = cartera_minima_varianza(cov, min_weight)
            w_sharpe, ok_sharpe = cartera_maximo_sharpe(mu, cov, rf, min_weight)
            w_eq = cartera_equally_weighted(cov)
            w_target, ok_target = (None, False)
            if target is not None:
                w_target, ok_target = cartera_retorno_objetivo(mu, cov, target, min_weight)

            ret_mv, vol_mv, sh_mv = performance(w_minvar.values, mu, cov, rf)
            ret_sh, vol_sh, sh_sh = performance(w_sharpe.values, mu, cov, rf)
            ret_eq, vol_eq, sh_eq = performance(w_eq.values, mu, cov, rf)

            vols_front, rets_front = frontera_eficiente(mu, cov, min_weight)
            nube = portafolios_aleatorios(mu, cov, rf, min_weight)

            cum_activos = (1 + returns).cumprod() - 1
            cum_minvar = (1 + (returns @ w_minvar)).cumprod() - 1
            cum_sharpe = (1 + (returns @ w_sharpe)).cumprod() - 1
            cum_eq = (1 + (returns @ w_eq)).cumprod() - 1

            dd_minvar, max_dd_minvar = calcular_drawdown(returns @ w_minvar)
            dd_sharpe, max_dd_sharpe = calcular_drawdown(returns @ w_sharpe)
            dd_eq, max_dd_eq = calcular_drawdown(returns @ w_eq)

        st.session_state["resultado"] = dict(
            tickers=tickers, descartados=descartados, prices=prices, returns=returns,
            mu=mu, cov=cov, corr=corr, cagr=cagr, rf=rf, min_weight=min_weight, target=target,
            w_minvar=w_minvar, w_sharpe=w_sharpe, w_eq=w_eq, w_target=w_target, ok_target=ok_target,
            ret_mv=ret_mv, vol_mv=vol_mv, sh_mv=sh_mv, ret_sh=ret_sh, vol_sh=vol_sh, sh_sh=sh_sh,
            ret_eq=ret_eq, vol_eq=vol_eq, sh_eq=sh_eq,
            vols_front=vols_front, rets_front=rets_front, nube=nube,
            cum_activos=cum_activos, cum_minvar=cum_minvar, cum_sharpe=cum_sharpe, cum_eq=cum_eq,
            dd_minvar=dd_minvar, dd_sharpe=dd_sharpe, dd_eq=dd_eq,
            max_dd_minvar=max_dd_minvar, max_dd_sharpe=max_dd_sharpe, max_dd_eq=max_dd_eq,
        )

    r = st.session_state["resultado"]

    if r["descartados"]:
        st.warning(f"No se pudieron descargar (o tienen muy poca historia) estos tickers, "
                   f"fueron excluidos: {', '.join(r['descartados'])}")

    # --- Tarjetas rápidas ---
    with col_resumen:
        st.metric("Activos comparables", len(r["tickers"]) - len(r["descartados"]))
        st.metric("Tasa libre de riesgo", f"{r['rf']:.2%}")

    # --- Tabla comparativa de carteras (retorno Y volatilidad, siempre visibles) ---
    st.subheader("Comparación de carteras")
    filas_comparacion = {
        "Sharpe Óptimo": (r["ret_sh"], r["vol_sh"], r["sh_sh"], r["max_dd_sharpe"]),
        "Mínima Volatilidad": (r["ret_mv"], r["vol_mv"], r["sh_mv"], r["max_dd_minvar"]),
        "Igual Ponderación (1/N)": (r["ret_eq"], r["vol_eq"], r["sh_eq"], r["max_dd_eq"]),
    }
    if r["target"] is not None and r["ok_target"]:
        ret_t, vol_t, sh_t = performance(r["w_target"].values, r["mu"], r["cov"], r["rf"])
        _, max_dd_t = calcular_drawdown(r["returns"] @ r["w_target"])
        filas_comparacion["Retorno Objetivo"] = (ret_t, vol_t, sh_t, max_dd_t)

    tabla_comparacion = pd.DataFrame(
        filas_comparacion, index=["Retorno", "Volatilidad", "Sharpe", "Máx. Drawdown"]
    ).T
    st.dataframe(
        tabla_comparacion.style.format({"Retorno": "{:.2%}", "Volatilidad": "{:.2%}",
                                        "Sharpe": "{:.2f}", "Máx. Drawdown": "{:.2%}"}),
        use_container_width=True,
    )
    st.caption("La cartera Igual Ponderación (1/N) es la referencia \"estándar\" sin optimizar — "
              "sirve para ver cuánto suma realmente el modelo de Markowitz por sobre repartir "
              "el capital en partes iguales.")

    st.divider()

    # --- Tabla CAGR + correlación ---
    col_tabla, col_heatmap = st.columns([1, 1.3])
    with col_tabla:
        st.subheader("CAGR por activo")
        tabla_cagr = r["cagr"].to_frame("CAGR anual")
        tabla_cagr["Tipo"] = "Activo"
        filas_extra = pd.DataFrame({
            "CAGR anual": [r["ret_sh"], r["ret_mv"], r["ret_eq"]],
            "Tipo": ["Portfolio", "Portfolio", "Portfolio"],
        }, index=["Sharpe Óptimo", "Mínima Volatilidad", "Igual Ponderación"])
        tabla_completa = pd.concat([tabla_cagr, filas_extra])
        st.dataframe(
            tabla_completa.style.format({"CAGR anual": "{:.2%}"}),
            use_container_width=True, height=380,
        )

    with col_heatmap:
        st.subheader("Matriz de correlación entre activos")
        fig_corr = px.imshow(r["corr"], text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, aspect="auto")
        fig_corr.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- Comparación CAGR en barras ---
    st.subheader("Comparación CAGR anual: Activos y Portfolios")
    fig_bar = go.Figure()
    fig_bar.add_bar(x=tabla_completa.index, y=tabla_completa["CAGR anual"],
                     marker_color=[COLOR_ACENTO if t == "Portfolio" else COLOR_PRIMARIO
                                   for t in tabla_completa["Tipo"]],
                     text=[f"{v:.1%}" for v in tabla_completa["CAGR anual"]],
                     textposition="outside")
    fig_bar.update_layout(height=380, yaxis_tickformat=".0%", showlegend=False,
                          margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Retornos acumulados ---
    st.subheader("Rendimientos acumulados")
    col_ra1, col_ra2 = st.columns(2)
    with col_ra1:
        st.markdown("**Por activo**")
        fig_cum_a = go.Figure()
        for col in r["cum_activos"].columns:
            fig_cum_a.add_scatter(x=r["cum_activos"].index, y=r["cum_activos"][col],
                                  mode="lines", name=col)
        fig_cum_a.update_layout(height=400, yaxis_tickformat=".0%",
                                margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_cum_a, use_container_width=True)

    with col_ra2:
        st.markdown("**Portfolios óptimos**")
        fig_cum_p = go.Figure()
        fig_cum_p.add_scatter(x=r["cum_sharpe"].index, y=r["cum_sharpe"], mode="lines",
                              name="Sharpe Óptimo", line=dict(color=COLOR_ACENTO, width=3))
        fig_cum_p.add_scatter(x=r["cum_minvar"].index, y=r["cum_minvar"], mode="lines",
                              name="Mínima Volatilidad", line=dict(color=COLOR_PRIMARIO, width=3))
        fig_cum_p.add_scatter(x=r["cum_eq"].index, y=r["cum_eq"], mode="lines",
                              name="Igual Ponderación", line=dict(color="gray", width=2, dash="dot"))
        fig_cum_p.update_layout(height=400, yaxis_tickformat=".0%",
                                margin=dict(l=10, r=10, t=10, b=10),
                                legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_cum_p, use_container_width=True)

    st.divider()

    # --- Drawdown ---
    st.subheader("Drawdown")
    st.caption("Caída desde el máximo acumulado en cada momento — el mismo eje temporal que "
              "los retornos acumulados, para ver de un vistazo cuándo y cuánto sufrió cada cartera.")
    fig_dd = go.Figure()
    fig_dd.add_scatter(x=r["dd_sharpe"].index, y=r["dd_sharpe"], mode="lines",
                       name="Sharpe Óptimo", line=dict(color=COLOR_ACENTO, width=2),
                       fill="tozeroy", fillcolor="rgba(201,168,76,0.15)")
    fig_dd.add_scatter(x=r["dd_minvar"].index, y=r["dd_minvar"], mode="lines",
                       name="Mínima Volatilidad", line=dict(color=COLOR_PRIMARIO, width=2),
                       fill="tozeroy", fillcolor="rgba(15,28,63,0.12)")
    fig_dd.add_scatter(x=r["dd_eq"].index, y=r["dd_eq"], mode="lines",
                       name="Igual Ponderación", line=dict(color="gray", width=1.5, dash="dot"))
    fig_dd.update_layout(height=350, yaxis_tickformat=".0%",
                         margin=dict(l=10, r=10, t=10, b=10),
                         legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_dd, use_container_width=True)

    st.divider()

    # --- Frontera eficiente + pesos óptimos ---
    col_front, col_pesos = st.columns([1.3, 1])
    with col_front:
        st.subheader("Espacio de Portfolios")
        st.caption("La curva marca la frontera eficiente. El punto dorado es Máximo Sharpe, "
                   "el gris es Mínima Volatilidad.")
        fig_frontier = go.Figure()
        fig_frontier.add_scatter(x=r["nube"]["vol"], y=r["nube"]["ret"], mode="markers",
                                 marker=dict(size=4, color=r["nube"]["sharpe"],
                                             colorscale="Turbo", showscale=True,
                                             colorbar=dict(title="Sharpe")),
                                 name="Portfolios aleatorios", opacity=0.5)
        mask = ~np.isnan(r["vols_front"])
        fig_frontier.add_scatter(x=r["vols_front"][mask], y=r["rets_front"][mask], mode="lines",
                                 name="Frontera eficiente", line=dict(color="blue", width=3))
        fig_frontier.add_scatter(x=[r["vol_sh"]], y=[r["ret_sh"]], mode="markers",
                                 marker=dict(size=16, color=COLOR_ACENTO, symbol="diamond"),
                                 name="Sharpe Óptimo")
        fig_frontier.add_scatter(x=[r["vol_mv"]], y=[r["ret_mv"]], mode="markers",
                                 marker=dict(size=14, color=COLOR_PRIMARIO, symbol="circle"),
                                 name="Mínima Volatilidad")
        fig_frontier.add_scatter(x=[r["vol_eq"]], y=[r["ret_eq"]], mode="markers",
                                 marker=dict(size=14, color="gray", symbol="square"),
                                 name="Igual Ponderación")
        fig_frontier.update_layout(height=480, xaxis_title="Volatilidad anual",
                                   yaxis_title="Rendimiento anual", xaxis_tickformat=".0%",
                                   yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_frontier, use_container_width=True)

    with col_pesos:
        st.subheader("Pesos óptimos")
        tabla_pesos = pd.DataFrame({
            "Sharpe Óptimo": r["w_sharpe"],
            "Mínima Volatilidad": r["w_minvar"],
            "Igual Ponderación": r["w_eq"],
        })
        if r["target"] is not None:
            if r["ok_target"]:
                tabla_pesos["Retorno Objetivo"] = r["w_target"]
            else:
                st.warning("El retorno objetivo pedido no es alcanzable con estos activos "
                          "(está fuera del rango que permite la frontera eficiente).")
        st.dataframe(tabla_pesos.style.format("{:.2%}"), use_container_width=True, height=460)

    st.caption(f"Fuente: Yahoo Finance. {len(r['tickers']) - len(r['descartados'])} activos, "
               f"{len(r['returns'])} ruedas comparables, período {years}y.")


if __name__ == "__main__":
    main()
