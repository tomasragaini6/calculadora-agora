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
st.set_page_config(page_title="Optimizador de Carteras · Ágora", layout="wide", page_icon="📈",
                   initial_sidebar_state="expanded")

TICKERS_DEFAULT = "AMZN, BRK-B, GOOGL, JPM, MCD, MELI, META, MSFT, NU, QQQ, SPY"

# --- Paleta Ágora (modo oscuro) ---
NAVY_OSCURO = "#0B1526"      # fondo principal
NAVY = "#0F1C3F"             # navy de marca — tarjetas, sidebar
NAVY_CLARO = "#1C2E5E"       # navy más claro — bordes, hover
DORADO = "#C9A84C"           # acento de marca
DORADO_SUAVE = "#E3CD8A"
TEXTO = "#E8E8E8"
GRIS_MEDIO = "#8891A5"
COLOR_PRIMARIO = DORADO      # color principal para gráficos (antes navy, ahora dorado sobre fondo oscuro)
COLOR_SECUNDARIO = "#5B7FD8"  # azul acero — segunda serie en gráficos

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_LAYOUT_BASE = dict(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=NAVY,
    plot_bgcolor=NAVY,
    font=dict(color=TEXTO, family="Lato, sans-serif"),
    margin=dict(l=10, r=10, t=30, b=10),
)


def aplicar_estilos():
    """CSS de marca: tipografías Ágora (Lato/Playfair Display), tarjetas,
    header y sidebar con la paleta navy/dorado."""
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Lato:wght@400;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Lato', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'Playfair Display', serif !important;
        color: {TEXTO} !important;
    }}
    h1 {{
        color: {DORADO} !important;
        border-bottom: 2px solid {NAVY_CLARO};
        padding-bottom: 0.4em;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
        border-right: 1px solid {NAVY_CLARO};
    }}

    /* Botón principal */
    .stButton > button[kind="primary"] {{
        background-color: {DORADO};
        color: {NAVY_OSCURO};
        font-weight: 700;
        border: none;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {DORADO_SUAVE};
        color: {NAVY_OSCURO};
    }}

    /* Tarjetas de métricas */
    div[data-testid="stMetric"] {{
        background-color: {NAVY};
        border: 1px solid {NAVY_CLARO};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: {GRIS_MEDIO} !important; }}
    div[data-testid="stMetricValue"] {{ color: {DORADO} !important; }}

    /* Tablas */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {NAVY_CLARO};
        border-radius: 8px;
    }}

    /* Pestañas */
    button[data-baseweb="tab"] {{
        font-family: 'Lato', sans-serif;
        font-weight: 600;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {DORADO} !important;
        border-bottom-color: {DORADO} !important;
    }}

    .agora-caption {{
        color: {GRIS_MEDIO};
        font-size: 0.85em;
    }}
    </style>
    """, unsafe_allow_html=True)


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


def cagr_real_cartera(retornos_diarios_cartera):
    """CAGR real de una cartera: compone la serie de retornos diarios
    (equity curve) y anualiza sobre los días calendario transcurridos.
    A diferencia de `ret` en `performance()` (que es la media aritmética
    de retornos diarios anualizada — el input que usa el optimizador de
    Markowitz), esto es lo que la cartera efectivamente hubiera rendido.
    Con activos volátiles ambas métricas pueden diferir mucho (volatility
    drag): CAGR ≈ media_aritmética − volatilidad²/2."""
    equity = (1 + retornos_diarios_cartera).cumprod()
    dias = (equity.index[-1] - equity.index[0]).days
    if dias <= 0:
        return np.nan
    return equity.iloc[-1] ** (365.25 / dias) - 1


def retornos_anuales_cartera(retornos_diarios_cartera):
    """Retorno compuesto por año calendario de una cartera. Devuelve una
    Series indexada por año, más un flag de qué años son parciales
    (menos de ~200 ruedas) para no leerlos como un año completo."""
    retorno_anual = (1 + retornos_diarios_cartera).groupby(
        retornos_diarios_cartera.index.year).prod() - 1
    ruedas = retornos_diarios_cartera.groupby(retornos_diarios_cartera.index.year).size()
    return retorno_anual, ruedas < 200


def tabla_retornos_anuales(carteras_retornos_diarios: dict):
    """carteras_retornos_diarios: {"Sharpe Óptimo": serie_retornos_diarios, ...}
    Devuelve (tabla, tabla_parciales) con años en filas y carteras en columnas."""
    columnas, parciales = {}, {}
    for nombre, serie in carteras_retornos_diarios.items():
        ret_anual, parcial = retornos_anuales_cartera(serie)
        columnas[nombre] = ret_anual
        parciales[nombre] = parcial
    return pd.DataFrame(columnas), pd.DataFrame(parciales)


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

    aplicar_estilos()
    _, col_centro, _ = st.columns([1, 1.2, 1])
    with col_centro:
        st.markdown(f"""
        <div style="text-align:center; margin-top:8vh; margin-bottom:1.5em;">
            <div style="font-size:2.2em;">📈</div>
            <h1 style="border:none; margin-bottom:0;">Ágora</h1>
            <p style="color:{GRIS_MEDIO}; margin-top:0;">Optimizador de Carteras</p>
        </div>
        """, unsafe_allow_html=True)
        pwd = st.text_input("Contraseña de acceso", type="password", label_visibility="collapsed",
                            placeholder="Contraseña de acceso")
        if st.button("Ingresar", type="primary", use_container_width=True):
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
def _fig_layout(fig, **overrides):
    """Aplica el layout base (tema oscuro Ágora) a una figura de Plotly,
    con overrides puntuales por gráfico."""
    fig.update_layout(**PLOTLY_LAYOUT_BASE)
    if overrides:
        fig.update_layout(**overrides)
    return fig


def main():
    aplicar_estilos()

    if not chequear_password():
        st.stop()

    # --- Header ---
    st.markdown(f"""
    <div style="display:flex; align-items:baseline; gap:14px; margin-bottom:0;">
        <h1 style="margin-bottom:0;">Optimizador de Carteras</h1>
        <span style="color:{GRIS_MEDIO}; font-size:1.1em;">· Ágora Asset Management</span>
    </div>
    <p class="agora-caption">Modelo de Markowitz — cargá tickers y obtené la cartera de mínima
    volatilidad y de máximo Sharpe, con sus gráficos y métricas.</p>
    """, unsafe_allow_html=True)

    # --- Sidebar: formulario ---
    with st.sidebar:
        st.markdown(f"<h2 style='font-size:1.3em; color:{DORADO};'>⚙️ Parámetros</h2>",
                    unsafe_allow_html=True)
        tickers_raw = st.text_area("Tickers (separados por coma)", TICKERS_DEFAULT, height=100)
        years = st.number_input("Años de historia", min_value=1, max_value=20, value=5)
        rf_pct = st.number_input("Tasa libre de riesgo anual (%)", min_value=0.0,
                                  max_value=20.0, value=4.0, step=0.5)

        with st.expander("Parámetros avanzados"):
            target_pct_raw = st.text_input("Retorno objetivo anual (%) — opcional", "")
            min_weight_pct = st.number_input("Peso mínimo por activo (%)", min_value=0.0,
                                              max_value=20.0, value=0.0, step=1.0)

        calcular = st.button("Calcular optimización ▶", type="primary", use_container_width=True)
        st.markdown(f"<p class='agora-caption'>Fuente: Yahoo Finance.</p>", unsafe_allow_html=True)

    if not calcular and "resultado" not in st.session_state:
        st.info("👈 Completá los parámetros en la barra lateral y presioná "
                 "**Calcular optimización** para empezar.")
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

            # --- Retornos diarios de cada cartera, CAGR real (equity curve) y
            # retorno año a año, para complementar el "Retorno" del optimizador
            # (media aritmética) con lo que cada cartera efectivamente hubiera
            # rendido. Se muestra aparte de "CAGR por activo" para no confundir
            # un CAGR acumulado de todo el período con un retorno anual.
            carteras_diarias = {
                "Sharpe Óptimo": returns @ w_sharpe,
                "Mínima Volatilidad": returns @ w_minvar,
                "Igual Ponderación": returns @ w_eq,
            }
            if target is not None and ok_target:
                carteras_diarias["Retorno Objetivo"] = returns @ w_target

            cagr_real = {nombre: cagr_real_cartera(serie) for nombre, serie in carteras_diarias.items()}
            tabla_anual, tabla_anual_parcial = tabla_retornos_anuales(carteras_diarias)

        st.session_state["resultado"] = dict(
            tickers=tickers, descartados=descartados, prices=prices, returns=returns,
            mu=mu, cov=cov, corr=corr, cagr=cagr, rf=rf, min_weight=min_weight, target=target,
            years=years,
            w_minvar=w_minvar, w_sharpe=w_sharpe, w_eq=w_eq, w_target=w_target, ok_target=ok_target,
            ret_mv=ret_mv, vol_mv=vol_mv, sh_mv=sh_mv, ret_sh=ret_sh, vol_sh=vol_sh, sh_sh=sh_sh,
            ret_eq=ret_eq, vol_eq=vol_eq, sh_eq=sh_eq,
            vols_front=vols_front, rets_front=rets_front, nube=nube,
            cum_activos=cum_activos, cum_minvar=cum_minvar, cum_sharpe=cum_sharpe, cum_eq=cum_eq,
            dd_minvar=dd_minvar, dd_sharpe=dd_sharpe, dd_eq=dd_eq,
            max_dd_minvar=max_dd_minvar, max_dd_sharpe=max_dd_sharpe, max_dd_eq=max_dd_eq,
            cagr_real=cagr_real, tabla_anual=tabla_anual, tabla_anual_parcial=tabla_anual_parcial,
        )

    r = st.session_state["resultado"]
    n_activos = len(r["tickers"]) - len(r["descartados"])

    if r["descartados"]:
        st.warning(f"No se pudieron descargar (o tienen muy poca historia) estos tickers, "
                   f"fueron excluidos: {', '.join(r['descartados'])}")

    # --- Tarjetas rápidas ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Activos comparables", n_activos)
    c2.metric("Tasa libre de riesgo", f"{r['rf']:.2%}")
    c3.metric("Máx. Sharpe", f"{r['sh_sh']:.2f}", help="Sharpe Óptimo")
    c4.metric("Ruedas comparables", len(r["returns"]))

    # --- Tabla comparativa de carteras (retorno, volatilidad, drawdown SIEMPRE visibles) ---
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
                                        "Sharpe": "{:.2f}", "Máx. Drawdown": "{:.2%}"})
        .background_gradient(cmap="RdYlGn", subset=["Sharpe"])
        .background_gradient(cmap="RdYlGn_r", subset=["Volatilidad", "Máx. Drawdown"]),
        use_container_width=True,
    )
    st.markdown('<p class="agora-caption">La cartera Igual Ponderación (1/N) es la referencia '
               '"estándar" sin optimizar — sirve para ver cuánto suma realmente el modelo de '
               'Markowitz por sobre repartir el capital en partes iguales. El "Retorno" de esta '
               'tabla es la media aritmética anualizada que usa el optimizador como objetivo — '
               'para ver lo que cada cartera efectivamente hubiera rendido, mirá el CAGR real '
               'en la pestaña "Retornos & Drawdown".</p>',
               unsafe_allow_html=True)

    tabla_cagr = r["cagr"].to_frame("CAGR anual")
    tabla_cagr["Tipo"] = "Activo"
    filas_extra = pd.DataFrame({
        "CAGR anual": [r["ret_sh"], r["ret_mv"], r["ret_eq"]],
        "Tipo": ["Portfolio", "Portfolio", "Portfolio"],
    }, index=["Sharpe Óptimo", "Mínima Volatilidad", "Igual Ponderación"])
    tabla_completa = pd.concat([tabla_cagr, filas_extra])

    tab_cagr, tab_ret, tab_front = st.tabs(
        ["📊 CAGR & Correlación", "📈 Retornos & Drawdown", "🎯 Frontera & Pesos"]
    )

    # ================== TAB 1: CAGR & CORRELACIÓN ==================
    with tab_cagr:
        col_tabla, col_heatmap = st.columns([1, 1.3])
        with col_tabla:
            st.subheader("CAGR por activo")
            st.dataframe(
                tabla_completa.style.format({"CAGR anual": "{:.2%}"})
                .background_gradient(cmap="RdYlGn", subset=["CAGR anual"]),
                use_container_width=True, height=380,
            )
        with col_heatmap:
            st.subheader("Matriz de correlación")
            st.markdown('<p class="agora-caption">Azul = menor correlación (más diversificación) '
                       '· Rojo = mayor correlación</p>', unsafe_allow_html=True)
            fig_corr = px.imshow(r["corr"], text_auto=".2f", color_continuous_scale="RdBu_r",
                                 zmin=-1, zmax=1, aspect="auto")
            _fig_layout(fig_corr, height=420)
            st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("Comparación CAGR anual: Activos y Portfolios")
        fig_bar = go.Figure()
        fig_bar.add_bar(x=tabla_completa.index, y=tabla_completa["CAGR anual"],
                        marker_color=[DORADO if t == "Portfolio" else COLOR_SECUNDARIO
                                     for t in tabla_completa["Tipo"]],
                        text=[f"{v:.1%}" for v in tabla_completa["CAGR anual"]],
                        textposition="outside")
        _fig_layout(fig_bar, height=380, yaxis_tickformat=".0%", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('<p class="agora-caption">El CAGR de las carteras acá es sobre TODO el '
                   'período descargado (uno solo compuesto de punta a punta). Para ver cómo se '
                   'compone año por año, mirá la pestaña "Retornos & Drawdown".</p>',
                   unsafe_allow_html=True)

    # ================== TAB 2: RETORNOS & DRAWDOWN ==================
    with tab_ret:
        st.subheader("CAGR real de la cartera (equity curve)")
        st.markdown('<p class="agora-caption">Compone la serie de retornos diarios real de cada '
                   'cartera — es lo que esa cartera efectivamente hubiera rendido, a diferencia '
                   'del "Retorno" de la tabla de comparación (que es el objetivo del optimizador, '
                   'una media aritmética). Con activos volátiles ambos números pueden diferir '
                   'bastante (volatility drag).</p>', unsafe_allow_html=True)
        cols_cagr = st.columns(len(r["cagr_real"]))
        for col, (nombre, valor) in zip(cols_cagr, r["cagr_real"].items()):
            col.metric(nombre, f"{valor:.2%}")

        st.subheader("Retorno año a año")
        tabla_anual_pct = r["tabla_anual"].copy()
        tabla_anual_pct.index.name = "Año"

        def _marcar_parcial(col):
            flags = r["tabla_anual_parcial"][col.name]
            return ["font-style: italic; color: #8891A5;" if flags.get(idx, False) else ""
                    for idx in col.index]

        st.dataframe(
            tabla_anual_pct.style
                .format("{:+.2%}", na_rep="—")
                .background_gradient(cmap="RdYlGn", axis=None, vmin=-0.5, vmax=0.5)
                .apply(_marcar_parcial, axis=0),
            use_container_width=True,
        )
        st.markdown('<p class="agora-caption">Los años en cursiva tienen menos de ~200 ruedas '
                   'operadas (primer y último año del período descargado) — son años parciales, '
                   'no comparables 1 a 1 contra un año calendario completo.</p>',
                   unsafe_allow_html=True)

        st.divider()

        st.subheader("Rendimientos acumulados")
        col_ra1, col_ra2 = st.columns(2)
        with col_ra1:
            st.markdown("**Por activo**")
            fig_cum_a = go.Figure()
            for col in r["cum_activos"].columns:
                fig_cum_a.add_scatter(x=r["cum_activos"].index, y=r["cum_activos"][col],
                                      mode="lines", name=col)
            _fig_layout(fig_cum_a, height=400, yaxis_tickformat=".0%",
                       legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_cum_a, use_container_width=True)

        with col_ra2:
            st.markdown("**Portfolios óptimos**")
            fig_cum_p = go.Figure()
            fig_cum_p.add_scatter(x=r["cum_sharpe"].index, y=r["cum_sharpe"], mode="lines",
                                  name="Sharpe Óptimo", line=dict(color=DORADO, width=3))
            fig_cum_p.add_scatter(x=r["cum_minvar"].index, y=r["cum_minvar"], mode="lines",
                                  name="Mínima Volatilidad", line=dict(color=COLOR_SECUNDARIO, width=3))
            fig_cum_p.add_scatter(x=r["cum_eq"].index, y=r["cum_eq"], mode="lines",
                                  name="Igual Ponderación", line=dict(color=GRIS_MEDIO, width=2, dash="dot"))
            _fig_layout(fig_cum_p, height=400, yaxis_tickformat=".0%",
                       legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_cum_p, use_container_width=True)

        st.subheader("Drawdown")
        st.markdown('<p class="agora-caption">Caída desde el máximo acumulado — mismo eje '
                   'temporal que los retornos acumulados.</p>', unsafe_allow_html=True)
        fig_dd = go.Figure()
        fig_dd.add_scatter(x=r["dd_sharpe"].index, y=r["dd_sharpe"], mode="lines",
                           name="Sharpe Óptimo", line=dict(color=DORADO, width=2),
                           fill="tozeroy", fillcolor="rgba(201,168,76,0.15)")
        fig_dd.add_scatter(x=r["dd_minvar"].index, y=r["dd_minvar"], mode="lines",
                           name="Mínima Volatilidad", line=dict(color=COLOR_SECUNDARIO, width=2),
                           fill="tozeroy", fillcolor="rgba(91,127,216,0.15)")
        fig_dd.add_scatter(x=r["dd_eq"].index, y=r["dd_eq"], mode="lines",
                           name="Igual Ponderación", line=dict(color=GRIS_MEDIO, width=1.5, dash="dot"))
        _fig_layout(fig_dd, height=350, yaxis_tickformat=".0%",
                   legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_dd, use_container_width=True)

    # ================== TAB 3: FRONTERA & PESOS ==================
    with tab_front:
        col_front, col_pesos = st.columns([1.3, 1])
        with col_front:
            st.subheader("Espacio de Portfolios")
            st.markdown('<p class="agora-caption">La curva marca la frontera eficiente. El '
                       'diamante dorado es Máximo Sharpe, el círculo azul es Mínima Volatilidad, '
                       'el cuadrado gris es Igual Ponderación.</p>', unsafe_allow_html=True)
            fig_frontier = go.Figure()
            fig_frontier.add_scatter(x=r["nube"]["vol"], y=r["nube"]["ret"], mode="markers",
                                     marker=dict(size=4, color=r["nube"]["sharpe"],
                                                 colorscale="Turbo", showscale=True,
                                                 colorbar=dict(title="Sharpe")),
                                     name="Portfolios aleatorios", opacity=0.5)
            mask = ~np.isnan(r["vols_front"])
            fig_frontier.add_scatter(x=r["vols_front"][mask], y=r["rets_front"][mask], mode="lines",
                                     name="Frontera eficiente", line=dict(color=DORADO_SUAVE, width=3))
            fig_frontier.add_scatter(x=[r["vol_sh"]], y=[r["ret_sh"]], mode="markers",
                                     marker=dict(size=16, color=DORADO, symbol="diamond",
                                                line=dict(color=NAVY_OSCURO, width=1)),
                                     name="Sharpe Óptimo")
            fig_frontier.add_scatter(x=[r["vol_mv"]], y=[r["ret_mv"]], mode="markers",
                                     marker=dict(size=14, color=COLOR_SECUNDARIO, symbol="circle",
                                                line=dict(color=NAVY_OSCURO, width=1)),
                                     name="Mínima Volatilidad")
            fig_frontier.add_scatter(x=[r["vol_eq"]], y=[r["ret_eq"]], mode="markers",
                                     marker=dict(size=14, color=GRIS_MEDIO, symbol="square",
                                                line=dict(color=NAVY_OSCURO, width=1)),
                                     name="Igual Ponderación")
            _fig_layout(fig_frontier, height=480, xaxis_title="Volatilidad anual",
                       yaxis_title="Rendimiento anual", xaxis_tickformat=".0%",
                       yaxis_tickformat=".0%")
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
            st.dataframe(
                tabla_pesos.style.format("{:.2%}").background_gradient(cmap="YlOrBr", axis=0),
                use_container_width=True, height=460,
            )

    st.divider()
    st.markdown(
        f'<p class="agora-caption">Fuente: Yahoo Finance · {n_activos} activos · '
        f'{len(r["returns"])} ruedas comparables · período {r["years"]}y · Ágora Asset Management</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


