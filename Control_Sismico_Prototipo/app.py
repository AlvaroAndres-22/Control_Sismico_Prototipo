"""
app.py — Control Sísmico: Plataforma de Análisis Geoespacial Avanzado
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from clustering import (
    ALGORITHMS, NOISE_LABEL, assign_risk, build_matrix, compare_territories,
    describe_cluster, evaluate, fit_model, forecast_poisson, generate_executive_report,
    k_sweep, log10_sum_energy, pca_2d, simulate_impact, trend_analysis,
)
from data_loader import (
    FEATURES, SEVERITY_COLORS, SEVERITY_LABELS,
    build_dataset, dictionary_frame, read_catalog_path,
)

# =========================================================================== #
# Configuración de página y tema oscuro geofísico
# =========================================================================== #
st.set_page_config(
    page_title="🌍 Control Sísmico · Plataforma Geoespacial",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS para tema oscuro geofísico profesional
st.markdown("""
<style>
    /* Fondo principal y sidebar */
    .stApp {
        background: linear-gradient(135deg, #0a0f1a 0%, #121a2e 50%, #0d1117 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #151d30 100%);
        border-right: 1px solid #1e3a5f;
    }
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00d4ff;
    }
    [data-testid="stMetricLabel"] {
        color: #8899aa;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(10, 15, 26, 0.7);
        padding: 4px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
    }
    /* Headers */
    h1, h2, h3 {
        color: #e0e8f0 !important;
    }
    /* Tarjetas de info */
    .info-card {
        background: linear-gradient(135deg, #1a2332 0%, #1e2d42 100%);
        border: 1px solid #2a3f5f;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .severity-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .severity-leve { background: #2ecc7133; color: #2ecc71; border: 1px solid #2ecc71; }
    .severity-medio { background: #f39c1233; color: #f39c12; border: 1px solid #f39c12; }
    .severity-grave { background: #e74c3c33; color: #e74c3c; border: 1px solid #e74c3c; }
</style>
""", unsafe_allow_html=True)

LEVEL_COLORS = {"Bajo": "#2e9e4f", "Medio": "#f0a020", "Alto": "#d62728", NOISE_LABEL: "#7f7f7f"}
PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.D3

# Ruta por defecto al catálogo real
DEFAULT_CSV = r"c:\Users\valanar\Downloads\Catalogo_Sismico.csv"

# Compatibilidad de Plotly (v5/v6 con mapbox vs v7+ con map)
HAS_PX_MAP = hasattr(px, "scatter_map")

def plot_scatter_map(d, **kwargs):
    if HAS_PX_MAP:
        kwargs["map_style"] = "open-street-map"
        return px.scatter_map(d, **kwargs)
    kwargs["mapbox_style"] = "open-street-map"
    return px.scatter_mapbox(d, **kwargs)

def plot_density_map(d, **kwargs):
    if hasattr(px, "density_map"):
        kwargs["map_style"] = "open-street-map"
        return px.density_map(d, **kwargs)
    kwargs["mapbox_style"] = "open-street-map"
    return px.density_mapbox(d, **kwargs)

ScatterMapTrace = getattr(go, "Scattermap", getattr(go, "Scattermapbox", None))


def cluster_color(c: int) -> str:
    return "#7f7f7f" if c == -1 else PALETTE[int(c) % len(PALETTE)]


# --------------------------------------------------------------------------- #
# Funciones cacheadas
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="🔄 Cargando y limpiando el catálogo sísmico…")
def load_data(csv_path: str | None, file_bytes: bytes | None):
    if file_bytes is not None:
        from data_loader import read_catalog_bytes
        raw = read_catalog_bytes(file_bytes)
    elif csv_path:
        raw = read_catalog_path(csv_path)
    else:
        raise ValueError("No se proporcionó fuente de datos.")
    return build_dataset(raw)


@st.cache_data(show_spinner="🤖 Entrenando modelo de clustering…")
def run_model(df: pd.DataFrame, features: tuple, algo: str, k: int,
              eps: float, min_samples: int, seed: int):
    X, _ = build_matrix(df, features)
    labels = fit_model(X, algo, k, eps, min_samples, seed)
    metrics = evaluate(X, labels, seed)
    coords, var = pca_2d(X, seed)
    return X, labels, metrics, coords, var


@st.cache_data(show_spinner="📊 Calculando curva del codo…")
def run_sweep(df: pd.DataFrame, features: tuple, k_max: int, seed: int):
    X, _ = build_matrix(df, features)
    return k_sweep(X, 2, k_max, seed)


def legend_html(items: dict):
    html = " &nbsp; ".join(f"<span style='color:{c}'>■</span> {n}" for n, c in items.items())
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar — Configuración
# --------------------------------------------------------------------------- #
st.sidebar.markdown("## ⚙️ Configuración")
st.sidebar.markdown("---")

st.sidebar.subheader("📂 Fuente de Datos")
source = st.sidebar.radio(
    "Selecciona la fuente",
    ["Catálogo local (Catalogo_Sismico.csv)", "Subir CSV personalizado"],
    index=0,
)
upload = None
if source == "Subir CSV personalizado":
    upload = st.sidebar.file_uploader("Catálogo SGC (.csv)", type=["csv"])

try:
    if source == "Subir CSV personalizado" and upload is not None:
        data, report = load_data(None, upload.getvalue())
    else:
        data, report = load_data(DEFAULT_CSV, None)
except (ValueError, FileNotFoundError) as e:
    st.error(f"❌ No se pudo cargar el catálogo: {e}")
    st.info("Asegúrate de que el archivo `Catalogo_Sismico.csv` esté en la carpeta de descargas.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filtros")

# Filtro de clasificación
clases_sel = st.sidebar.multiselect(
    "Clasificación sísmica",
    SEVERITY_LABELS,
    default=SEVERITY_LABELS,
)

mag_min, mag_max_f = st.sidebar.slider(
    "Rango de magnitud",
    float(np.floor(data.magnitud.min())),
    float(np.ceil(data.magnitud.max())),
    (float(np.floor(data.magnitud.min())), float(np.ceil(data.magnitud.max()))),
    0.1,
)
d_min, d_max = data.fecha_hora_utc.min().date(), data.fecha_hora_utc.max().date()
date_range = st.sidebar.date_input("Rango de fechas", (d_min, d_max), min_value=d_min, max_value=d_max)
if len(date_range) != 2:
    st.sidebar.info("Selecciona fecha inicial y final.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 Modelo No Supervisado")
algo = st.sidebar.selectbox("Algoritmo", ALGORITHMS)
k = st.sidebar.slider("Número de clústeres (k)", 2, 12, 4, disabled=(algo == "DBSCAN"))
eps = st.sidebar.slider("DBSCAN · eps", 0.5, 6.0, 2.0, 0.1, disabled=(algo != "DBSCAN"))
min_samples = st.sidebar.slider("DBSCAN · min_samples", 3, 60, 15, disabled=(algo != "DBSCAN"))
features = st.sidebar.multiselect("Variables del modelo", FEATURES, default=FEATURES)
seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, 42)

if len(features) < 2:
    st.warning("⚠️ Selecciona al menos 2 variables.")
    st.stop()

# Filtrado
mask = (
    (data.magnitud >= mag_min) &
    (data.magnitud <= mag_max_f) &
    (data.fecha_hora_utc.dt.date >= date_range[0]) &
    (data.fecha_hora_utc.dt.date <= date_range[1]) &
    (data.clasificacion.isin(clases_sel))
)
df = data[mask].reset_index(drop=True)
if len(df) < 50:
    st.warning("⚠️ Muy pocos eventos con estos filtros. Amplía el rango.")
    st.stop()

X, labels, metrics, coords, var = run_model(df, tuple(features), algo, k, eps, min_samples, int(seed))
dfr, risk = assign_risk(df, labels)

# --------------------------------------------------------------------------- #
# PESTAÑAS PRINCIPALES
# --------------------------------------------------------------------------- #
tab_home, tab_map, tab_eda, tab_ml, tab_pred, tab_sim, tab_comp, tab_report, tab_dict = st.tabs([
    "🏠 Inicio",
    "🗺️ Mapa Interactivo",
    "📊 Análisis Exploratorio",
    "🤖 Clustering",
    "📈 Predicción y Tendencia",
    "💥 Simulador de Impacto",
    "⚔️ Comparador Territorial",
    "📋 Reporte Ejecutivo",
    "📖 Diccionario & Export",
])

# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: INICIO
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_home:
    st.markdown("# 🌍 Control Sísmico — Plataforma de Análisis Geoespacial")
    st.markdown(
        "> **Sistema de vigilancia sísmica** con clustering no supervisado, análisis predictivo "
        "y herramientas de simulación para la gestión integral del riesgo."
    )

    # Métricas principales
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("📊 Total Eventos", f"{len(df):,}")
    c2.metric("⚡ Mag. Máxima", f"{df.magnitud.max():.1f}")
    c3.metric("📏 Prof. Media", f"{df.profundidad_km.mean():.0f} km")
    c4.metric("🔥 Energía Total", f"10^{log10_sum_energy(df.energia_log10_J):.1f} J")
    c5.metric("🔢 Variables", len(features))
    n_grave = (df.clasificacion == "Grave").sum()
    c6.metric("🔴 Sismos Graves", f"{n_grave:,}")

    st.markdown("---")

    # Clasificación visual
    st.subheader("Distribución por Clasificación Sísmica")
    cl_counts = df.clasificacion.value_counts()
    cols_cls = st.columns(3)
    for i, (label, css_class) in enumerate([
        ("Leve", "severity-leve"), ("Medio", "severity-medio"), ("Grave", "severity-grave")
    ]):
        cnt = cl_counts.get(label, 0)
        pct = 100 * cnt / len(df) if len(df) > 0 else 0
        cols_cls[i].markdown(
            f'<div class="info-card" style="text-align:center">'
            f'<span class="severity-badge {css_class}">{label}</span><br>'
            f'<span style="font-size:2.2rem;font-weight:800;color:#e0e8f0">{cnt:,}</span><br>'
            f'<span style="color:#8899aa">{pct:.1f}% del total</span></div>',
            unsafe_allow_html=True,
        )

    # Panel de búsqueda de lugares
    st.markdown("---")
    st.subheader("🔎 Búsqueda de Lugar — Métricas Dinámicas")
    all_places = sorted(df["municipio"].dropna().unique().tolist())
    if all_places:
        selected_place = st.selectbox("Busca un municipio o localidad:", [""] + all_places)
        if selected_place:
            place_df = df[df.municipio == selected_place]
            if not place_df.empty:
                p1, p2, p3, p4, p5 = st.columns(5)
                p1.metric("Total sismos", f"{len(place_df):,}")
                p2.metric("Mag. máxima", f"{place_df.magnitud.max():.1f}")
                p3.metric("Prof. media", f"{place_df.profundidad_km.mean():.0f} km")

                # Nivel de riesgo predominante
                if selected_place in dfr["municipio"].values:
                    place_risk = dfr[dfr.municipio == selected_place]
                    risk_mode = place_risk["nivel_riesgo"].mode()
                    risk_label = risk_mode.iloc[0] if len(risk_mode) > 0 else "N/A"
                else:
                    risk_label = "N/A"
                p4.metric("Riesgo predominante", risk_label)
                p5.metric("Energía acum.", f"10^{log10_sum_energy(place_df.energia_log10_J):.1f} J")

                # Clasificación del lugar
                place_cls = place_df.clasificacion.value_counts()
                st.markdown(
                    f"**Clasificación:** "
                    f'🟢 Leve: {place_cls.get("Leve", 0)} · '
                    f'🟡 Medio: {place_cls.get("Medio", 0)} · '
                    f'🔴 Grave: {place_cls.get("Grave", 0)}'
                )
            else:
                st.info("Sin datos para esta ubicación.")

    st.markdown("---")
    st.subheader("📝 Reporte de Limpieza del Catálogo")
    st.dataframe(
        pd.DataFrame({"Concepto": [k_.replace("_", " ").title() for k_ in report],
                       "Valor": list(report.values())}),
        hide_index=True, use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: MAPA INTERACTIVO
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_map:
    st.header("🗺️ Mapa Interactivo Mundial — OpenStreetMap")
    st.caption("Mapa 100% libre de claves API · Navegación libre por zoom y paneo")

    map_color = st.radio(
        "Colorear por:", ["Clasificación (Leve/Medio/Grave)", "Nivel de riesgo (cluster)", "Clúster"],
        horizontal=True,
    )
    sample_map = dfr.sample(min(len(dfr), 6000), random_state=0).copy()

    if map_color.startswith("Clasificación"):
        color_col = "clasificacion"
        color_map = SEVERITY_COLORS
        cat_order = {"clasificacion": ["Leve", "Medio", "Grave"]}
    elif map_color.startswith("Nivel"):
        color_col = "nivel_riesgo"
        color_map = LEVEL_COLORS
        cat_order = {"nivel_riesgo": ["Bajo", "Medio", "Alto", NOISE_LABEL]}
    else:
        sample_map["cluster_str"] = sample_map["cluster"].astype(str)
        color_col = "cluster_str"
        color_map = None
        cat_order = {}

    fig_map = plot_scatter_map(
        sample_map,
        lat="latitud",
        lon="longitud",
        color=color_col,
        color_discrete_map=color_map,
        category_orders=cat_order,
        size=np.clip(sample_map["magnitud"] * 2, 3, 18).tolist(),
        hover_name="municipio",
        hover_data={
            "magnitud": ":.1f",
            "profundidad_km": ":.0f",
            "clasificacion": True,
            "latitud": ":.3f",
            "longitud": ":.3f",
        },
        zoom=7,
        center={"lat": sample_map.latitud.mean(), "lon": sample_map.longitud.mean()},
        height=700,
        title="Mapa Sísmico Interactivo",
    )
    fig_map.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e8f0",
        legend=dict(bgcolor="rgba(10,15,26,0.8)", font=dict(size=12)),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Mapa de calor de densidad
    st.subheader("Mapa de Densidad (Heatmap)")
    fig_heat = plot_density_map(
        sample_map,
        lat="latitud",
        lon="longitud",
        z="magnitud",
        radius=15,
        zoom=7,
        center={"lat": sample_map.latitud.mean(), "lon": sample_map.longitud.mean()},
        height=500,
        title="Densidad de Energía Sísmica",
        color_continuous_scale="YlOrRd",
    )
    fig_heat.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e8f0",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: ANÁLISIS EXPLORATORIO
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_eda:
    st.header("📊 Análisis Exploratorio de Datos")
    a, b = st.columns(2)

    var_sel = a.selectbox("Variable a explorar", FEATURES, index=0)
    fig_hist = px.histogram(
        df, x=var_sel, nbins=50, marginal="box",
        color="clasificacion", color_discrete_map=SEVERITY_COLORS,
        title=f"Distribución de {var_sel}",
    )
    fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                           font_color="#e0e8f0")
    a.plotly_chart(fig_hist, use_container_width=True)

    fig_scatter = px.scatter(
        df.sample(min(len(df), 4000), random_state=0),
        x="magnitud", y="profundidad_km",
        color="clasificacion", color_discrete_map=SEVERITY_COLORS,
        opacity=0.5, title="Magnitud vs Profundidad",
    )
    fig_scatter.update_yaxes(autorange="reversed")
    fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                              font_color="#e0e8f0")
    b.plotly_chart(fig_scatter, use_container_width=True)

    # Correlación
    corr = df[features].corr()
    fig_corr = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
        title="Matriz de Correlación",
    )
    fig_corr.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e8f0")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Eventos por mes
    monthly = df.set_index("fecha_hora_utc").resample("MS").size().rename("eventos").reset_index()
    fig_month = px.bar(
        monthly, x="fecha_hora_utc", y="eventos",
        color_discrete_sequence=["#00d4ff"],
        title="Eventos por Mes",
    )
    fig_month.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                            font_color="#e0e8f0")
    st.plotly_chart(fig_month, use_container_width=True)

    # Distribución por clasificación
    fig_pie = px.pie(
        df, names="clasificacion", color="clasificacion",
        color_discrete_map=SEVERITY_COLORS,
        title="Proporción de Clasificación Sísmica",
    )
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e8f0")
    st.plotly_chart(fig_pie, use_container_width=True)

    # Top departamentos
    if "departamento" in df.columns:
        dep_counts = df["departamento"].value_counts().head(15).reset_index()
        dep_counts.columns = ["departamento", "eventos"]
        fig_dep = px.bar(
            dep_counts, x="eventos", y="departamento", orientation="h",
            color_discrete_sequence=["#f39c12"],
            title="Top 15 Departamentos por Actividad Sísmica",
        )
        fig_dep.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                              font_color="#e0e8f0", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_dep, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: CLUSTERING
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_ml:
    st.header(f"🤖 Clustering No Supervisado — {algo}")

    m = st.columns(5)
    m[0].metric("Clústeres", metrics["n_clusters"])
    m[1].metric("Silueta ↑", "—" if np.isnan(metrics["silhouette"]) else f"{metrics['silhouette']:.3f}")
    m[2].metric("Davies-Bouldin ↓", "—" if np.isnan(metrics["davies_bouldin"]) else f"{metrics['davies_bouldin']:.2f}")
    m[3].metric("Calinski-Harabasz ↑", "—" if np.isnan(metrics["calinski_harabasz"]) else f"{metrics['calinski_harabasz']:,.0f}")
    m[4].metric("Ruido", f"{metrics['pct_ruido']:.1f}%")

    if metrics["n_clusters"] < 2:
        st.warning("El modelo produjo <2 clústeres: ajusta eps / min_samples / k.")

    # Codo y silueta
    st.subheader("Selección de k: Codo y Silueta (K-Means)")
    k_max_sweep = st.slider("k máximo a evaluar", 4, 12, 10)
    sweep = run_sweep(df, tuple(features), k_max_sweep, int(seed))
    s1, s2 = st.columns(2)
    fig_elbow = px.line(sweep, x="k", y="inercia", markers=True, title="Método del Codo (Inercia)")
    fig_elbow.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)", font_color="#e0e8f0")
    s1.plotly_chart(fig_elbow, use_container_width=True)

    fig_sil = px.line(sweep, x="k", y="silhouette", markers=True, title="Coeficiente de Silueta")
    fig_sil.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)", font_color="#e0e8f0")
    s2.plotly_chart(fig_sil, use_container_width=True)

    best_k = int(sweep.loc[sweep.silhouette.idxmax(), "k"])
    st.info(f"📌 Mejor silueta en k = {best_k}. Combina con el codo y la interpretabilidad geológica.")

    # PCA
    st.subheader("Proyección PCA 2D")
    pc = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": dfr["cluster"].astype(str)})
    fig_pca = px.scatter(
        pc.sample(min(len(pc), 5000), random_state=0),
        x="PC1", y="PC2", color="cluster", opacity=0.6,
        title=f"PCA (varianza explicada: {var.sum() * 100:.1f}%)",
    )
    fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)", font_color="#e0e8f0")
    st.plotly_chart(fig_pca, use_container_width=True)

    # Perfil por clúster
    st.subheader("Perfil de Cada Clúster (z-score)")
    z = pd.DataFrame(X, columns=features).assign(cluster=labels)
    z = z[z.cluster != -1].groupby("cluster").mean()
    if not z.empty:
        fig_prof = px.imshow(
            z.T, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
            labels=dict(color="z-score"),
        )
        fig_prof.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e8f0")
        st.plotly_chart(fig_prof, use_container_width=True)

    # Interpretación territorial
    st.subheader("Interpretación de Riesgo por Clúster")
    if not risk.empty:
        for cid, row in risk.sort_values("score_riesgo", ascending=False).iterrows():
            st.markdown("- " + describe_cluster(cid, row))
        st.dataframe(risk.round(2), use_container_width=True)

    # Mapa por riesgo
    st.subheader("Mapa por Nivel de Riesgo")
    legend_html(LEVEL_COLORS)
    sample_risk = dfr.sample(min(len(dfr), 5000), random_state=0)
    fig_risk_map = plot_scatter_map(
        sample_risk, lat="latitud", lon="longitud",
        color="nivel_riesgo", color_discrete_map=LEVEL_COLORS,
        category_orders={"nivel_riesgo": ["Bajo", "Medio", "Alto", NOISE_LABEL]},
        size=np.clip(sample_risk["magnitud"] * 2, 3, 15).tolist(),
        hover_name="municipio",
        hover_data={"magnitud": ":.1f", "profundidad_km": ":.0f"},
        zoom=7,
        center={"lat": sample_risk.latitud.mean(), "lon": sample_risk.longitud.mean()},
        height=550,
    )
    fig_risk_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                               font_color="#e0e8f0")
    st.plotly_chart(fig_risk_map, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: PREDICCIÓN Y TENDENCIA
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_pred:
    st.header("📈 Predicción y Tendencia de Futuros Sismos")
    st.markdown(
        "> Análisis de tendencia basado en estadísticas de Poisson y medias móviles "
        "sobre la tasa de eventos históricos, proyectando la tendencia a corto plazo."
    )

    trend = trend_analysis(df, window_days=30)

    # Tasa de eventos con media móvil
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(
        x=trend["fecha_hora_utc"], y=trend["eventos"],
        name="Eventos diarios", marker_color="#1e5fa0", opacity=0.5,
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend["fecha_hora_utc"], y=trend["media_movil_eventos"],
        name="Media móvil 30d", line=dict(color="#00d4ff", width=2.5),
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend["fecha_hora_utc"], y=trend["tasa_poisson"],
        name="Tasa Poisson (λ acumulada)", line=dict(color="#f39c12", width=1.5, dash="dash"),
    ))
    fig_trend.update_layout(
        title="Tasa de Eventos Diarios y Tendencia",
        xaxis_title="Fecha", yaxis_title="Eventos / día",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
        font_color="#e0e8f0", height=450,
        legend=dict(bgcolor="rgba(10,15,26,0.8)"),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Energía con media móvil
    fig_energy = go.Figure()
    fig_energy.add_trace(go.Scatter(
        x=trend["fecha_hora_utc"], y=trend["energia_log10"],
        name="Energía diaria (log10 J)", fill="tozeroy",
        fillcolor="rgba(231,76,60,0.2)", line=dict(color="#e74c3c"),
    ))
    fig_energy.add_trace(go.Scatter(
        x=trend["fecha_hora_utc"], y=trend["media_movil_energia"],
        name="Media móvil 30d", line=dict(color="#2ecc71", width=2.5),
    ))
    fig_energy.update_layout(
        title="Energía Sísmica Diaria Acumulada",
        xaxis_title="Fecha", yaxis_title="log₁₀(E) [J]",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
        font_color="#e0e8f0", height=400,
    )
    st.plotly_chart(fig_energy, use_container_width=True)

    # Pronóstico de Poisson
    st.subheader("📊 Pronóstico Estadístico (Poisson)")
    horizon = st.slider("Horizonte de pronóstico (días)", 7, 90, 30)
    forecast = forecast_poisson(df, horizon)

    fc1, fc2, fc3, fc4 = st.columns(4)
    fc1.metric("λ diaria (reciente)", f"{forecast['lambda_diaria']:.2f} ev/día")
    fc2.metric(f"Eventos esperados ({horizon}d)", f"{forecast['eventos_esperados']:.0f}")
    fc3.metric("IC 95% inferior", f"{forecast['ic_inferior_95']:.0f}")
    fc4.metric("IC 95% superior", f"{forecast['ic_superior_95']:.0f}")

    st.info(
        f"📌 Basado en la tasa Poisson de los últimos 90 días, se esperan "
        f"**{forecast['eventos_esperados']:.0f}** sismos en los próximos **{horizon}** días "
        f"(IC 95%: [{forecast['ic_inferior_95']:.0f}, {forecast['ic_superior_95']:.0f}])."
    )


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: SIMULADOR DE IMPACTO
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_sim:
    st.header("💥 Simulador de Escenarios de Impacto Sísmico")
    st.markdown(
        "> Simula un sismo hipotético seleccionando magnitud y profundidad para calcular "
        "dinámicamente el radio de afectación y los municipios/clústeres expuestos."
    )

    sim_c1, sim_c2 = st.columns(2)
    sim_mag = sim_c1.slider("Magnitud del sismo hipotético", 2.0, 9.0, 5.5, 0.1)
    sim_depth = sim_c1.slider("Profundidad del foco (km)", 0.0, 300.0, 30.0, 5.0)

    # Epicentro: usar el centroide de los datos o selección manual
    sim_lat = sim_c2.number_input("Latitud del epicentro", value=float(df.latitud.mean()), format="%.3f")
    sim_lon = sim_c2.number_input("Longitud del epicentro", value=float(df.longitud.mean()), format="%.3f")

    if st.button("🚀 Ejecutar Simulación", type="primary"):
        result = simulate_impact(sim_mag, sim_depth, sim_lat, sim_lon, dfr)

        # Resultados
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Radio de Afectación", f"{result['radio_afectacion_km']:.1f} km")
        r2.metric("Intensidad Estimada", result["intensidad_estimada"])
        r3.metric("Municipios Afectados", result["n_municipios_afectados"])
        r4.metric("Sismos Históricos en Zona", f"{result['n_sismos_historicos_en_zona']:,}")

        st.metric("Energía del Evento", f"10^{result['energia_log10_J']:.2f} J")

        # Municipios afectados
        if result["municipios_afectados"]:
            st.subheader("Municipios dentro del Radio de Afectación")
            mun_df = pd.DataFrame({"Municipio": result["municipios_afectados"]})
            st.dataframe(mun_df, hide_index=True, use_container_width=True)

        # Mapa de la simulación
        st.subheader("Mapa de Simulación")
        fig_sim = plot_scatter_map(
            dfr, lat="latitud", lon="longitud",
            color="clasificacion", color_discrete_map=SEVERITY_COLORS,
            size=np.clip(dfr["magnitud"] * 1.5, 2, 10).tolist(),
            opacity=0.4,
            zoom=7,
            center={"lat": sim_lat, "lon": sim_lon},
            height=500,
        )
        # Añadir epicentro simulado
        fig_sim.add_trace(ScatterMapTrace(
            lat=[sim_lat], lon=[sim_lon],
            mode="markers+text",
            marker=dict(size=20, color="red"),
            text=[f"M{sim_mag} SIMULADO"],
            textposition="top center",
            name="Epicentro Simulado",
        ))
        # Añadir círculo de afectación (aproximado con puntos)
        angles = np.linspace(0, 2 * np.pi, 72)
        r_deg = result["radio_afectacion_km"] / 111.0
        circle_lat = sim_lat + r_deg * np.cos(angles)
        circle_lon = sim_lon + r_deg / np.cos(np.radians(sim_lat)) * np.sin(angles)
        fig_sim.add_trace(ScatterMapTrace(
            lat=circle_lat.tolist(), lon=circle_lon.tolist(),
            mode="lines",
            line=dict(color="red", width=2),
            name=f"Radio {result['radio_afectacion_km']:.0f} km",
        ))
        fig_sim.update_layout(
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#e0e8f0",
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: COMPARADOR TERRITORIAL
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_comp:
    st.header("⚔️ Comparador Bivariado Territorial (Zona A vs. Zona B)")
    st.markdown(
        "> Enfrenta dos unidades territoriales para comparar métricas estadísticas lado a lado."
    )

    comp_by = st.radio("Comparar por:", ["Municipio", "Departamento"], horizontal=True)
    col_comp = "municipio" if comp_by == "Municipio" else "departamento"

    available = sorted(dfr[col_comp].dropna().unique().tolist())
    if len(available) < 2:
        st.warning("Se necesitan al menos 2 territorios para comparar.")
    else:
        cc1, cc2 = st.columns(2)
        zone_a = cc1.selectbox("🅰️ Zona A", available, index=0)
        zone_b = cc2.selectbox("🅱️ Zona B", available, index=min(1, len(available) - 1))

        if zone_a and zone_b and zone_a != zone_b:
            comp_df = compare_territories(dfr, col_comp, zone_a, zone_b)
            st.dataframe(comp_df.T, use_container_width=True)

            # Gráficos comparativos
            gc1, gc2 = st.columns(2)
            # Box plot de magnitud
            comp_data = dfr[dfr[col_comp].isin([zone_a, zone_b])]
            fig_box = px.box(
                comp_data, x=col_comp, y="magnitud", color=col_comp,
                title="Distribución de Magnitud",
            )
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                                  font_color="#e0e8f0")
            gc1.plotly_chart(fig_box, use_container_width=True)

            # Composición de clasificación
            ct = (
                pd.crosstab(comp_data[col_comp], comp_data["clasificacion"], normalize="index") * 100
            ).reset_index().melt(id_vars=col_comp, value_name="%")
            fig_ct = px.bar(
                ct, x=col_comp, y="%", color="clasificacion",
                color_discrete_map=SEVERITY_COLORS,
                title="Composición de Clasificación Sísmica (%)",
            )
            fig_ct.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                                 font_color="#e0e8f0")
            gc2.plotly_chart(fig_ct, use_container_width=True)

            # Profundidad comparada
            fig_depth = px.violin(
                comp_data, x=col_comp, y="profundidad_km", color=col_comp,
                box=True, title="Distribución de Profundidad",
            )
            fig_depth.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                                    font_color="#e0e8f0")
            st.plotly_chart(fig_depth, use_container_width=True)

            # Timeline comparado
            for zone_label in [zone_a, zone_b]:
                zone_ts = dfr[dfr[col_comp] == zone_label].set_index("fecha_hora_utc").resample("MS").size().rename("eventos").reset_index()
                fig_ts = px.bar(
                    zone_ts, x="fecha_hora_utc", y="eventos",
                    title=f"Eventos Mensuales — {zone_label}",
                    color_discrete_sequence=["#00d4ff" if zone_label == zone_a else "#f39c12"],
                )
                fig_ts.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,0.5)",
                                     font_color="#e0e8f0", height=300)
                st.plotly_chart(fig_ts, use_container_width=True)
        elif zone_a == zone_b:
            st.warning("Selecciona dos zonas diferentes para comparar.")


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: REPORTE EJECUTIVO
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_report:
    st.header("📋 Generador de Reportes Ejecutivos")
    st.markdown(
        "> Genera un informe ejecutivo descargable en Markdown o CSV, "
        "listo para presentar ante autoridades de gestión del riesgo."
    )

    if st.button("📄 Generar Reporte Ejecutivo", type="primary"):
        md_report = generate_executive_report(df, dfr, risk, metrics, algo)
        st.markdown(md_report)

        # Descargas
        dc1, dc2 = st.columns(2)
        dc1.download_button(
            "⬇️ Descargar Reporte (Markdown)",
            md_report.encode("utf-8"),
            "informe_ejecutivo_sismico.md",
            "text/markdown",
        )
        # CSV del dataset completo
        csv_data = dfr.to_csv(index=False).encode("utf-8-sig")
        dc2.download_button(
            "⬇️ Descargar Dataset Completo (CSV)",
            csv_data,
            "dataset_sismico_procesado.csv",
            "text/csv",
        )

    st.markdown("---")
    st.subheader("Vista Previa del Dataset Procesado")
    st.dataframe(dfr.head(100), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════ #
# TAB: DICCIONARIO & EXPORTACIÓN
# ═══════════════════════════════════════════════════════════════════════════ #
with tab_dict:
    st.header("📖 Diccionario de Datos")
    st.dataframe(dictionary_frame(), hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Diccionario (CSV)",
        dictionary_frame().to_csv(index=False).encode("utf-8-sig"),
        "diccionario_datos.csv",
        "text/csv",
    )

    st.markdown("---")
    st.header("📥 Dataset Preprocesado Completo")
    st.dataframe(dfr.head(200), use_container_width=True)
    st.download_button(
        "⬇️ Dataset + Clúster (CSV)",
        dfr.to_csv(index=False).encode("utf-8-sig"),
        "sismos_preprocesado_cluster.csv",
        "text/csv",
    )

    st.markdown("---")
    st.subheader("🎯 Guía de Pitch (10 minutos)")
    st.markdown("""
    ### Estructura sugerida para la presentación de la hackathon:

    | Minuto | Sección | Contenido |
    |--------|---------|-----------|
    | 0-1 | 🎯 Hook | Dato impactante: magnitud máxima, total de sismos, zonas críticas |
    | 1-3 | 📊 Problema | Contexto sismológico colombiano, necesidad de monitoreo inteligente |
    | 3-5 | 🤖 Solución | Clustering, clasificación Leve/Medio/Grave, variables de modelado |
    | 5-7 | 🗺️ Demo | Mapa interactivo, búsqueda de lugares, simulador de impacto |
    | 7-8 | 📈 Predicción | Tendencias, tasa Poisson, proyección a 30 días |
    | 8-9 | ⚔️ Comparador | Comparación territorial A vs B, métricas lado a lado |
    | 9-10 | 🏁 Cierre | Impacto social, escalabilidad, próximos pasos |

    **Recomendaciones:**
    - Enfócate en el mapa interactivo como wow-factor visual.
    - Usa el simulador en vivo para generar engagement del jurado.
    - El reporte ejecutivo demuestra utilidad real para gestión del riesgo.
    """)
