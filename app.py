"""
app.py — Sismicidad de Santander: clustering no supervisado (Hackathon SGC, categoría Medio)
Ejecutar:  streamlit run app.py
"""
from __future__ import annotations

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

from clustering import (ALGORITHMS, NOISE_LABEL, assign_risk, build_matrix, describe_cluster,
                        evaluate, fit_model, k_sweep, log10_sum_energy, pca_2d)
from data_loader import (BUCARAMANGA, FEATURES, MESA_LOS_SANTOS, MESA_RADIO_KM, build_dataset,
                         dictionary_frame, generate_mock_catalog, read_catalog_bytes)

st.set_page_config(page_title="Sismicidad Santander · Clustering", page_icon="🌋", layout="wide")

LEVEL_COLORS = {"Bajo": "#2e9e4f", "Medio": "#f0a020", "Alto": "#d62728", NOISE_LABEL: "#7f7f7f"}
PALETTE = px.colors.qualitative.Plotly + px.colors.qualitative.D3


def cluster_color(c: int) -> str:
    return "#7f7f7f" if c == -1 else PALETTE[int(c) % len(PALETTE)]


# --------------------------------------------------------------------------- #
# Funciones cacheadas
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Cargando y limpiando el catálogo…")
def load_data(use_upload: bool, file_bytes: bytes | None, n_mock: int, seed: int):
    raw = read_catalog_bytes(file_bytes) if (use_upload and file_bytes) else generate_mock_catalog(n_mock, seed)
    return build_dataset(raw)


@st.cache_data(show_spinner="Entrenando modelo…")
def run_model(df: pd.DataFrame, features: tuple, algo: str, k: int, eps: float, min_samples: int, seed: int):
    X, _ = build_matrix(df, features)
    labels = fit_model(X, algo, k, eps, min_samples, seed)
    metrics = evaluate(X, labels, seed)
    coords, var = pca_2d(X, seed)
    return X, labels, metrics, coords, var


@st.cache_data(show_spinner="Calculando curva del codo…")
def run_sweep(df: pd.DataFrame, features: tuple, k_max: int, seed: int):
    X, _ = build_matrix(df, features)
    return k_sweep(X, 2, k_max, seed)


def make_map(d: pd.DataFrame, color_fn, max_points: int = 3000):
    """Mapa folium con círculos proporcionales a la magnitud."""
    d = d.sample(min(len(d), max_points), random_state=0)
    m = folium.Map(location=[6.9, -73.3], zoom_start=8, tiles="CartoDB positron")
    for r in d.itertuples():
        folium.CircleMarker(
            location=(r.latitud, r.longitud), radius=float(np.clip(1.5 * r.magnitud, 2, 12)),
            color=color_fn(r), weight=0.5, fill=True, fill_color=color_fn(r), fill_opacity=0.65,
            tooltip=f"M{r.magnitud:.1f} · {r.profundidad_km:.0f} km · {r.municipio}",
        ).add_to(m)
    folium.Circle(MESA_LOS_SANTOS, radius=MESA_RADIO_KM * 1000, color="#444", weight=1.5,
                  dash_array="6", fill=False, tooltip="Zona Mesa de Los Santos (25 km)").add_to(m)
    folium.Marker(BUCARAMANGA, tooltip="Bucaramanga", icon=folium.Icon(color="black", icon="home")).add_to(m)
    return m


def legend(items: dict):
    html = " &nbsp; ".join(f"<span style='color:{c}'>■</span> {n}" for n, c in items.items())
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Configuración")
st.sidebar.subheader("1 · Datos")
source = st.sidebar.radio("Fuente", ["Datos simulados (demo)", "CSV del SGC"], index=0)
upload = st.sidebar.file_uploader("Catálogo SGC (.csv)", type=["csv"]) if source == "CSV del SGC" else None
n_mock = st.sidebar.slider("Eventos simulados", 2000, 20000, 6000, 1000) if source.startswith("Datos") else 6000
seed = st.sidebar.number_input("Semilla aleatoria", 0, 9999, 42)

use_upload = source == "CSV del SGC" and upload is not None
if source == "CSV del SGC" and upload is None:
    st.sidebar.warning("Sube un CSV; mientras tanto se muestran datos simulados.")
try:
    data, report = load_data(use_upload, upload.getvalue() if use_upload else None, n_mock, int(seed))
except ValueError as e:
    st.error(f"No se pudo procesar el archivo: {e}")
    st.stop()

st.sidebar.subheader("2 · Filtros")
mag_min = st.sidebar.slider("Magnitud mínima", float(np.floor(data.magnitud.min())), float(np.ceil(data.magnitud.max())),
                            float(np.floor(data.magnitud.min())), 0.1)
d_min, d_max = data.fecha_hora_utc.min().date(), data.fecha_hora_utc.max().date()
date_range = st.sidebar.date_input("Rango de fechas", (d_min, d_max), min_value=d_min, max_value=d_max)
if len(date_range) != 2:
    st.sidebar.info("Selecciona fecha inicial y final.")
    st.stop()

st.sidebar.subheader("3 · Modelo no supervisado")
algo = st.sidebar.selectbox("Algoritmo", ALGORITHMS)
k = st.sidebar.slider("Número de clústeres (k)", 2, 12, 4, disabled=(algo == "DBSCAN"))
eps = st.sidebar.slider("DBSCAN · eps", 0.5, 6.0, 2.0, 0.1, disabled=(algo != "DBSCAN"))
min_samples = st.sidebar.slider("DBSCAN · min_samples", 3, 60, 15, disabled=(algo != "DBSCAN"))
features = st.sidebar.multiselect("Variables del modelo", FEATURES, default=FEATURES)
if len(features) < 2:
    st.warning("Selecciona al menos 2 variables (la rúbrica exige ≥15).")
    st.stop()

# Filtrado (las variables de vecindad se calcularon sobre el catálogo completo)
mask = (data.magnitud >= mag_min) & (data.fecha_hora_utc.dt.date >= date_range[0]) & (data.fecha_hora_utc.dt.date <= date_range[1])
df = data[mask].reset_index(drop=True)
if len(df) < 50:
    st.warning("Muy pocos eventos con estos filtros; amplía el rango.")
    st.stop()

X, labels, metrics, coords, var = run_model(df, tuple(features), algo, k, eps, min_samples, int(seed))
dfr, risk = assign_risk(df, labels)

# --------------------------------------------------------------------------- #
# Pestañas
# --------------------------------------------------------------------------- #
tab_home, tab_eda, tab_ml, tab_terr, tab_exp = st.tabs(
    ["📌 Inicio & Contexto", "📊 Análisis Exploratorio", "🤖 Modelo No Supervisado",
     "🗺️ Interpretación Territorial", "📄 Diccionario & Exportación"])

# ---- Inicio ---------------------------------------------------------------- #
with tab_home:
    st.title("🌋 Sismicidad de Santander: clustering no supervisado")
    st.markdown(
        "Agrupamos sismos en regímenes con comportamiento similar (profundidad, energía, densidad, "
        "recurrencia y cercanía al **Nido Sísmico de Bucaramanga**) **sin usar etiquetas**, y traducimos "
        "los clústeres a un nivel de riesgo comparativo por territorio."
    )
    if not use_upload:
        st.warning("⚠️ Estás viendo **datos simulados** (formato SGC) para la demo. Para resultados reales, "
                   "sube el CSV del catálogo en la barra lateral.")
    c = st.columns(5)
    c[0].metric("Eventos", f"{len(df):,}")
    c[1].metric("Magnitud máx.", f"{df.magnitud.max():.1f}")
    c[2].metric("Profundidad media", f"{df.profundidad_km.mean():.0f} km")
    c[3].metric("Energía acumulada", f"10^{log10_sum_energy(df.energia_log10_J):.1f} J")
    c[4].metric("Variables del modelo", len(features))
    st.subheader("Fuente de datos")
    st.markdown(
        "- **Catálogo de la Red Sismológica Nacional de Colombia — Servicio Geológico Colombiano (SGC)**: "
        "<https://www.sgc.gov.co/sismos> (verifica en el portal la ruta actual de descarga del catálogo).\n"
        "- **Filtros usados al descargar**: departamento/región Santander (aprox. lat 5.7–8.2, lon −74.6 a −72.4), "
        "rango de fechas y magnitud mínima según el CSV cargado.\n"
        "- **Hora**: el SGC reporta en UTC; la app la convierte a hora local (UTC−5) solo para variables cíclicas."
    )
    st.subheader("Reporte de limpieza")
    st.dataframe(pd.DataFrame({"concepto": [k_.replace("_", " ") for k_ in report], "valor": list(report.values())}),
                 hide_index=True, width="stretch")

# ---- EDA ------------------------------------------------------------------- #
with tab_eda:
    st.header("Análisis exploratorio")
    a, b = st.columns(2)
    var_sel = a.selectbox("Variable a explorar", FEATURES, index=0)
    a.plotly_chart(px.histogram(df, x=var_sel, nbins=50, marginal="box", title=f"Distribución de {var_sel}"),
                   width="stretch")
    b.plotly_chart(px.scatter(df.sample(min(len(df), 4000), random_state=0), x="magnitud", y="profundidad_km",
                              color="zona_territorial", opacity=0.5, title="Magnitud vs profundidad").update_yaxes(autorange="reversed"),
                   width="stretch")
    corr = df[features].corr()
    st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                              title="Matriz de correlación"), width="stretch")
    st.caption("Las correlaciones altas (p. ej. magnitud-energía, posición-distancias) justifican revisar "
               "redundancia; el clustering lo mitiga con estandarización y PCA para visualizar.")
    monthly = df.set_index("fecha_hora_utc").resample("MS").size().rename("eventos").reset_index()
    st.plotly_chart(px.bar(monthly, x="fecha_hora_utc", y="eventos", title="Eventos por mes"), width="stretch")
    st.subheader("Mapa preliminar (mapa de calor)")
    hm = folium.Map(location=[6.9, -73.3], zoom_start=8, tiles="CartoDB positron")
    w = (df.magnitud - df.magnitud.min() + 0.1).tolist()
    HeatMap([[la, lo, ww] for la, lo, ww in zip(df.latitud, df.longitud, w)], radius=10, blur=14).add_to(hm)
    st_folium(hm, height=450, use_container_width=True, returned_objects=[])

# ---- Modelo ---------------------------------------------------------------- #
with tab_ml:
    st.header(f"Clustering · {algo}")
    m = st.columns(5)
    m[0].metric("Clústeres", metrics["n_clusters"])
    m[1].metric("Silueta ↑", "—" if np.isnan(metrics["silhouette"]) else f"{metrics['silhouette']:.3f}")
    m[2].metric("Davies-Bouldin ↓", "—" if np.isnan(metrics["davies_bouldin"]) else f"{metrics['davies_bouldin']:.2f}")
    m[3].metric("Calinski-Harabasz ↑", "—" if np.isnan(metrics["calinski_harabasz"]) else f"{metrics['calinski_harabasz']:,.0f}")
    m[4].metric("Ruido", f"{metrics['pct_ruido']:.1f}%")
    if metrics["n_clusters"] < 2:
        st.warning("El modelo produjo <2 clústeres: ajusta eps / min_samples / k.")

    st.subheader("Selección de k: codo y silueta (K-Means)")
    k_max = st.slider("k máximo a evaluar", 4, 12, 10)
    sweep = run_sweep(df, tuple(features), k_max, int(seed))
    s1, s2 = st.columns(2)
    s1.plotly_chart(px.line(sweep, x="k", y="inercia", markers=True, title="Método del codo (inercia)"), width="stretch")
    s2.plotly_chart(px.line(sweep, x="k", y="silhouette", markers=True, title="Coeficiente de silueta"), width="stretch")
    st.info(f"Mejor silueta en k = {int(sweep.loc[sweep.silhouette.idxmax(), 'k'])}. "
            "Combina silueta con el codo y con la interpretabilidad geológica antes de elegir k.")

    st.subheader("Proyección PCA 2D")
    pc = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": dfr["cluster"].astype(str)})
    st.plotly_chart(px.scatter(pc.sample(min(len(pc), 5000), random_state=0), x="PC1", y="PC2", color="cluster", opacity=0.6,
                               title=f"PCA (varianza explicada: {var.sum() * 100:.1f}%)"), width="stretch")

    st.subheader("Perfil de cada clúster (variables estandarizadas)")
    z = pd.DataFrame(X, columns=features).assign(cluster=labels)
    z = z[z.cluster != -1].groupby("cluster").mean()
    if not z.empty:
        st.plotly_chart(px.imshow(z.T, color_continuous_scale="RdBu_r", zmin=-2, zmax=2, aspect="auto",
                                  labels=dict(color="z-score")), width="stretch")

# ---- Territorial ----------------------------------------------------------- #
with tab_terr:
    st.header("Interpretación territorial y riesgo sísmico")
    st.caption("El nivel de riesgo es un índice **comparativo** entre clústeres (energía, magnitud p90, somerosidad "
               "y densidad); no es una amenaza probabilística ni reemplaza la norma sismorresistente NSR-10.")
    view = st.radio("Colorear mapa por", ["Nivel de riesgo", "Clúster"], horizontal=True)
    if view == "Nivel de riesgo":
        legend(LEVEL_COLORS)
        mp = make_map(dfr, lambda r: LEVEL_COLORS[r.nivel_riesgo])
    else:
        legend({("Ruido" if c == -1 else f"Clúster {c}"): cluster_color(c) for c in sorted(dfr.cluster.unique())})
        mp = make_map(dfr, lambda r: cluster_color(r.cluster))
    st_folium(mp, height=560, use_container_width=True, returned_objects=[])

    if not risk.empty:
        st.subheader("Lectura de cada clúster")
        for cid, row in risk.sort_values("score_riesgo", ascending=False).iterrows():
            st.markdown("- " + describe_cluster(cid, row))
        st.dataframe(risk.round(2), width="stretch")

    st.subheader("Mesa de Los Santos vs. resto de Santander")
    comp = dfr.groupby("zona_territorial").agg(
        eventos=("magnitud", "size"), mag_media=("magnitud", "mean"), mag_max=("magnitud", "max"),
        prof_media=("profundidad_km", "mean"), dens_25km=("dens_25km", "mean"),
        pct_riesgo_alto=("nivel_riesgo", lambda s: 100 * (s == "Alto").mean()),
        energia_total_log10=("energia_log10_J", log10_sum_energy)).round(2)
    st.dataframe(comp, width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.box(dfr, x="zona_territorial", y="magnitud", color="zona_territorial", title="Magnitud por zona"), width="stretch")
    ct = (pd.crosstab(dfr.zona_territorial, dfr.nivel_riesgo, normalize="index") * 100).reset_index().melt(id_vars="zona_territorial", value_name="%")
    c2.plotly_chart(px.bar(ct, x="zona_territorial", y="%", color="nivel_riesgo", color_discrete_map=LEVEL_COLORS,
                           title="Composición de riesgo por zona"), width="stretch")

    st.subheader("Municipios con más actividad")
    mun = dfr.groupby("municipio").agg(eventos=("magnitud", "size"), mag_media=("magnitud", "mean"),
                                       prof_media=("profundidad_km", "mean"),
                                       pct_riesgo_alto=("nivel_riesgo", lambda s: 100 * (s == "Alto").mean())
                                       ).sort_values("eventos", ascending=False).head(12).round(2)
    st.dataframe(mun, width="stretch")

# ---- Diccionario & exportación -------------------------------------------- #
with tab_exp:
    st.header("Diccionario de datos")
    st.dataframe(dictionary_frame(), hide_index=True, width="stretch")
    st.download_button("⬇️ Diccionario (CSV)", dictionary_frame().to_csv(index=False).encode("utf-8-sig"),
                       "diccionario_datos.csv", "text/csv")
    st.header("Dataset preprocesado")
    st.dataframe(dfr.head(200), width="stretch")
    st.download_button("⬇️ Dataset preprocesado + clúster (CSV)", dfr.to_csv(index=False).encode("utf-8-sig"),
                       "sismos_santander_preprocesado.csv", "text/csv")
