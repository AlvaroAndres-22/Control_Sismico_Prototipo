"""
clustering.py
=============
Modelos NO supervisados (K-Means, GMM, DBSCAN), métricas de validación,
índice de riesgo por clúster, módulo de predicción/tendencia y simulador
de escenarios de impacto sísmico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ALGORITHMS = ["K-Means", "Gaussian Mixture (GMM)", "DBSCAN"]
NOISE_LABEL = "Atípico (ruido)"


# --------------------------------------------------------------------------- #
# Construcción de la matriz y entrenamiento
# --------------------------------------------------------------------------- #
def build_matrix(df: pd.DataFrame, features, winsor=(0.01, 0.99)):
    """Winsoriza (1%-99%) y estandariza con StandardScaler."""
    X = df[list(features)].astype(float)
    X = X.clip(X.quantile(winsor[0]), X.quantile(winsor[1]), axis=1)
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def fit_model(X, algo: str, k: int = 4, eps: float = 2.0, min_samples: int = 15, seed: int = 42):
    """Entrena el algoritmo y devuelve etiquetas (-1 = ruido en DBSCAN)."""
    if algo == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if algo.startswith("Gaussian"):
        return GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit_predict(X)
    if algo == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    raise ValueError(f"Algoritmo desconocido: {algo}")


def evaluate(X, labels, seed: int = 42, max_sample: int = 5000) -> dict:
    """Silueta, Davies-Bouldin y Calinski-Harabasz sin contar ruido."""
    mask = labels != -1
    Xv, lv = X[mask], labels[mask]
    out = {
        "n_clusters": int(len(set(lv))),
        "pct_ruido": float(100 * (~mask).mean()),
        "silhouette": np.nan,
        "davies_bouldin": np.nan,
        "calinski_harabasz": np.nan,
    }
    if out["n_clusters"] >= 2 and len(lv) > out["n_clusters"]:
        ss = min(max_sample, len(lv))
        out["silhouette"] = float(
            silhouette_score(Xv, lv, sample_size=None if ss == len(lv) else ss, random_state=seed)
        )
        out["davies_bouldin"] = float(davies_bouldin_score(Xv, lv))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xv, lv))
    return out


def k_sweep(X, k_min: int = 2, k_max: int = 10, seed: int = 42) -> pd.DataFrame:
    """Método del codo (inercia) y silueta para K-Means."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
        ss = min(3000, len(X))
        sil = silhouette_score(X, km.labels_, sample_size=None if ss == len(X) else ss, random_state=seed)
        rows.append({"k": k, "inercia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


def pca_2d(X, seed: int = 42):
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X), pca.explained_variance_ratio_


def log10_sum_energy(e) -> float:
    """log10 de la suma de energías dado log10 E por evento."""
    e = np.asarray(e, dtype=float)
    if len(e) == 0:
        return 0.0
    emax = e.max()
    return float(emax + np.log10(np.sum(10.0 ** (e - emax))))


# --------------------------------------------------------------------------- #
# Riesgo por clúster
# --------------------------------------------------------------------------- #
def _minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def risk_table(df: pd.DataFrame, labels) -> pd.DataFrame:
    """
    Perfil por clúster + índice compuesto de riesgo sísmico:
        score = media(energía, mag_p90, somerosidad, densidad)
    """
    d = df.assign(cluster=labels)
    d = d[d["cluster"] != -1]
    if d.empty:
        return pd.DataFrame()
    g = d.groupby("cluster")
    t = pd.DataFrame({
        "n_eventos": g.size(),
        "mag_media": g["magnitud"].mean(),
        "mag_p90": g["magnitud"].quantile(0.9),
        "mag_max": g["magnitud"].max(),
        "prof_media": g["profundidad_km"].mean(),
        "dens_25km": g["dens_25km"].mean(),
        "tasa_30d": g["tasa_30d"].mean(),
        "energia_total_log10": g["energia_log10_J"].apply(log10_sum_energy),
    })
    comp = pd.concat([
        _minmax(t["energia_total_log10"]), _minmax(t["mag_p90"]),
        _minmax(np.exp(-t["prof_media"] / 50.0)), _minmax(t["dens_25km"]),
    ], axis=1)
    t["score_riesgo"] = _minmax(comp.mean(axis=1))
    t["nivel_riesgo"] = pd.cut(
        t["score_riesgo"], [-0.01, 1 / 3, 2 / 3, 1.01],
        labels=["Bajo", "Medio", "Alto"]
    ).astype(str)
    return t


def assign_risk(df: pd.DataFrame, labels):
    """Devuelve (df con cluster y nivel_riesgo, tabla de riesgo)."""
    table = risk_table(df, labels)
    out = df.copy()
    out["cluster"] = labels
    mapping = table["nivel_riesgo"].to_dict() if not table.empty else {}
    out["nivel_riesgo"] = out["cluster"].map(mapping).fillna(NOISE_LABEL)
    return out, table


def describe_cluster(cid: int, row: pd.Series) -> str:
    """Texto interpretativo automático para un clúster."""
    p = row["prof_media"]
    reg = (
        "profunda (≥100 km)" if p >= 100
        else "intermedia" if p >= 40
        else "superficial (corteza, mayor peligro)"
    )
    return (
        f"**Clúster {cid} — riesgo {row['nivel_riesgo']}**: sismicidad {reg}; "
        f"{int(row['n_eventos']):,} eventos, magnitud media {row['mag_media']:.2f} "
        f"(máx. {row['mag_max']:.1f}), profundidad media {p:.0f} km, "
        f"{row['dens_25km']:.0f} vecinos promedio a 25 km."
    )


# --------------------------------------------------------------------------- #
# Módulo de Predicción y Tendencia
# --------------------------------------------------------------------------- #
def trend_analysis(df: pd.DataFrame, window_days: int = 30) -> pd.DataFrame:
    """
    Análisis de tendencia: tasa de eventos y energía acumulada por ventana temporal,
    con media móvil y tasa de Poisson esperada.
    """
    ts = df.set_index("fecha_hora_utc").sort_index()
    # Resample por día
    daily = ts.resample("D").agg(
        eventos=("magnitud", "size"),
        mag_max=("magnitud", "max"),
        energia_log10=("energia_log10_J", lambda x: log10_sum_energy(x) if len(x) > 0 else 0),
    ).fillna({"eventos": 0, "mag_max": 0, "energia_log10": 0})

    daily["media_movil_eventos"] = daily["eventos"].rolling(window_days, min_periods=1).mean()
    daily["media_movil_energia"] = daily["energia_log10"].rolling(window_days, min_periods=1).mean()
    daily["tasa_poisson"] = daily["eventos"].expanding().mean()

    return daily.reset_index()


def forecast_poisson(df: pd.DataFrame, horizon_days: int = 30) -> dict:
    """
    Proyección estadística de Poisson sobre la tasa de eventos.
    Calcula lambda (tasa diaria promedio) y proyecta el número esperado
    de eventos con intervalo de confianza al 95%.
    """
    ts = df.set_index("fecha_hora_utc").sort_index()
    daily_counts = ts.resample("D").size()
    # Usar los últimos 90 días para estimar lambda reciente
    recent = daily_counts.tail(90)
    lambda_hat = float(recent.mean())
    expected = lambda_hat * horizon_days
    # IC 95% para Poisson (aproximación normal para lambda grande)
    std = np.sqrt(lambda_hat * horizon_days)
    return {
        "lambda_diaria": round(lambda_hat, 2),
        "eventos_esperados": round(expected, 0),
        "ic_inferior_95": round(max(0, expected - 1.96 * std), 0),
        "ic_superior_95": round(expected + 1.96 * std, 0),
        "horizonte_dias": horizon_days,
    }


# --------------------------------------------------------------------------- #
# Simulador de Escenarios de Impacto
# --------------------------------------------------------------------------- #
def simulate_impact(
    mag: float,
    depth_km: float,
    epicenter_lat: float,
    epicenter_lon: float,
    df: pd.DataFrame,
) -> dict:
    """
    Simula un sismo hipotético y calcula:
    - Radio de afectación estimado (fórmula empírica simplificada)
    - Municipios/regiones dentro del radio de afectación
    - Energía del evento
    - Intensidad estimada (escala simplificada)
    """
    from data_loader import haversine_km

    # Radio de afectación empírico (simplificado):
    #   R ≈ 10^(0.5*M - 1.0) * factor_profundidad
    # Sismos superficiales afectan área mayor en superficie.
    depth_factor = np.clip(1.0 / (1.0 + depth_km / 50.0), 0.2, 1.0)
    radius_km = (10 ** (0.5 * mag - 1.0)) * depth_factor * 30  # km
    radius_km = np.clip(radius_km, 5, 500)

    # Energía
    energia_log10 = 4.8 + 1.5 * mag

    # Municipios afectados
    dist = haversine_km(
        df["latitud"].values, df["longitud"].values,
        epicenter_lat, epicenter_lon,
    )
    mask = dist <= radius_km
    affected_df = df[mask]

    municipios_afectados = []
    if "municipio" in affected_df.columns:
        municipios_afectados = affected_df["municipio"].unique().tolist()

    # Intensidad estimada (simplificada basada en magnitud y profundidad)
    if mag >= 7.0:
        intensidad = "Extrema (IX-XII MMI)"
    elif mag >= 6.0:
        intensidad = "Muy fuerte (VII-IX MMI)"
    elif mag >= 5.0:
        intensidad = "Fuerte (V-VII MMI)"
    elif mag >= 4.0:
        intensidad = "Moderada (IV-V MMI)"
    elif mag >= 3.0:
        intensidad = "Leve (II-III MMI)"
    else:
        intensidad = "Imperceptible (I MMI)"

    return {
        "magnitud": mag,
        "profundidad_km": depth_km,
        "radio_afectacion_km": round(float(radius_km), 1),
        "energia_log10_J": round(energia_log10, 2),
        "intensidad_estimada": intensidad,
        "n_municipios_afectados": len(municipios_afectados),
        "municipios_afectados": municipios_afectados[:30],  # top 30
        "n_sismos_historicos_en_zona": int(mask.sum()),
        "epicentro": (epicenter_lat, epicenter_lon),
    }


# --------------------------------------------------------------------------- #
# Comparador Territorial
# --------------------------------------------------------------------------- #
def compare_territories(
    df: pd.DataFrame, col: str, zone_a: str, zone_b: str,
) -> pd.DataFrame:
    """
    Compara métricas estadísticas de dos territorios (Zona A vs Zona B).
    """
    da = df[df[col] == zone_a]
    db = df[df[col] == zone_b]

    def _stats(d: pd.DataFrame, name: str) -> dict:
        if d.empty:
            return {"zona": name, "total_sismos": 0}
        return {
            "zona": name,
            "total_sismos": len(d),
            "mag_media": round(d["magnitud"].mean(), 2),
            "mag_max": round(d["magnitud"].max(), 1),
            "mag_p90": round(d["magnitud"].quantile(0.9), 2),
            "prof_media_km": round(d["profundidad_km"].mean(), 1),
            "prof_std_km": round(d["profundidad_km"].std(), 1),
            "dens_25km_media": round(d["dens_25km"].mean(), 1),
            "energia_total_log10": round(log10_sum_energy(d["energia_log10_J"]), 2),
            "pct_grave": round(100 * (d["clasificacion"] == "Grave").mean(), 1),
            "pct_medio": round(100 * (d["clasificacion"] == "Medio").mean(), 1),
            "pct_leve": round(100 * (d["clasificacion"] == "Leve").mean(), 1),
            "tasa_30d_media": round(d["tasa_30d"].mean(), 1),
        }

    return pd.DataFrame([_stats(da, zone_a), _stats(db, zone_b)]).set_index("zona")


# --------------------------------------------------------------------------- #
# Generador de Reportes Ejecutivos
# --------------------------------------------------------------------------- #
def generate_executive_report(
    df: pd.DataFrame,
    dfr: pd.DataFrame,
    risk_table_df: pd.DataFrame,
    metrics: dict,
    algo: str,
) -> str:
    """
    Genera un informe ejecutivo en formato Markdown listo para descarga.
    """
    n = len(dfr)
    date_min = df["fecha_hora_utc"].min().strftime("%Y-%m-%d")
    date_max = df["fecha_hora_utc"].max().strftime("%Y-%m-%d")

    clasificacion = dfr["clasificacion"].value_counts().to_dict() if "clasificacion" in dfr.columns else {}
    grave = clasificacion.get("Grave", 0)
    medio = clasificacion.get("Medio", 0)
    leve = clasificacion.get("Leve", 0)

    report = f"""# 📋 INFORME EJECUTIVO — Control Sísmico

**Fecha de generación:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
**Período analizado:** {date_min} a {date_max}
**Total de eventos analizados:** {n:,}

---

## 1. Resumen General

| Métrica | Valor |
|---------|-------|
| Total eventos | {n:,} |
| Magnitud máxima | {dfr['magnitud'].max():.1f} |
| Magnitud media | {dfr['magnitud'].mean():.2f} |
| Profundidad media | {dfr['profundidad_km'].mean():.1f} km |
| Energía acumulada | 10^{log10_sum_energy(dfr['energia_log10_J']):.1f} J |

## 2. Clasificación de Sismos

| Categoría | Cantidad | Porcentaje |
|-----------|----------|------------|
| 🔴 Grave (M≥5.0) | {grave:,} | {100*grave/max(n,1):.1f}% |
| 🟡 Medio (3.0≤M<5.0) | {medio:,} | {100*medio/max(n,1):.1f}% |
| 🟢 Leve (M<3.0) | {leve:,} | {100*leve/max(n,1):.1f}% |

## 3. Modelo de Clustering

- **Algoritmo:** {algo}
- **Clústeres identificados:** {metrics.get('n_clusters', 'N/A')}
- **Coeficiente de Silueta:** {metrics.get('silhouette', 'N/A')}
- **Índice Davies-Bouldin:** {metrics.get('davies_bouldin', 'N/A')}
- **Ruido:** {metrics.get('pct_ruido', 0):.1f}%

## 4. Perfil de Riesgo por Clúster

"""
    if not risk_table_df.empty:
        report += risk_table_df.to_markdown() + "\n\n"
    else:
        report += "Sin datos de riesgo por clúster.\n\n"

    # Top municipios
    if "municipio" in dfr.columns:
        top = dfr.groupby("municipio").agg(
            eventos=("magnitud", "size"),
            mag_max=("magnitud", "max"),
            mag_media=("magnitud", "mean"),
        ).sort_values("eventos", ascending=False).head(10).round(2)
        report += "## 5. Top 10 Municipios por Actividad Sísmica\n\n"
        report += top.to_markdown() + "\n\n"

    report += """## 6. Recomendaciones

1. **Monitoreo continuo** de las zonas con clústeres de riesgo Alto.
2. **Revisión de infraestructura** en municipios con mayor densidad sísmica.
3. **Actualizar planes de emergencia** basados en los patrones de clustering identificados.
4. **Análisis de tendencia** sugiere seguimiento especial a las zonas con incremento en tasa de eventos.

---

*Informe generado automáticamente por el Sistema de Control Sísmico.*
*Este análisis es de carácter investigativo y no reemplaza los estudios oficiales de amenaza sísmica.*
"""
    return report
