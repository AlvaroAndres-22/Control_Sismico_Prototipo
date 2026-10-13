"""
clustering.py
=============
Modelos NO supervisados (K-Means, Gaussian Mixture, DBSCAN), métricas de
calidad e índice interpretativo de riesgo por clúster.
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


def build_matrix(df: pd.DataFrame, features, winsor=(0.01, 0.99)):
    """Winsoriza (1%-99%) para controlar outliers y estandariza con StandardScaler."""
    X = df[list(features)].astype(float)
    X = X.clip(X.quantile(winsor[0]), X.quantile(winsor[1]), axis=1)
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler


def fit_model(X, algo: str, k: int = 4, eps: float = 2.0, min_samples: int = 15, seed: int = 42):
    """Entrena el algoritmo elegido y devuelve las etiquetas (-1 = ruido en DBSCAN)."""
    if algo == "K-Means":
        return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
    if algo.startswith("Gaussian"):
        return GaussianMixture(n_components=k, covariance_type="full", n_init=3, random_state=seed).fit_predict(X)
    if algo == "DBSCAN":
        return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
    raise ValueError(f"Algoritmo desconocido: {algo}")


def evaluate(X, labels, seed: int = 42, max_sample: int = 5000) -> dict:
    """Silueta (con submuestreo), Davies-Bouldin y Calinski-Harabasz sin contar el ruido."""
    mask = labels != -1
    Xv, lv = X[mask], labels[mask]
    out = {"n_clusters": int(len(set(lv))), "pct_ruido": float(100 * (~mask).mean()),
           "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    if out["n_clusters"] >= 2 and len(lv) > out["n_clusters"]:
        ss = min(max_sample, len(lv))
        out["silhouette"] = float(silhouette_score(Xv, lv, sample_size=None if ss == len(lv) else ss, random_state=seed))
        out["davies_bouldin"] = float(davies_bouldin_score(Xv, lv))
        out["calinski_harabasz"] = float(calinski_harabasz_score(Xv, lv))
    return out


def k_sweep(X, k_min: int = 2, k_max: int = 10, seed: int = 42) -> pd.DataFrame:
    """Método del codo (inercia) y silueta para K-Means en un rango de k."""
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
    """log10 de la suma de energías, dado log10 E por evento (estable numéricamente)."""
    e = np.asarray(e, dtype=float)
    emax = e.max()
    return float(emax + np.log10(np.sum(10.0 ** (e - emax))))


def _minmax(s: pd.Series) -> pd.Series:
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def risk_table(df: pd.DataFrame, labels) -> pd.DataFrame:
    """
    Perfil por clúster + índice compuesto de riesgo sísmico INTERPRETATIVO:
        score = media( energía acumulada, magnitud p90, somerosidad, densidad )
    (todas min-max entre clústeres). La somerosidad = exp(-prof/50): un sismo
    superficial genera más movimiento fuerte en superficie. Niveles por terciles
    del score normalizado. Es un índice comparativo, NO una amenaza probabilística.
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
        "dist_mesa_km": g["dist_mesa_km"].mean(),
        "energia_total_log10": g["energia_log10_J"].apply(log10_sum_energy),
    })
    comp = pd.concat([
        _minmax(t["energia_total_log10"]), _minmax(t["mag_p90"]),
        _minmax(np.exp(-t["prof_media"] / 50.0)), _minmax(t["dens_25km"]),
    ], axis=1)
    t["score_riesgo"] = _minmax(comp.mean(axis=1))
    t["nivel_riesgo"] = pd.cut(t["score_riesgo"], [-0.01, 1 / 3, 2 / 3, 1.01], labels=["Bajo", "Medio", "Alto"]).astype(str)
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
    reg = ("profunda (≥100 km, consistente con el Nido Sísmico de Bucaramanga)" if p >= 100
           else "intermedia" if p >= 40 else "superficial (corteza, falla activa)")
    return (f"**Clúster {cid} — riesgo {row['nivel_riesgo']}**: sismicidad {reg}; "
            f"{int(row['n_eventos']):,} eventos, magnitud media {row['mag_media']:.2f} "
            f"(máx. {row['mag_max']:.1f}), profundidad media {p:.0f} km, "
            f"{row['dens_25km']:.0f} vecinos promedio a 25 km, a {row['dist_mesa_km']:.0f} km de la Mesa de Los Santos.")
