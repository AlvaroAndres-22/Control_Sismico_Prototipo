"""
data_loader.py
==============
Ingesta, limpieza rigurosa, clasificación sísmica (Leve / Medio / Grave) e
ingeniería de 18+ variables del catálogo sísmico colombiano.

Flujo:
    CSV  →  standardize_columns  →  clean_catalog  →  engineer_features
         →  DataFrame único (1 fila = 1 sismo)

Todas las funciones son puras para cachearlas con ``st.cache_data``.
"""
from __future__ import annotations

import io
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# --------------------------------------------------------------------------- #
# Constantes geográficas y de modelado
# --------------------------------------------------------------------------- #
EARTH_KM = 6371.0088
K_VECINOS = 30

# --------------------------------------------------------------------------- #
# Clasificación sísmica  (Leve / Medio / Grave)
# --------------------------------------------------------------------------- #
# Regla de negocio basada en la escala Richter simplificada:
#   Leve  : M < 3.0   → sismos menores, baja energía, rara vez percibidos.
#   Medio : 3.0 ≤ M < 5.0 → sismos moderados, monitoreo regular.
#   Grave : M ≥ 5.0   → alta magnitud, alto potencial de daño.
SEVERITY_BINS   = [-np.inf, 3.0, 5.0, np.inf]
SEVERITY_LABELS = ["Leve", "Medio", "Grave"]
SEVERITY_COLORS = {"Leve": "#2ecc71", "Medio": "#f39c12", "Grave": "#e74c3c"}

# Variables usadas por el modelo no supervisado (18 ≥ 15 exigidas)
FEATURES = [
    "magnitud", "profundidad_km", "latitud", "longitud",
    "energia_log10_J", "dt_prev_log_h", "tasa_30d", "hora_sin", "hora_cos",
    "dens_25km", "dens_knn_log", "mag_media_vec", "mag_max_vec",
    "prof_media_vec", "prof_std_vec", "energia_vec_log10",
    "dist_centroide_km", "profundidad_norm",
]

# --------------------------------------------------------------------------- #
# Diccionario de datos
# --------------------------------------------------------------------------- #
DATA_DICTIONARY = [
    ("fecha_hora_utc", "datetime64[ns, UTC]", "UTC", "SGC (directa)", "Marca temporal del origen del sismo."),
    ("anio", "int", "año", "Derivada", "Año del evento."),
    ("mes", "int", "1-12", "Derivada", "Mes del evento."),
    ("municipio", "str", "—", "SGC / derivada", "Municipio extraído de la columna Región."),
    ("departamento", "str", "—", "SGC / derivada", "Departamento extraído de la columna Región."),
    ("region_raw", "str", "—", "SGC (directa)", "Texto original de la columna Región del SGC."),
    ("clasificacion", "str", "Leve/Medio/Grave", "Regla de negocio", "Clasificación por magnitud: Leve (<3), Medio (3-5), Grave (≥5)."),
    ("magnitud", "float", "Mw/ML", "SGC (directa)", "Magnitud del sismo."),
    ("profundidad_km", "float", "km", "SGC (directa)", "Profundidad del foco sísmico."),
    ("latitud", "float", "grados", "SGC (directa)", "Latitud del epicentro."),
    ("longitud", "float", "grados", "SGC (directa)", "Longitud del epicentro."),
    ("energia_log10_J", "float", "log10(J)", "Derivada", "Energía: log10 E = 4.8 + 1.5·M."),
    ("dt_prev_log_h", "float", "log(1+h)", "Derivada", "Intervalo desde el sismo anterior."),
    ("tasa_30d", "int", "eventos/30d", "Derivada", "Eventos en los 30 días previos."),
    ("hora_sin", "float", "[-1,1]", "Derivada", "Hora local codificada (seno)."),
    ("hora_cos", "float", "[-1,1]", "Derivada", "Hora local codificada (coseno)."),
    ("dens_25km", "int", "eventos", "Derivada", "Sismos a ≤25 km."),
    ("dens_knn_log", "float", "log10(ev/km²)", "Derivada", "Densidad local k-NN."),
    ("mag_media_vec", "float", "magnitud", "Derivada", "Magnitud media del vecindario."),
    ("mag_max_vec", "float", "magnitud", "Derivada", "Magnitud máxima del vecindario."),
    ("prof_media_vec", "float", "km", "Derivada", "Profundidad media del vecindario."),
    ("prof_std_vec", "float", "km", "Derivada", "Dispersión de profundidad del vecindario."),
    ("energia_vec_log10", "float", "log10(J)", "Derivada", "Energía acumulada del vecindario."),
    ("dist_centroide_km", "float", "km", "Derivada", "Distancia al centroide geográfico del catálogo."),
    ("profundidad_norm", "float", "[0,1]", "Derivada", "Profundidad min-max normalizada."),
    ("cluster", "int", "—", "Modelo", "Etiqueta de clúster."),
    ("nivel_riesgo", "str", "Bajo/Medio/Alto", "Modelo", "Nivel interpretativo por clúster."),
]


def dictionary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        DATA_DICTIONARY,
        columns=["variable", "tipo", "unidad", "fuente", "justificacion_tecnica"],
    )


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia gran círculo (km), vectorizada."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))


def _norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Estandarización de columnas del SGC
# --------------------------------------------------------------------------- #
COL_ALIASES = {
    "fecha_hora":      ["fecha_hora_utc", "fecha_hora__utc", "fecha_hora", "fecha_utc", "fecha", "origin_time", "time"],
    "hora":            ["hora_utc", "hora"],
    "latitud":         ["latitud", "lat", "latitude"],
    "longitud":        ["longitud", "lon", "lng", "longitude", "long"],
    "profundidad_km":  ["profundidad_km", "profundidad", "depth", "prof", "prof_km"],
    "magnitud":        ["magnitud", "mag", "magnitude", "ml"],
    "municipio":       ["region", "municipio_departamento", "municipio", "localizacion", "lugar", "ubicacion"],
}
REQUIRED = ("fecha_hora", "latitud", "longitud", "profundidad_km", "magnitud")


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del CSV del SGC a nombres canónicos."""
    norm = {c: _norm(c) for c in raw.columns}
    found: dict[str, str] = {}
    for target, aliases in COL_ALIASES.items():
        hit = None
        for a in aliases:
            hit = next((c for c, n in norm.items() if n == a and c not in found.values()), None)
            if hit:
                break
        if hit is None:
            for a in aliases:
                hit = next((c for c, n in norm.items() if n.startswith(a) and c not in found.values()), None)
                if hit:
                    break
        if hit:
            found[target] = hit
    missing = [k for k in REQUIRED if k not in found]
    if missing:
        raise ValueError(
            f"Faltan columnas obligatorias {missing}. Columnas recibidas: {list(raw.columns)}"
        )
    df = raw[list(found.values())].copy()
    df.columns = list(found.keys())
    if "hora" in df.columns:
        df["fecha_hora"] = df["fecha_hora"].astype(str).str[:10] + " " + df["hora"].astype(str)
        df = df.drop(columns="hora")
    return df


def read_catalog_bytes(data: bytes) -> pd.DataFrame:
    """Lee un CSV subido por el usuario probando codificaciones y separadores."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("No fue posible leer el archivo CSV.")


def read_catalog_path(path: str) -> pd.DataFrame:
    """Lee un CSV desde ruta en disco."""
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError("No fue posible leer el archivo CSV.")


# --------------------------------------------------------------------------- #
# Extracción de municipio y departamento del campo Region del SGC
# --------------------------------------------------------------------------- #
def _parse_region(region_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Parsea 'Municipio - Departamento, Colombia' → (municipio, departamento).
    """
    clean = region_series.fillna("").astype(str)
    # Quitar ', Colombia' al final
    clean = clean.str.replace(r",\s*Colombia$", "", regex=True).str.strip()
    # Separar por ' - '
    parts = clean.str.split(r"\s*-\s*", n=1, expand=True)
    municipio = parts[0].str.strip() if 0 in parts.columns else pd.Series("", index=region_series.index)
    departamento = parts[1].str.strip() if 1 in parts.columns else pd.Series("", index=region_series.index)
    municipio = municipio.replace({"": "Desconocido"})
    departamento = departamento.replace({"": "Desconocido", None: "Desconocido"})
    return municipio, departamento


# --------------------------------------------------------------------------- #
# Limpieza
# --------------------------------------------------------------------------- #
def clean_catalog(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpieza: tipos, fechas, duplicados, rangos físicos, faltantes y outliers."""
    rep: dict = {"filas_originales": len(raw)}
    df = standardize_columns(raw)

    # Guardar región cruda para parsing posterior
    if "municipio" in df.columns:
        df = df.rename(columns={"municipio": "region_raw"})
    else:
        df["region_raw"] = ""

    # 1) Tipos numéricos (coma decimal → punto)
    for c in ("latitud", "longitud", "profundidad_km", "magnitud"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce", utc=True)

    # 2) Faltantes críticos
    n0 = len(df)
    df = df.dropna(subset=["fecha_hora", "latitud", "longitud", "magnitud"])
    rep["eliminadas_por_faltantes_criticos"] = n0 - len(df)

    # 3) Duplicados
    n0 = len(df)
    key = pd.DataFrame({
        "t": df["fecha_hora"], "la": df["latitud"].round(3),
        "lo": df["longitud"].round(3), "m": df["magnitud"].round(1),
    })
    df = df[~key.duplicated(keep="first")]
    rep["duplicados_eliminados"] = n0 - len(df)

    # 4) Rango de magnitud físicamente posible
    n0 = len(df)
    df = df[df["magnitud"].between(-1.0, 10.0)]
    rep["magnitudes_imposibles"] = n0 - len(df)

    # 5) Outliers de profundidad (>700 km o <0)
    bad = (df["profundidad_km"] < 0) | (df["profundidad_km"] > 700)
    rep["profundidades_atipicas_corregidas"] = int(bad.sum())
    df.loc[bad, "profundidad_km"] = np.nan

    # 6) Imputación de profundidad
    rep["profundidades_imputadas"] = int(df["profundidad_km"].isna().sum())
    cell = (df["latitud"] * 4).round().astype(int).astype(str) + "_" + (df["longitud"] * 4).round().astype(int).astype(str)
    cell_med = df.groupby(cell)["profundidad_km"].transform("median")
    df["profundidad_km"] = df["profundidad_km"].fillna(cell_med).fillna(df["profundidad_km"].median())

    df = df.sort_values("fecha_hora").reset_index(drop=True)
    rep["filas_tras_limpieza"] = len(df)
    return df, rep


# --------------------------------------------------------------------------- #
# Ingeniería de características
# --------------------------------------------------------------------------- #
def _log10_sum_pow10(e: np.ndarray, axis: int = 1) -> np.ndarray:
    """log10(sum(10**e)) numéricamente estable."""
    emax = e.max(axis=axis, keepdims=True)
    return (emax + np.log10(np.sum(10.0 ** (e - emax), axis=axis, keepdims=True))).squeeze(axis)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye 18+ variables del modelo + clasificación + territorio."""
    df = df.sort_values("fecha_hora").reset_index(drop=True).copy()
    n = len(df)
    if n < 20:
        raise ValueError(f"Muy pocos eventos tras la limpieza ({n}); se necesitan al menos 20.")

    out = df.rename(columns={"fecha_hora": "fecha_hora_utc"})
    out["anio"] = out["fecha_hora_utc"].dt.year
    out["mes"] = out["fecha_hora_utc"].dt.month

    # Parsear municipio y departamento desde region_raw
    mun, dep = _parse_region(out["region_raw"])
    out["municipio"] = mun
    out["departamento"] = dep

    # --- Clasificación sísmica (Leve / Medio / Grave)
    out["clasificacion"] = pd.cut(
        out["magnitud"], bins=SEVERITY_BINS, labels=SEVERITY_LABELS, right=False,
    ).astype(str)

    # --- Energía (Gutenberg-Richter): log10 E[J] = 4.8 + 1.5 M
    out["energia_log10_J"] = 4.8 + 1.5 * out["magnitud"]

    # --- Variables temporales
    t = (out["fecha_hora_utc"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    dt_h = np.diff(t, prepend=np.nan) / 3600.0
    dt_h[0] = np.nanmedian(dt_h)
    out["dt_prev_log_h"] = np.log1p(np.clip(dt_h, 0, None))
    left = np.searchsorted(t, t - 30 * 86400, side="left")
    out["tasa_30d"] = np.arange(n) - left
    hora_local = ((out["fecha_hora_utc"].dt.hour + out["fecha_hora_utc"].dt.minute / 60) - 5) % 24
    out["hora_sin"] = np.sin(2 * np.pi * hora_local / 24)
    out["hora_cos"] = np.cos(2 * np.pi * hora_local / 24)

    # --- Vecindad espacial (haversine)
    coords = np.radians(out[["latitud", "longitud"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    out["dens_25km"] = tree.query_radius(coords, r=25.0 / EARTH_KM, count_only=True) - 1

    k = min(K_VECINOS, n - 1)
    dist, ind = tree.query(coords, k=k + 1)
    ind = ind[:, 1:]
    r_k = np.maximum(dist[:, -1] * EARTH_KM, 0.5)
    out["dens_knn_log"] = np.log10(k / (np.pi * r_k**2))

    mag = out["magnitud"].to_numpy()[ind]
    dep_arr = out["profundidad_km"].to_numpy()[ind]
    en = out["energia_log10_J"].to_numpy()[ind]
    out["mag_media_vec"] = mag.mean(axis=1)
    out["mag_max_vec"] = mag.max(axis=1)
    out["prof_media_vec"] = dep_arr.mean(axis=1)
    out["prof_std_vec"] = dep_arr.std(axis=1)
    out["energia_vec_log10"] = _log10_sum_pow10(en)

    # --- Variables geográficas genéricas
    centroid_lat = out["latitud"].mean()
    centroid_lon = out["longitud"].mean()
    out["dist_centroide_km"] = haversine_km(
        out["latitud"], out["longitud"], centroid_lat, centroid_lon
    )
    prof_min = out["profundidad_km"].min()
    prof_max = out["profundidad_km"].max()
    out["profundidad_norm"] = (out["profundidad_km"] - prof_min) / (prof_max - prof_min + 1e-9)

    cols = [
        "fecha_hora_utc", "anio", "mes", "municipio", "departamento",
        "region_raw", "clasificacion",
    ] + FEATURES
    return out[cols]


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo: crudo → tabla única limpia con 18+ variables de modelado."""
    clean, report = clean_catalog(raw)
    final = engineer_features(clean)
    report["variables_modelo"] = len(FEATURES)
    return final, report
