"""
data_loader.py
==============
Carga, limpieza e ingeniería de características del catálogo sísmico del
Servicio Geológico Colombiano (SGC) para Santander.

Flujo:
    raw (CSV SGC o simulación) -> standardize_columns -> clean_catalog
        -> engineer_features -> DataFrame de UNA sola tabla (1 fila = 1 sismo)

Todas las funciones son puras (sin efectos secundarios) para poder cachearlas
con ``st.cache_data`` en la aplicación.
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

# Caja geográfica aproximada de Santander (no es el polígono oficial).
SANTANDER_BBOX = {"lat_min": 5.70, "lat_max": 8.20, "lon_min": -74.60, "lon_max": -72.40}

BUCARAMANGA = (7.119, -73.122)        # exposición urbana principal
MESA_LOS_SANTOS = (6.755, -73.105)    # zona foco del Nido Sísmico de Bucaramanga
MESA_RADIO_KM = 25.0                  # radio que define la "zona Mesa de Los Santos"
K_VECINOS = 30                        # vecinos para estadísticas locales

# Municipios de referencia (coordenadas aproximadas de cabeceras). Se usan
# SOLO si el CSV no trae municipio. Ajústelas si dispone de un shapefile DANE.
MUNICIPIOS_REF = {
    "Bucaramanga": (7.119, -73.122), "Floridablanca": (7.062, -73.086),
    "Girón": (7.068, -73.169), "Piedecuesta": (6.987, -73.053),
    "Los Santos": (6.755, -73.105), "Zapatoca": (6.815, -73.268),
    "Lebrija": (7.117, -73.218), "San Gil": (6.556, -73.134),
    "Socorro": (6.467, -73.260), "Barrancabermeja": (7.065, -73.854),
    "Barbosa": (5.934, -73.617), "Vélez": (6.014, -73.674),
    "Málaga": (6.702, -72.732), "Charalá": (6.287, -73.150),
    "Sabana de Torres": (7.390, -73.500), "Rionegro": (7.265, -73.150),
    "Cepitá": (6.753, -73.012), "Aratoca": (6.695, -73.022),
}

# Variables usadas por el modelo no supervisado (18 > 15 exigidas)
FEATURES = [
    "magnitud", "profundidad_km", "latitud", "longitud",
    "energia_log10_J", "dt_prev_log_h", "tasa_30d", "hora_sin", "hora_cos",
    "dens_25km", "dens_knn_log", "mag_media_vec", "mag_max_vec",
    "prof_media_vec", "prof_std_vec", "energia_vec_log10",
    "dist_bucaramanga_km", "dist_mesa_km",
]

# --------------------------------------------------------------------------- #
# Diccionario de datos (fuente única; la app y el .md se generan desde aquí)
# --------------------------------------------------------------------------- #
DATA_DICTIONARY = [
    # name, tipo, unidad, fuente, justificación
    ("fecha_hora_utc", "datetime64[ns, UTC]", "UTC", "SGC (directa)", "Marca temporal del origen del sismo; base de intervalos y tasas."),
    ("anio", "int", "año", "Derivada", "Año del evento; permite filtrar y revisar tendencias."),
    ("mes", "int", "1-12", "Derivada", "Mes del evento; revisa estacionalidad del registro."),
    ("municipio", "str", "—", "SGC o referencia", "Municipio reportado (SGC) o municipio de referencia más cercano si falta."),
    ("zona_territorial", "str", "—", "Derivada", "'Mesa de Los Santos' (≤25 km del foco) vs 'Resto de Santander'."),
    ("magnitud", "float", "Mw/ML (adim.)", "SGC (directa)", "Tamaño del sismo; principal indicador de severidad."),
    ("profundidad_km", "float", "km", "SGC (directa)", "Profundidad del foco; separa sismicidad profunda (nido) de la superficial (corteza), más dañina."),
    ("latitud", "float", "grados", "SGC (directa)", "Posición norte-sur; define la estructura espacial."),
    ("longitud", "float", "grados", "SGC (directa)", "Posición este-oeste; define la estructura espacial."),
    ("energia_log10_J", "float", "log10(J)", "Derivada", "Energía sísmica aproximada: log10 E = 4.8 + 1.5·M (Gutenberg-Richter). Linealiza el rango enorme de energías."),
    ("dt_prev_log_h", "float", "log(1+horas)", "Derivada", "Intervalo desde el sismo anterior; capta enjambres y réplicas (intervalos cortos)."),
    ("tasa_30d", "int", "eventos/30 días", "Derivada", "Eventos regionales en los 30 días previos; actividad reciente."),
    ("hora_sin", "float", "[-1, 1]", "Derivada", "Hora local (UTC-5) codificada cíclicamente (seno)."),
    ("hora_cos", "float", "[-1, 1]", "Derivada", "Hora local (UTC-5) codificada cíclicamente (coseno)."),
    ("dens_25km", "int", "eventos", "Derivada", "Conteo de sismos a ≤25 km: densidad de eventos por radio fijo."),
    ("dens_knn_log", "float", "log10(ev/km²)", "Derivada", "Densidad local con los 30 vecinos más cercanos; adaptativa a zonas densas/dispersas."),
    ("mag_media_vec", "float", "magnitud", "Derivada", "Magnitud media de los 30 vecinos; nivel típico de sismicidad local."),
    ("mag_max_vec", "float", "magnitud", "Derivada", "Magnitud máxima entre los vecinos; potencial sísmico local observado."),
    ("prof_media_vec", "float", "km", "Derivada", "Profundidad media del vecindario; régimen tectónico local."),
    ("prof_std_vec", "float", "km", "Derivada", "Dispersión de profundidad del vecindario; mezcla de regímenes somero/profundo."),
    ("energia_vec_log10", "float", "log10(J)", "Derivada", "Energía acumulada de los 30 vecinos (suma log-estable); carga energética local."),
    ("dist_bucaramanga_km", "float", "km", "Territorial", "Distancia a Bucaramanga: proxy de exposición del área metropolitana."),
    ("dist_mesa_km", "float", "km", "Territorial", "Distancia a la Mesa de Los Santos: proximidad al foco del Nido Sísmico."),
    ("cluster", "int", "—", "Salida del modelo", "Etiqueta de clúster (-1 = ruido en DBSCAN). Se añade al exportar."),
    ("nivel_riesgo", "str", "Bajo/Medio/Alto", "Salida del modelo", "Nivel interpretativo por clúster (índice compuesto, no es una amenaza probabilística)."),
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


# Alias aceptados para cada columna del catálogo (el SGC cambia nombres entre exportes)
COL_ALIASES = {
    "fecha_hora": ["fecha_hora_utc", "fecha_hora", "fecha_utc", "fecha", "origin_time", "time"],
    "hora": ["hora_utc", "hora"],
    "latitud": ["latitud", "lat", "latitude"],
    "longitud": ["longitud", "lon", "lng", "longitude"],
    "profundidad_km": ["profundidad_km", "profundidad", "depth", "prof"],
    "magnitud": ["magnitud", "mag", "magnitude", "ml"],
    "municipio": ["municipio_departamento", "municipio", "localizacion", "lugar", "ubicacion"],
}
REQUIRED = ("fecha_hora", "latitud", "longitud", "profundidad_km", "magnitud")


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Renombra columnas del CSV del SGC a nombres canónicos."""
    norm = {c: _norm(c) for c in raw.columns}
    found: dict[str, str] = {}
    for target, aliases in COL_ALIASES.items():
        hit = None
        for a in aliases:                       # 1) coincidencia exacta
            hit = next((c for c, n in norm.items() if n == a and c not in found.values()), None)
            if hit:
                break
        if hit is None:
            for a in aliases:                   # 2) coincidencia por prefijo
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
    if "hora" in df.columns:  # fecha y hora en columnas separadas
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


# --------------------------------------------------------------------------- #
# Limpieza
# --------------------------------------------------------------------------- #
def clean_catalog(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpieza: tipos, fechas, duplicados, rangos físicos, faltantes y outliers."""
    rep: dict = {"filas_originales": len(raw)}
    df = standardize_columns(raw)

    # 1) Homogeneización de tipos y unidades (coma decimal -> punto)
    for c in ("latitud", "longitud", "profundidad_km", "magnitud"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    df["fecha_hora"] = pd.to_datetime(df["fecha_hora"], errors="coerce", utc=True)

    # 2) Faltantes críticos (sin fecha, coordenadas o magnitud no se puede imputar)
    n0 = len(df)
    df = df.dropna(subset=["fecha_hora", "latitud", "longitud", "magnitud"])
    rep["eliminadas_por_faltantes_criticos"] = n0 - len(df)

    # 3) Duplicados (mismo instante, posición y magnitud)
    n0 = len(df)
    key = pd.DataFrame({
        "t": df["fecha_hora"], "la": df["latitud"].round(3),
        "lo": df["longitud"].round(3), "m": df["magnitud"].round(1),
    })
    df = df[~key.duplicated(keep="first")]
    rep["duplicados_eliminados"] = n0 - len(df)

    # 4) Rango físico/geográfico
    n0 = len(df)
    b = SANTANDER_BBOX
    df = df[df["latitud"].between(b["lat_min"], b["lat_max"]) & df["longitud"].between(b["lon_min"], b["lon_max"])]
    rep["fuera_de_santander_bbox"] = n0 - len(df)
    n0 = len(df)
    df = df[df["magnitud"].between(-1.0, 9.0)]
    rep["magnitudes_imposibles"] = n0 - len(df)

    # 5) Outliers de profundidad (>300 km o <0 son errores de captura en Santander):
    #    se anulan y se imputan. NO se eliminan magnitudes altas: son sismos reales.
    bad = (df["profundidad_km"] < 0) | (df["profundidad_km"] > 300)
    rep["profundidades_atipicas_corregidas"] = int(bad.sum())
    df.loc[bad, "profundidad_km"] = np.nan

    # 6) Imputación de profundidad: mediana de la celda 0.25° y, si no hay, mediana global
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
    """Construye las 18 variables del modelo + variables territoriales."""
    df = df.sort_values("fecha_hora").reset_index(drop=True).copy()
    n = len(df)
    if n < 20:
        raise ValueError(f"Muy pocos eventos tras la limpieza ({n}); se necesitan al menos 20.")

    out = df.rename(columns={"fecha_hora": "fecha_hora_utc"})
    out["anio"] = out["fecha_hora_utc"].dt.year
    out["mes"] = out["fecha_hora_utc"].dt.month

    # --- Energía (Gutenberg-Richter): log10 E[J] = 4.8 + 1.5 M
    out["energia_log10_J"] = 4.8 + 1.5 * out["magnitud"]

    # --- Variables temporales
    t = (out["fecha_hora_utc"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()
    dt_h = np.diff(t, prepend=np.nan) / 3600.0
    dt_h[0] = np.nanmedian(dt_h)
    out["dt_prev_log_h"] = np.log1p(np.clip(dt_h, 0, None))
    left = np.searchsorted(t, t - 30 * 86400, side="left")
    out["tasa_30d"] = np.arange(n) - left                      # eventos previos en 30 días
    hora_local = ((out["fecha_hora_utc"].dt.hour + out["fecha_hora_utc"].dt.minute / 60) - 5) % 24
    out["hora_sin"] = np.sin(2 * np.pi * hora_local / 24)
    out["hora_cos"] = np.cos(2 * np.pi * hora_local / 24)

    # --- Vecindad espacial (haversine)
    coords = np.radians(out[["latitud", "longitud"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    out["dens_25km"] = tree.query_radius(coords, r=25.0 / EARTH_KM, count_only=True) - 1

    k = min(K_VECINOS, n - 1)
    dist, ind = tree.query(coords, k=k + 1)                    # la col. 0 es el propio evento
    ind = ind[:, 1:]
    r_k = np.maximum(dist[:, -1] * EARTH_KM, 0.5)               # km al k-ésimo vecino
    out["dens_knn_log"] = np.log10(k / (np.pi * r_k**2))

    mag = out["magnitud"].to_numpy()[ind]
    dep = out["profundidad_km"].to_numpy()[ind]
    en = out["energia_log10_J"].to_numpy()[ind]
    out["mag_media_vec"] = mag.mean(axis=1)
    out["mag_max_vec"] = mag.max(axis=1)
    out["prof_media_vec"] = dep.mean(axis=1)
    out["prof_std_vec"] = dep.std(axis=1)
    out["energia_vec_log10"] = _log10_sum_pow10(en)

    # --- Variables territoriales
    out["dist_bucaramanga_km"] = haversine_km(out["latitud"], out["longitud"], *BUCARAMANGA)
    out["dist_mesa_km"] = haversine_km(out["latitud"], out["longitud"], *MESA_LOS_SANTOS)
    out["zona_territorial"] = np.where(
        out["dist_mesa_km"] <= MESA_RADIO_KM, "Mesa de Los Santos", "Resto de Santander"
    )

    # Municipio: el del SGC si existe; si no, el de referencia más cercano
    names = list(MUNICIPIOS_REF)
    ref = np.array([MUNICIPIOS_REF[m] for m in names])
    d = haversine_km(out["latitud"].to_numpy()[:, None], out["longitud"].to_numpy()[:, None],
                     ref[None, :, 0], ref[None, :, 1])
    nearest = np.array(names)[d.argmin(axis=1)]
    if "municipio" in out.columns:
        mun = out["municipio"].astype(str).str.split(r"[-,/]").str[0].str.strip()
        mun = mun.replace({"": np.nan, "nan": np.nan, "None": np.nan})
        out["municipio"] = mun.fillna(pd.Series(nearest, index=out.index))
    else:
        out["municipio"] = nearest

    cols = ["fecha_hora_utc", "anio", "mes", "municipio", "zona_territorial"] + FEATURES
    return out[cols]


def build_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Pipeline completo: crudo -> tabla única limpia con 18 variables de modelado."""
    clean, report = clean_catalog(raw)
    final = engineer_features(clean)
    report["variables_modelo"] = len(FEATURES)
    return final, report


# --------------------------------------------------------------------------- #
# Simulación realista (Nido Sísmico de Bucaramanga + corteza + Magdalena Medio)
# --------------------------------------------------------------------------- #
def generate_mock_catalog(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    """
    Catálogo sintético con el formato de columnas del SGC e imperfecciones
    deliberadas (duplicados, faltantes, profundidades erróneas) para demostrar
    la limpieza. Magnitudes ~ Gutenberg-Richter (b=1). NO son datos reales.
    """
    rng = np.random.default_rng(seed)
    comp = rng.choice(4, size=n, p=[0.55, 0.20, 0.15, 0.10])
    lat, lon, dep, mmin = (np.zeros(n) for _ in range(4))

    m = comp == 0   # Nido Sísmico de Bucaramanga (profundo, muy activo)
    lat[m] = rng.normal(6.82, 0.07, m.sum()); lon[m] = rng.normal(-73.10, 0.06, m.sum())
    dep[m] = np.clip(rng.normal(150, 12, m.sum()), 100, 185); mmin[m] = 1.7
    m = comp == 1   # Mesa de Los Santos / Cañón del Chicamocha (somero)
    lat[m] = rng.normal(6.75, 0.08, m.sum()); lon[m] = rng.normal(-73.00, 0.08, m.sum())
    dep[m] = rng.gamma(4, 4, m.sum()); mmin[m] = 1.4
    m = comp == 2   # traza de la Falla Bucaramanga-Santa Marta (somera, lineal)
    lat[m] = rng.uniform(6.3, 7.6, m.sum())
    lon[m] = -73.05 - 0.2 * (lat[m] - 6.8) + rng.normal(0, 0.04, m.sum())
    dep[m] = rng.gamma(3, 5, m.sum()); mmin[m] = 1.4
    m = comp == 3   # Magdalena Medio / Barrancabermeja
    lat[m] = rng.normal(7.10, 0.25, m.sum()); lon[m] = rng.normal(-73.80, 0.15, m.sum())
    dep[m] = rng.gamma(4, 6, m.sum()) + 5; mmin[m] = 1.5

    mag = np.clip(mmin - np.log10(1 - rng.random(n)), None, 6.4).round(1)

    t0, t1 = pd.Timestamp("2018-01-01").timestamp(), pd.Timestamp("2025-12-31").timestamp()
    t = rng.uniform(t0, t1, n)
    swarm = rng.random(n) < 0.08                                   # enjambres / réplicas
    t[swarm] = t[rng.integers(0, n, swarm.sum())] + rng.exponential(48 * 3600, swarm.sum())

    df = pd.DataFrame({
        "FECHA - HORA UTC": pd.to_datetime(t, unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "LATITUD": lat.round(3), "LONGITUD": lon.round(3),
        "PROFUNDIDAD (Km)": dep.round(1), "MAGNITUD": mag,
    })
    # Imperfecciones deliberadas
    df.loc[rng.random(n) < 0.03, "PROFUNDIDAD (Km)"] = np.nan
    df.loc[rng.random(n) < 0.003, "MAGNITUD"] = np.nan
    df.loc[rng.random(n) < 0.002, "PROFUNDIDAD (Km)"] = 999.0
    df = pd.concat([df, df.sample(int(0.02 * n), random_state=seed)], ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
