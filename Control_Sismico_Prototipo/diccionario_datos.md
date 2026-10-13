# Diccionario de datos

Fuente: Catálogo de la Red Sismológica Nacional — Servicio Geológico Colombiano (SGC). Una fila = un sismo; una columna = una variable.

| Variable | Tipo | Unidad | Fuente | Justificación técnica |
|---|---|---|---|---|
| `fecha_hora_utc` | datetime64[ns, UTC] | UTC | SGC (directa) | Marca temporal del origen del sismo; base de intervalos y tasas. |
| `anio` | int | año | Derivada | Año del evento; permite filtrar y revisar tendencias. |
| `mes` | int | 1-12 | Derivada | Mes del evento; revisa estacionalidad del registro. |
| `municipio` | str | — | SGC o referencia | Municipio reportado (SGC) o municipio de referencia más cercano si falta. |
| `zona_territorial` | str | — | Derivada | 'Mesa de Los Santos' (≤25 km del foco) vs 'Resto de Santander'. |
| `magnitud` | float | Mw/ML (adim.) | SGC (directa) | Tamaño del sismo; principal indicador de severidad. |
| `profundidad_km` | float | km | SGC (directa) | Profundidad del foco; separa sismicidad profunda (nido) de la superficial (corteza), más dañina. |
| `latitud` | float | grados | SGC (directa) | Posición norte-sur; define la estructura espacial. |
| `longitud` | float | grados | SGC (directa) | Posición este-oeste; define la estructura espacial. |
| `energia_log10_J` | float | log10(J) | Derivada | Energía sísmica aproximada: log10 E = 4.8 + 1.5·M (Gutenberg-Richter). Linealiza el rango enorme de energías. |
| `dt_prev_log_h` | float | log(1+horas) | Derivada | Intervalo desde el sismo anterior; capta enjambres y réplicas (intervalos cortos). |
| `tasa_30d` | int | eventos/30 días | Derivada | Eventos regionales en los 30 días previos; actividad reciente. |
| `hora_sin` | float | [-1, 1] | Derivada | Hora local (UTC-5) codificada cíclicamente (seno). |
| `hora_cos` | float | [-1, 1] | Derivada | Hora local (UTC-5) codificada cíclicamente (coseno). |
| `dens_25km` | int | eventos | Derivada | Conteo de sismos a ≤25 km: densidad de eventos por radio fijo. |
| `dens_knn_log` | float | log10(ev/km²) | Derivada | Densidad local con los 30 vecinos más cercanos; adaptativa a zonas densas/dispersas. |
| `mag_media_vec` | float | magnitud | Derivada | Magnitud media de los 30 vecinos; nivel típico de sismicidad local. |
| `mag_max_vec` | float | magnitud | Derivada | Magnitud máxima entre los vecinos; potencial sísmico local observado. |
| `prof_media_vec` | float | km | Derivada | Profundidad media del vecindario; régimen tectónico local. |
| `prof_std_vec` | float | km | Derivada | Dispersión de profundidad del vecindario; mezcla de regímenes somero/profundo. |
| `energia_vec_log10` | float | log10(J) | Derivada | Energía acumulada de los 30 vecinos (suma log-estable); carga energética local. |
| `dist_bucaramanga_km` | float | km | Territorial | Distancia a Bucaramanga: proxy de exposición del área metropolitana. |
| `dist_mesa_km` | float | km | Territorial | Distancia a la Mesa de Los Santos: proximidad al foco del Nido Sísmico. |
| `cluster` | int | — | Salida del modelo | Etiqueta de clúster (-1 = ruido en DBSCAN). Se añade al exportar. |
| `nivel_riesgo` | str | Bajo/Medio/Alto | Salida del modelo | Nivel interpretativo por clúster (índice compuesto, no es una amenaza probabilística). |
