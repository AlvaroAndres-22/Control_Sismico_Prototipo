# Guía de pitch (10 minutos) — Sismicidad de Santander con clustering no supervisado

**Mensaje central (una frase):** *"Sin etiquetas, dejamos que los datos del SGC nos digan qué regímenes sísmicos existen en Santander, y los convertimos en un mapa de riesgo comparativo que cualquiera puede explorar."*

| Min | Bloque | Qué decir / mostrar |
|---|---|---|
| 0:00–1:00 | **Problema y gancho** | Santander alberga el Nido Sísmico de Bucaramanga, de los más activos del mundo. Pregunta: ¿qué zonas se comportan distinto y cuáles merecen más atención? |
| 1:00–2:30 | **Datos** | Catálogo de la Red Sismológica Nacional (SGC). Una tabla, una fila por sismo. Limpieza: duplicados, faltantes, profundidades erróneas, UTC→hora local. Muestra la pestaña *Inicio* con el reporte de limpieza. |
| 2:30–4:00 | **Variables (18 > 15)** | Cuatro familias: sísmicas puras (magnitud, profundidad, coordenadas, energía), temporales (intervalo, tasa 30 días, hora cíclica), de vecindad (densidad, magnitud/profundidad/energía de los 30 vecinos) y territoriales (distancia a Bucaramanga y a la Mesa de Los Santos). Muestra el diccionario. |
| 4:00–6:30 | **Modelo (demo en vivo)** | Pestaña *Modelo*: elbow + silueta para elegir k; compara K-Means, GMM y DBSCAN; explica por qué estandarizamos y winsorizamos; muestra PCA y el heatmap de perfiles. Cambia k en vivo. |
| 6:30–8:30 | **Interpretación territorial** | Mapa por nivel de riesgo; frase automática por clúster (profundo = nido, somero = corteza); comparativa Mesa de Los Santos vs. resto. Este es el momento de mayor valor práctico. |
| 8:30–9:30 | **Honestidad técnica** | Índice comparativo, no amenaza probabilística; municipio por cercanía a referencia; silueta moderada es normal en sismicidad continua; datos simulados si no se cargó el CSV real. |
| 9:30–10:00 | **Cierre** | Próximos pasos: shapefile DANE, fallas activas, mapas de aceleración, validación con expertos del SGC. Repite el mensaje central. |

## Puntos fuertes para el jurado
1. **Solidez técnica:** tres algoritmos, métricas (silueta, Davies-Bouldin, Calinski-Harabasz), codo, manejo de outliers y semilla reproducible.
2. **Valor práctico:** cada clúster se traduce a lenguaje territorial (profundidad, energía, densidad, nivel de riesgo).
3. **Prototipo en vivo:** todo interactivo, con exportación del dataset y diccionario.

## Preguntas probables
- *¿Por qué no supervisado?* No existe etiqueta de "riesgo" objetiva en el catálogo.
- *¿Cómo eligieron k?* Codo + silueta + interpretabilidad geológica.
- *¿El riesgo es predictivo?* No; es un índice comparativo de la sismicidad observada.
- *¿Y los outliers?* Los sismos grandes son reales y se conservan; se winsoriza solo la matriz del modelo.

## Antes de subir al escenario
- Prueba con el CSV real del SGC (revisa que los nombres de columnas se reconozcan) y ten un plan B con los datos simulados.
- Ensaya el recorrido de pestañas con cronómetro.
