[English](README.md) | **Español**

# Satisfacción de pasajeros: un 0.961 en el leaderboard que en realidad es un modelo de 0.994

![CI](https://github.com/Rxyxs/airline-satisfaction-kaggle/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.7-green)
![CatBoost](https://img.shields.io/badge/CatBoost-GPU-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

Competencia de Kaggle [Playground Series S6E10, Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10):
699.635 pasajeros con etiqueta y 299.844 por predecir, evaluados con ROC AUC. Las filas son
sintéticas. Se generaron a partir de una encuesta real a 129.880 pasajeros de aerolíneas de
EE.UU. ([Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)).

Armé el pipeline habitual de una competencia: cinco modelos, CV de 5 folds, dos blends y siete
envíos hechos desde una sola corrida. Después evalué cada modelo también sobre la encuesta
**real**. Los modelos son mucho mejores de lo que muestra el leaderboard. Un stack que saca 0.961
en la competencia saca **0.994 sobre los pasajeros reales**, y esa brecha se comporta como si
cerca del 3% de las etiquetas de la competencia estuvieran invertidas al azar.

**Mejor puntaje público en Kaggle: 0.96051** (stack de cuatro modelos de boosting; CV 0.96109).
Puesto 1.192 de 2.914 equipos el 2026-10-24. El primero está en 0.96407 y el equipo mediano en
0.95959.

**Página del proyecto:** https://rxyxs.github.io/airline-satisfaction-kaggle/

## Resultados

Los siete archivos salieron de una sola corrida y se enviaron juntos. No se ajustó nada mirando el
leaderboard. "Encuesta real" son los 129.880 pasajeros originales. Los modelos nunca entrenaron con
ellos, y cada número es la predicción promedio de los cinco modelos de fold.

| Envío | CV 5 folds | Kaggle público | Encuesta real |
|---|---|---|---|
| Regresión logística (one-hot, escalada) | 0.92156 | 0.92227 | 0.92000 |
| LightGBM, las 21 columnas tal cual | 0.95884 | 0.95813 | 0.99300 |
| LightGBM + features de valores | 0.96064 | 0.96017 | 0.99327 |
| XGBoost (GPU) + features de valores | 0.96079 | 0.96037 | 0.99333 |
| CatBoost (GPU), cada columna también como categoría | 0.96069 | 0.96038 | 0.99354 |
| Promedio de rangos de los cuatro modelos de árboles | 0.96097 | 0.96029 | 0.99375 |
| **Stack logístico de los cuatro modelos de árboles** | **0.96109** | **0.96051** | 0.99369 |

La desviación estándar entre folds es 0.0004-0.0005 en todos los modelos de árboles. El CV y el
leaderboard público coinciden con diferencias de hasta 0.0007 y ordenan los modelos casi igual.

![modelos](docs/img/models.png)

## Hallazgos

**1. Las etiquetas de la competencia parecen ~3% más ruidosas que las reales.** Un modelo
entrenado solo con filas sintéticas ordena a los pasajeros reales casi tan bien (0.9937) como un
LightGBM entrenado con la encuesta real y evaluado en un 20% reservado de ella (0.9950). La
estructura está en los datos sintéticos. Lo que limita el puntaje son las etiquetas. Invertí al azar
una fracción de las etiquetas de la encuesta real y volví a evaluar el stack. Con 3.0% invertidas,
su AUC sobre pasajeros reales es 0.963. Con 3.5%, es 0.958. El CV de la competencia (0.961) queda
entre ambos, así que las etiquetas sintéticas se comportan como **cerca de un 3% de inversiones
aleatorias**. Visto así, todo el leaderboard está apretado contra un piso de ruido. El primer equipo
está 0.0036 sobre mi stack y el equipo mediano 0.0009 bajo él.

![ruido](docs/img/noise.png)

**2. Tratar los valores como categorías vale +0.0018, y no es solo un artefacto sintético.** Los
modelos "de valores" ven cada valor exacto (y cada par de valores de dos columnas) como una
categoría, mediante target encoding fuera de fold o el manejo categórico propio de CatBoost. Eso
sube el CV de 0.9588 a 0.9606-0.9608 y el puntaje público de 0.9581 a 0.9602-0.9604. Mi hipótesis
era que estos modelos aprendían manías del generador, lo que empeoraría el resultado sobre datos
reales. No pasa. También mejoran sobre la encuesta real (0.9930 a 0.9933-0.9935).

**3. Agregar la encuesta real como filas de entrenamiento no ayuda.** Sumar los 129.880 pasajeros
originales a cada fold de entrenamiento mueve el CV de 0.95884 a 0.95866. El generador cambió los
datos lo suficiente como para que las filas reales sean otra distribución. El 91% de los pasajeros
sintéticos salió a la hora, contra 56% en la encuesta. La satisfacción con vuelos atrasados es más
baja en los datos sintéticos (33% vs 38% con atrasos de 16-60 minutos), y se eliminó la nota
"Inflight service" de la encuesta. Un modelo entrenado solo con la encuesta saca 0.9549 sobre las
filas de la competencia, menos que cualquier modelo de boosting entrenado con las propias filas de
la competencia.

![cambio](docs/img/shift.png)

**4. El CV fue una guía confiable.** Cada modelo de boosting y cada blend perdió 0.0003-0.0007 del CV
al leaderboard público, así que las diferencias entre ellos se mantuvieron. Los dos blends se
diferenciaron en 0.0001 en CV y 0.0002 en el leaderboard, en la misma dirección. Con 700 mil filas
de entrenamiento, el optimismo del early stopping sobre el fold de validación (estándar en stacking
fuera de fold) es pequeño e igual para todos los modelos.

## Método

- **Datos** (`src/airline/data.py`): 21 features. 13 son notas de servicio de 0 a 5, más edad,
  distancia del vuelo, dos atrasos (292 atrasos de llegada faltantes, que los árboles reciben como
  faltantes) y cuatro columnas de texto. La encuesta original se lleva al mismo esquema.
- **Folds**: un solo `StratifiedKFold(5, shuffle, seed 0)` compartido por todos los modelos, los
  blends y el estudio.
- **Features de valores** (`src/airline/features.py`): la frecuencia de cada valor en train+test
  (sin etiquetas), más `TargetEncoder` de sklearn (cross-fitting de 5 folds dentro de cada fold de
  entrenamiento) sobre las 21 columnas y los 210 pares de columnas como claves de texto. El
  encoder solo ve la parte de entrenamiento del fold.
- **Modelos** (`src/airline/models.py`): early stopping sobre el fold de validación (200 rondas;
  300 en CatBoost). LightGBM con 63 hojas y lr 0.05. XGBoost con profundidad 7 en CUDA. CatBoost con
  profundidad 7 en GPU, con las 21 columnas como texto categórico junto a las cuatro numéricas
  originales.
- **Blends** (`src/airline/blend.py`): un promedio de rangos sin pesos y una regresión logística
  sobre los logits de los cuatro modelos. El CV del stack se valida cruzado sobre las filas fuera de
  fold. Pesos finales: CatBoost 0.42, XGBoost 0.27, LightGBM-valores 0.27, LightGBM-crudo 0.04.
- **Estudio con la encuesta real** (`src/airline/study.py`): entrenamiento solo con el original,
  filas originales agregadas a cada fold, comparación de distribuciones y la curva de inversión de
  etiquetas (5 inversiones aleatorias por tasa, tasas de 0 a 8% en pasos de 0.5%).

## Reproducir

```bash
python -m venv venv && venv/Scripts/pip install -r requirements.txt   # ruta de Windows
# requiere una API key de Kaggle y haber aceptado las reglas de la competencia
PYTHONPATH=src python -m airline.data      # descarga ambos datasets en data/
PYTHONPATH=src python -m airline.run       # 5 modelos x 5 folds (~37 min; XGBoost/CatBoost usan GPU)
PYTHONPATH=src python -m airline.blend     # blends + submissions/*.csv
PYTHONPATH=src python -m airline.study     # estudio con la encuesta real
PYTHONPATH=src python -m airline.figures
PYTHONPATH=src python -m airline.submit stack   # opcional
PYTHONPATH=src pytest -q tests
```

Los tiempos son de una RTX 4060 Ti y una CPU de 16 hilos: LightGBM-crudo 1,6 min, LightGBM-valores
12,6, XGBoost 9,5, CatBoost 12,8. Todas las métricas están en `reports/*.json`.

## Datos y licencia

- Datos de la competencia: [playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10),
  bajo las reglas de la competencia. No se redistribuyen aquí (los CSV están en `.gitignore`).
  Descargados el 2026-10-23.
- Encuesta original: [teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction),
  licencia indicada como "other". No se redistribuye. Descargada el 2026-10-23.
- `submissions/` (en `.gitignore`) se regenera con `airline.blend`.
- Código: MIT, ver [LICENSE](LICENSE).
