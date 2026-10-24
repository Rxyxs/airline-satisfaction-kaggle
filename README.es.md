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

Armé el pipeline habitual de una competencia: árboles con boosting, una red neuronal, CV de 5
folds y blends apilados. Después evalué cada modelo también sobre la encuesta **real**. Los
modelos son mucho mejores de lo que muestra el leaderboard. Un stack que saca 0.961 en la
competencia saca **0.994 sobre los pasajeros reales**, y esa brecha se comporta como si cerca
del 3% de las etiquetas de la competencia estuvieran invertidas al azar.

Esa encuesta real permite además una prueba que el leaderboard no puede hacer. Para cada truco
de features puedo preguntar si su ganancia se mantiene con pasajeros reales o si viene solo de
cómo se generaron las filas sintéticas. Algunos de los trucos populares no pasan esa prueba.

**Mejor puntaje público en Kaggle: 0.96055** (stack de cinco modelos de boosting y un RealMLP;
CV 0.96113). Puesto 1.179 de 2.924 equipos el 2026-10-24. El primero está en 0.96407 y el equipo
mediano en 0.95958.

**Página del proyecto:** https://rxyxs.github.io/airline-satisfaction-kaggle/

## Resultados

Los primeros siete archivos salieron de una sola corrida y se enviaron juntos. Los dos últimos
(RealMLP y el stack extendido) se eligieron por CV y se enviaron juntos unas horas después. No se
ajustó nada mirando el leaderboard. "Encuesta real" son los 129.880 pasajeros originales. Los
modelos nunca entrenaron con ellos, y cada número es la predicción promedio de los cinco
modelos de fold.

| Envío | CV 5 folds | Kaggle público | Encuesta real |
|---|---|---|---|
| Regresión logística (one-hot, escalada) | 0.92156 | 0.92227 | 0.92000 |
| LightGBM, las 21 columnas tal cual | 0.95884 | 0.95813 | 0.99300 |
| LightGBM + features de valores | 0.96064 | 0.96017 | 0.99327 |
| XGBoost (GPU) + features de valores | 0.96079 | 0.96037 | 0.99333 |
| CatBoost (GPU), cada columna también como categoría | 0.96064* | 0.96038 | 0.99353 |
| RealMLP (GPU) + features de valores | 0.96051 | 0.95995 | 0.99331 |
| Promedio de rangos de los cuatro modelos de árboles | 0.96096* | 0.96029 | 0.99375 |
| Stack logístico de los cuatro modelos de árboles | 0.96107* | 0.96051 | 0.99369 |
| **Stack logístico extendido** (cuatro árboles + RealMLP + LightGBM con todos los grupos de features) | **0.96113** | **0.96055** | 0.99376 |

La desviación estándar entre folds es 0.0004-0.0005 en todos los modelos. El CV y el
leaderboard público coinciden con diferencias de hasta 0.0007 y ordenan los modelos casi igual.

\* CatBoost en GPU no es reproducible bit a bit. Al volver a correr el pipeline, su CV pasó de
0.96069 a 0.96064, y los blends que lo usan se movieron 0.00001-0.00002. Los puntajes de Kaggle
corresponden a los archivos de la primera corrida. Todos los demás modelos se reproducen
exactamente.

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
está 0.0035 sobre mi mejor envío y el equipo mediano 0.0010 bajo él.

![ruido](docs/img/noise.png)

**2. Algunos trucos de features solo sirven con datos sintéticos.** Los notebooks públicos de esta
competencia acumulan features: valores exactos como categorías, conteos de valores, los dígitos
de cada número, tasas tomadas de la encuesta original. Agregué cada grupo por separado al LightGBM
con las columnas crudas, sobre los mismos folds, y evalué cada modelo de fold dos veces. Para saber
qué cuenta como diferencia real, repetí tres configuraciones con otras tres semillas de LightGBM.
La desviación estándar entre semillas es 0.00004 en ambos puntajes. La tabla muestra las corridas
con semilla 0.

| Grupo agregado | Ganancia en CV de la competencia | Ganancia en la encuesta real |
|---|---|---|
| Conteos de valores | +0.00067 | +0.00013 |
| Dígitos decimales de los cuatro números | +0.00061 | 0.00000 |
| Target encoding de los valores exactos | +0.00128 | **-0.00025** |
| Target encoding de valores y de los 210 pares de columnas | +0.00182 | +0.00018 |
| Tasa de satisfacción de cada valor y par en la encuesta real | +0.00104 | +0.00024 |
| Conteos + ambos target encodings (la vista "de valores") | +0.00181 | +0.00027 |
| Todo lo anterior | +0.00180 | +0.00013 |

![ablación](docs/img/ablation.png)

- **Los dígitos son un artefacto puro del generador.** El último dígito de la distancia de vuelo
  no dice nada de un pasajero real. Promediando cuatro semillas, los dígitos suman +0.0005 en CV
  (unas 12 desviaciones estándar de semilla) y -0.00007 en la encuesta real (dentro de 2).
- **El target encoding de valores exactos, por sí solo, empeora el resultado con pasajeros
  reales.** Es la mayor ganancia de un solo grupo en el CV de la competencia, y deja al modelo
  peor sobre la encuesta real (-0.00027 promediando cuatro semillas, unas 6 desviaciones
  estándar). El modelo aprende tasas para edades y distancias individuales que el generador
  inventó. Al agregar los pares de columnas, el efecto sobre la encuesta real pasa a ser positivo
  (+0.00018): las combinaciones llevan señal que sí se transfiere.
- **Las tasas de la propia encuesta real ayudan en ambos lados**, pero menos en la competencia
  (+0.0010) que los encodings ajustados con las etiquetas de la competencia.
- **Juntarlo todo no suma nada.** El LightGBM con todos los grupos empata con la vista de valores
  en CV (0.96064) y es peor sobre la encuesta real (0.99313 vs 0.99327).

En el leaderboard, un truco que explota al generador y uno que captura el comportamiento de los
pasajeros se ven iguales. Solo la encuesta real permite distinguirlos.

**3. Agregar la encuesta real como filas de entrenamiento no ayuda.** Sumar los 129.880 pasajeros
originales a cada fold de entrenamiento mueve el CV de 0.95884 a 0.95866. El generador cambió los
datos lo suficiente como para que las filas reales sean otra distribución. El 91% de los pasajeros
sintéticos salió a la hora, contra 56% en la encuesta. La satisfacción con vuelos atrasados es más
baja en los datos sintéticos (33% vs 38% con atrasos de 16-60 minutos), y se eliminó la nota
"Inflight service" de la encuesta. Un modelo entrenado solo con la encuesta saca 0.9549 sobre las
filas de la competencia, menos que cualquier modelo de boosting entrenado con las propias filas de
la competencia.

![cambio](docs/img/shift.png)

**4. Una red neuronal aporta diversidad, no un mejor puntaje.** RealMLP (pytabkit) sobre la vista
de valores saca 0.96051 en CV, cerca de los modelos de boosting (0.96064-0.96079). Sus predicciones
fuera de fold se correlacionan 0.969-0.974 (Spearman) con los otros modelos de valores, tan bajo
como CatBoost (0.969-0.975) y por debajo de los dos LightGBM de valores y XGBoost entre sí
(0.983-0.984). El stack extendido pone la mayor parte de su peso en esos dos (CatBoost 0.37,
RealMLP 0.36). Todo esto sube el stack en 0.00006 de CV y 0.00004 en el leaderboard. Con un piso
de ruido tan cerca, agregar modelos casi no rinde.

**5. El CV fue una guía confiable.** Cada modelo y cada blend perdió 0.0003-0.0007 del CV al
leaderboard público, así que las diferencias entre ellos se mantuvieron. El promedio de rangos y
el stack de cuatro árboles se diferenciaron en 0.0001 en CV y 0.0002 en el leaderboard, en la
misma dirección. Con 700 mil filas de entrenamiento, el optimismo del early stopping sobre el fold
de validación (estándar en stacking fuera de fold) es pequeño e igual para todos los modelos.

## Método

- **Datos** (`src/airline/data.py`): 21 features. 13 son notas de servicio de 0 a 5, más edad,
  distancia del vuelo, dos atrasos (292 atrasos de llegada faltantes, que los árboles reciben como
  faltantes) y cuatro columnas de texto. La encuesta original se lleva al mismo esquema.
- **Folds**: un solo `StratifiedKFold(5, shuffle, seed 0)` compartido por todos los modelos, los
  blends, el estudio y la ablación.
- **Features de valores** (`src/airline/features.py`): la frecuencia de cada valor en train+test
  (sin etiquetas), más `TargetEncoder` de sklearn (cross-fitting de 5 folds dentro de cada fold de
  entrenamiento) sobre las 21 columnas y los 210 pares de columnas como claves de texto. El
  encoder solo ve la parte de entrenamiento del fold.
- **Modelos** (`src/airline/models.py`): early stopping sobre el fold de validación (200 rondas;
  300 en CatBoost). LightGBM con 63 hojas y lr 0.05. XGBoost con profundidad 7 en CUDA. CatBoost con
  profundidad 7 en GPU, con las 21 columnas como texto categórico junto a las cuatro numéricas
  originales. RealMLP: 512-256-128 con SiLU, ensamble de 4, 4 épocas fijas (sin early stopping).
- **Blends** (`src/airline/blend.py`): un promedio de rangos sin pesos y una regresión logística
  sobre los logits de los modelos. El CV del stack se valida cruzado sobre las filas fuera de
  fold. Pesos del stack de cuatro árboles: CatBoost 0.40, XGBoost 0.28, LightGBM-valores 0.28,
  LightGBM-crudo 0.05. Stack extendido: CatBoost 0.37, RealMLP 0.36, XGBoost 0.17,
  LightGBM-valores 0.12, LightGBM-todo 0.08, LightGBM-crudo -0.01.
- **Ablación** (`src/airline/ablation.py`): un LightGBM por grupo de features sobre los folds
  compartidos, cada uno evaluado en CV y sobre la encuesta real. Cuando se evalúa la propia
  encuesta, el grupo de tasas de la encuesta se calcula fuera de fold dentro de ella, así que
  ninguna fila ve su propia etiqueta. Las configuraciones crudo, target encoding de valores
  exactos y dígitos se repiten con tres semillas más.
- **Estudio con la encuesta real** (`src/airline/study.py`): entrenamiento solo con el original,
  filas originales agregadas a cada fold, comparación de distribuciones y la curva de inversión de
  etiquetas (5 inversiones aleatorias por tasa, tasas de 0 a 8% en pasos de 0.5%).

## Reproducir

```bash
python -m venv venv && venv/Scripts/pip install -r requirements-nn.txt   # ruta de Windows; requirements.txt solo omite RealMLP
# requiere una API key de Kaggle y haber aceptado las reglas de la competencia
PYTHONPATH=src python -m airline.data      # descarga ambos datasets en data/
PYTHONPATH=src python -m airline.run       # 7 modelos x 5 folds (XGBoost, CatBoost y RealMLP usan GPU)
PYTHONPATH=src python -m airline.blend     # blends + submissions/*.csv
PYTHONPATH=src python -m airline.study     # estudio con la encuesta real
PYTHONPATH=src python -m airline.ablation  # ablación por grupo de features (~75 min)
PYTHONPATH=src python -m airline.figures
PYTHONPATH=src python -m airline.submit stack   # opcional
PYTHONPATH=src pytest -q tests
```

En una RTX 4060 Ti con una CPU de 16 hilos y sin nada más corriendo, los primeros cinco modelos
tardan unos 37 minutos (LightGBM-crudo 1,6 min, LightGBM-valores 12,6, XGBoost 9,5, CatBoost 12,8).
Los minutos en `reports/cv.json` son de la última corrida, que compartió la máquina con otros
trabajos. Todas las métricas están en `reports/*.json`.

## Datos y licencia

- Datos de la competencia: [playground-series-s6e10](https://www.kaggle.com/competitions/playground-series-s6e10),
  bajo las reglas de la competencia. No se redistribuyen aquí (los CSV están en `.gitignore`).
  Descargados el 2026-10-23.
- Encuesta original: [teejmahal20/airline-passenger-satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction),
  licencia indicada como "other". No se redistribuye. Descargada el 2026-10-23.
- `submissions/` (en `.gitignore`) se regenera con `airline.blend`.
- Código: MIT, ver [LICENSE](LICENSE).
