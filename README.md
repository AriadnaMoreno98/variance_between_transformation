# variance_between: transformación MML para clasificación de texto en ciberseguridad

Paquete de reproducibilidad del paper que propone `variance_between`
(`VarianceBetweenProjection`), una transformación para el pipeline
Minimalist Machine Learning (MML) aplicado a clasificación binaria de
texto en el dominio de ciberseguridad. El pipeline reduce una representación
TF-IDF a un plano bidimensional interpretable `(z1, z2)` y clasifica
mediante una frontera lineal sobre ese plano.

`variance_between` se compara contra otras cinco transformaciones MML
(`discriminative`, `polarity`, `weighted`, `split_halves`,
`centroid_distance`) y contra siete modelos de referencia clásicos
(regresión logística, SVM lineal, KNN, Naive Bayes, MLP, y las variantes
balanceadas de regresión logística y SVM), sobre dos familias de tareas
de clasificación binaria:

- **CTI-to-MITRE** (`data/cti_is_*.csv`, 15 tareas): clasificación de
  reportes CTI según la táctica ATT&CK asociada.
- **ATT&CK Enterprise** (`data/attck_*.csv`, 10 tareas): clasificación de
  técnicas ATT&CK Enterprise según táctica, plataforma, o condición de
  sub-técnica.

El diseño experimental cubre 25 tareas × 6 transformaciones × 3 valores
de `k` (50, 100, 200), evaluadas con validación cruzada estratificada de
5 particiones (`StratifiedKFold`).

Los resultados y su discusión se presentan en el paper. Este repositorio
provee el código, los datos y el procedimiento necesarios para
reproducirlos y verificarlos de manera independiente.

## Estructura del repositorio

```
src/          código fuente del pipeline MML (representación, selección
              de features, transformaciones, búsqueda de frontera) y de
              los modelos de referencia
data/         conjuntos de datos de las 25 tareas (columnas: text, label)
scripts/      script de ejecución del experimento completo
notebooks/    notebook de análisis (analysis.ipynb), que agrega los
              resultados crudos en las tablas y figuras reportadas
```

Cada módulo de `src/` implementa un contrato común (`fit`/`transform`)
definido en su respectivo `base.py`:

- `src/representations/lexical.py` — `TFIDFRepresentation`.
- `src/selection/selectors.py` — `DMeansSelector`, selección de las `k`
  features con mayor diferencia de medias entre clases.
- `src/transformations/transformations.py` — las seis transformaciones
  evaluadas, incluida `VarianceBetweenProjection`.
- `src/search/search.py` — `HorizontalSearch`, búsqueda exacta de la
  frontera `z2 = threshold` que maximiza `balanced_accuracy` en
  entrenamiento.
- `src/mml/pipeline_v2.py` — `MMLPipeline`, que integra los componentes
  anteriores.
- `src/baselines/baseline_pipeline.py` — interfaz común para los modelos
  de referencia clásicos.
- `src/evaluation/metrics.py` — cálculo de las métricas reportadas.

## Requisitos

Python 3.10 o superior (verificado con 3.13).

## Reproducción del experimento

### 1. Configuración del entorno

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` cubre las dependencias del pipeline y del script de
experimento (`pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`). La
ejecución del notebook de análisis requiere además:

```bash
pip install matplotlib nbclient ipykernel jupyter
```

### 2. Ejecución del experimento completo

```bash
python scripts/run_experiment.py
```

Este comando ejecuta las 6 transformaciones × `k` ∈ {50, 100, 200} con
validación cruzada estratificada (5 particiones) sobre las 25 tareas de
`data/`, en paralelo (4 procesos por defecto), y almacena un CSV por
combinación en `results/runs/`. La ejecución es tolerante a
interrupciones: una combinación ya calculada no se repite.

Dado el volumen de combinaciones (450, cada una con vectorización TF-IDF,
selección de features y validación de 5 particiones), la ejecución
completa puede requerir varias horas; se recomienda ejecutarla en un
entorno con recursos dedicados.

Parámetros disponibles:

```bash
# validación cruzada repetida (5 particiones x 3 repeticiones)
python scripts/run_experiment.py --validator repeated_kfold --n_splits 5 --n_repeats 3

# subconjunto reducido, para verificar que la instalación funciona
python scripts/run_experiment.py --k_values 50 --max_workers 1 --task_prefix attck_tactic_discovery

# inclusión de los modelos de referencia clásicos
python scripts/run_experiment.py --run_baselines --k_values 100
```

### 3. Ejecución del análisis

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```

`notebooks/analysis.ipynb` es la única fuente de las tablas y figuras del
paper: no existen agregaciones manuales fuera del notebook. Cada sección
documenta en markdown qué calcula. El notebook:

- Carga los CSVs generados en el paso anterior.
- Reporta explícitamente la cobertura de combinaciones esperadas frente a
  las encontradas, para detectar corridas incompletas.
- Calcula las tablas y figuras del análisis y las almacena en
  `results/tables/` y `results/figures/`.

El notebook versionado en este repositorio conserva las tablas y figuras
de su última ejecución como referencia, para verificar que una nueva
ejecución produce salidas equivalentes.

## Organización de `results/`

El script y el notebook esperan encontrar los CSVs crudos en:

```
results/experiments/v2_exp1_transformaciones/results/runs/   # resultados MML
results/experiments/v2_baselines/results/runs/                # resultados de referencia
```

Si `scripts/run_experiment.py` se ejecuta sin modificar `--results_dir`,
los CSVs se generan en `results/runs/`. En ese caso, deben moverse (o
enlazarse) a las rutas anteriores antes de ejecutar el notebook, o bien
ajustarse las variables `RUNS_DIR` / `BASELINES_RUNS_DIR` definidas en la
Sección 1.1 y 2.1 del notebook.
