

## Estructura

- `src/paper1_minimal/`: logica del experimento.
- `scripts/run_minimal_experiment.py`: entrada principal.
- `tests/`: tests pequenos de consistencia.
- `outputs/`: carpeta de resultados generados localmente.

## Instalacion

```bash
python -m pip install -e .
```

## Ejecucion

```bash
python scripts/run_minimal_experiment.py
```

Esto genera:

- `outputs/evaluation_by_condition.csv`
- `outputs/reference_instability_detail.csv`
- `outputs/reference_instability_summary.csv`
- `outputs/rule_condition_summary.csv`
- `outputs/rule_test_detail.csv`
- `outputs/rule_benchmark_summary.csv`
- `outputs/report.json`

## Diseño experimental minimo

- Dataset: grafo sintetico pequeno con `5` seeds por defecto.
- Baselines:
  - `persistence`
  - `moving_average`
  - `neighbor_last`
  - `linear_trend`
- Referencias perturbadas:
  - `clean`
  - `lag_p1`
  - `lag_p3`
  - `node_shift_1hop`
  - `missing_30`
  - `support_drop_30`
- Familia plausible por defecto para la regla robusta:
  - `node_shift_1hop`

## Lectura recomendada

El repo esta pensado para mostrar una idea concreta:

- el ganador bajo referencia limpia no tiene por que seguir siendo el mejor bajo una referencia plausible pero imperfecta;
- si las predicciones se dejan fijas y solo cambia la referencia, el cambio de ganador puede medirse como un problema de decision y no de reentrenamiento.

## Tests

```bash
python -m unittest discover -s tests
```


