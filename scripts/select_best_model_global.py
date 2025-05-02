# scripts/select_best_model_global.py

# Importar librerias necesarias
import mlflow
import pandas as pd

# Lista de experimentos a comparar
experiment_names = [
    "Iris_KNN_AutoTest_CV",
    "Iris_DecisionTree_CV",
    "Iris_RandomForest_CV",
    "Iris_SVM_CV"
]

# Inicializar cliente de MLflow
client = mlflow.tracking.MlflowClient()

# Lista para guardar los mejores runs por modelo
best_runs = []

# Paso 1: Obtener el mejor run de cada experimento (modelo)
for exp_name in experiment_names:
    experiment = client.get_experiment_by_name(exp_name)
    experiment_id = experiment.experiment_id

    # Obtener todos los runs del experimento
    runs = mlflow.search_runs(experiment_ids=experiment_id)

    # Asegurar datatypes correctos
    runs['metrics.mean_accuracy'] = runs['metrics.mean_accuracy'].astype(float)
    runs['metrics.std_accuracy'] = runs['metrics.std_accuracy'].astype(float)

    # Ordenar por mayor mean_accuracy, menor std_accuracy, menor param (si aplica)
    if 'params.n_neighbors' in runs.columns:
        runs['params.n_neighbors'] = runs['params.n_neighbors'].astype(int)
        sort_cols = ['metrics.mean_accuracy',
                     'metrics.std_accuracy', 'params.n_neighbors']
    else:
        sort_cols = ['metrics.mean_accuracy', 'metrics.std_accuracy']

    runs_sorted = runs.sort_values(
        by=sort_cols,
        ascending=[False, True, True]
    )

    # Seleccionar el mejor run (primera fila)
    best_run = runs_sorted.iloc[0]
    best_run['experiment_name'] = exp_name
    best_runs.append(best_run)

# Paso 2: Crear DataFrame resumen
best_df = pd.DataFrame(best_runs)

print("\nMejores modelos por experimento:")
print(best_df[['experiment_name', 'params.model',
      'metrics.mean_accuracy', 'metrics.std_accuracy']])

# Paso 3: Seleccionar el mejor modelo global
best_overall = best_df.sort_values(
    by=['metrics.mean_accuracy', 'metrics.std_accuracy'],
    ascending=[False, True]
).iloc[0]

print("\nMejor modelo global seleccionado:")
print(f"Modelo: {best_overall['params.model']}")
print(f"Desde experimento: {best_overall['experiment_name']}")
print(f"mean_accuracy: {best_overall['metrics.mean_accuracy']}")
print(f"std_accuracy: {best_overall['metrics.std_accuracy']}")

# Paso 4: Registrar en el Model Registry
model_name = "BestIrisModel"
model_uri = f"runs:/{best_overall['run_id']}/model_artifact"

# Crear o actualizar registro
mlflow.register_model(model_uri, model_name)
print(
    f"\nModelo registrado en el Model Registry como '{model_name}'. Revisa MLflow UI para ver su estado.")
