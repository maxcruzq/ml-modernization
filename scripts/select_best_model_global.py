# scripts/select_best_model_global.py

# Importar librerias necesarias
import json
import mlflow
import pandas as pd
import json

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
        ascending = [False, True, True]
    else:
        sort_cols = ['metrics.mean_accuracy', 'metrics.std_accuracy']
        ascending = [False, True]

    runs_sorted = runs.sort_values(
        by=sort_cols,
        ascending=ascending
    )

    # Seleccionar el mejor run (primera fila)
    best_run = runs_sorted.iloc[0].copy()  # hacemos una copia segura
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

# Paso 4: Generar archivo JSON con info del mejor modelo
output_data = {
    "run_id": best_overall['run_id'],
    "artifact_path": "model_artifact",
    "model_name": "BestIrisModel"
}

with open("best_model_info.json", "w") as f:
    json.dump(output_data, f, indent=4)

print("\nArchivo 'best_model_info.json' generado con exito.")
print("Contiene: run_id, artifact_path, model_name del mejor modelo global")
