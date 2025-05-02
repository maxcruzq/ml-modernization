# scripts/select_best_knn.py

# Importar librerias necesarias
import mlflow
import pandas as pd

# Nombre del experimento que queremos analizar
experiment_name = "Iris_KNN_AutoTest_CV"

# Obtebemos el ID del experimento
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Obtenemos todos los runs
runs = mlflow.search_runs(experiment_ids=experiment_id)

# Convertimos a DataFrame y aseguramos los datatype correctos
df = runs[['params.n_neighbors', 'metrics.mean_accuracy', 'metrics.std_accuracy']]
df['params.n_neighbors'] = df['params.n_neighbors'].astype(int)
df['metrics.mean_accuracy'] = df['metrics.mean_accuracy'].astype(float)
df['metrics.std_accuracy'] = df['metrics.std_accuracy'].astype(float)

# Ordenamos por mayor mean_accuracy, menor std_accuracy y menor n_neighbors
df_sorted = df.sort_values(
    by=['metrics.mean_accuracy', 'metrics.std_accuracy', 'params.n_neighbors'],
    ascending=[False, True, True]
)

# Mostramos los top 3
print("\nTop 3 modelos segun criterios combinados:")
print(df_sorted.head(3))

# Mostramos el mejor seleccionado
best = df_sorted.iloc[0]
print("\nMejor modelo seleccionado:")
print(f"n_neighbors = {best['params.n_neighbors']}")
print(f"mean_accuracy = {best['metrics.mean_accuracy']}")
print(f"std_accuracy = {best['metrics.std_accuracy']}")
