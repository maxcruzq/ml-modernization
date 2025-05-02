# scripts/anaylize_mlflow_results.py

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Nombre del experimento que queremos analizar
experiment_name = "Iris_KNN_AutoTest_CV"

# Obtenemos el ID del experimento
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(experiment_name)
experiment_id = experiment.experiment_id

# Obtenemos todos los runs
runs = mlflow.search_runs(experiment_ids=experiment_id)

# Convertimos a DataFrame y ordenamos por n_neighbors
df = runs[['params.n_neighbors', 'metrics.mean_accuracy', 'metrics.std_accuracy']]
df['params.n_neighbors'] = df['params.n_neighbors'].astype(int)
df = df.sort_values('params.n_neighbors')

print(df)

# Graficar mean accuracy vs n_neighbors
plt.figure()
plt.plot(
    df['params.n_neighbors'],
    df['metrics.mean_accuracy'],
    marker='o'
)
plt.xlabel('n_neighbors')
plt.ylabel('Mean Accuracy')
plt.title('KNN Mean Accuracy (5-Fold CV)')
plt.show()

# Graficar std_accuracy vs n_neighbors
plt.figure()
plt.plot(
    df['params.n_neighbors'],
    df['metrics.std_accuracy'],
    marker='o',
    color='orange'
)
plt.xlabel('n_neighbors')
plt.ylabel('Std Accuracy')
plt.title('KNN Std Accuracy (5-Fold)')
plt.show()
