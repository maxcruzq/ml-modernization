# scripts/log_dataset.py

# Importar las librerias necesarias
import mlflow
import pandas as pd
from sklearn.datasets import load_iris

# Cargamos el dataset Iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Guardamos localmente como CSV
dataset_path = "iris_dataset.csv"
df.to_csv(dataset_path, index=False)

# Loggeamos en MLflow como artefacto
mlflow.set_experiment("Iris_Dataset_Logging")

with mlflow.start_run():
    mlflow.log_artifact(dataset_path)
    print(f"Dataset loggeado como artefacto en MLflow: {dataset_path}")
