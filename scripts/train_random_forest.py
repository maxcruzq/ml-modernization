# Scripts/train_random_forest.py

# Importar librerias necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Creamos o apuntamos al experimento en MLflow
mlflow.set_experiment("Iris_RandomForest_CV")

# Importamos el dataset
iris = load_iris()
X, y = iris.data, iris.target

# Instanciamos el modelo
n_estimators = 100
model = RandomForestClassifier(n_estimators=n_estimators)

# "Entrenamos" el modelo y luego registramos en MLflow
with mlflow.start_run(run_name="RandomForest"):
    mlflow.set_tag("phase", "cross-validation")

    # Calculamos metricas de interes
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    # Registramos metricas en MLflow
    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("mean_accuracy", mean_acc)
    mlflow.log_metric("std_accuracy", std_acc)

    # Entrenar explicitamente antes de loggear
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model_artifact")

    print(
        f"Random Forest completado: mean_acc = {mean_acc}, std_acc = {std_acc}")
