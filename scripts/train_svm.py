# scripts/train_svm.py

# Importar librerias necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np

# Crear o apuntar el experimento en MLflow
mlflow.set_experiment("Iris_SVM_CV")

# Cargamos el dataset
iris = load_iris()
X, y = iris.data, iris.target

# Instanciamos el modelo
model = SVC()

# "Entrenamos" y registramos en MLflow
with mlflow.start_run(run_name="SVM"):
    mlflow.set_tag("phase", "cross-validation")

    # Calculamos las metricas de interes
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    # Registramos las metricas en MLflow
    mlflow.log_param("model", "SVM")
    mlflow.log_param("kernel", "rbf")
    mlflow.log_metric("mean_accuracy", mean_acc)
    mlflow.log_metric("std_accuracy", std_acc)
    mlflow.sklearn.log_model(model, "model_artifact")

    print(f"SVM completado: mean_acc = {mean_acc}, std_acc = {std_acc}")
