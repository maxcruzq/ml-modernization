# scripts/train_decision_tree.py

# Importar las librerias necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Crear o apuntar experimento en MLflow
mlflow.set_experiment("Iris_DecisionTree_CV")

# Importamos Dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Instanciamos el modelo
model = DecisionTreeClassifier()

# "Entrenamos" el modelo y registramos en MLflow
with mlflow.start_run(run_name="DecisionTree"):
    mlflow.set_tag("phase", "cross-validation_test")

    # Calculamos metricas de interes
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    # Registramos las informacion en MLflow
    mlflow.log_param("model", "DecisionTree")
    mlflow.log_metric("mean_accuracy", mean_acc)
    mlflow.log_metric("std_accuracy", std_acc)
    mlflow.sklearn.log_model(model, "model_artifact")

    # Imprimimos metricas
    print(
        f"Decision Tree completado: mean_acc = {mean_acc}, std_acc = {std_acc}")
