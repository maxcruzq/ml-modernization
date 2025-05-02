# scripts/train_multiple.py

# Importamos las librerias necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np


def load_data():
    """
    Carga el dataset Iris y lo separa de X (features) e y (target)
    """
    iris = load_iris()
    return iris.data, iris.target


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Divide los datos en set de entrenamiento y prueba
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_and_evaluate(X, y, n_neighbors, cv_folds=5):
    """
    Entrena el modelo KNN usando K-Fold Cross-Validation
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Realizamos Cross-Validation
    scores = cross_val_score(model, X, y, cv=cv_folds, scoring='accuracy')

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    # Entrenar explicitamente el modelo sobre todos los datos antes de loggear
    model.fit(X, y)

    return model, mean_acc, std_acc


def main():
    # Configuramos el nombre del experimento
    mlflow.set_experiment("Iris_KNN_AutoTest_CV")

    # Cargamos y preparamos los datos
    X, y = load_data()

    # Recorremos valores de n_neighbors de 1 a 30
    for k in range(1, 31):
        with mlflow.start_run():
            mlflow.set_tag("phase", "cross_validation_test")

            # Entrenamos y evaluamos el modelo
            model, mean_acc, std_acc = train_and_evaluate(X, y, n_neighbors=k)

            # Loggeamos los resultados en MLflow
            mlflow.log_param("n_neighbors", k)
            mlflow.log_metric("mean_accuracy", mean_acc)
            mlflow.log_metric("std_accuracy", std_acc)
            mlflow.sklearn.log_model(model, "model_artifact")

            print(
                f"Run completado: n_neighbors = {k}, mean_acc = {mean_acc}, std_acc= {std_acc}")


if __name__ == "__main__":
    main()
