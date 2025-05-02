# scripts/train_multiple.py

# Importamos las librerias necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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


def train_and_evaluate(X_train, X_test, y_train, y_test, n_neighbors):
    """
    Entrena el modelo KNN y calcula accuracy
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return model, acc


def main():
    # Configuramos el nombre del experimento
    mlflow.set_experiment("Iris_KNN_AutoTest")

    # Cargamos y preparamos los datos
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Recorremos valores de n_neighbors de 1 a 30
    for k in range(1, 31):
        with mlflow.start_run():
            mlflow.set_tag("phase", "diagnostic_test")

            # Entrenamos y evaluamos el modelo
            model, acc = train_and_evaluate(
                X_train, X_test, y_train, y_test, n_neighbors=k)

            # BLOQUE DE DIAGNOSTICO
            predictions = model.predict(X_test)
            print(f"Run con k = {k}")
            print(f"Test set classes: {set(y_test)}")
            print(f"Predicciones únicas: {set(predictions)}")
            print(f"Cantidad muestras test: {len(y_test)}\n")

            # Loggeamos los resultados en MLflow
            mlflow.log_param("n_neighbors", k)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "knn_model")

            print(f"Run completado: n_neighbors = {k}, accuracy = {acc}")


if __name__ == "__main__":
    main()
