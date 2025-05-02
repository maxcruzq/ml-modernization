# scripts/train.py

# Importamos las librerias necesarias
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def load_data():
    """
    Carga el dataset Iris y lo separa en variables X (features) e y (target)
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Separa los datos en entrenamiento y prueba
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train, y_train, n_neighbors):
    """
    Crea y entrena un modelo KNN con los datos de entrenamiento
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evalua el modelo usando accuracy sobre el set de prueba
    """
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc


def main():
    # Cargamos y separamos los datos
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Definimos el parametro a experimentar
    n_neighbors = 6

    # Asignar nombre el experimento que creamremos en MLflow
    mlflow.set_experiment("Iris_KNN_SingleRun")
    # Iniciamos un experimento de MLflow
    with mlflow.start_run():
        # Entrenamos el modelo
        model = train_model(X_train, y_train, n_neighbors)

        # Evaluamos el modelo
        accuracy = evaluate_model(model, X_test, y_test)

        # Loggeamos en MLflow
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model_artifact")

        print(
            f"Modelo entrenado con n_neighbors = {n_neighbors}, precision = {accuracy}")


if __name__ == "__main__":
    main()
