# scripts/register_model.py

# Importamos librerias necesarias
import json
import mlflow

# Leer archivo JSON generado previamente
with open("best_model_info.json", "r") as f:
    data = json.load(f)

run_id = data["run_id"]
artifact_path = data["artifact_path"]
model_name = data["model_name"]

# Construir URI del modelo
model_uri = f"runs:/{run_id}/{artifact_path}"

# Inicializar cliente de MLflow y registrar modelo
client = mlflow.tracking.MlflowClient()

result = mlflow.register_model(model_uri, model_name)

# Mostrar confirmacion
print(f"\nModelo registrado en el Model Registry como '{model_name}'.")
print(f"Run ID: {run_id}")
print(f"Version creada: {result.version}")
print("Revisa MLflow UI para ver su estado.")
