# scripts/promote_to_production.py

# Importar librerias necesarias
import mlflow
import argparse

# Argumentos de lineas de comando
parser = argparse.ArgumentParser(
    description="Promote MLflow model version to Production")
parser.add_argument('--model', type=str, required=True,
                    help='Model name in MLflow')
parser.add_argument('--version', type=int, required=True,
                    help='Model version to promote')
args = parser.parse_args()

# Inicializamos cliente
client = mlflow.tracking.MlflowClient()

# Movemos a Production
client.transition_model_version_stage(
    name=args.model,
    version=args.version,
    stage="Production"
)

print(f"Modelo '{args.model}' version {args.version} ahora esta en Production")
