# scripts/test_mlflow_batch.py

# Importar librerias necesarias
import requests
import pandas as pd
import json

# Paso 1: Cargar los datos batch desde un archivo CSV

# Definir ubicacion del archivo
data_path = "data/batch_data.csv"

# Leer el archivo CSV en un DataFrame de pandas
df = pd.read_csv(data_path)

# Mostrar en consola los primeros registros
print("Datos cargados para el batch:")
print(df.head())

# Paso 2: Convertir los datos a formato JSON esperado por MLflow

# MLflow espera un JSON con la clave "dataframe_split"
# que tiene las columnas y los datos separados
data_json = json.dumps({
    "dataframe_split": json.loads(df.to_json(orient="split"))
})

# Paso 3: Configurar la URL del endpoint de MLflow

# Nos conectamos al servior local levantando el puerto 1234
url = "http://127.0.0.1:1234/invocations"

# Paso 4: Enviar solicitud POST con el batch completo

# Encabezado de la solicitud, indicando que estamos enviando JSON
headers = {"Content-Type": "application/json"}

try:

    # Realizamos la solicitud POST al endpoint del modelo
    response = requests.post(url, headers=headers, data=data_json)

    # Paso 5: Verificar la respuesta del modelo
    if response.status_code == 200:
        # Si todo salio bien, imprimimos las predicciones recibidas
        predictions = response.json()
        print("Predicciones para el batch:")
        print(predictions)
    else:
        # Si hubo algun error, mostramos el codigo y el mensaje
        print(f"Error al invocar el modelo: {response.status_code}")
        print(response.text)

except requests.exceptions.RequestException as e:
    # Si hubo un error de conexion o red lo mostramos
    print(f"Error al conectar con el servidor MLflow: {e}")
