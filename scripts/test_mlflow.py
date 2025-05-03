# Importamos la librería requests para hacer solicitudes HTTP
import requests

# Importamos la librería json para convertir datos Python a formato JSON
import json

# Definimos la URL del servidor MLflow donde está corriendo el modelo
# Este es el endpoint local que levantaste con mlflow models serve
url = "http://127.0.0.1:1234/invocations"

# Preparamos los datos de entrada en formato compatible con el modelo MLflow
# Usamos el formato "dataframe_split" que espera columnas y datos por separado
data = {
    "dataframe_split": {
        "columns": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "data": [
            [5.1, 3.5, 1.4, 0.2],  # Primer ejemplo: características de una flor
            [6.7, 3.1, 4.7, 1.5]   # Segundo ejemplo: otra flor
        ]
    }
}

# Definimos los headers (cabeceras) de la solicitud
# Indicamos que estamos enviando datos en formato JSON
headers = {"Content-Type": "application/json"}

# Hacemos una solicitud POST al endpoint del modelo
# Enviamos los datos serializados como JSON usando json.dumps
response = requests.post(url, headers=headers, data=json.dumps(data))

# Verificamos si la respuesta fue exitosa (código HTTP 200)
if response.status_code == 200:
    # Si todo salió bien, imprimimos las predicciones recibidas desde el modelo
    print("Predicciones del modelo:")
    print(response.json())  # Mostramos el contenido JSON de la respuesta
else:
    # Si hubo algún error, mostramos el código de error y el contenido del mensaje
    print(f"Error al hacer la petición: código {response.status_code}")
    print(response.text)
