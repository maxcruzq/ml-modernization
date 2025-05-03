# Usa una imagen base oficial de Python
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias al contenedor
COPY requirements.txt .

# Instala las dependencias especificas
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo el contenido del proyecto al contenedor
COPY . .

# Expone el puerto 1234 (el mismo que usamos localmente)
EXPOSE 1234

# Comando por defecto: levanta MLflow sirviendo el modelo localmente
CMD ["mlflow", "models", "serve", "-m", "model_artifact","models:/BestIrisModel/Production", "--no-conda", "-h", "0.0.0.0", "-p", "1234"]
