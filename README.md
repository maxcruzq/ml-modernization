# ML Modernization Project

## 1. Nombre del Proyecto

ML Modernization Project  
Modernización y operacionalización de modelos de machine learning usando MLflow, Docker y Git.

---

## 2. Descripción y Objetivos del Proyecto

Este proyecto tiene como propósito demostrar un flujo moderno y reproducible para desarrollar, entrenar, evaluar, registrar, servir y probar modelos de machine learning usando herramientas de MLOps.

Los principales objetivos son:
- Aplicar buenas prácticas de MLOps en un flujo real.
- Entrenar y comparar múltiples modelos clásicos de clasificación usando el dataset Iris.
- Automatizar el registro y selección del mejor modelo usando MLflow.
- Desplegar el modelo ganador como servicio REST usando `mlflow models serve`.
- Validar las predicciones del modelo mediante scripts automáticos (single y batch).
- Integrar todos los servicios y testers bajo una orquestación reproducible usando Docker Compose.
- Mantener un flujo de trabajo limpio, con ramas separadas (`dev` para desarrollo y `main` para producción).

Este repositorio sirve como base para proyectos donde se requiera modernizar pipelines de machine learning hacia flujos más escalables, reproducibles y auditables.

---

## 3. Resumen

Este repositorio contiene scripts, artefactos y configuraciones para:
- Entrenar múltiples modelos de clasificación (KNN, Decision Tree, Random Forest, SVM) usando el dataset Iris.
- Registrar experimentos y métricas usando **MLflow**.
- Seleccionar y promover automáticamente el mejor modelo a producción.
- Servir el modelo como servicio REST vía API usando **MLflow models serve**.
- Orquestar servicios y pruebas automáticas usando **Docker Compose**.
- Mantener un flujo limpio de desarrollo usando ramas `dev` y `main` gestionadas con Git.

Este proyecto es una demostración completa de buenas prácticas en MLOps, despliegue reproducible y automatización.

---

## 4. Estructura del Repositorio

```
ml-modernization/
├── data/                     → Datos de prueba (batch_data.csv)
├── mlruns/                   → Experimentos y runs de MLflow
├── model_artifact/           → Artefactos exportados del modelo (MLmodel, pkl, yaml)
├── scripts/                  → Scripts de entrenamiento, prueba, análisis y promoción
├── venv/                     → Entorno virtual local (excluido de Git)
├── .dockerignore             → Exclusiones para builds Docker
├── .gitignore                → Exclusiones para Git
├── best_model_info.json      → Resultado del modelo seleccionado como mejor
├── docker-compose.yaml       → Orquestación de servicios con Docker Compose
├── Dockerfile                → Build del servicio del modelo
├── Dockerfile-tester-single  → Build del tester de predicciones individuales
├── Dockerfile-tester-batch   → Build del tester de predicciones batch
├── iris_dataset.csv          → Dataset base (iris)
├── README.md                 → Este documento
└── requirements.txt          → Dependencias del proyecto
```

---

## 5. Dataset

- **Nombre**: Iris Dataset
- **Descripción**: Dataset clásico de clasificación que contiene 150 registros de flores Iris con 4 características numéricas:
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Target**: Especie (Iris setosa, versicolor, virginica)
- **Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)

---

## 6. Modelos utilizados

Los siguientes modelos fueron entrenados y comparados:
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Se registra automáticamente el mejor modelo (según precisión) en MLflow y se expone como servicio REST para consumo externo.

---

## 7. Requisitos Previos

- Python 3.9
- Docker
- Docker Compose
- Git

---

## 8. Configuración Local (paso a paso)

### a. Clonar el repositorio

```bash
git clone https://github.com/maxcruzq/ml-modernization.git
cd ml-modernization
```

### b. Crear y activar entorno virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### c. Instalar dependencias

```bash
pip install --no-cache-dir -r requirements.txt
```

### d. Entrenar y registrar modelos

```bash
python scripts/train.py
```

### e. Servir el modelo localmente

```bash
mlflow models serve -m models:/BestIrisModel/Production -p 1234
```

### f. Ejecutar pruebas manuales

```bash
python scripts/test_mlflow.py
python scripts/test_mlflow_batch.py
```

---

## 9. Uso con Docker Compose

### a. Levantar todo con un solo comando

```bash
docker-compose up --build
```

Esto:
- Levanta el contenedor `mlflow-model` que sirve el modelo como API REST.
- Corre automáticamente los testers `mlflow-tester-single` y `mlflow-tester-batch` (incluyen espera de 10 segundos para asegurar que el modelo esté disponible).

### b. Ver resultados en consola

Revisa las salidas del log:
- Predicciones individuales desde `scripts/test_mlflow.py`.
- Predicciones batch desde `scripts/test_mlflow_batch.py`.

---

## 10. Flujo Git recomendado

### a. Desarrollar siempre en `dev`

```bash
git checkout dev
git add .
git commit -m "Descripción del cambio"
git push origin dev
```

### b. Cuando esté listo para producción

```bash
git checkout main
git pull origin main
git merge dev
git push origin main
```

---

## 11. Estado final

Este proyecto es completamente reproducible en cualquier entorno con Docker, permitiendo:
- Entrenamiento y registro automático de modelos.
- Despliegue de modelos como servicio REST.
- Validación automatizada vía scripts.
- Orquestación robusta usando Docker Compose.