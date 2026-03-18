# Proyecto de Inferencia y Regresión Polinómica
**Análisis de Ingresos en E-commerce para el Politécnico Grancolombiano**

## 📜 Introducción

Este proyecto es un ejercicio académico para la materia **Inferencia y Regresión**. Implementa un pipeline completo de ciencia de datos en Python, desde la ingesta de datos hasta el modelado y la generación de reportes. El objetivo principal es analizar un dataset de e-commerce para predecir los ingresos (`revenue`) utilizando un **modelo de regresión polinómica de grado 2 (Optimizado)**.

El pipeline incluye:
- Descarga automática de datos desde Kaggle.
- Análisis Exploratorio de Datos (EDA).
- Análisis Inferencial, incluyendo una demostración del **Teorema del Límite Central (TLC)** y una **Prueba de Hipótesis (T-Test)**.
- Un paso de filtrado de datos basado en evidencia estadística.
- Entrenamiento y evaluación de un modelo de regresión polinómica (Grado 2 para evitar sobreajuste).
- Pruebas de diagnóstico sobre los residuos del modelo.
- Generación de reportes completos en formatos `.txt` y `.xlsx`.

## 📊 Dataset

- **Nombre**: E-commerce Marketing and Sales Revenue Prediction
- **Fuente**: Kaggle
- **Automatización**: El script `data_loader.py` gestiona la descarga y el almacenamiento en caché del dataset. En la primera ejecución, descargará los datos y guardará una copia local (`dataset_ecommerce_local.csv`) para acelerar ejecuciones futuras.

## 📂 Estructura del Proyecto

El proyecto está modularizado en varios archivos de Python para mayor claridad y mantenibilidad:

- `regresion.py`: El script principal que orquesta todo el pipeline.
- `data_loader.py`: Gestiona la ingesta de datos desde Kaggle.
- `analysis.py`: Contiene funciones para el EDA y el análisis inferencial.
- `model_pipeline.py`: Incluye el preprocesamiento de datos, el entrenamiento del modelo y el diagnóstico de residuos.
- `reporting.py`: Responsable de generar los reportes finales en TXT y Excel.
- `utils.py`: Un conjunto de funciones de ayuda para el logging formateado en consola y para guardar gráficos.
- `config.py`: Configuraciones globales, como estilos de gráficos y directorios de salida.

## ⚙️ Requisitos

Para ejecutar este proyecto, necesitas Python 3 y las siguientes librerías. Puedes instalarlas usando `pip`:

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn kagglehub joblib xlsxwriter
```

## 🚀 Cómo Ejecutar

1.  Clona este repositorio.
2.  Instala las librerías requeridas (ver sección anterior).
3.  Asegúrate de tener tus credenciales de la API de Kaggle configuradas para que `kagglehub` funcione. Puedes hacerlo colocando tu archivo `kaggle.json` en la carpeta `~/.kaggle/` (en Windows, `C:\Users\<TuUsuario>\.kaggle`).
4.  Ejecuta el script principal del pipeline desde tu terminal:

```bash
python regresion.py
```

El script ejecutará todos los pasos secuencialmente e imprimirá un progreso detallado en la consola.

## 👣 Pasos del Pipeline (Explicación Detallada)

La función `main` en `regresion.py` ejecuta los siguientes pasos:

### 1. Ingesta de Datos (`data_loader.py`)
- El proceso comienza llamando a `cargar_datos()`.
- Esta función primero comprueba si existe una copia local `dataset_ecommerce_local.csv`.
- Si no existe, utiliza `kagglehub` para descargar el dataset, lo guarda localmente y lo carga en un DataFrame de pandas.
- Los nombres de las columnas se normalizan automáticamente (minúsculas, guiones bajos en lugar de espacios).

### 2. Análisis Exploratorio de Datos (EDA) (`analysis.py`)
- La función `analisis_exploratorio()` realiza un EDA básico.
- **Valores Nulos**: Comprueba si hay valores faltantes. Si los encuentra, imputa las columnas numéricas con la media y las categóricas usando `forward-fill`.
- **Estadísticas Descriptivas**: Calcula y muestra estadísticas de resumen del dataset.
- **Matriz de Correlación**: Genera y guarda un mapa de calor de la matriz de correlación de Pearson para todas las variables numéricas. El gráfico se guarda en `graficos_resultados/1_matriz_correlacion.png`.

### 3. Análisis Inferencial (`analysis.py`)
- La función `analisis_inferencial_clt()` demuestra conceptos estadísticos clave.
- **Teorema del Límite Central (TLC)**: Valida el TLC visualmente. Toma 1000 muestras aleatorias (con reemplazo) de tamaño 50 de la variable objetivo (`revenue`), calcula la media de cada muestra y grafica la distribución de estas medias. Como predice el TLC, esta distribución se aproxima a una normal. El gráfico se guarda en `graficos_resultados/2_teorema_limite_central.png`.
- **Prueba de Hipótesis (T-Test)**: El script identifica la primera columna categórica (en este caso, `channel`) y realiza una prueba T para muestras independientes. La prueba compara la media de `revenue` entre las dos categorías más frecuentes. El objetivo es determinar si existe una diferencia estadísticamente significativa en los ingresos entre estos dos grupos.

### 4. Filtrado Inteligente (`analysis.py`)
- La función `filtrar_mejor_canal()` toma una decisión basada en datos a partir del resultado de la prueba T.
- Si el p-valor de la prueba T es menor que 0.05, indica una diferencia significativa.
- La función identifica el canal con el ingreso promedio más alto y filtra el DataFrame para conservar solo los datos correspondientes a este canal "óptimo".
- Si el p-valor no es significativo (p >= 0.05), se utiliza el DataFrame original y sin filtrar para el modelado.

### 5. Preprocesamiento y Modelado (`model_pipeline.py`)
- **`preprocesamiento()`**: Los datos se dividen en conjuntos de entrenamiento (80%) y prueba (20%). Solo se utilizan las características numéricas para este modelo.
- **`pipeline_regresion_q2()` (Optimizado)**: Esta es la función principal de modelado. Se utiliza grado 2 en lugar de 4 para mejorar la generalización.
    1.  **Transformación Polinómica**: Utiliza `PolynomialFeatures(degree=2)` para crear términos de interacción y características cuadráticas.
    2.  **Escalado**: Se utiliza `StandardScaler` para estandarizar las características polinómicas recién creadas. Esto es crucial para la estabilidad del modelo.
    3.  **Ajuste del Modelo**: Se ajusta un modelo de Mínimos Cuadrados Ordinarios (OLS) utilizando la librería `statsmodels` sobre los datos de entrenamiento.
    4.  **Evaluación**: El modelo entrenado se utiliza para hacer predicciones sobre el conjunto de prueba. Se calculan y muestran métricas clave como el Error Cuadrático Medio (RMSE) y el R-cuadrado (R²).

### 6. Diagnóstico y Exportación (`model_pipeline.py`)
- **`exportar_modelo()`**: El objeto del modelo `statsmodels` entrenado se guarda en un archivo (`modelo_ecommerce_Q2.pkl`) usando `joblib`.
- **`diagnostico_residuos()`**: Esta función genera gráficos para verificar los supuestos del modelo de regresión.
    - **Homocedasticidad**: Un gráfico de dispersión de los residuos frente a los valores predichos. Un patrón aleatorio alrededor de la línea cero sugiere que se cumple este supuesto.
    - **Normalidad de los Residuos**: Un histograma de los residuos para comprobar si siguen una distribución normal.
    - El gráfico se guarda como `graficos_resultados/4_diagnostico_residuos.png`.

### 7. Generación de Reportes (`reporting.py`)
- Finalmente, `generar_reportes_finales()` crea dos archivos de salida completos:
- **`reporte_analisis_regresion_Q2.txt`**: Un archivo de texto que contiene las métricas de rendimiento (RMSE, R²) y el resumen completo y detallado del modelo OLS de `statsmodels`.
- **`resultados_regresion_Q2_organizados.xlsx`**: Un libro de Excel con múltiples hojas para un análisis fácil:
    - `Métricas`: RMSE y R².
    - `Coeficientes`: Los coeficientes del modelo y sus estadísticas.
    - `Datos`: Los datos (potencialmente filtrados) utilizados para el modelado.
    - `Descriptivos`: Las estadísticas descriptivas del EDA.
    - `Correlaciones`: La matriz de correlación.
    - `Predicciones`: Una comparación lado a lado de los valores reales vs. los predichos y los residuos correspondientes para el conjunto de prueba.

---
*Proyecto realizado como parte del programa de Maestría del Politécnico Grancolombiano.*