import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, print_step, print_substep, print_info, print_kv_table, print_success, print_warning

def preprocesamiento(df, target_col='revenue'):
    """Prepara los datos para el modelado científico."""
    print_step(4, "PREPROCESAMIENTO DE DATOS")
    print_substep(f"Definiendo variable objetivo: {target_col}")
    
    # 1. Limpieza: Eliminar ID y Fecha (no predictivos para este modelo)
    cols_drop = [target_col]
    for col in ['id', 'date']:
        if col in df.columns: cols_drop.append(col)
        
    X = df.drop(columns=cols_drop)
    y = df[target_col]
    
    # 2. Ingeniería de Características: One-Hot Encoding para categorías
    print_substep("Codificando variables categóricas (One-Hot Encoding)...")
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0) # Seguridad para nulos post-dummies
    
    print_substep("Dividiendo conjunto de datos Train/Test (80/20)...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def pipeline_modelado_avanzado(X_train, y_train, X_test, y_test):
    """Modelo avanzado utilizando Random Forest Regressor."""
    print_step(5, "MODELADO: MACHINE LEARNING (RANDOM FOREST)")
    
    # Random Forest no necesita escalado ni creación manual de polinomios.
    # Maneja interacciones no lineales internamente.
    
    # Limpiamos nombres de columnas para evitar errores de caracteres especiales en sklearn
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()

    # Chequeo de seguridad: Evitar R2 negativo por falta de datos
    # MEJORA CLAVE: Transformación Logarítmica del Target
    # Esto estabiliza la varianza para montos monetarios (ingresos)
    print_substep("Aplicando Log-Transform a la variable objetivo (Log-Lin)...")
    
    # Filtro de outliers: Suavizamos el umbral a 4 desviaciones estándar para no perder "ballenas"
    y_train_log = np.log1p(y_train.clip(lower=0))
    mask_outliers = np.abs((y_train_log - y_train_log.mean()) / y_train_log.std()) < 4
    X_train_final = X_train_final[mask_outliers]
    y_train_log = y_train_log[mask_outliers]
    print_info("Outliers removidos", f"{len(X_train) - len(X_train_final)}")
    
    # 4. MODELADO CON RANDOM FOREST
    # Usamos un bosque con suficientes árboles para estabilizar la predicción
    print_substep("Entrenando Random Forest (200 árboles)...")
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=15,       # Limitamos profundidad para evitar overfitting extremo
        min_samples_leaf=5, # Regularización: al menos 5 datos por hoja
        random_state=42,
        n_jobs=-1           # Usar todos los procesadores
    )
    model.fit(X_train_final, y_train_log)
    
    # 5. Predicciones y Métricas
    y_pred_log = model.predict(X_test_final)
    # Revertir logaritmo para métricas reales (Exponencial)
    y_pred = np.expm1(y_pred_log)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Hack para que reporting.py pueda leer los nombres de las variables
    model.feature_names_in_ = X_train_final.columns.tolist()
    
    print_success(f"Random Forest ajustado. R2 Test: {r2:.4f}")
    
    print_kv_table({
        "Modelo": "Random Forest Regressor",
        "RMSE (Test)": f"{rmse:,.2f}",
        "R^2 (Test)": f"{r2:.4f}",
        "Observaciones Train": len(X_train),
        "Observaciones Test": len(X_test)
    }, title="Métricas de Desempeño")
    
    return model, y_test, y_pred, rmse, r2

def diagnostico_residuos(y_test, y_pred):
    """Verificación de supuestos del modelo."""
    print_step(6, "DIAGNÓSTICO DE RESIDUOS")
    residuals = y_test - y_pred
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Homocedasticidad
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0], alpha=0.6)
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title("Homocedasticidad: Residuos vs Predichos")
    
    # Normalidad
    sns.histplot(residuals, kde=True, ax=ax[1], color='green')
    ax[1].set_title("Normalidad: Distribución de Residuos")
    
    plt.tight_layout()
    guardar_grafico("4_diagnostico_residuos.png")
    print_success("Gráficos de diagnóstico mostrados.")
    plt.close()

def exportar_modelo(model, filename="modelo_random_forest.pkl"):
    """Guarda el modelo entrenado."""
    print_substep(f"Guardando objeto del modelo en disco ({filename})...")
    joblib.dump(model, filename)
    print_success("Modelo exportado.")