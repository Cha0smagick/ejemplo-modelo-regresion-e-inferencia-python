import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, print_step, print_substep, print_info, print_kv_table, print_success, print_warning

def preprocesamiento(df, target_col='revenue'):
    """Prepara los datos para el modelado científico."""
    print_step(4, "PREPROCESAMIENTO DE DATOS")
    print_substep(f"Definiendo variable objetivo: {target_col}")
    
    # 1. Ingeniería de Fechas (Feature Engineering)
    # Las tendencias temporales son vitales en e-commerce (navidad, temporadas)
    if 'date' in df.columns:
        print_substep("Extrayendo características temporales (Mes, Año, Día)...")
        # Convertir a datetime con inferencia de formato
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')
        
        df['month'] = df['date_dt'].dt.month.fillna(0).astype(int)
        df['year'] = df['date_dt'].dt.year.fillna(0).astype(int)
        df['dow'] = df['date_dt'].dt.dayofweek.fillna(0).astype(int)
        df = df.drop(columns=['date_dt'])

    # 2. Limpieza: Eliminar ID y Fecha original
    cols_drop = [target_col]
    for col in ['id', 'date']:
        if col in df.columns: cols_drop.append(col)
        
    X = df.drop(columns=cols_drop)
    y = df[target_col]
    
    # 3. Ingeniería de Características: One-Hot Encoding para categorías
    print_substep("Codificando variables categóricas (One-Hot Encoding)...")
    X = pd.get_dummies(X, drop_first=True)
    
    # Imputación profesional de características (rellenar huecos en X)
    # Usamos la mediana porque es resistente a outliers
    print_substep("Imputando valores faltantes en características (Mediana)...")
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    
    print_substep("Dividiendo conjunto de datos Train/Test (80/20)...")
    return train_test_split(X_imputed, y, test_size=0.2, random_state=42)

def pipeline_modelado_avanzado(X_train, y_train, X_test, y_test):
    """Modelo avanzado utilizando Gradient Boosting."""
    print_step(5, "MODELADO: MACHINE LEARNING (GRADIENT BOOSTING)")
    
    # Random Forest no necesita escalado ni creación manual de polinomios.
    # Maneja interacciones no lineales internamente.
    
    # Limpiamos nombres de columnas para evitar errores de caracteres especiales en sklearn
    X_train_final = X_train.copy()
    X_test_final = X_test.copy()

    # Chequeo de seguridad: Evitar R2 negativo por falta de datos
    # MEJORA CLAVE: Transformación Logarítmica del Target
    # Esto estabiliza la varianza para montos monetarios (ingresos)
    print_substep("Aplicando Log-Transform a la variable objetivo (Log-Lin)...")
    y_train_log = np.log1p(y_train.clip(lower=0))
    
    # MEJORA: NO FILTRAR OUTLIERS EN EL TARGET
    # En e-commerce, los 'whales' son reales e importantes. El RF es robusto a ellos.
    print_info("Estrategia Outliers", "Conservando todos los datos (Whales incluidos)")

    # 4. MODELADO CON GRADIENT BOOSTING
    # Boosting suele superar a Random Forest en datos tabulares ruidosos
    print_substep("Entrenando Gradient Boosting Regressor...")
    model = GradientBoostingRegressor(
        n_estimators=500,       # Más árboles, pero más pequeños
        learning_rate=0.05,     # Aprender lento para generalizar mejor
        max_depth=5,            # Profundidad controlada para evitar overfitting
        min_samples_leaf=10,    # Regularización
        loss='absolute_error',  # Clave: Usar error absoluto para ignorar outliers extremos
        random_state=42,
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
    
    print_success(f"Gradient Boosting ajustado. R2 Test: {r2:.4f}")
    
    print_kv_table({
        "Modelo": "Gradient Boosting (LAD Loss)",
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