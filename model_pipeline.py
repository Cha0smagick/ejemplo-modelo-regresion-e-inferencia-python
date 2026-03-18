import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from utils import guardar_grafico, print_step, print_substep, print_info, print_kv_table, print_success

def preprocesamiento(df, target_col='revenue'):
    """Prepara los datos para el modelado científico."""
    print_step(4, "PREPROCESAMIENTO DE DATOS")
    print_substep(f"Definiendo variable objetivo: {target_col}")
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    print_substep("Dividiendo conjunto de datos Train/Test (80/20)...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def pipeline_regresion_q4(X_train, y_train, X_test, y_test):
    """Modelo de regresión polinómica grado 4."""
    print_step(5, "MODELADO: REGRESIÓN POLINÓMICA Q4")
    
    # 1. Transformación Polinómica
    print_substep("Aplicando Transformación Polinómica (Grado 4)...")
    poly = PolynomialFeatures(degree=4, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 2. Escalado
    print_substep("Estandarizando características (StandardScaler)...")
    scaler = StandardScaler()
    X_train_poly_scaled = scaler.fit_transform(X_train_poly)
    X_test_poly_scaled = scaler.transform(X_test_poly)
    
    # 3. OLS con constante
    X_train_ols = sm.add_constant(X_train_poly_scaled)
    X_test_ols = sm.add_constant(X_test_poly_scaled)
    
    print_substep("Ajustando modelo OLS (Ordinary Least Squares)...")
    model = sm.OLS(y_train, X_train_ols).fit()
    
    # 4. Predicciones y Métricas
    y_pred = model.predict(X_test_ols)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Mostrar resumen del modelo (primeras líneas si es muy largo, o completo)
    # print(model.summary()) # Comentado para no saturar si es muy largo, se guarda en txt
    print_success("Modelo ajustado correctamente.")
    
    print_kv_table({
        "Modelo": "Regresión Polinómica Grado 4",
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
    plt.show()

def exportar_modelo(model, filename="modelo_regresion_Q4.pkl"):
    """Guarda el modelo entrenado."""
    print_substep(f"Guardando objeto del modelo en disco ({filename})...")
    joblib.dump(model, filename)
    print_success("Modelo exportado.")