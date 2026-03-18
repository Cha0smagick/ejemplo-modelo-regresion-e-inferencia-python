import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from utils import guardar_grafico, print_step, print_substep, print_info, print_warning, print_success, print_kv_table

def analisis_exploratorio(df):
    """Realiza un EDA básico y retorna estadísticos."""
    print_step(2, "ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    
    # Verificar nulos
    print_substep("Verificando valores nulos...")
    if df.isnull().sum().sum() > 0:
        print_warning("Valores nulos detectados. Imputando con la media/moda...")
        df = df.fillna(df.mean(numeric_only=True))
        # Para columnas no numéricas
        df = df.fillna(method='ffill').fillna(method='bfill')
    else:
        print_success("Dataset limpio: No se encontraron valores nulos.")
    
    # Estadísticos descriptivos
    print_substep("Calculando estadísticas descriptivas...")
    desc_stats = df.describe()
    
    # Mostrar tabla resumen de descriptivos para algunas columnas clave
    cols_mostrar = desc_stats.columns[:3] # Primeras 3 para no saturar consola
    print_info("Resumen Estadístico (Primeras cols)", cols_mostrar.tolist())
    
    # Matriz de correlación
    print_substep("Generando matriz de correlación de Pearson...")
    plt.figure(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Matriz de Correlación de Pearson")
    plt.tight_layout()
    guardar_grafico("1_matriz_correlacion.png")
    print_success("Gráfico de correlación generado y guardado.")
    plt.show()
    
    return df, desc_stats, corr

def analisis_inferencial_clt(df, target_col):
    """
    Paso 3: Análisis Inferencial (TLC y T-Test).
    """
    print_step(3, "ANÁLISIS INFERENCIAL (TLC & HIPÓTESIS)")
    
    # 1. Visualización del Teorema del Límite Central (Bootstrapping)
    p_val = None
    print_substep(f"Validando Teorema del Límite Central para '{target_col}'...")
    medias_muestrales = []
    n_samples = 1000
    sample_size = 50 
    
    for _ in range(n_samples):
        muestra = df[target_col].sample(sample_size, replace=True)
        medias_muestrales.append(muestra.mean())
        
    plt.figure(figsize=(10, 6))
    sns.histplot(medias_muestrales, kde=True, color='purple', edgecolor='black')
    plt.title(f"Teorema del Límite Central: Distribución de Medias de '{target_col}'")
    plt.xlabel("Media Muestral")
    plt.ylabel("Frecuencia")
    plt.axvline(np.mean(medias_muestrales), color='red', linestyle='--', label='Gran Media')
    plt.legend()
    guardar_grafico("2_teorema_limite_central.png")
    plt.show()
    print_success("Gráfico de distribución muestral (TLC) generado.")

    # 2. Prueba de Hipótesis (T-Test)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if cat_cols:
        group_col = cat_cols[0]
        top_vals = df[group_col].value_counts().head(2).index.tolist()
        
        if len(top_vals) >= 2:
            val_a, val_b = top_vals[0], top_vals[1]
            print_substep(f"Prueba de Hipótesis (T-Test) en variable: '{group_col}'")
            print_info("Grupos a comparar", f"{val_a} vs {val_b}")
            
            grupo_a = df[df[group_col] == val_a][target_col]
            grupo_b = df[df[group_col] == val_b][target_col]
            t_stat, p_val = stats.ttest_ind(grupo_a, grupo_b, equal_var=False)
            
            print_kv_table({
                "Grupo A": val_a,
                "Grupo B": val_b,
                "Estadístico t": f"{t_stat:.4f}",
                "Valor P (P-value)": f"{p_val:.4e}",
                "Significativo (alpha=0.05)": "SÍ" if p_val < 0.05 else "NO"
            }, title="Resultados T-Test")
    
    return p_val

def filtrar_mejor_canal(df, target_col, p_val, col_canal='channel'):
    """Filtra el DataFrame si hay evidencia estadística."""
    # print_step no necesario aquí si se considera parte del flujo lógico, 
    # pero usaremos substep para indicar la decisión.
    print_substep("Evaluando segmentación por canal rentable...")
    
    if p_val is None or p_val >= 0.05:
        print_info("Decisión", "No se filtra (No hay evidencia estadística suficiente p>=0.05)")
        return df
    
    if col_canal not in df.columns: return df
    
    mejor_canal = df.groupby(col_canal)[target_col].mean().sort_values(ascending=False).index[0]
    print_success(f"Filtrando dataset por canal: '{mejor_canal}' (P-val < 0.05)")
    
    df_filtrado = df[df[col_canal] == mejor_canal].copy()
    # (Opcional) Aquí podrías volver a graficar la comparativa si lo deseas
    
    return df_filtrado