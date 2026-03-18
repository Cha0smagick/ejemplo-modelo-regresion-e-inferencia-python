"""
Pipeline de Análisis Científico (Random Forest)
Dataset: E-commerce Marketing and Sales Revenue Prediction
MODULARIZADO
"""

import traceback

# Importación de Módulos Locales
from config import configurar_estilos
from data_loader import cargar_datos
from analysis import analisis_exploratorio, analisis_inferencial_clt, filtrar_mejor_canal
from model_pipeline import preprocesamiento, pipeline_modelado_avanzado, diagnostico_residuos, exportar_modelo
from reporting import generar_reportes_finales
from utils import print_warning, print_info

def main():
    try:
        # 0. Config
        configurar_estilos()

        # 1. Carga
        df = cargar_datos()
        
        # 2. Análisis
        df, desc_stats, corr = analisis_exploratorio(df)
        
        # Identificar columna objetivo (Target)
        # El dataset suele tener una columna 'revenue' o 'sales'. Intentamos detectarla.
        posibles_targets = ['revenue', 'sales', 'ventas']
        target_col = next((col for col in df.columns if col in posibles_targets), None)
        
        if not target_col:
            # Fallback: Usar la última columna si no se encuentra nombre conocido
            target_col = df.columns[-1]
            print_warning(f"No se detectó nombre estándar de target. Usando la última columna: '{target_col}'")
        else:
            print_info("Variable objetivo detectada", target_col)
            
        # NUEVO: Eliminar filas donde el Target sea nulo (Vital para no entrenar con basura)
        initial_len = len(df)
        df = df.dropna(subset=[target_col])
        if len(df) < initial_len:
            print_warning(f"Se eliminaron {initial_len - len(df)} filas con Target nulo para garantizar calidad.")

        # 3. Análisis Inferencial (TLC y Hipótesis)
        p_val = analisis_inferencial_clt(df, target_col)
            
        # FILTRADO INTELIGENTE (Basado en la pregunta del usuario)
        # Si hay diferencias significativas, nos enfocamos en el canal más rentable.
        df = filtrar_mejor_canal(df, target_col, p_val, col_canal='channel')

        # 4. Preprocesamiento
        X_train, X_test, y_train, y_test = preprocesamiento(df, target_col=target_col)
        
        # 5. Modelado Avanzado (Random Forest)
        model, y_test_out, y_pred_out, rmse, r2 = pipeline_modelado_avanzado(X_train, y_train, X_test, y_test)
        
        # 6. Exportación y Diagnóstico
        exportar_modelo(model, "modelo_ecommerce_RF.pkl")
        diagnostico_residuos(y_test_out, y_pred_out)
        
        # 7. Generación de Salidas
        generar_reportes_finales(df, desc_stats, corr, model, rmse, r2, y_test_out, y_pred_out)

    except Exception as e:
        traceback.print_exc()
        print(f"\nOcurrió un error en el pipeline: {e}")

if __name__ == "__main__":
    main()
