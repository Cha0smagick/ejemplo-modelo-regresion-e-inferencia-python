import pandas as pd
from utils import print_step, print_success, print_error, print_substep

def generar_reportes_finales(df_limpio, desc_stats, corr, model, rmse, r2, y_test, y_pred):
    """Genera salidas en TXT y Excel."""
    print_step(7, "GENERACIÓN DE REPORTES Y EXPORTACIÓN")
    
    # --- Reporte TXT ---
    txt_filename = "reporte_analisis_regresion_Q2.txt"
    print_substep("Generando reporte científico en texto plano...")
    try:
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write("REPORTE CIENTÍFICO DE REGRESIÓN POLINÓMICA (Q2 - Optimizado)\n")
            f.write("="*50 + "\n\n")
            f.write(f"RMSE: {rmse:,.2f}\n")
            f.write(f"R^2: {r2:.4f}\n\n")
            f.write("RESUMEN DEL MODELO (Statsmodels)\n")
            f.write(model.summary().as_text())
        print_success(f"Archivo guardado: {txt_filename}")
    except Exception as e:
        print_error(f"Error generando TXT: {e}")

    # --- Reporte EXCEL ---
    xlsx_filename = "resultados_regresion_Q2_organizados.xlsx"
    print_substep("Compilando resultados en libro de Excel...")
    
    # Extraer coeficientes
    try:
        table_data = model.summary().tables[1].data
        coef_df = pd.DataFrame(table_data[1:], columns=table_data[0])
        for col in coef_df.columns:
            coef_df[col] = pd.to_numeric(coef_df[col], errors='ignore')
    except:
        coef_df = pd.DataFrame(model.params, columns=['Coeficiente Estimado'])

    # Predicciones
    pred_df = pd.DataFrame({
        'Real': y_test.values,
        'Predicho': y_pred,
        'Residuo': y_test.values - y_pred
    })

    try:
        with pd.ExcelWriter(xlsx_filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            fmt_header = workbook.add_format({
                'bold': True, 'fg_color': '#4F81BD', 'font_color': 'white', 'border': 1
            })
            
            sheets = {
                'Métricas': pd.DataFrame({'Métrica': ['RMSE', 'R2'], 'Valor': [rmse, r2]}),
                'Coeficientes': coef_df,
                'Datos': df_limpio,
                'Descriptivos': desc_stats,
                'Correlaciones': corr,
                'Predicciones': pred_df
            }
            
            for sheet_name, data in sheets.items():
                data.to_excel(writer, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                for col_num, value in enumerate(data.columns.values):
                    worksheet.write(0, col_num + 1, value, fmt_header)
        print_success(f"Archivo guardado: {xlsx_filename}")
    except Exception as e:
        print_error(f"Error Excel: {e}")