import os
import glob
import pandas as pd
import kagglehub
from utils import print_step, print_substep, print_success, print_info, print_kv_table

def cargar_datos():
    """Descarga y carga el dataset automáticamente."""
    print_step(1, "INGESTA DE DATOS")
    
    local_filename = "dataset_ecommerce_local.csv"
    if os.path.exists(local_filename):
        print_substep(f"Archivo local encontrado: {local_filename}")
        df = pd.read_csv(local_filename)
        # Asegurar nombres normalizados
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        print_success("Datos cargados exitosamente.")
        print_info("Dimensiones", df.shape)
        return df

    # Descargar última versión
    path = kagglehub.dataset_download("amineipad/e-commerce-marketing-and-sales-revenue-prediction")
    print_substep(f"Dataset descargado de Kaggle en: {path}")

    # Buscar el archivo CSV en el directorio descargado
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en la ruta descargada.")
    
    file_path = csv_files[0]
    print_substep(f"Cargando archivo: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # Normalizar nombres de columnas (minusculas, sin espacios)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Guardar copia local para uso futuro
    df.to_csv(local_filename, index=False)
    print_success(f"Copia local guardada como: {local_filename}")
    
    # Mostrar resumen
    print_info("Dimensiones", str(df.shape))
    print_kv_table({
        "Columnas": len(df.columns),
        "Filas": len(df),
        "Ejemplos Cols": ", ".join(df.columns[:3]) + "..."
    }, title="Metadatos del Dataset")
    
    return df