import os
import glob
import pandas as pd
import kagglehub

def cargar_datos():
    """Descarga y carga el dataset automáticamente."""
    print("--- 1. INGESTA DE DATOS ---")
    
    local_filename = "dataset_ecommerce_local.csv"
    if os.path.exists(local_filename):
        print(f"-> Archivo local encontrado: {local_filename}")
        df = pd.read_csv(local_filename)
        # Asegurar nombres normalizados
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        print("-> Datos cargados exitosamente.")
        print(f"-> Dimensiones: {df.shape}")
        return df

    # Descargar última versión
    path = kagglehub.dataset_download("amineipad/e-commerce-marketing-and-sales-revenue-prediction")
    print(f"Dataset descargado en: {path}")

    # Buscar el archivo CSV en el directorio descargado
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No se encontró ningún archivo CSV en la ruta descargada.")
    
    file_path = csv_files[0]
    print(f"-> Cargando archivo: {os.path.basename(file_path)}")
    df = pd.read_csv(file_path)
    
    # Normalizar nombres de columnas (minusculas, sin espacios)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Guardar copia local para uso futuro
    df.to_csv(local_filename, index=False)
    print(f"-> Copia local guardada como: {local_filename}")
    
    print(f"-> Dimensiones del dataset: {df.shape}")
    print(f"-> Columnas detectadas: {df.columns.tolist()}")
    return df