import os
import matplotlib.pyplot as plt
from config import PLOTS_DIR

def guardar_grafico(nombre_archivo):
    """
    Guarda el gráfico actual en la carpeta designada en config.
    """
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    ruta_completa = os.path.join(PLOTS_DIR, nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight')
    print(f"-> Gráfico guardado en: {ruta_completa}")