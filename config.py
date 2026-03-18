import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de carpeta para gráficos
PLOTS_DIR = "graficos_resultados"

def configurar_estilos():
    """Aplica el estilo global para los gráficos."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")