import os
import matplotlib.pyplot as plt
from config import PLOTS_DIR

# --- SISTEMA DE LOGGING VISUAL Y TRAZABILIDAD ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(step, title):
    """Imprime un encabezado de paso principal con bordes."""
    border = "═" * 65
    print(f"\n{Colors.HEADER}{Colors.BOLD}╔{border}╗")
    print(f"║ PASO {str(step).zfill(2)}: {title.center(55)} ║")
    print(f"╚{border}╝{Colors.ENDC}")

def print_substep(message):
    """Imprime un sub-paso de proceso."""
    print(f"{Colors.CYAN}   ➤ {message}{Colors.ENDC}")

def print_success(message):
    """Imprime un mensaje de éxito."""
    print(f"{Colors.GREEN}   ✔ {message}{Colors.ENDC}")

def print_warning(message):
    """Imprime una advertencia."""
    print(f"{Colors.WARNING}   ⚠ {message}{Colors.ENDC}")

def print_error(message):
    """Imprime un error."""
    print(f"{Colors.FAIL}   ✖ {message}{Colors.ENDC}")

def print_info(label, value):
    """Imprime un par clave-valor informativo."""
    print(f"   • {Colors.BOLD}{label}:{Colors.ENDC} {value}")

def print_kv_table(data_dict, title="Resumen de Datos"):
    """Imprime un diccionario como una tabla formateada."""
    if not data_dict: return
    width_k = 30
    width_v = 30
    print(f"\n   {Colors.BOLD}{Colors.UNDERLINE}{title}{Colors.ENDC}")
    print(f"   {Colors.BLUE}┌{'─'*width_k}┬{'─'*width_v}┐{Colors.ENDC}")
    for k, v in data_dict.items():
        val_str = str(v)
        # Truncar si es muy largo
        k_str = (k[:width_k-3] + '..') if len(k) > width_k else k
        v_str = (val_str[:width_v-3] + '..') if len(val_str) > width_v else val_str
        print(f"   {Colors.BLUE}│{Colors.ENDC} {k_str.ljust(width_k-2)} {Colors.BLUE}│{Colors.ENDC} {v_str.ljust(width_v-2)} {Colors.BLUE}│{Colors.ENDC}")
    print(f"   {Colors.BLUE}└{'─'*width_k}┴{'─'*width_v}┘{Colors.ENDC}")
# ----------------------------------------------------

def guardar_grafico(nombre_archivo):
    """
    Guarda el gráfico actual en la carpeta designada en config.
    """
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    ruta_completa = os.path.join(PLOTS_DIR, nombre_archivo)
    plt.savefig(ruta_completa, bbox_inches='tight')
    print_success(f"Gráfico guardado en: {ruta_completa}")