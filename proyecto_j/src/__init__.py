# Este archivo hace que el directorio src sea reconocido como un paquete de Python
# Configuración para imports absolutos
import os
import sys

# Agregar el directorio actual al path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir) 