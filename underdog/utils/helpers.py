import yaml
import os
from typing import Dict, Any

def load_zmq_config() -> Dict[str, Any]:
    """
    Carga la configuración ZMQ/MT5 desde el archivo mt5_credentials.yaml.
    Busca el archivo en 'config/runtime/env/mt5_credentials.yaml' asumiendo que el
    script se ejecuta desde la raíz del proyecto.
    """
    # Construcción robusta de la ruta del archivo de configuración
    # Este método intenta ser agnóstico a la ubicación de ejecución
    
    # Asumimos que la carpeta raíz del proyecto es dos niveles arriba de underdog/utils
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
    config_path = os.path.join(base_dir, 'config', 'runtime', 'env', 'mt5_credentials.yaml')
    
    # Fallback si no encuentra la ruta relativa al script actual
    if not os.path.exists(config_path):
        config_path = os.path.join('config', 'runtime', 'env', 'mt5_credentials.yaml')
        
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Esto lanzará un error crítico si el archivo falta
        raise FileNotFoundError(f"Archivo de configuración ZMQ/MT5 no encontrado en: {config_path}")
    except Exception as e:
        raise IOError(f"Error al leer o parsear el archivo de configuración YAML: {e}")

# Aquí irían otras funciones de ayuda genéricas del proyecto.