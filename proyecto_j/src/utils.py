import yaml
import json
import os


def load_config(path):
    """Carga configuración desde YAML o JSON."""
    ext = os.path.splitext(path)[-1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in [".yml", ".yaml"]:
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Formato de configuración no soportado: {ext}")
