import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "proyecto_j", "src")))
import project_j_pipeline as pjp


def test_pipeline_functions_exist():
    assert hasattr(pjp, "cargar_datos")
    assert hasattr(pjp, "limpiar_datos")
    assert hasattr(pjp, "transformar_datos")
    assert hasattr(pjp, "modelar")
    assert hasattr(pjp, "visualizar")
    assert hasattr(pjp, "generar_reporte")
