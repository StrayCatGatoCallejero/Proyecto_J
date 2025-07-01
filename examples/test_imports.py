#!/usr/bin/env python3
"""
Script de prueba para verificar imports
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

try:
    print("Testing imports...")
    from processing.io import DataLoader

    print("✅ processing.io imported successfully")

    from processing.types import SchemaValidator

    print("✅ processing.types imported successfully")

    from processing.filters import DataFilter

    print("✅ processing.filters imported successfully")

    from processing.stats import summary_statistics_advanced

    print("✅ processing.stats imported successfully")

    from processing.features import compute_ratios

    print("✅ processing.features imported successfully")

    from processing.visualization import VisualizationGenerator

    print("✅ processing.visualization imported successfully")

    from processing.logging import setup_logging

    print("✅ processing.logging imported successfully")

    from orchestrator.pipeline_orchestrator import PipelineOrchestrator

    print("✅ orchestrator.pipeline_orchestrator imported successfully")

    print("\n🎉 All imports successful! Streamlit should work now.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    sys.exit(1)
