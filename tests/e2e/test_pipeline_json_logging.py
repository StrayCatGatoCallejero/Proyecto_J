"""
Test End-to-End para Pipeline Orchestrator con Logging JSON
==========================================================

Este test verifica que el PipelineOrchestrator funcione correctamente con logging JSON
y que genere logs válidos para cada paso del pipeline.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any, List

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from processing.json_logging import LogLevel, LogCategory


class TestPipelineJsonLogging:
    """Test suite para verificar el logging JSON del pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Crear directorio temporal para los tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Crear datos de ejemplo para el test."""
        np.random.seed(42)
        
        data = {
            'edad': np.random.normal(35, 12, 100).astype(int),
            'ingresos': np.random.lognormal(10, 0.5, 100),
            'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], 100),
            'satisfaccion': np.random.randint(1, 11, 100),
            'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 100)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_file(self, temp_dir, sample_data):
        """Crear archivo de ejemplo para el test."""
        file_path = os.path.join(temp_dir, "test_data.csv")
        sample_data.to_csv(file_path, index=False)
        return file_path
    
    @pytest.fixture
    def config(self, temp_dir):
        """Crear configuración para el test."""
        return {
            "config_path": "test_config",
            "logging": {
                "json_logging": {
                    "enabled": True,
                    "log_level": "DEBUG",
                    "log_file": os.path.join(temp_dir, "pipeline_test.json"),
                    "max_file_size_mb": 5,
                    "backup_count": 2,
                    "rotation": "time",
                    "rotation_interval": "1 hour",
                    "include_system_info": True,
                    "include_session_id": True,
                    "format": "json",
                    "compression": False
                }
            },
            "data_processing": {
                "chunk_size": 1000,
                "parallel_processing": False,
                "memory_limit_mb": 256
            },
            "validation": {
                "strict_mode": False,
                "auto_fix": True,
                "report_errors": True
            },
            "visualization": {
                "theme": "default",
                "figure_size": [10, 6],
                "dpi": 100,
                "save_format": "png"
            }
        }
    
    def test_pipeline_execution_with_json_logging(self, config, sample_file):
        """Test que verifica la ejecución completa del pipeline con logging JSON."""
        
        # 1. Inicializar Pipeline Orchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Verificar que se creó el logger JSON
        assert hasattr(orchestrator, 'json_logger')
        assert orchestrator.json_logger is not None
        
        # 2. Definir filtros y esquema para el test
        filters = {
            "edad": {"min": 18, "max": 65},
            "educacion": {"values": ["Secundaria", "Universitaria"]}
        }
        
        schema = {
            "edad": {"type": "int", "min": 0, "max": 120},
            "ingresos": {"type": "float", "min": 0},
            "satisfaccion": {"type": "int", "min": 1, "max": 10}
        }
        
        # 3. Ejecutar pipeline completo
        results = orchestrator.run_pipeline(
            path=sample_file,
            filters=filters,
            schema=schema
        )
        
        # 4. Verificar resultados del pipeline
        assert results is not None
        assert "session_id" in results
        assert "pipeline_status" in results
        assert results["pipeline_status"] == "completed"
        assert "pipeline_metrics" in results
        assert results["pipeline_metrics"]["completed_steps"] > 0
        assert results["pipeline_metrics"]["failed_steps"] == 0
        
        # 5. Verificar que se generó el archivo de log
        log_file = config["logging"]["json_logging"]["log_file"]
        assert os.path.exists(log_file), f"Archivo de log no encontrado: {log_file}"
        
        # 6. Verificar contenido del archivo de log
        self._verify_log_file_content(log_file, orchestrator.session_id)
        
        # 7. Verificar métricas específicas
        assert results["pipeline_metrics"]["total_steps"] >= 6  # Mínimo de pasos esperados
        assert results["pipeline_metrics"]["total_execution_time"] > 0
        assert results["pipeline_metrics"]["error_rate"] == 0.0
        
        # 8. Verificar estados de los pasos
        assert len(results["step_statuses"]) > 0
        for step_name, step_info in results["step_statuses"].items():
            assert step_info["status"] == "completed"
            assert "timestamp" in step_info
    
    def test_log_file_structure(self, config, sample_file):
        """Test que verifica la estructura del archivo de log JSON."""
        
        orchestrator = PipelineOrchestrator(config)
        
        # Ejecutar pipeline
        orchestrator.run_pipeline(path=sample_file)
        
        # Leer archivo de log
        log_file = config["logging"]["json_logging"]["log_file"]
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Verificar que hay líneas de log
        assert len(log_lines) > 0, "El archivo de log está vacío"
        
        # Verificar estructura de cada línea JSON
        for i, line in enumerate(log_lines):
            try:
                log_entry = json.loads(line.strip())
                self._verify_log_entry_structure(log_entry, i)
            except json.JSONDecodeError as e:
                pytest.fail(f"Línea {i+1} no es JSON válido: {e}")
    
    def test_log_entry_types(self, config, sample_file):
        """Test que verifica que se generen todos los tipos de logs esperados."""
        
        orchestrator = PipelineOrchestrator(config)
        
        # Ejecutar pipeline
        orchestrator.run_pipeline(path=sample_file)
        
        # Leer archivo de log
        log_file = config["logging"]["json_logging"]["log_file"]
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Extraer tipos de logs
        log_types = []
        log_categories = []
        log_steps = []
        
        for line in log_lines:
            log_entry = json.loads(line.strip())
            log_types.append(log_entry.get("level"))
            log_categories.append(log_entry.get("category"))
            log_steps.append(log_entry.get("step"))
        
        # Verificar que hay logs de diferentes tipos
        assert "INFO" in log_types, "No hay logs de nivel INFO"
        assert "DEBUG" in log_types or "INFO" in log_types, "No hay logs de nivel DEBUG o INFO"
        
        # Verificar categorías esperadas
        expected_categories = [
            LogCategory.DATA_LOAD.value,
            LogCategory.VALIDATION.value,
            LogCategory.BUSINESS_RULES.value,
            LogCategory.PROCESSING.value,
            LogCategory.ANALYSIS.value,
            LogCategory.VISUALIZATION.value
        ]
        
        for category in expected_categories:
            assert category in log_categories, f"No hay logs de categoría {category}"
        
        # Verificar pasos esperados
        expected_steps = [
            "data_loading",
            "schema_validation", 
            "business_rules_validation",
            "data_filtering",
            "statistical_analysis",
            "visualization_generation"
        ]
        
        for step in expected_steps:
            assert step in log_steps, f"No hay logs del paso {step}"
    
    def test_session_id_consistency(self, config, sample_file):
        """Test que verifica que el session_id sea consistente en todos los logs."""
        
        orchestrator = PipelineOrchestrator(config)
        
        # Ejecutar pipeline
        results = orchestrator.run_pipeline(path=sample_file)
        session_id = results["session_id"]
        
        # Leer archivo de log
        log_file = config["logging"]["json_logging"]["log_file"]
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Verificar que todos los logs tienen el mismo session_id
        for i, line in enumerate(log_lines):
            log_entry = json.loads(line.strip())
            log_session_id = log_entry.get("session_id")
            assert log_session_id == session_id, f"Session ID inconsistente en línea {i+1}"
    
    def test_metrics_capture(self, config, sample_file):
        """Test que verifica que se capturen métricas en los logs."""
        
        orchestrator = PipelineOrchestrator(config)
        
        # Ejecutar pipeline
        orchestrator.run_pipeline(path=sample_file)
        
        # Leer archivo de log
        log_file = config["logging"]["json_logging"]["log_file"]
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Verificar que hay logs con métricas
        metrics_found = False
        for line in log_lines:
            log_entry = json.loads(line.strip())
            if "before_metrics" in log_entry or "after_metrics" in log_entry:
                metrics_found = True
                break
        
        assert metrics_found, "No se encontraron logs con métricas"
    
    def test_error_logging(self, config, temp_dir):
        """Test que verifica el logging de errores."""
        
        orchestrator = PipelineOrchestrator(config)
        
        # Intentar cargar un archivo inexistente
        non_existent_file = os.path.join(temp_dir, "non_existent.csv")
        
        try:
            orchestrator.load_data(non_existent_file)
            pytest.fail("Debería haber lanzado una excepción")
        except Exception:
            # Verificar que se generó un log de error
            log_file = config["logging"]["json_logging"]["log_file"]
            assert os.path.exists(log_file)
            
            with open(log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            # Buscar logs de error
            error_logs = []
            for line in log_lines:
                log_entry = json.loads(line.strip())
                if log_entry.get("level") == "ERROR":
                    error_logs.append(log_entry)
            
            assert len(error_logs) > 0, "No se encontraron logs de error"
    
    def _verify_log_file_content(self, log_file: str, session_id: str):
        """Verifica el contenido del archivo de log."""
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        # Verificar que hay logs
        assert len(log_lines) > 0, "El archivo de log está vacío"
        
        # Verificar que todos los logs son JSON válidos
        for i, line in enumerate(log_lines):
            try:
                log_entry = json.loads(line.strip())
                assert "timestamp" in log_entry, f"Log {i+1} no tiene timestamp"
                assert "level" in log_entry, f"Log {i+1} no tiene level"
                assert "message" in log_entry, f"Log {i+1} no tiene message"
                assert "session_id" in log_entry, f"Log {i+1} no tiene session_id"
                assert log_entry["session_id"] == session_id, f"Session ID incorrecto en log {i+1}"
            except json.JSONDecodeError as e:
                pytest.fail(f"Log {i+1} no es JSON válido: {e}")
    
    def _verify_log_entry_structure(self, log_entry: Dict[str, Any], line_number: int):
        """Verifica la estructura de una entrada de log."""
        
        # Campos obligatorios
        required_fields = ["timestamp", "level", "message", "session_id"]
        for field in required_fields:
            assert field in log_entry, f"Campo obligatorio '{field}' faltante en línea {line_number+1}"
        
        # Verificar tipos de datos
        assert isinstance(log_entry["timestamp"], str), f"timestamp debe ser string en línea {line_number+1}"
        assert isinstance(log_entry["level"], str), f"level debe ser string en línea {line_number+1}"
        assert isinstance(log_entry["message"], str), f"message debe ser string en línea {line_number+1}"
        assert isinstance(log_entry["session_id"], str), f"session_id debe ser string en línea {line_number+1}"
        
        # Verificar que el timestamp es válido
        try:
            datetime.fromisoformat(log_entry["timestamp"].replace('Z', '+00:00'))
        except ValueError:
            pytest.fail(f"Timestamp inválido en línea {line_number+1}: {log_entry['timestamp']}")
        
        # Verificar que el level es válido
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert log_entry["level"] in valid_levels, f"Level inválido en línea {line_number+1}: {log_entry['level']}"
        
        # Campos opcionales pero comunes
        if "module" in log_entry:
            assert isinstance(log_entry["module"], str), f"module debe ser string en línea {line_number+1}"
        
        if "function" in log_entry:
            assert isinstance(log_entry["function"], str), f"function debe ser string en línea {line_number+1}"
        
        if "step" in log_entry:
            assert isinstance(log_entry["step"], str), f"step debe ser string en línea {line_number+1}"
        
        if "category" in log_entry:
            assert isinstance(log_entry["category"], str), f"category debe ser string en línea {line_number+1}"
        
        if "execution_time" in log_entry:
            assert isinstance(log_entry["execution_time"], (int, float)), f"execution_time debe ser numérico en línea {line_number+1}"
        
        if "tags" in log_entry:
            assert isinstance(log_entry["tags"], list), f"tags debe ser lista en línea {line_number+1}"
            for tag in log_entry["tags"]:
                assert isinstance(tag, str), f"tags debe contener strings en línea {line_number+1}"
        
        if "metadata" in log_entry:
            assert isinstance(log_entry["metadata"], dict), f"metadata debe ser dict en línea {line_number+1}"


if __name__ == "__main__":
    # Ejecutar tests manualmente
    pytest.main([__file__, "-v"]) 