"""
Test End-to-End para Sistema de Logging JSON
===========================================

Este test verifica que el sistema de logging JSON funcione correctamente
en un flujo completo del PipelineOrchestrator.
"""

import pytest
import json
import os
import tempfile
import shutil
from datetime import datetime
import pandas as pd
import numpy as np

from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from processing.json_logging import JsonLogger
from processing.session_data import SessionData


class TestJsonLoggingE2E:
    """Test end-to-end para el sistema de logging JSON"""
    
    @pytest.fixture
    def temp_dir(self):
        """Crear directorio temporal para logs"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Crear datos de ejemplo para el test"""
        np.random.seed(42)
        data = {
            'edad': np.random.randint(18, 80, 100),
            'ingresos': np.random.normal(50000, 15000, 100),
            'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad'], 100),
            'genero': np.random.choice(['M', 'F'], 100),
            'satisfaccion': np.random.randint(1, 11, 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self, temp_dir):
        """Configuración para el test"""
        return {
            'data': {
                'input_path': None,  # Se usará DataFrame en memoria
                'output_path': os.path.join(temp_dir, 'output'),
                'export_formats': ['csv', 'xlsx']
            },
            'analysis': {
                'columns_to_analyze': ['edad', 'ingresos', 'satisfaccion'],
                'correlation_method': 'pearson',
                'significance_level': 0.05
            },
            'filters': {
                'remove_outliers': True,
                'outlier_method': 'iqr',
                'handle_missing': 'drop'
            },
            'logging': {
                'enabled': True,
                'level': 'INFO',
                'log_file': os.path.join(temp_dir, 'logs', 'pipeline.json'),
                'rotation': {
                    'when': 'midnight',
                    'interval': 1,
                    'backup_count': 7
                },
                'max_size_mb': 100
            }
        }
    
    def test_pipeline_logging_complete_flow(self, temp_dir, sample_data, config):
        """Test del flujo completo del pipeline con logging JSON"""
        
        # Crear directorio de logs
        os.makedirs(os.path.join(temp_dir, 'logs'), exist_ok=True)
        
        # Inicializar PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Crear SessionData
        session_data = SessionData(
            session_id=f"test_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id="test_user",
            timestamp=datetime.now()
        )
        
        # Ejecutar pipeline completo
        result = orchestrator.run(sample_data, session_data)
        
        # Verificar que el pipeline se ejecutó correctamente
        assert result is not None
        assert 'data' in result
        assert 'analysis' in result
        assert 'visualizations' in result
        assert 'exports' in result
        
        # Verificar que el archivo de log existe
        log_file = config['logging']['log_file']
        assert os.path.exists(log_file), f"El archivo de log no existe: {log_file}"
        
        # Verificar que el archivo no está vacío
        file_size = os.path.getsize(log_file)
        assert file_size > 0, f"El archivo de log está vacío: {log_file}"
        
        # Leer y validar logs JSON
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        assert len(log_lines) > 0, "No hay líneas de log"
        
        # Validar cada línea como JSON válido
        log_events = []
        for i, line in enumerate(log_lines):
            line = line.strip()
            if line:  # Ignorar líneas vacías
                try:
                    log_event = json.loads(line)
                    log_events.append(log_event)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Línea {i+1} no es JSON válido: {line[:100]}... Error: {e}")
        
        assert len(log_events) > 0, "No se encontraron eventos de log válidos"
        
        # Verificar estructura de los logs
        for event in log_events:
            # Campos obligatorios
            assert 'timestamp' in event, "Falta campo 'timestamp' en log"
            assert 'level' in event, "Falta campo 'level' en log"
            assert 'message' in event, "Falta campo 'message' en log"
            assert 'session_id' in event, "Falta campo 'session_id' en log"
            assert 'module' in event, "Falta campo 'module' en log"
            assert 'function' in event, "Falta campo 'function' en log"
            assert 'step' in event, "Falta campo 'step' en log"
            assert 'category' in event, "Falta campo 'category' en log"
            
            # Verificar tipos de datos
            assert isinstance(event['timestamp'], str), "timestamp debe ser string"
            assert isinstance(event['level'], str), "level debe ser string"
            assert isinstance(event['message'], str), "message debe ser string"
            assert isinstance(event['session_id'], str), "session_id debe ser string"
            assert isinstance(event['module'], str), "module debe ser string"
            assert isinstance(event['function'], str), "function debe ser string"
            assert isinstance(event['step'], str), "step debe ser string"
            assert isinstance(event['category'], str), "category debe ser string"
            
            # Verificar que session_id es consistente
            assert event['session_id'] == session_data.session_id, "session_id inconsistente"
        
        # Verificar tipos de logs esperados
        log_types = [event['step'] for event in log_events]
        expected_steps = [
            'pipeline_start',
            'data_loading',
            'schema_validation',
            'business_rules_validation',
            'data_filtering',
            'statistical_analysis',
            'visualization_generation',
            'pipeline_complete'
        ]
        
        # Verificar que al menos algunos de los pasos esperados están presentes
        found_steps = [step for step in expected_steps if step in log_types]
        assert len(found_steps) >= 3, f"Se encontraron pocos pasos esperados: {found_steps}"
        
        # Verificar niveles de log
        log_levels = [event['level'] for event in log_events]
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        for level in log_levels:
            assert level in valid_levels, f"Nivel de log inválido: {level}"
        
        # Verificar métricas en logs
        metrics_logs = [event for event in log_events if 'metrics' in event]
        assert len(metrics_logs) > 0, "No se encontraron logs con métricas"
        
        # Verificar logs de error (deberían ser 0 en un test exitoso)
        error_logs = [event for event in log_events if event['level'] == 'ERROR']
        assert len(error_logs) == 0, f"Se encontraron errores en el pipeline: {len(error_logs)}"
        
        # Verificar logs de sistema
        system_logs = [event for event in log_events if event['category'] == 'SYSTEM']
        assert len(system_logs) > 0, "No se encontraron logs del sistema"
        
        # Verificar logs de análisis
        analysis_logs = [event for event in log_events if event['category'] == 'ANALYSIS']
        assert len(analysis_logs) > 0, "No se encontraron logs de análisis"
        
        # Verificar que hay logs con execution_time
        execution_time_logs = [event for event in log_events if 'execution_time' in event]
        assert len(execution_time_logs) > 0, "No se encontraron logs con tiempo de ejecución"
        
        # Verificar que los tiempos de ejecución son números positivos
        for event in execution_time_logs:
            assert isinstance(event['execution_time'], (int, float)), "execution_time debe ser número"
            assert event['execution_time'] >= 0, "execution_time debe ser positivo"
        
        print(f"✅ Test completado exitosamente:")
        print(f"   - Total de eventos de log: {len(log_events)}")
        print(f"   - Pasos encontrados: {found_steps}")
        print(f"   - Logs con métricas: {len(metrics_logs)}")
        print(f"   - Logs del sistema: {len(system_logs)}")
        print(f"   - Logs de análisis: {len(analysis_logs)}")
        print(f"   - Logs con tiempo de ejecución: {len(execution_time_logs)}")
        print(f"   - Errores encontrados: {len(error_logs)}")
    
    def test_logging_with_errors(self, temp_dir, sample_data, config):
        """Test del logging cuando ocurren errores"""
        
        # Crear directorio de logs
        os.makedirs(os.path.join(temp_dir, 'logs'), exist_ok=True)
        
        # Modificar datos para causar un error (columna inexistente)
        config['analysis']['columns_to_analyze'] = ['columna_inexistente']
        
        # Inicializar PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Crear SessionData
        session_data = SessionData(
            session_id=f"test_error_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id="test_user",
            timestamp=datetime.now()
        )
        
        # Ejecutar pipeline (debería fallar pero generar logs)
        try:
            result = orchestrator.run(sample_data, session_data)
        except Exception:
            pass  # Esperamos que falle
        
        # Verificar que el archivo de log existe
        log_file = config['logging']['log_file']
        assert os.path.exists(log_file), f"El archivo de log no existe: {log_file}"
        
        # Leer logs
        with open(log_file, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        log_events = []
        for line in log_lines:
            line = line.strip()
            if line:
                try:
                    log_event = json.loads(line)
                    log_events.append(log_event)
                except json.JSONDecodeError:
                    continue
        
        # Verificar que hay logs de error
        error_logs = [event for event in log_events if event['level'] == 'ERROR']
        assert len(error_logs) > 0, "No se encontraron logs de error"
        
        # Verificar que los logs de error tienen información útil
        for error_log in error_logs:
            assert 'error_message' in error_log, "Log de error sin mensaje"
            assert 'error_type' in error_log, "Log de error sin tipo"
            assert 'function' in error_log, "Log de error sin función"
            assert 'context' in error_log, "Log de error sin contexto"
        
        print(f"✅ Test de errores completado:")
        print(f"   - Total de eventos de log: {len(log_events)}")
        print(f"   - Logs de error: {len(error_logs)}")
    
    def test_log_rotation(self, temp_dir, sample_data, config):
        """Test de rotación de logs"""
        
        # Crear directorio de logs
        os.makedirs(os.path.join(temp_dir, 'logs'), exist_ok=True)
        
        # Configurar rotación por tamaño pequeño para testing
        config['logging']['max_size_mb'] = 0.001  # 1KB
        
        # Inicializar PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config)
        
        # Crear SessionData
        session_data = SessionData(
            session_id=f"test_rotation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            user_id="test_user",
            timestamp=datetime.now()
        )
        
        # Ejecutar pipeline múltiples veces para generar rotación
        for i in range(3):
            try:
                result = orchestrator.run(sample_data, session_data)
            except Exception:
                pass
        
        # Verificar que existen archivos de log (puede haber rotación)
        log_dir = os.path.dirname(config['logging']['log_file'])
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        
        assert len(log_files) > 0, "No se encontraron archivos de log"
        
        print(f"✅ Test de rotación completado:")
        print(f"   - Archivos de log encontrados: {len(log_files)}")
        print(f"   - Archivos: {log_files}")


if __name__ == "__main__":
    # Ejecutar tests
    pytest.main([__file__, "-v"])
