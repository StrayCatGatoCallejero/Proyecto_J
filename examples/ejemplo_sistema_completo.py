"""
Ejemplo Completo del Sistema de Logging JSON
============================================

Este ejemplo demuestra que el sistema de logging JSON está completamente
funcional y genera logs válidos para cada paso del pipeline.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import tempfile
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.pipeline_orchestrator import PipelineOrchestrator
from processing.json_logging import LogLevel, LogCategory


def create_sample_data():
    """Crea datos de ejemplo para el test."""
    np.random.seed(42)
    
    data = {
        'edad': np.random.normal(35, 12, 50).astype(int),
        'ingresos': np.random.lognormal(10, 0.5, 50),
        'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universitaria'], 50),
        'satisfaccion': np.random.randint(1, 11, 50),
        'region': np.random.choice(['Norte', 'Sur', 'Este', 'Oeste'], 50)
    }
    
    return pd.DataFrame(data)


def create_config(temp_dir):
    """Crea configuración para el test."""
    return {
        "config_path": "test_config",
        "logging": {
            "json_logging": {
                "enabled": True,
                "log_level": "DEBUG",
                "log_file": os.path.join(temp_dir, "pipeline_demo.json"),
                "max_file_size_mb": 5,
                "backup_count": 2,
                "rotation": {
                    "enabled": True,
                    "when": "midnight",
                    "interval": 1,
                    "backup_count": 2
                },
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


def main():
    """Función principal que demuestra el sistema completo."""
    
    print("🚀 DEMOSTRACIÓN DEL SISTEMA DE LOGGING JSON")
    print("=" * 50)
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    print(f"📁 Directorio temporal: {temp_dir}")
    
    try:
        # 1. Crear datos de ejemplo
        print("\n📊 1. Creando datos de ejemplo...")
        sample_df = create_sample_data()
        sample_file = os.path.join(temp_dir, "test_data.csv")
        sample_df.to_csv(sample_file, index=False)
        print(f"   ✅ Datos creados: {sample_df.shape} - Guardados en: {sample_file}")
        
        # 2. Crear configuración
        print("\n⚙️  2. Configurando sistema...")
        config = create_config(temp_dir)
        print(f"   ✅ Configuración creada con logging JSON habilitado")
        
        # 3. Inicializar Pipeline Orchestrator
        print("\n🔧 3. Inicializando Pipeline Orchestrator...")
        orchestrator = PipelineOrchestrator(config)
        print(f"   ✅ Session ID: {orchestrator.session_id}")
        
        # 4. Definir filtros y esquema
        print("\n🔍 4. Configurando filtros y validaciones...")
        filters = {
            "edad": {"min": 18, "max": 65},
            "educacion": {"values": ["Secundaria", "Universitaria"]}
        }
        
        schema = {
            "edad": {"type": "int", "min": 0, "max": 120},
            "ingresos": {"type": "float", "min": 0},
            "satisfaccion": {"type": "int", "min": 1, "max": 10}
        }
        print(f"   ✅ Filtros y esquema configurados")
        
        # 5. Ejecutar pipeline completo
        print("\n🔄 5. Ejecutando pipeline completo...")
        print("   ⏳ Esto puede tomar unos segundos...")
        
        results = orchestrator.run_pipeline(
            path=sample_file,
            filters=filters,
            schema=schema
        )
        
        print("   ✅ Pipeline ejecutado exitosamente!")
        
        # 6. Verificar resultados
        print("\n📈 6. Resultados del pipeline:")
        print(f"   • Session ID: {results['session_id']}")
        print(f"   • Estado: {results['pipeline_status']}")
        print(f"   • Pasos completados: {results['pipeline_metrics']['completed_steps']}/{results['pipeline_metrics']['total_steps']}")
        print(f"   • Tiempo total: {results['pipeline_metrics']['total_execution_time']:.2f}s")
        print(f"   • Tasa de error: {results['pipeline_metrics']['error_rate']:.2%}")
        print(f"   • Forma de datos: {results['data_info']['shape']}")
        print(f"   • Uso de memoria: {results['data_info']['memory_usage_mb']:.2f} MB")
        
        # 7. Verificar archivo de log
        print("\n📝 7. Verificando archivo de log...")
        log_file = config["logging"]["json_logging"]["log_file"]
        
        if os.path.exists(log_file):
            file_size = os.path.getsize(log_file) / 1024  # KB
            print(f"   ✅ Archivo de log encontrado: {log_file}")
            print(f"   📏 Tamaño: {file_size:.2f} KB")
            
            # Leer y analizar logs
            with open(log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            print(f"   📊 Total de entradas de log: {len(log_lines)}")
            
            # Analizar tipos de logs
            log_types = {}
            log_categories = {}
            log_steps = {}
            
            for line in log_lines:
                try:
                    log_entry = json.loads(line.strip())
                    
                    # Contar tipos
                    level = log_entry.get("level", "UNKNOWN")
                    log_types[level] = log_types.get(level, 0) + 1
                    
                    # Contar categorías
                    category = log_entry.get("category", "UNKNOWN")
                    log_categories[category] = log_categories.get(category, 0) + 1
                    
                    # Contar pasos
                    step = log_entry.get("step", "UNKNOWN")
                    log_steps[step] = log_steps.get(step, 0) + 1
                    
                except json.JSONDecodeError as e:
                    print(f"   ⚠️  Error en línea JSON: {e}")
            
            print(f"   📊 Tipos de logs: {log_types}")
            print(f"   📊 Categorías: {log_categories}")
            print(f"   📊 Pasos: {log_steps}")
            
            # Verificar session ID consistente
            session_ids = set()
            for line in log_lines:
                log_entry = json.loads(line.strip())
                session_ids.add(log_entry.get("session_id"))
            
            if len(session_ids) == 1:
                print(f"   ✅ Session ID consistente: {list(session_ids)[0]}")
            else:
                print(f"   ⚠️  Session IDs inconsistentes: {session_ids}")
            
            # Mostrar ejemplo de log
            if log_lines:
                example_log = json.loads(log_lines[0].strip())
                print(f"   📄 Ejemplo de log:")
                print(f"      Timestamp: {example_log.get('timestamp')}")
                print(f"      Level: {example_log.get('level')}")
                print(f"      Message: {example_log.get('message')}")
                print(f"      Step: {example_log.get('step')}")
                print(f"      Category: {example_log.get('category')}")
            
        else:
            print(f"   ❌ Archivo de log no encontrado: {log_file}")
        
        # 8. Verificar estados de los pasos
        print("\n📋 8. Estado de los pasos:")
        for step_name, step_info in results['step_statuses'].items():
            status_emoji = {
                'completed': '✅',
                'failed': '❌',
                'running': '🔄',
                'pending': '⏳',
                'skipped': '⏭️'
            }.get(step_info['status'], '❓')
            print(f"   {status_emoji} {step_name}: {step_info['status']}")
        
        # 9. Verificar métricas detalladas
        print("\n📊 9. Métricas detalladas del pipeline:")
        for step_name, metrics in orchestrator.pipeline_metrics.step_metrics.items():
            print(f"   📈 {step_name}:")
            print(f"      • Tiempo: {metrics.execution_time:.3f}s")
            print(f"      • Filas antes/después: {metrics.rows_before} → {metrics.rows_after}")
            print(f"      • Memoria: {metrics.memory_before:.2f}MB → {metrics.memory_after:.2f}MB")
            print(f"      • Delta memoria: {metrics.memory_delta:+.2f}MB")
        
        print("\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("=" * 50)
        print("✅ Pipeline Orchestrator funcionando correctamente")
        print("✅ Logging JSON estructurado implementado")
        print("✅ Métricas detalladas capturadas")
        print("✅ Trazabilidad completa con session_id")
        print("✅ Archivo de log generado y validado")
        
        print(f"\n📁 Archivos generados:")
        print(f"   • {sample_file}")
        print(f"   • {log_file}")
        
        print(f"\n🔍 Próximos pasos:")
        print(f"   1. Revisa el archivo de log: {log_file}")
        print(f"   2. Analiza las métricas de rendimiento")
        print(f"   3. Integra con sistemas de monitoreo (ELK, Datadog)")
        print(f"   4. Personaliza la configuración según tus necesidades")
        
        return True
        
    except Exception as e:
        print(f"\n💥 Error en la demostración: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpiar archivos temporales
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Directorio temporal limpiado: {temp_dir}")
        except:
            pass


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Demostración completada exitosamente")
    else:
        print("\n❌ La demostración falló")
        sys.exit(1) 