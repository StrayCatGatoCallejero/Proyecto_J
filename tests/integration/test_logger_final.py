"""
Test Final del JsonLogger con Serialización Robusta
==================================================

Verifica que el logger funciona correctamente en un escenario real.
"""

import os
import tempfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import shutil

# Importar el JsonLogger
from processing.json_logging import JsonLogger, LogLevel, LogCategory


def test_real_world_scenario():
    """Test del JsonLogger en un escenario real"""
    
    print("🧪 Probando JsonLogger en escenario real...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'real_world_test.json')
    
    try:
        # Crear logger
        logger = JsonLogger(
            file_path=log_file,
            level='INFO',
            session_id='real_world_session',
            console_output=False  # Sin output de consola para limpiar
        )
        
        # Simular datos reales del pipeline
        df = pd.DataFrame({
            'edad': np.random.randint(18, 80, 50),
            'ingresos': np.random.normal(50000, 15000, 50),
            'educacion': np.random.choice(['Primaria', 'Secundaria', 'Universidad'], 50)
        })
        
        # Estadísticas típicas del pipeline
        stats = {
            'media_edad': float(df['edad'].mean()),
            'std_ingresos': float(df['ingresos'].std()),
            'conteo_educacion': df['educacion'].value_counts().to_dict(),
            'percentiles_edad': df['edad'].quantile([0.25, 0.5, 0.75]).to_dict()
        }
        
        # Log de carga de datos
        logger.log_event(
            level=LogLevel.INFO,
            message="Datos cargados exitosamente",
            module="data_loader",
            function="load_data",
            step="data_loading",
            category=LogCategory.DATA_LOAD.value,
            parameters={"source": "memory", "rows": len(df)},
            before_metrics={"expected_rows": 50},
            after_metrics={
                "actual_rows": len(df),
                "actual_columns": len(df.columns),
                "memory_mb": df.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            execution_time=0.15,
            tags=["data_load", "success"],
            metadata={"data_types": df.dtypes.to_dict()}
        )
        
        # Log de análisis estadístico
        logger.log_event(
            level=LogLevel.INFO,
            message="Análisis estadístico completado",
            module="statistical_analyzer",
            function="analyze_data",
            step="statistical_analysis",
            category=LogCategory.ANALYSIS.value,
            parameters={"methods": ["descriptive", "correlation"]},
            before_metrics={"input_size": len(df)},
            after_metrics={
                "output_size": len(stats),
                "statistics_generated": list(stats.keys())
            },
            execution_time=0.25,
            tags=["analysis", "statistics"],
            metadata={"statistics": stats}
        )
        
        # Verificar que el archivo se creó y es válido
        if os.path.exists(log_file):
            print("✅ Archivo de log creado exitosamente!")
            
            # Leer logs
            with open(log_file, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            print(f"📄 Total de líneas de log: {len(log_lines)}")
            
            # Validar cada línea
            valid_logs = 0
            for i, line in enumerate(log_lines):
                line = line.strip()
                if line:
                    try:
                        log_event = json.loads(line)
                        valid_logs += 1
                        
                        # Verificar estructura básica
                        required_fields = ['timestamp', 'level', 'message', 'session_id', 'step']
                        for field in required_fields:
                            if field not in log_event:
                                print(f"❌ Falta campo requerido '{field}' en línea {i+1}")
                                return False
                        
                        # Verificar que no hay errores de serialización
                        if 'error' in log_event.get('message', '').lower():
                            print(f"❌ Error detectado en log: {log_event['message']}")
                            return False
                            
                    except json.JSONDecodeError as e:
                        print(f"❌ Error en línea {i+1}: {e}")
                        return False
            
            print(f"✅ Logs válidos: {valid_logs}/{len(log_lines)}")
            
            if valid_logs == len(log_lines) and valid_logs > 0:
                print("🎉 ¡Todos los logs son válidos!")
                return True
            else:
                print("❌ Algunos logs no son válidos")
                return False
                
        else:
            print("❌ No se creó el archivo de log")
            return False
            
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Limpiar
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


def test_error_handling():
    """Test del manejo de errores en el logger"""
    
    print("\n🧪 Probando manejo de errores...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    log_file = os.path.join(temp_dir, 'error_test.json')
    
    try:
        # Crear logger
        logger = JsonLogger(
            file_path=log_file,
            level='INFO',
            session_id='error_test_session',
            console_output=False
        )
        
        # Simular un error
        try:
            # Intentar dividir por cero
            result = 1 / 0
        except Exception as e:
            logger.log_error(
                function="test_function",
                error=e,
                context="test_error_handling",
                execution_time=0.01,
                additional_data={"test_param": "test_value"}
            )
        
        # Verificar que se creó el log de error
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            if 'ZeroDivisionError' in log_content:
                print("✅ Log de error creado correctamente!")
                return True
            else:
                print("❌ No se detectó el error en el log")
                return False
        else:
            print("❌ No se creó el archivo de log de error")
            return False
            
    except Exception as e:
        print(f"❌ Error en test de errores: {e}")
        return False
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    print("🚀 Test Final del JsonLogger con Serialización Robusta")
    print("=" * 65)
    
    # Ejecutar tests
    test1 = test_real_world_scenario()
    test2 = test_error_handling()
    
    print(f"\n📊 Resumen de tests:")
    print(f"  Escenario real: {'✅ PASÓ' if test1 else '❌ FALLÓ'}")
    print(f"  Manejo de errores: {'✅ PASÓ' if test2 else '❌ FALLÓ'}")
    
    if all([test1, test2]):
        print("\n🎉 ¡Todos los tests pasaron!")
        print("💡 El JsonLogger está listo para producción")
        print("🚀 Puedes proceder con la integración en los módulos clave")
    else:
        print("\n⚠️ Algunos tests fallaron. Revisar antes de continuar.") 