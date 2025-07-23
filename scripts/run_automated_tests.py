#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de automatización para ejecutar pruebas críticas del sistema
Este script se ejecuta automáticamente para verificar que las funcionalidades principales funcionen correctamente
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import traceback

class AutomatedTestRunner:
    """Ejecutor automático de pruebas del sistema"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        self.log_file = self.project_root / "logs" / "automated_tests.log"
        self.results_file = self.project_root / "logs" / "test_results.json"
        
        # Asegurar que el directorio de logs existe
        self.log_file.parent.mkdir(exist_ok=True)
    
    def log(self, message, level="INFO"):
        """Registrar mensaje en el log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {level}: {message}"
        print(log_message)
        
        # Guardar en archivo de log
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def run_command(self, command, description, timeout=300):
        """Ejecutar comando y capturar resultado"""
        self.log(f"Ejecutando: {description}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root
            )
            
            success = result.returncode == 0
            output = result.stdout + result.stderr
            
            if success:
                self.log(f"✅ {description} - EXITOSO")
            else:
                self.log(f"❌ {description} - FALLÓ")
                self.log(f"Error: {output}")
            
            return {
                'success': success,
                'output': output,
                'returncode': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            self.log(f"⏰ {description} - TIMEOUT")
            return {
                'success': False,
                'output': 'Timeout expired',
                'returncode': -1
            }
        except Exception as e:
            self.log(f"💥 {description} - ERROR: {str(e)}")
            return {
                'success': False,
                'output': str(e),
                'returncode': -1
            }
    
    def test_file_upload_functionality(self):
        """Probar funcionalidad de carga de archivos"""
        self.log("🧪 Iniciando pruebas de carga de archivos")
        
        # Ejecutar pruebas específicas de carga de archivos
        result = self.run_command(
            "python -m pytest tests/test_file_upload_integration.py -v --tb=short",
            "Pruebas de integración de carga de archivos"
        )
        
        self.results['tests']['file_upload'] = result
        return result['success']
    
    def test_streamlit_app_imports(self):
        """Probar que la aplicación Streamlit se puede importar correctamente"""
        self.log("🧪 Verificando imports de la aplicación Streamlit")
        
        try:
            # Cambiar al directorio del proyecto
            os.chdir(self.project_root / "proyecto_j")
            
            # Intentar importar módulos críticos
            import sys
            sys.path.insert(0, str(self.project_root / "proyecto_j"))
            
            # Importar módulos principales
            from streamlit_app import load_file, check_system_status
            from src import core, estadistica, ciencias_sociales
            
            self.log("✅ Imports de Streamlit - EXITOSO")
            result = {'success': True, 'output': 'Imports exitosos', 'returncode': 0}
            
        except Exception as e:
            self.log(f"❌ Imports de Streamlit - FALLÓ: {str(e)}")
            result = {'success': False, 'output': str(e), 'returncode': -1}
        
        self.results['tests']['streamlit_imports'] = result
        return result['success']
    
    def test_dependencies(self):
        """Verificar que las dependencias críticas estén disponibles"""
        self.log("🧪 Verificando dependencias críticas")
        
        dependencies = [
            ('pandas', 'pandas'),
            ('numpy', 'numpy'),
            ('streamlit', 'streamlit'),
            ('plotly', 'plotly'),
            ('matplotlib', 'matplotlib'),
            ('seaborn', 'seaborn'),
            ('chardet', 'chardet'),
            ('openpyxl', 'openpyxl'),
            ('pyreadstat', 'pyreadstat')
        ]
        
        results = {}
        all_available = True
        
        for package_name, import_name in dependencies:
            try:
                __import__(import_name)
                self.log(f"✅ {package_name} - DISPONIBLE")
                results[package_name] = {'available': True, 'error': None}
            except ImportError as e:
                self.log(f"⚠️ {package_name} - NO DISPONIBLE: {str(e)}")
                results[package_name] = {'available': False, 'error': str(e)}
                if package_name in ['pandas', 'numpy', 'streamlit']:
                    all_available = False
        
        result = {
            'success': all_available,
            'output': json.dumps(results, indent=2),
            'returncode': 0 if all_available else 1
        }
        
        self.results['tests']['dependencies'] = result
        return result['success']
    
    def test_data_loading_simulation(self):
        """Simular carga de datos sin Streamlit"""
        self.log("🧪 Simulando carga de datos")
        
        try:
            import pandas as pd
            import numpy as np
            import tempfile
            from pathlib import Path
            
            # Crear datos de prueba
            test_data = pd.DataFrame({
                'id': range(1, 11),
                'nombre': [f'Test {i}' for i in range(1, 11)],
                'valor': np.random.randn(10)
            })
            
            # Probar diferentes formatos
            formats_tested = []
            
            # CSV
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                test_data.to_csv(f.name, index=False)
                loaded_csv = pd.read_csv(f.name)
                formats_tested.append(('CSV', loaded_csv.equals(test_data)))
                os.unlink(f.name)
            
            # Excel
            try:
                with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
                    test_data.to_excel(f.name, index=False)
                    loaded_excel = pd.read_excel(f.name)
                    formats_tested.append(('Excel', loaded_excel.equals(test_data)))
                    os.unlink(f.name)
            except ImportError:
                formats_tested.append(('Excel', False))
            
            # Verificar resultados
            all_formats_work = all(success for _, success in formats_tested)
            
            if all_formats_work:
                self.log("✅ Simulación de carga de datos - EXITOSO")
            else:
                self.log(f"⚠️ Simulación de carga de datos - PARCIAL: {formats_tested}")
            
            result = {
                'success': all_formats_work,
                'output': str(formats_tested),
                'returncode': 0 if all_formats_work else 1
            }
            
        except Exception as e:
            self.log(f"❌ Simulación de carga de datos - FALLÓ: {str(e)}")
            result = {'success': False, 'output': str(e), 'returncode': -1}
        
        self.results['tests']['data_loading_simulation'] = result
        return result['success']
    
    def test_code_quality(self):
        """Verificar calidad del código"""
        self.log("🧪 Verificando calidad del código")
        
        # Verificar sintaxis de archivos principales
        main_files = [
            "proyecto_j/streamlit_app.py",
            "proyecto_j/src/core.py",
            "proyecto_j/src/estadistica.py",
            "proyecto_j/src/ciencias_sociales.py"
        ]
        
        syntax_errors = []
        
        for file_path in main_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        compile(f.read(), str(full_path), 'exec')
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}: {str(e)}")
        
        if not syntax_errors:
            self.log("✅ Calidad del código - EXITOSO")
            result = {'success': True, 'output': 'Sin errores de sintaxis', 'returncode': 0}
        else:
            self.log(f"❌ Calidad del código - FALLÓ: {syntax_errors}")
            result = {'success': False, 'output': str(syntax_errors), 'returncode': 1}
        
        self.results['tests']['code_quality'] = result
        return result['success']
    
    def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        self.log("🚀 Iniciando ejecución automática de pruebas")
        
        test_functions = [
            ('Dependencias', self.test_dependencies),
            ('Calidad del código', self.test_code_quality),
            ('Imports de Streamlit', self.test_streamlit_app_imports),
            ('Simulación de carga de datos', self.test_data_loading_simulation),
            ('Carga de archivos', self.test_file_upload_functionality)
        ]
        
        for test_name, test_func in test_functions:
            self.results['summary']['total'] += 1
            
            try:
                success = test_func()
                if success:
                    self.results['summary']['passed'] += 1
                else:
                    self.results['summary']['failed'] += 1
            except Exception as e:
                self.log(f"💥 Error en {test_name}: {str(e)}")
                self.results['summary']['failed'] += 1
                self.results['tests'][test_name.lower().replace(' ', '_')] = {
                    'success': False,
                    'output': str(e),
                    'returncode': -1
                }
        
        # Guardar resultados
        self.save_results()
        
        # Mostrar resumen
        self.show_summary()
        
        return self.results['summary']['failed'] == 0
    
    def save_results(self):
        """Guardar resultados en archivo JSON"""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            self.log(f"💾 Resultados guardados en {self.results_file}")
        except Exception as e:
            self.log(f"⚠️ Error guardando resultados: {str(e)}")
    
    def show_summary(self):
        """Mostrar resumen de resultados"""
        summary = self.results['summary']
        
        self.log("=" * 60)
        self.log("📊 RESUMEN DE PRUEBAS AUTOMATIZADAS")
        self.log("=" * 60)
        self.log(f"📅 Fecha: {self.results['timestamp']}")
        self.log(f"📈 Total de pruebas: {summary['total']}")
        self.log(f"✅ Exitosas: {summary['passed']}")
        self.log(f"❌ Fallidas: {summary['failed']}")
        self.log(f"⏭️ Omitidas: {summary['skipped']}")
        
        if summary['failed'] == 0:
            self.log("🎉 ¡TODAS LAS PRUEBAS PASARON!")
        else:
            self.log(f"⚠️ {summary['failed']} prueba(s) fallaron")
        
        self.log("=" * 60)

def main():
    """Función principal"""
    runner = AutomatedTestRunner()
    
    try:
        success = runner.run_all_tests()
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n⏹️ Ejecución interrumpida por el usuario")
        return 1
    except Exception as e:
        print(f"💥 Error fatal: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 