#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hook de pre-commit para ejecutar pruebas críticas antes de cada commit
Este script se ejecuta automáticamente antes de cada commit para verificar que no se rompa nada
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

class PreCommitHook:
    """Hook de pre-commit para verificar la calidad del código"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.failed_checks = []
        self.passed_checks = []
    
    def log(self, message, level="INFO"):
        """Registrar mensaje"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
    
    def run_check(self, name, check_function, critical=True):
        """Ejecutar una verificación"""
        self.log(f"🔍 Ejecutando: {name}")
        
        try:
            start_time = time.time()
            result = check_function()
            end_time = time.time()
            
            if result:
                self.passed_checks.append(name)
                self.log(f"✅ {name} - EXITOSO ({end_time - start_time:.2f}s)")
                return True
            else:
                self.failed_checks.append(name)
                self.log(f"❌ {name} - FALLÓ ({end_time - start_time:.2f}s)")
                return False
                
        except Exception as e:
            self.failed_checks.append(name)
            self.log(f"💥 {name} - ERROR: {str(e)}")
            return False
    
    def check_syntax(self):
        """Verificar sintaxis de archivos Python modificados"""
        try:
            # Obtener archivos modificados
            result = subprocess.run(
                ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode != 0:
                return False
            
            modified_files = result.stdout.strip().split('\n')
            python_files = [f for f in modified_files if f.endswith('.py') and f]
            
            if not python_files:
                self.log("📝 No hay archivos Python modificados")
                return True
            
            # Verificar sintaxis de cada archivo
            for file_path in python_files:
                full_path = self.project_root / file_path
                if full_path.exists():
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            compile(f.read(), str(full_path), 'exec')
                    except SyntaxError as e:
                        self.log(f"❌ Error de sintaxis en {file_path}: {e}")
                        return False
            
            return True
            
        except Exception as e:
            self.log(f"Error verificando sintaxis: {e}")
            return False
    
    def check_file_upload_functionality(self):
        """Verificar que la funcionalidad de carga de archivos funciona"""
        try:
            # Verificar que el archivo principal existe y se puede importar
            streamlit_app_path = self.project_root / "proyecto_j" / "streamlit_app.py"
            
            if not streamlit_app_path.exists():
                self.log("⚠️ streamlit_app.py no encontrado")
                return True  # No es crítico si no existe
            
            # Verificar que la función load_file existe
            with open(streamlit_app_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'def load_file(' not in content:
                self.log("❌ Función load_file no encontrada en streamlit_app.py")
                return False
            
            # Verificar que no hay return None mal posicionado
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'return None' in line and i > 0:
                    # Verificar que no esté mal posicionado
                    prev_line = lines[i-1].strip()
                    if not prev_line.endswith('except Exception as e:') and not prev_line.endswith('return None'):
                        # Verificar contexto más amplio
                        context = '\n'.join(lines[max(0, i-5):i+1])
                        if 'def load_file(' in context and 'return None' in line:
                            # Verificar que esté dentro de un bloque de manejo de errores
                            if not any('except' in lines[j] for j in range(max(0, i-10), i)):
                                self.log(f"⚠️ Posible return None mal posicionado en línea {i+1}")
                                return False
            
            return True
            
        except Exception as e:
            self.log(f"Error verificando funcionalidad de carga: {e}")
            return False
    
    def check_dependencies(self):
        """Verificar dependencias críticas"""
        try:
            critical_deps = ['pandas', 'numpy', 'streamlit']
            missing_deps = []
            
            for dep in critical_deps:
                try:
                    __import__(dep)
                except ImportError:
                    missing_deps.append(dep)
            
            if missing_deps:
                self.log(f"❌ Dependencias críticas faltantes: {missing_deps}")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Error verificando dependencias: {e}")
            return False
    
    def check_test_files(self):
        """Verificar que los archivos de prueba existen"""
        try:
            test_files = [
                "tests/test_file_upload_integration.py",
                "scripts/run_automated_tests.py"
            ]
            
            missing_files = []
            for test_file in test_files:
                if not (self.project_root / test_file).exists():
                    missing_files.append(test_file)
            
            if missing_files:
                self.log(f"⚠️ Archivos de prueba faltantes: {missing_files}")
                return False
            
            return True
            
        except Exception as e:
            self.log(f"Error verificando archivos de prueba: {e}")
            return False
    
    def check_requirements_file(self):
        """Verificar que el archivo de requirements existe"""
        try:
            requirements_files = [
                "requirements_unified.txt",
                "requirements.txt"
            ]
            
            for req_file in requirements_files:
                if (self.project_root / req_file).exists():
                    return True
            
            self.log("⚠️ No se encontró archivo de requirements")
            return False
            
        except Exception as e:
            self.log(f"Error verificando requirements: {e}")
            return False
    
    def run_all_checks(self):
        """Ejecutar todas las verificaciones"""
        self.log("🚀 Iniciando verificaciones de pre-commit")
        
        checks = [
            ("Sintaxis de archivos Python", self.check_syntax, True),
            ("Funcionalidad de carga de archivos", self.check_file_upload_functionality, True),
            ("Dependencias críticas", self.check_dependencies, True),
            ("Archivos de prueba", self.check_test_files, False),
            ("Archivo de requirements", self.check_requirements_file, False)
        ]
        
        for name, check_func, critical in checks:
            success = self.run_check(name, check_func, critical)
            if not success and critical:
                self.log(f"❌ Verificación crítica falló: {name}")
                return False
        
        return len(self.failed_checks) == 0
    
    def show_summary(self):
        """Mostrar resumen de verificaciones"""
        self.log("=" * 60)
        self.log("📊 RESUMEN DE VERIFICACIONES PRE-COMMIT")
        self.log("=" * 60)
        
        if self.passed_checks:
            self.log("✅ Verificaciones exitosas:")
            for check in self.passed_checks:
                self.log(f"   • {check}")
        
        if self.failed_checks:
            self.log("❌ Verificaciones fallidas:")
            for check in self.failed_checks:
                self.log(f"   • {check}")
        
        self.log(f"📈 Total: {len(self.passed_checks)} exitosas, {len(self.failed_checks)} fallidas")
        
        if not self.failed_checks:
            self.log("🎉 ¡Todas las verificaciones pasaron!")
        else:
            self.log("⚠️ Algunas verificaciones fallaron")
        
        self.log("=" * 60)

def main():
    """Función principal"""
    hook = PreCommitHook()
    
    try:
        success = hook.run_all_checks()
        hook.show_summary()
        
        if success:
            print("\n✅ Pre-commit hook completado exitosamente")
            return 0
        else:
            print("\n❌ Pre-commit hook falló. Por favor, corrige los errores antes de hacer commit.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️ Pre-commit hook interrumpido")
        return 1
    except Exception as e:
        print(f"\n💥 Error en pre-commit hook: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 