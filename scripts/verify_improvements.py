#!/usr/bin/env python3
"""
Script de Verificación de Mejoras - Proyecto J
==============================================

Este script verifica que todas las mejoras de seguridad y robustez
implementadas funcionan correctamente.

Uso:
    python scripts/verify_improvements.py
"""

import sys
import os
import yaml
import importlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ImprovementVerifier:
    """Verificador de mejoras implementadas"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
    
    def log_result(self, test_name: str, success: bool, message: str = ""):
        """Registra un resultado de test"""
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        self.results.append({
            'test': test_name,
            'success': success,
            'message': message,
            'status': status
        })
        
        if not success:
            self.errors.append(f"{test_name}: {message}")
        elif message:
            self.warnings.append(f"{test_name}: {message}")
    
    def verify_pyproject_toml(self) -> bool:
        """Verifica que pyproject.toml esté configurado correctamente"""
        try:
            pyproject_path = project_root / "pyproject.toml"
            if not pyproject_path.exists():
                self.log_result("pyproject.toml", False, "Archivo no existe")
                return False
            
            # Verificar que no esté vacío
            content = pyproject_path.read_text()
            if len(content.strip()) < 100:
                self.log_result("pyproject.toml", False, "Archivo parece estar vacío o incompleto")
                return False
            
            # Verificar secciones importantes
            required_sections = ["[build-system]", "[project]", "[tool.black]", "[tool.mypy]"]
            missing_sections = []
            
            for section in required_sections:
                if section not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                self.log_result("pyproject.toml", False, f"Secciones faltantes: {missing_sections}")
                return False
            
            self.log_result("pyproject.toml", True, "Configuración moderna implementada")
            return True
            
        except Exception as e:
            self.log_result("pyproject.toml", False, f"Error verificando: {e}")
            return False
    
    def verify_requirements_optimized(self) -> bool:
        """Verifica que requirements_unified.txt esté optimizado"""
        try:
            req_path = project_root / "requirements_unified.txt"
            if not req_path.exists():
                self.log_result("requirements_unified.txt", False, "Archivo no existe")
                return False
            
            content = req_path.read_text()
            
            # Verificar que no hay duplicados obvios
            lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
            duplicates = []
            seen = set()
            
            for line in lines:
                if '>=' in line:
                    package = line.split('>=')[0].strip()
                    if package in seen:
                        duplicates.append(package)
                    seen.add(package)
            
            if duplicates:
                self.log_result("requirements_unified.txt", False, f"Dependencias duplicadas: {duplicates}")
                return False
            
            # Verificar organización por categorías
            if "# Core Data Science" not in content:
                self.log_result("requirements_unified.txt", False, "Falta organización por categorías")
                return False
            
            self.log_result("requirements_unified.txt", True, "Dependencias optimizadas")
            return True
            
        except Exception as e:
            self.log_result("requirements_unified.txt", False, f"Error verificando: {e}")
            return False
    
    def verify_security_config(self) -> bool:
        """Verifica que la configuración de seguridad esté presente"""
        try:
            security_path = project_root / "config" / "security.yml"
            if not security_path.exists():
                self.log_result("security.yml", False, "Archivo de configuración de seguridad no existe")
                return False
            
            # Verificar que es un YAML válido
            with open(security_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Verificar secciones importantes (nueva estructura)
            required_sections = ['security', 'data_validation', 'secure_processing']
            missing_sections = []
            
            for section in required_sections:
                if section not in config:
                    missing_sections.append(section)
            
            if missing_sections:
                self.log_result("security.yml", False, f"Secciones faltantes: {missing_sections}")
                return False
            
            # Verificar patrones de seguridad en la nueva estructura
            if 'patterns' not in config['security']:
                self.log_result("security.yml", False, "Patrones de seguridad no configurados")
                return False
            
            self.log_result("security.yml", True, "Configuración de seguridad válida")
            return True
            
        except Exception as e:
            self.log_result("security.yml", False, f"Error verificando: {e}")
            return False
    
    def verify_dependency_manager(self) -> bool:
        """Verifica que el gestor de dependencias funcione"""
        try:
            # Intentar importar el módulo
            from processing.dependency_manager import dependency_manager
            
            # Verificar que tiene dependencias configuradas
            if not hasattr(dependency_manager, 'dependencies'):
                self.log_result("dependency_manager", False, "No tiene atributo 'dependencies'")
                return False
            
            if len(dependency_manager.dependencies) == 0:
                self.log_result("dependency_manager", False, "No hay dependencias configuradas")
                return False
            
            # Verificar funciones de conveniencia
            from processing.dependency_manager import is_spss_available, is_encoding_detection_available
            
            # Estas funciones deben existir y ser callables
            if not callable(is_spss_available):
                self.log_result("dependency_manager", False, "is_spss_available no es callable")
                return False
            
            if not callable(is_encoding_detection_available):
                self.log_result("dependency_manager", False, "is_encoding_detection_available no es callable")
                return False
            
            self.log_result("dependency_manager", True, "Gestor de dependencias funcional")
            return True
            
        except ImportError as e:
            self.log_result("dependency_manager", False, f"No se puede importar: {e}")
            return False
        except Exception as e:
            self.log_result("dependency_manager", False, f"Error verificando: {e}")
            return False
    
    def verify_io_module(self) -> bool:
        """Verifica que el módulo de I/O mejorado funcione"""
        try:
            from processing.io import (
                validate_file_security, 
                calculate_file_hash, 
                detect_encoding,
                FileSecurityError,
                FileValidationError
            )
            
            # Verificar que las funciones existen y son callables
            functions_to_check = [
                validate_file_security,
                calculate_file_hash,
                detect_encoding
            ]
            
            for func in functions_to_check:
                if not callable(func):
                    self.log_result("io_module", False, f"Función {func.__name__} no es callable")
                    return False
            
            # Verificar que las excepciones existen
            if not issubclass(FileSecurityError, Exception):
                self.log_result("io_module", False, "FileSecurityError no es una excepción válida")
                return False
            
            if not issubclass(FileValidationError, Exception):
                self.log_result("io_module", False, "FileValidationError no es una excepción válida")
                return False
            
            self.log_result("io_module", True, "Módulo de I/O mejorado funcional")
            return True
            
        except ImportError as e:
            self.log_result("io_module", False, f"No se puede importar: {e}")
            return False
        except Exception as e:
            self.log_result("io_module", False, f"Error verificando: {e}")
            return False
    
    def verify_data_validators(self) -> bool:
        """Verifica que el módulo de validación de datos funcione"""
        try:
            from processing.data_validators import (
                DataValidator, 
                ValidationRule, 
                ValidationResult,
                create_validator
            )
            
            # Verificar que las clases existen
            if not hasattr(DataValidator, '__init__'):
                self.log_result("data_validators", False, "DataValidator no tiene __init__")
                return False
            
            # Verificar que se puede crear una instancia
            validator = DataValidator()
            if not hasattr(validator, 'add_rule'):
                self.log_result("data_validators", False, "DataValidator no tiene método add_rule")
                return False
            
            # Verificar función de conveniencia
            if not callable(create_validator):
                self.log_result("data_validators", False, "create_validator no es callable")
                return False
            
            self.log_result("data_validators", True, "Módulo de validación funcional")
            return True
            
        except ImportError as e:
            self.log_result("data_validators", False, f"No se puede importar: {e}")
            return False
        except Exception as e:
            self.log_result("data_validators", False, f"Error verificando: {e}")
            return False
    
    def verify_streamlit_improvements(self) -> bool:
        """Verifica que las mejoras en Streamlit estén implementadas"""
        try:
            streamlit_path = project_root / "proyecto_j" / "streamlit_app.py"
            if not streamlit_path.exists():
                self.log_result("streamlit_improvements", False, "streamlit_app.py no existe")
                return False
            
            # Leer el archivo con manejo robusto de encoding
            content = None
            encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'mac_roman', 'iso-8859-1']
            
            for encoding in encodings_to_try:
                try:
                    with open(streamlit_path, 'r', encoding=encoding, errors='replace') as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                self.log_result("streamlit_improvements", False, "No se pudo leer el archivo con ningún encoding")
                return False
            
            # Verificar que tiene el gestor de dependencias
            if "dependency_manager" not in content:
                self.log_result("streamlit_improvements", False, "No se detecta integración con dependency_manager")
                return False
            
            # Verificar manejo de errores mejorado
            if "DEPENDENCY_MANAGER_LOADED" not in content:
                self.log_result("streamlit_improvements", False, "No se detecta manejo de dependencias")
                return False
            
            # Verificar funciones de fallback
            if "safe_import_chardet" not in content:
                self.log_result("streamlit_improvements", False, "No se detecta importación segura de chardet")
                return False
            
            self.log_result("streamlit_improvements", True, "Mejoras en Streamlit implementadas")
            return True
            
        except Exception as e:
            self.log_result("streamlit_improvements", False, f"Error verificando: {e}")
            return False
    
    def verify_test_configuration(self) -> bool:
        """Verifica que la configuración de tests esté mejorada"""
        try:
            # Verificar pytest.ini
            pytest_path = project_root / "tests" / "pytest.ini"
            if not pytest_path.exists():
                self.log_result("test_configuration", False, "pytest.ini no existe")
                return False
            
            # Verificar conftest.py
            conftest_path = project_root / "tests" / "conftest.py"
            if not conftest_path.exists():
                self.log_result("test_configuration", False, "conftest.py no existe")
                return False
            
            # Verificar que conftest.py tiene fixtures útiles
            conftest_content = conftest_path.read_text()
            if "sample_data" not in conftest_content:
                self.log_result("test_configuration", False, "conftest.py no tiene fixture sample_data")
                return False
            
            self.log_result("test_configuration", True, "Configuración de tests mejorada")
            return True
            
        except Exception as e:
            self.log_result("test_configuration", False, f"Error verificando: {e}")
            return False
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """Ejecuta todas las verificaciones"""
        print("🔍 Iniciando verificación de mejoras implementadas...")
        print("=" * 60)
        
        # Ejecutar verificaciones
        self.verify_pyproject_toml()
        self.verify_requirements_optimized()
        self.verify_security_config()
        self.verify_dependency_manager()
        self.verify_io_module()
        self.verify_data_validators()
        self.verify_streamlit_improvements()
        self.verify_test_configuration()
        
        # Mostrar resultados
        print("\n📊 RESULTADOS DE VERIFICACIÓN")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for result in self.results:
            print(f"{result['status']} {result['test']}")
            if result['message']:
                print(f"    {result['message']}")
            
            if result['success']:
                passed += 1
            else:
                failed += 1
        
        print("\n" + "=" * 60)
        print(f"✅ PASARON: {passed}")
        print(f"❌ FALLARON: {failed}")
        print(f"📈 TOTAL: {len(self.results)}")
        
        if self.warnings:
            print(f"\n⚠️ ADVERTENCIAS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.errors:
            print(f"\n❌ ERRORES ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        # Resumen final
        success_rate = (passed / len(self.results)) * 100 if self.results else 0
        
        print(f"\n🎯 TASA DE ÉXITO: {success_rate:.1f}%")
        
        if success_rate >= 90:
            print("🎉 ¡Excelente! Las mejoras están implementadas correctamente.")
        elif success_rate >= 70:
            print("👍 Bueno. La mayoría de las mejoras están implementadas.")
        else:
            print("⚠️ Necesita atención. Varias mejoras no están implementadas correctamente.")
        
        return {
            'passed': passed,
            'failed': failed,
            'total': len(self.results),
            'success_rate': success_rate,
            'errors': self.errors,
            'warnings': self.warnings,
            'results': self.results
        }


def main():
    """Función principal"""
    verifier = ImprovementVerifier()
    results = verifier.run_all_verifications()
    
    # Retornar código de salida apropiado
    if results['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 