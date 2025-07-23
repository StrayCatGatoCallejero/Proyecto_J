"""
Gestor de Dependencias Opcionales - Proyecto J
==============================================

Responsabilidades:
- Detección automática de dependencias opcionales
- Manejo graceful de dependencias faltantes
- Configuración de funcionalidades según disponibilidad
- Logging de estado de dependencias
- Sugerencias de instalación para funcionalidades faltantes
"""

import importlib
import sys
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import logging
from pathlib import Path

# Configurar logging
logger = logging.getLogger(__name__)


@dataclass
class DependencyInfo:
    """Información de una dependencia"""
    name: str
    package_name: str
    version: Optional[str] = None
    is_available: bool = False
    import_name: Optional[str] = None
    install_command: str = ""
    description: str = ""
    critical: bool = False
    features_enabled: List[str] = field(default_factory=list)


class DependencyManager:
    """
    Gestor de dependencias opcionales con detección automática.
    """
    
    def __init__(self):
        """Inicializa el gestor de dependencias"""
        self.dependencies: Dict[str, DependencyInfo] = {}
        self.feature_map: Dict[str, List[str]] = {}
        self._setup_dependencies()
        self._detect_availability()
    
    def _setup_dependencies(self) -> None:
        """Configura las dependencias conocidas"""
        
        # Dependencias para SPSS/Stata
        self.dependencies['pyreadstat'] = DependencyInfo(
            name="pyreadstat",
            package_name="pyreadstat",
            import_name="pyreadstat",
            install_command="pip install pyreadstat",
            description="Soporte para archivos SPSS (.sav) y Stata (.dta)",
            features_enabled=["spss_loading", "stata_loading", "spss_metadata"]
        )
        
        # Dependencias para detección de encoding
        self.dependencies['chardet'] = DependencyInfo(
            name="chardet",
            package_name="chardet",
            import_name="chardet",
            install_command="pip install chardet",
            description="Detección automática de encoding de archivos",
            features_enabled=["encoding_detection", "robust_csv_loading"]
        )
        
        # Dependencias para machine learning avanzado
        self.dependencies['tensorflow'] = DependencyInfo(
            name="TensorFlow",
            package_name="tensorflow",
            import_name="tensorflow",
            install_command="pip install tensorflow",
            description="Machine learning avanzado y deep learning",
            features_enabled=["deep_learning", "neural_networks", "advanced_ml"]
        )
        
        self.dependencies['torch'] = DependencyInfo(
            name="PyTorch",
            package_name="torch",
            import_name="torch",
            install_command="pip install torch",
            description="Machine learning con PyTorch",
            features_enabled=["pytorch_models", "neural_networks", "advanced_ml"]
        )
        
        # Dependencias para procesamiento de lenguaje natural
        self.dependencies['transformers'] = DependencyInfo(
            name="Transformers",
            package_name="transformers",
            import_name="transformers",
            install_command="pip install transformers",
            description="Procesamiento de lenguaje natural avanzado",
            features_enabled=["nlp_advanced", "text_analysis", "language_models"]
        )
        
        # Dependencias para visualización avanzada
        self.dependencies['plotly'] = DependencyInfo(
            name="Plotly",
            package_name="plotly",
            import_name="plotly",
            install_command="pip install plotly",
            description="Visualizaciones interactivas avanzadas",
            features_enabled=["interactive_plots", "3d_visualizations", "dashboards"]
        )
        
        # Dependencias para análisis de series temporales
        self.dependencies['statsmodels'] = DependencyInfo(
            name="Statsmodels",
            package_name="statsmodels",
            import_name="statsmodels",
            install_command="pip install statsmodels",
            description="Análisis estadístico avanzado y series temporales",
            features_enabled=["time_series", "statistical_tests", "regression_advanced"]
        )
        
        # Dependencias para procesamiento paralelo
        self.dependencies['joblib'] = DependencyInfo(
            name="Joblib",
            package_name="joblib",
            import_name="joblib",
            install_command="pip install joblib",
            description="Procesamiento paralelo y caching",
            features_enabled=["parallel_processing", "caching", "performance_optimization"]
        )
        
        # Dependencias para compresión de archivos
        self.dependencies['lz4'] = DependencyInfo(
            name="LZ4",
            package_name="lz4",
            import_name="lz4",
            install_command="pip install lz4",
            description="Compresión rápida de archivos",
            features_enabled=["fast_compression", "large_file_handling"]
        )
        
        # Dependencias para monitoreo de rendimiento
        self.dependencies['psutil'] = DependencyInfo(
            name="psutil",
            package_name="psutil",
            import_name="psutil",
            install_command="pip install psutil",
            description="Monitoreo de recursos del sistema",
            features_enabled=["system_monitoring", "memory_tracking", "performance_metrics"]
        )
        
        # Configurar mapeo de características
        for dep_name, dep_info in self.dependencies.items():
            for feature in dep_info.features_enabled:
                if feature not in self.feature_map:
                    self.feature_map[feature] = []
                self.feature_map[feature].append(dep_name)
    
    def _detect_availability(self) -> None:
        """Detecta la disponibilidad de todas las dependencias"""
        for dep_name, dep_info in self.dependencies.items():
            try:
                if dep_info.import_name:
                    module = importlib.import_module(dep_info.import_name)
                    dep_info.is_available = True
                    
                    # Intentar obtener versión
                    if hasattr(module, '__version__'):
                        dep_info.version = module.__version__
                    elif hasattr(module, 'version'):
                        dep_info.version = module.version
                    
                    logger.debug(f"✅ {dep_name} disponible (v{dep_info.version or 'unknown'})")
                else:
                    # Verificar si el paquete está instalado
                    spec = importlib.util.find_spec(dep_info.package_name)
                    dep_info.is_available = spec is not None
                    
                    if dep_info.is_available:
                        logger.debug(f"✅ {dep_name} disponible")
                    else:
                        logger.debug(f"❌ {dep_name} no disponible")
                        
            except ImportError:
                dep_info.is_available = False
                logger.debug(f"❌ {dep_name} no disponible")
            except Exception as e:
                dep_info.is_available = False
                logger.warning(f"⚠️ Error detectando {dep_name}: {e}")
    
    def is_available(self, dependency_name: str) -> bool:
        """
        Verifica si una dependencia está disponible.
        
        Args:
            dependency_name: Nombre de la dependencia
            
        Returns:
            True si la dependencia está disponible
        """
        if dependency_name not in self.dependencies:
            logger.warning(f"Dependencia no registrada: {dependency_name}")
            return False
        
        return self.dependencies[dependency_name].is_available
    
    def is_feature_available(self, feature_name: str) -> bool:
        """
        Verifica si una característica está disponible.
        
        Args:
            feature_name: Nombre de la característica
            
        Returns:
            True si la característica está disponible
        """
        if feature_name not in self.feature_map:
            return False
        
        # Una característica está disponible si al menos una de sus dependencias lo está
        dependencies = self.feature_map[feature_name]
        return any(self.is_available(dep) for dep in dependencies)
    
    def get_missing_dependencies(self, feature_name: str) -> List[str]:
        """
        Obtiene las dependencias faltantes para una característica.
        
        Args:
            feature_name: Nombre de la característica
            
        Returns:
            Lista de dependencias faltantes
        """
        if feature_name not in self.feature_map:
            return []
        
        missing = []
        for dep_name in self.feature_map[feature_name]:
            if not self.is_available(dep_name):
                missing.append(dep_name)
        
        return missing
    
    def get_install_commands(self, feature_name: str) -> List[str]:
        """
        Obtiene los comandos de instalación para una característica.
        
        Args:
            feature_name: Nombre de la característica
            
        Returns:
            Lista de comandos de instalación
        """
        missing_deps = self.get_missing_dependencies(feature_name)
        commands = []
        
        for dep_name in missing_deps:
            if dep_name in self.dependencies:
                commands.append(self.dependencies[dep_name].install_command)
        
        return commands
    
    def safe_import(self, dependency_name: str, fallback: Optional[Callable] = None):
        """
        Importa una dependencia de forma segura con fallback.
        
        Args:
            dependency_name: Nombre de la dependencia
            fallback: Función de fallback si la dependencia no está disponible
            
        Returns:
            Módulo importado o resultado del fallback
        """
        if not self.is_available(dependency_name):
            if fallback:
                logger.warning(f"⚠️ {dependency_name} no disponible, usando fallback")
                return fallback()
            else:
                raise ImportError(f"Dependencia requerida no disponible: {dependency_name}")
        
        try:
            dep_info = self.dependencies[dependency_name]
            if dep_info.import_name:
                return importlib.import_module(dep_info.import_name)
            else:
                return importlib.import_module(dep_info.package_name)
        except Exception as e:
            logger.error(f"Error importando {dependency_name}: {e}")
            if fallback:
                return fallback()
            raise
    
    def get_status_report(self) -> Dict[str, Any]:
        """
        Genera un reporte del estado de todas las dependencias.
        
        Returns:
            Diccionario con el estado de las dependencias
        """
        report = {
            'total_dependencies': len(self.dependencies),
            'available_dependencies': sum(1 for d in self.dependencies.values() if d.is_available),
            'missing_dependencies': sum(1 for d in self.dependencies.values() if not d.is_available),
            'dependencies': {},
            'features': {},
            'recommendations': []
        }
        
        # Estado de dependencias
        for dep_name, dep_info in self.dependencies.items():
            report['dependencies'][dep_name] = {
                'available': dep_info.is_available,
                'version': dep_info.version,
                'description': dep_info.description,
                'install_command': dep_info.install_command,
                'features': dep_info.features_enabled
            }
        
        # Estado de características
        for feature_name in self.feature_map.keys():
            report['features'][feature_name] = {
                'available': self.is_feature_available(feature_name),
                'dependencies': self.feature_map[feature_name],
                'missing_dependencies': self.get_missing_dependencies(feature_name)
            }
        
        # Recomendaciones
        for feature_name, feature_info in report['features'].items():
            if not feature_info['available']:
                missing_deps = feature_info['missing_dependencies']
                if missing_deps:
                    commands = self.get_install_commands(feature_name)
                    report['recommendations'].append({
                        'feature': feature_name,
                        'missing_dependencies': missing_deps,
                        'install_commands': commands
                    })
        
        return report
    
    def print_status(self) -> None:
        """Imprime el estado de las dependencias en consola"""
        report = self.get_status_report()
        
        print("\n" + "="*60)
        print("📦 ESTADO DE DEPENDENCIAS - PROYECTO J")
        print("="*60)
        
        print(f"\n📊 Resumen:")
        print(f"   • Total de dependencias: {report['total_dependencies']}")
        print(f"   • Disponibles: {report['available_dependencies']}")
        print(f"   • Faltantes: {report['missing_dependencies']}")
        
        print(f"\n✅ Dependencias Disponibles:")
        for dep_name, dep_info in report['dependencies'].items():
            if dep_info['available']:
                version = dep_info['version'] or 'unknown'
                print(f"   • {dep_name} (v{version})")
        
        print(f"\n❌ Dependencias Faltantes:")
        for dep_name, dep_info in report['dependencies'].items():
            if not dep_info['available']:
                print(f"   • {dep_name}: {dep_info['description']}")
                print(f"     Comando: {dep_info['install_command']}")
        
        print(f"\n🔧 Características:")
        for feature_name, feature_info in report['features'].items():
            status = "✅" if feature_info['available'] else "❌"
            print(f"   {status} {feature_name}")
            if not feature_info['available']:
                missing = ", ".join(feature_info['missing_dependencies'])
                print(f"      Dependencias faltantes: {missing}")
        
        if report['recommendations']:
            print(f"\n💡 Recomendaciones:")
            for rec in report['recommendations']:
                print(f"   • Para habilitar '{rec['feature']}':")
                for cmd in rec['install_commands']:
                    print(f"     {cmd}")
        
        print("\n" + "="*60)


# Instancia global del gestor de dependencias
dependency_manager = DependencyManager()


# Funciones de conveniencia
def is_spss_available() -> bool:
    """Verifica si el soporte para SPSS está disponible"""
    return dependency_manager.is_available('pyreadstat')


def is_encoding_detection_available() -> bool:
    """Verifica si la detección de encoding está disponible"""
    return dependency_manager.is_available('chardet')


def is_advanced_ml_available() -> bool:
    """Verifica si el machine learning avanzado está disponible"""
    return (dependency_manager.is_available('tensorflow') or 
            dependency_manager.is_available('torch'))


def is_nlp_advanced_available() -> bool:
    """Verifica si el NLP avanzado está disponible"""
    return dependency_manager.is_available('transformers')


def is_interactive_plots_available() -> bool:
    """Verifica si las visualizaciones interactivas están disponibles"""
    return dependency_manager.is_available('plotly')


def is_time_series_available() -> bool:
    """Verifica si el análisis de series temporales está disponible"""
    return dependency_manager.is_available('statsmodels')


def is_parallel_processing_available() -> bool:
    """Verifica si el procesamiento paralelo está disponible"""
    return dependency_manager.is_available('joblib')


def safe_import_pyreadstat():
    """Importa pyreadstat de forma segura"""
    def fallback():
        logger.warning("pyreadstat no disponible. Los archivos SPSS no serán soportados.")
        return None
    
    return dependency_manager.safe_import('pyreadstat', fallback)


def safe_import_chardet():
    """Importa chardet de forma segura"""
    def fallback():
        logger.warning("chardet no disponible. La detección de encoding será limitada.")
        return None
    
    return dependency_manager.safe_import('chardet', fallback)


def safe_import_tensorflow():
    """Importa tensorflow de forma segura"""
    def fallback():
        logger.warning("tensorflow no disponible. El deep learning no estará disponible.")
        return None
    
    return dependency_manager.safe_import('tensorflow', fallback)


def safe_import_torch():
    """Importa torch de forma segura"""
    def fallback():
        logger.warning("torch no disponible. Los modelos PyTorch no estarán disponibles.")
        return None
    
    return dependency_manager.safe_import('torch', fallback)


def safe_import_transformers():
    """Importa transformers de forma segura"""
    def fallback():
        logger.warning("transformers no disponible. El NLP avanzado no estará disponible.")
        return None
    
    return dependency_manager.safe_import('transformers', fallback)


def safe_import_plotly():
    """Importa plotly de forma segura"""
    def fallback():
        logger.warning("plotly no disponible. Las visualizaciones interactivas serán limitadas.")
        return None
    
    return dependency_manager.safe_import('plotly', fallback)


def safe_import_statsmodels():
    """Importa statsmodels de forma segura"""
    def fallback():
        logger.warning("statsmodels no disponible. El análisis estadístico avanzado será limitado.")
        return None
    
    return dependency_manager.safe_import('statsmodels', fallback)


def safe_import_joblib():
    """Importa joblib de forma segura"""
    def fallback():
        logger.warning("joblib no disponible. El procesamiento paralelo será limitado.")
        return None
    
    return dependency_manager.safe_import('joblib', fallback)


# Inicializar dependencias al importar el módulo
if __name__ == "__main__":
    dependency_manager.print_status() 