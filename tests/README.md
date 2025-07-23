# 🧪 Tests Unificados - Proyecto J

## 📋 Descripción

Esta carpeta contiene todos los tests unificados del proyecto, organizados de manera estructurada y siguiendo las mejores prácticas de testing.

## 📁 Estructura

```
tests/
├── unit/                          # 🧪 Tests unitarios
│   ├── test_imports.py           # ✅ Importaciones del proyecto
│   ├── test_imports_legacy.py    # 📜 Tests legacy de importaciones
│   ├── test_excel_load_legacy.py # 📜 Tests legacy de carga Excel
│   └── test_variable_classifier_legacy.py # 📜 Tests legacy de clasificador
├── integration/                   # 🔗 Tests de integración
│   ├── test_pipeline.py          # ✅ Pipeline completo
│   ├── test_complex_grouping_legacy.py # 📜 Tests legacy de agrupaciones
│   ├── test_nl_query_legacy.py   # 📜 Tests legacy de consultas NL
│   └── test_streamlit_fix_legacy.py # 📜 Tests legacy de Streamlit
├── e2e/                          # 🌐 Tests end-to-end
│   └── test_app_workflow.py      # ✅ Flujo completo de aplicación
├── fixtures/                      # 📦 Datos de prueba
│   ├── sample_data.csv           # ✅ Datos de ejemplo
│   ├── test_config.yml           # ✅ Configuración de prueba
│   └── expected_outputs/         # ✅ Salidas esperadas
├── conftest.py                   # ⚙️ Configuración pytest
├── pytest.ini                   # ⚙️ Configuración pytest
├── run_tests.py                  # 🚀 Script de ejecución
└── README.md                     # 📚 Este archivo
```

## 🚀 Ejecución de Tests

### Ejecución Básica

```bash
# Ejecutar todos los tests
python tests/run_tests.py

# Ejecutar solo tests unitarios
python tests/run_tests.py --type unit

# Ejecutar solo tests de integración
python tests/run_tests.py --type integration

# Ejecutar solo tests E2E
python tests/run_tests.py --type e2e
```

### Ejecución Avanzada

```bash
# Con cobertura de código
python tests/run_tests.py --coverage

# Con reporte HTML
python tests/run_tests.py --html-report

# En modo verbose
python tests/run_tests.py --verbose

# En paralelo
python tests/run_tests.py --parallel

# Limpiar archivos temporales antes de ejecutar
python tests/run_tests.py --clean
```

### Ejecución Directa con pytest

```bash
# Ejecutar tests específicos
pytest tests/unit/test_imports.py -v

# Ejecutar tests con marcadores
pytest -m unit -v
pytest -m integration -v
pytest -m e2e -v

# Ejecutar tests con cobertura
pytest --cov=proyecto_j --cov=processing --cov=orchestrator
```

## 📊 Tipos de Tests

### 🧪 Tests Unitarios (`unit/`)

- **Propósito**: Probar funciones y módulos individuales
- **Alcance**: Funciones específicas, clases, métodos
- **Velocidad**: Rápidos (< 1 segundo cada uno)
- **Dependencias**: Mínimas, principalmente fixtures

**Ejemplos**:
- Importaciones de módulos
- Funciones de cálculo estadístico
- Validadores de datos
- Clasificadores de variables

### 🔗 Tests de Integración (`integration/`)

- **Propósito**: Probar interacción entre módulos
- **Alcance**: Múltiples componentes trabajando juntos
- **Velocidad**: Moderados (1-10 segundos cada uno)
- **Dependencias**: Fixtures, datos de prueba

**Ejemplos**:
- Pipeline completo de procesamiento
- Integración con Streamlit
- Sistema de logging
- Procesamiento de datos con validación

### 🌐 Tests E2E (`e2e/`)

- **Propósito**: Probar flujos completos de la aplicación
- **Alcance**: Toda la aplicación desde entrada hasta salida
- **Velocidad**: Lentos (10-60 segundos cada uno)
- **Dependencias**: Datos reales, configuración completa

**Ejemplos**:
- Flujo completo de análisis de datos
- Exportación de reportes
- Consultas en lenguaje natural
- Procesamiento de archivos grandes

## 📦 Fixtures

### Datos de Prueba

- `sample_data`: DataFrame pequeño con datos de ejemplo
- `sample_data_large`: DataFrame grande para tests de rendimiento
- `sample_data_missing`: DataFrame con valores faltantes
- `sample_csv_file`: Archivo CSV temporal
- `sample_excel_file`: Archivo Excel temporal

### Configuración

- `test_config`: Configuración de prueba
- `temp_dir`: Directorio temporal
- `chile_geographic_data`: Datos geográficos de Chile
- `expected_outputs`: Salidas esperadas para validación

### Mocks

- `mock_streamlit`: Mock de Streamlit para tests

## 🏷️ Marcadores

### Marcadores Automáticos

Los tests se marcan automáticamente según su ubicación:

- `@pytest.mark.unit`: Tests en `unit/`
- `@pytest.mark.integration`: Tests en `integration/`
- `@pytest.mark.e2e`: Tests en `e2e/`

### Marcadores Manuales

- `@pytest.mark.slow`: Tests lentos (performance, datos grandes)
- `@pytest.mark.performance`: Tests de rendimiento

## 📈 Cobertura

### Generar Reporte de Cobertura

```bash
# Con reporte HTML
pytest --cov=proyecto_j --cov=processing --cov=orchestrator --cov-report=html:tests/coverage_html

# Con reporte XML (para CI/CD)
pytest --cov=proyecto_j --cov=processing --cov=orchestrator --cov-report=xml:tests/coverage.xml
```

### Ver Cobertura

- **HTML**: Abrir `tests/coverage_html/index.html`
- **XML**: Usar con herramientas de CI/CD

## 🔧 Configuración

### pytest.ini

Configuración centralizada de pytest con:
- Marcadores personalizados
- Configuración de cobertura
- Filtros de warnings
- Timeouts
- Paralelización

### conftest.py

Fixtures y configuración común:
- Datos de prueba
- Mocks
- Configuración de paths
- Marcadores automáticos

## 📋 Convenciones

### Nomenclatura

- Archivos: `test_*.py`
- Clases: `Test*`
- Métodos: `test_*`
- Fixtures: `*_fixture`

### Estructura de Tests

```python
class TestModuleName:
    """Tests para el módulo ModuleName"""
    
    def test_specific_functionality(self, fixture_name):
        """Test de funcionalidad específica"""
        # Arrange
        # Act
        # Assert
```

### Assertions

- Usar assertions específicos de pytest
- Incluir mensajes descriptivos
- Verificar tipos y valores esperados

## 🚨 Troubleshooting

### Problemas Comunes

1. **ImportError**: Verificar que el path del proyecto está configurado
2. **FixtureNotFound**: Verificar que las fixtures están definidas en conftest.py
3. **Timeout**: Aumentar timeout en pytest.ini para tests lentos
4. **MemoryError**: Reducir tamaño de datos de prueba

### Debugging

```bash
# Ejecutar tests con más información
pytest -v -s --tb=long

# Ejecutar test específico con debug
pytest tests/unit/test_imports.py::TestImports::test_core_imports -v -s
```

## 🔄 Mantenimiento

### Agregar Nuevos Tests

1. Crear archivo en la carpeta apropiada (`unit/`, `integration/`, `e2e/`)
2. Seguir convenciones de nomenclatura
3. Usar fixtures existentes cuando sea posible
4. Agregar marcadores apropiados

### Actualizar Fixtures

1. Modificar `conftest.py`
2. Verificar que todos los tests siguen funcionando
3. Actualizar documentación si es necesario

### Limpiar Tests Legacy

Los tests marcados como "legacy" son versiones anteriores que se mantienen por compatibilidad. Se pueden eliminar cuando:

1. Los nuevos tests cubren la misma funcionalidad
2. Se verifica que no hay regresiones
3. Se actualiza la documentación

## 📚 Recursos

- [Documentación de pytest](https://docs.pytest.org/)
- [Mejores prácticas de testing](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [Fixtures avanzadas](https://docs.pytest.org/en/stable/explanation/fixtures.html)
- [Marcadores personalizados](https://docs.pytest.org/en/stable/how-to/mark.html) 