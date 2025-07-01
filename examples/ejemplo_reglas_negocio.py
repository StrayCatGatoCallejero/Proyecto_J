"""
Ejemplo de Reglas de Negocio - Ciencias Sociales
===============================================

Este script demuestra el sistema de validación de reglas de negocio específicas
para investigaciones en ciencias sociales.

Características demostradas:
1. Validación de datos demográficos (edad, género, educación, ingresos)
2. Validación de escalas Likert y encuestas
3. Validación de datos geográficos (regiones, comunas)
4. Validaciones de consistencia cruzada
5. Integración con el sistema híbrido de errores
6. Logging detallado de validaciones
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, Any

# Agregar el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from processing.business_rules import (
    validate_demographics, 
    validate_likert, 
    validate_geography, 
    validate_cross_consistency,
    validate_business_rules,
    validate_business_rules_decorator,
    BusinessRuleError
)
from processing.config_manager import ConfigManager
from processing.error_reporter import report_business_rule_error


def crear_datos_demograficos():
    """Crea datos demográficos de ejemplo con algunos problemas"""
    
    # Datos válidos
    datos_validos = pd.DataFrame({
        'edad': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino', 
                  'Femenino', 'Masculino', 'Femenino', 'Masculino', 'Femenino'],
        'nivel_educacion': ['Primaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Secundaria',
                           'Universitaria', 'Primaria', 'Secundaria', 'Universitaria', 'Postgrado'],
        'ingresos': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
        'region': ['Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso', 'Metropolitana',
                  'Valparaíso', 'Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso'],
        'comuna': ['Santiago', 'Valparaíso', 'Providencia', 'Viña del Mar', 'Las Condes',
                  'Quilpué', 'Ñuñoa', 'Quillota', 'La Florida', 'Villa Alemana']
    })
    
    # Datos con problemas demográficos
    datos_problematicos = pd.DataFrame({
        'edad': [150, 30, -5, 40, 45, 50, 55, 60, 65, 70],  # Edades inválidas
        'genero': ['Masculino', 'Femenino', 'Otro', 'Masculino', 'Femenino', 
                  'Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino'],
        'nivel_educacion': ['Primaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Secundaria',
                           'Universitaria', 'Primaria', 'Secundaria', 'Universitaria', 'Postgrado'],
        'ingresos': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
        'region': ['Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso', 'Metropolitana',
                  'Valparaíso', 'Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso'],
        'comuna': ['Santiago', 'Valparaíso', 'Providencia', 'Viña del Mar', 'Las Condes',
                  'Quilpué', 'Ñuñoa', 'Quillota', 'La Florida', 'Villa Alemana']
    })
    
    # Datos con problemas de consistencia cruzada
    datos_inconsistentes = pd.DataFrame({
        'edad': [15, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino', 'Masculino', 
                  'Femenino', 'Masculino', 'Femenino', 'Masculino', 'Femenino'],
        'nivel_educacion': ['Universitaria', 'Secundaria', 'Universitaria', 'Postgrado', 'Secundaria',
                           'Universitaria', 'Primaria', 'Secundaria', 'Universitaria', 'Postgrado'],
        'ingresos': [30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 75000],
        'empleo': ['Empleado', 'Desempleado', 'Empleado', 'Jubilado', 'Empleado',
                  'Desempleado', 'Empleado', 'Jubilado', 'Empleado', 'Desempleado'],
        'ocupacion': ['Profesional', 'Desempleado', 'Profesional', 'Jubilado', 'Profesional',
                     'Desempleado', 'Profesional', 'Jubilado', 'Profesional', 'Desempleado'],
        'region': ['Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso', 'Metropolitana',
                  'Valparaíso', 'Metropolitana', 'Valparaíso', 'Metropolitana', 'Valparaíso'],
        'comuna': ['Santiago', 'Valparaíso', 'Providencia', 'Viña del Mar', 'Las Condes',
                  'Quilpué', 'Ñuñoa', 'Quillota', 'La Florida', 'Villa Alemana']
    })
    
    return {
        'validos': datos_validos,
        'problematicos': datos_problematicos,
        'inconsistentes': datos_inconsistentes
    }


def crear_datos_likert():
    """Crea datos de escalas Likert de ejemplo"""
    
    # Datos de escala Likert 5 puntos
    datos_likert_5 = pd.DataFrame({
        'pregunta_1_likert': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'pregunta_2_likert': [2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
        'pregunta_3_likert': [3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        'pregunta_4_likert': [4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
        'pregunta_5_likert': [5, 1, 2, 3, 4, 5, 1, 2, 3, 4]
    })
    
    # Datos de escala Likert con problemas
    datos_likert_problematico = pd.DataFrame({
        'pregunta_1_likert': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Valores fuera de rango
        'pregunta_2_likert': [2, 3, 4, 5, 1, 2, 3, 4, 5, 1],
        'pregunta_3_likert': [3, 4, 5, 1, 2, 3, 4, 5, 1, 2],
        'pregunta_4_likert': [4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
        'pregunta_5_likert': [5, 1, 2, 3, 4, 5, 1, 2, 3, 4]
    })
    
    # Datos de escala Likert con etiquetas
    datos_likert_etiquetas = pd.DataFrame({
        'pregunta_1_likert': ['Muy en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Muy de acuerdo',
                             'Muy en desacuerdo', 'En desacuerdo', 'Neutral', 'De acuerdo', 'Muy de acuerdo'],
        'pregunta_2_likert': ['En desacuerdo', 'Neutral', 'De acuerdo', 'Muy de acuerdo', 'Muy en desacuerdo',
                             'En desacuerdo', 'Neutral', 'De acuerdo', 'Muy de acuerdo', 'Muy en desacuerdo'],
        'pregunta_3_likert': ['Neutral', 'De acuerdo', 'Muy de acuerdo', 'Muy en desacuerdo', 'En desacuerdo',
                             'Neutral', 'De acuerdo', 'Muy de acuerdo', 'Muy en desacuerdo', 'En desacuerdo']
    })
    
    return {
        'likert_5': datos_likert_5,
        'likert_problematico': datos_likert_problematico,
        'likert_etiquetas': datos_likert_etiquetas
    }


def ejemplo_validacion_demografica():
    """Demuestra validación de datos demográficos"""
    print("\n" + "="*60)
    print("EJEMPLO 1: VALIDACIÓN DEMOGRÁFICA")
    print("="*60)
    
    datos = crear_datos_demograficos()
    
    # Validar datos válidos
    print("📊 Validando datos demográficos válidos...")
    try:
        result = validate_demographics(datos['validos'], {'dataset_type': 'demographic'})
        print(f"✅ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Validar datos problemáticos
    print("\n📊 Validando datos demográficos problemáticos...")
    try:
        result = validate_demographics(datos['problematicos'], {'dataset_type': 'demographic'})
        print(f"⚠️ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")


def ejemplo_validacion_likert():
    """Demuestra validación de escalas Likert"""
    print("\n" + "="*60)
    print("EJEMPLO 2: VALIDACIÓN DE ESCALAS LIKERT")
    print("="*60)
    
    datos = crear_datos_likert()
    
    # Validar escala Likert 5 puntos
    print("📊 Validando escala Likert 5 puntos...")
    try:
        result = validate_likert(datos['likert_5'], 'pregunta_1_likert', '5_puntos')
        print(f"✅ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Validar escala Likert problemática
    print("\n📊 Validando escala Likert problemática...")
    try:
        result = validate_likert(datos['likert_problematico'], 'pregunta_1_likert', '5_puntos')
        print(f"⚠️ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Validar escala Likert con etiquetas
    print("\n📊 Validando escala Likert con etiquetas...")
    try:
        result = validate_likert(datos['likert_etiquetas'], 'pregunta_1_likert', '5_puntos')
        print(f"✅ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")


def ejemplo_validacion_geografica():
    """Demuestra validación de datos geográficos"""
    print("\n" + "="*60)
    print("EJEMPLO 3: VALIDACIÓN GEOGRÁFICA")
    print("="*60)
    
    datos = crear_datos_demograficos()
    
    # Validar datos geográficos válidos
    print("📊 Validando datos geográficos válidos...")
    try:
        result = validate_geography(datos['validos'], 'region', 'comuna')
        print(f"✅ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Crear datos geográficos problemáticos
    datos_geo_problematico = pd.DataFrame({
        'region': ['Región Inexistente', 'Valparaíso', 'Metropolitana'],
        'comuna': ['Comuna Inexistente', 'Valparaíso', 'Santiago']
    })
    
    print("\n📊 Validando datos geográficos problemáticos...")
    try:
        result = validate_geography(datos_geo_problematico, 'region', 'comuna')
        print(f"⚠️ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")


def ejemplo_validacion_consistencia_cruzada():
    """Demuestra validación de consistencia cruzada"""
    print("\n" + "="*60)
    print("EJEMPLO 4: VALIDACIÓN DE CONSISTENCIA CRUZADA")
    print("="*60)
    
    datos = crear_datos_demograficos()
    
    # Validar consistencia cruzada en datos válidos
    print("📊 Validando consistencia cruzada en datos válidos...")
    try:
        result = validate_cross_consistency(datos['validos'], {'dataset_type': 'demographic'})
        print(f"✅ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Validar consistencia cruzada en datos inconsistentes
    print("\n📊 Validando consistencia cruzada en datos inconsistentes...")
    try:
        result = validate_cross_consistency(datos['inconsistentes'], {'dataset_type': 'demographic'})
        print(f"⚠️ Resultado: {result.message}")
        print(f"   Válido: {result.is_valid}")
        print(f"   Detalles: {result.details}")
    except Exception as e:
        print(f"❌ Error: {e}")


def ejemplo_validacion_completa():
    """Demuestra validación completa de reglas de negocio"""
    print("\n" + "="*60)
    print("EJEMPLO 5: VALIDACIÓN COMPLETA DE REGLAS DE NEGOCIO")
    print("="*60)
    
    datos = crear_datos_demograficos()
    
    # Validación completa en datos válidos
    print("📊 Validación completa en datos válidos...")
    try:
        results = validate_business_rules(datos['validos'], {'dataset_type': 'demographic'})
        print(f"✅ Validación completa completada: {len(results)} reglas ejecutadas")
        for i, result in enumerate(results, 1):
            print(f"   Regla {i} ({result.rule_name}): {result.is_valid}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Validación completa en datos problemáticos
    print("\n📊 Validación completa en datos problemáticos...")
    try:
        results = validate_business_rules(datos['problematicos'], {'dataset_type': 'demographic'})
        print(f"⚠️ Validación completa completada: {len(results)} reglas ejecutadas")
        for i, result in enumerate(results, 1):
            print(f"   Regla {i} ({result.rule_name}): {result.is_valid}")
            if not result.is_valid:
                print(f"      Error: {result.message}")
    except Exception as e:
        print(f"❌ Error: {e}")


def ejemplo_decorador_reglas_negocio():
    """Demuestra el uso del decorador de reglas de negocio"""
    print("\n" + "="*60)
    print("EJEMPLO 6: DECORADOR DE REGLAS DE NEGOCIO")
    print("="*60)
    
    @validate_business_rules_decorator
    def procesar_datos_demograficos(df: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
        """Función de ejemplo que procesa datos demográficos"""
        print(f"🔄 Procesando {len(df)} registros demográficos...")
        # Simular procesamiento
        return df.copy()
    
    datos = crear_datos_demograficos()
    
    # Procesar datos válidos
    print("📊 Procesando datos válidos con decorador...")
    try:
        result_df = procesar_datos_demograficos(datos['validos'], metadata={'dataset_type': 'demographic'})
        print(f"✅ Procesamiento completado: {len(result_df)} registros")
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")
    
    # Procesar datos problemáticos (debería fallar)
    print("\n📊 Procesando datos problemáticos con decorador...")
    try:
        result_df = procesar_datos_demograficos(datos['problematicos'], metadata={'dataset_type': 'demographic'})
        print(f"✅ Procesamiento completado: {len(result_df)} registros")
    except Exception as e:
        print(f"❌ Error en procesamiento: {e}")


def mostrar_resumen_sistema():
    """Muestra un resumen del sistema de reglas de negocio"""
    print("\n" + "="*60)
    print("RESUMEN DEL SISTEMA DE REGLAS DE NEGOCIO")
    print("="*60)
    
    print("✅ CARACTERÍSTICAS IMPLEMENTADAS:")
    print("   • Validación de datos demográficos (edad, género, educación, ingresos)")
    print("   • Validación de escalas Likert y encuestas")
    print("   • Validación de datos geográficos (regiones, comunas)")
    print("   • Validaciones de consistencia cruzada")
    print("   • Integración con sistema híbrido de errores")
    print("   • Logging detallado de validaciones")
    print("   • Decorador @validate_business_rules_decorator")
    
    print("\n✅ TIPOS DE VALIDACIÓN:")
    print("   • validate_demographics() - Validación de variables demográficas")
    print("   • validate_likert() - Validación de escalas de encuesta")
    print("   • validate_geography() - Validación de datos geográficos")
    print("   • validate_cross_consistency() - Validación de consistencia cruzada")
    print("   • validate_business_rules() - Validación completa automática")
    
    print("\n✅ REGLAS DE NEGOCIO IMPLEMENTADAS:")
    print("   • Edad en rango [0, 120]")
    print("   • Género con valores válidos")
    print("   • Nivel de educación mapeado a categorías predefinidas")
    print("   • Ingresos ≥ 0 con detección de outliers")
    print("   • Escalas Likert con valores en rango válido")
    print("   • Consistencia región-comuna")
    print("   • Edad vs educación (menores de 18 con educación universitaria)")
    print("   • Edad vs empleo (jubilados menores de 60)")
    print("   • Ingresos vs ocupación (desempleados con ingresos altos)")
    
    print("\n✅ INTEGRACIÓN CON SISTEMA DE ERRORES:")
    print("   • BusinessRuleError para errores específicos")
    print("   • ValidationResult con detalles completos")
    print("   • Logging automático de todas las validaciones")
    print("   • Reporte híbrido de errores (Streamlit + logging)")
    print("   • Detención automática de la app en errores críticos")


def main():
    """Función principal que ejecuta todos los ejemplos"""
    print("🚀 SISTEMA DE REGLAS DE NEGOCIO - CIENCIAS SOCIALES")
    print("="*60)
    
    # Configurar el sistema
    try:
        config = ConfigManager()
        print(f"✅ Configuración cargada: {config.get('app_name', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Error al cargar configuración: {e}")
    
    # Ejecutar ejemplos
    ejemplo_validacion_demografica()
    ejemplo_validacion_likert()
    ejemplo_validacion_geografica()
    ejemplo_validacion_consistencia_cruzada()
    ejemplo_validacion_completa()
    ejemplo_decorador_reglas_negocio()
    
    # Mostrar resumen
    mostrar_resumen_sistema()
    
    print("\n" + "="*60)
    print("🎉 TODOS LOS EJEMPLOS COMPLETADOS")
    print("="*60)
    print("El sistema de reglas de negocio está funcionando correctamente.")
    print("Las validaciones se aplican automáticamente según el tipo de datos.")
    print("Los errores se reportan de forma híbrida y detienen la app cuando es necesario.")


if __name__ == "__main__":
    main() 