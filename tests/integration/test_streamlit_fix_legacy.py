#!/usr/bin/env python3
"""
Script de prueba para verificar que los problemas de Streamlit están solucionados
"""

import sys
import os

def test_imports():
    """Prueba las importaciones problemáticas"""
    print("🔍 Probando importaciones...")
    
    # Test scipy
    try:
        import scipy
        print(f"✅ scipy version: {scipy.__version__}")
    except Exception as e:
        print(f"❌ Error con scipy: {e}")
    
    # Test fpdf2
    try:
        from fpdf import FPDF
        print("✅ fpdf2 importado correctamente")
    except Exception as e:
        print(f"❌ Error con fpdf2: {e}")
    
    # Test streamlit
    try:
        import streamlit as st
        print(f"✅ streamlit version: {st.__version__}")
    except Exception as e:
        print(f"❌ Error con streamlit: {e}")

def test_streamlit_app():
    """Prueba la importación del módulo de Streamlit"""
    print("\n🔍 Probando módulo de Streamlit...")
    
    try:
        # Agregar el directorio del proyecto al path
        sys.path.insert(0, os.path.join(os.getcwd(), 'proyecto_j'))
        
        # Importar el módulo
        import streamlit_app
        print("✅ Módulo streamlit_app importado correctamente")
        
        # Verificar variables importantes
        if hasattr(streamlit_app, 'MODULES_LOADED'):
            print(f"✅ MODULES_LOADED: {streamlit_app.MODULES_LOADED}")
        
        if hasattr(streamlit_app, 'MISSING_MODULES'):
            print(f"✅ MISSING_MODULES: {len(streamlit_app.MISSING_MODULES)} módulos faltantes")
            
        if hasattr(streamlit_app, 'OPTIONAL_DEPS'):
            print(f"✅ OPTIONAL_DEPS: {len(streamlit_app.OPTIONAL_DEPS)} dependencias opcionales")
            
    except Exception as e:
        print(f"❌ Error importando streamlit_app: {e}")

def test_pdf_generation():
    """Prueba la generación de PDF"""
    print("\n🔍 Probando generación de PDF...")
    
    try:
        import pandas as pd
        from fpdf import FPDF
        
        # Crear datos de prueba
        df = pd.DataFrame({
            'columna1': [1, 2, 3],
            'columna2': ['a', 'b', 'c']
        })
        
        # Crear PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16)
        pdf.cell(200, 10, txt="Test PDF", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Filas: {len(df)}", ln=True)
        pdf.cell(200, 10, txt=f"Columnas: {len(df.columns)}", ln=True)
        
        # Guardar
        test_file = "test_report.pdf"
        pdf.output(test_file)
        
        # Verificar que se creó
        if os.path.exists(test_file):
            print(f"✅ PDF generado correctamente: {test_file}")
            os.remove(test_file)  # Limpiar
        else:
            print("❌ PDF no se generó")
            
    except Exception as e:
        print(f"❌ Error generando PDF: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de corrección...\n")
    
    test_imports()
    test_streamlit_app()
    test_pdf_generation()
    
    print("\n✅ Pruebas completadas!")
    print("\n💡 Si todo está bien, la aplicación Streamlit debería funcionar sin errores.")
    print("🌐 Abre http://localhost:8502 en tu navegador") 