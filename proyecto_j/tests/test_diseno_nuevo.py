#!/usr/bin/env python3
"""
🎨 PROYECTO J - PRUEBAS DEL NUEVO DISEÑO VISUAL
==================================================

Script para verificar que el nuevo diseño visual está correctamente implementado:
- Fondo azul claro (#C7DCE5) en el área de contenido principal
- Fondo oscuro (#333333) en el panel lateral izquierdo
- Fondo general claro (#FBF7F2) en toda la aplicación
"""

import os
import re
import sys
from pathlib import Path

def verificar_configuracion_tema():
    """Verificar que la configuración del tema en .streamlit/config.toml es correcta."""
    print("🔍 Verificando configuración del tema...")
    
    config_path = Path(".streamlit/config.toml")
    if not config_path.exists():
        print("❌ Archivo .streamlit/config.toml no encontrado")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar configuraciones requeridas
    required_configs = [
        'base = "light"',
        'primaryColor = "#648DA5"',
        'backgroundColor = "#FBF7F2"',
        'secondaryBackgroundColor = "#F5E3D3"',
        'textColor = "#333333"',
        'font = "sans serif"'
    ]
    
    for config in required_configs:
        if config not in content:
            print(f"❌ Configuración faltante: {config}")
            return False
    
    print("✅ Configuración del tema correcta")
    return True

def verificar_css_app_encuestas():
    """Verificar que app_encuestas.py tiene el CSS correcto para el nuevo diseño."""
    print("\n🎨 Verificando CSS en app_encuestas.py...")
    
    if not os.path.exists("app_encuestas.py"):
        print("❌ Archivo app_encuestas.py no encontrado")
        return False
    
    with open("app_encuestas.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar elementos CSS requeridos
    css_checks = [
        ('color-scheme: light !important', 'Forzar modo claro'),
        ('--color-azul-claro: #C7DCE5', 'Variable azul claro'),
        ('background-color: var(--color-azul-claro) !important', 'Área de contenido azul claro'),
        ('background-color: #333333 !important', 'Sidebar fondo oscuro'),
        ('color: #FFFFFF !important', 'Texto blanco en sidebar'),
        ('color: #CCCCCC !important', 'Texto gris en sidebar'),
        ('border-radius: 10px', 'Border radius en área de contenido'),
        ('box-shadow: 0 2px 8px', 'Sombra en área de contenido')
    ]
    
    all_passed = True
    for css_pattern, description in css_checks:
        if css_pattern not in content:
            print(f"❌ CSS faltante: {description}")
            all_passed = False
        else:
            print(f"✅ {description}")
    
    return all_passed

def verificar_css_app_front():
    """Verificar que app_front.py tiene el CSS correcto para el nuevo diseño."""
    print("\n🎨 Verificando CSS en app_front.py...")
    
    if not os.path.exists("app_front.py"):
        print("❌ Archivo app_front.py no encontrado")
        return False
    
    with open("app_front.py", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar elementos CSS requeridos
    css_checks = [
        ('color-scheme: light !important', 'Forzar modo claro'),
        ('--color-azul-claro: #C7DCE5', 'Variable azul claro'),
        ('background-color: var(--color-azul-claro)', 'Área de contenido azul claro'),
        ('background-color: #333333 !important', 'Sidebar fondo oscuro'),
        ('color: #FFFFFF !important', 'Texto blanco en sidebar'),
        ('color: #CCCCCC !important', 'Texto gris en sidebar')
    ]
    
    all_passed = True
    for css_pattern, description in css_checks:
        if css_pattern not in content:
            print(f"❌ CSS faltante: {description}")
            all_passed = False
        else:
            print(f"✅ {description}")
    
    return all_passed

def verificar_documentacion():
    """Verificar que la documentación del tema está actualizada."""
    print("\n📚 Verificando documentación del tema...")
    
    if not os.path.exists("TEMA_FIJO.md"):
        print("❌ Archivo TEMA_FIJO.md no encontrado")
        return False
    
    with open("TEMA_FIJO.md", 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verificar elementos de documentación
    doc_checks = [
        ('#C7DCE5', 'Azul claro en documentación'),
        ('#333333', 'Fondo oscuro en documentación'),
        ('azul claro en el área de contenido', 'Descripción del diseño'),
        ('fondo oscuro en el sidebar', 'Descripción del sidebar')
    ]
    
    all_passed = True
    for doc_pattern, description in doc_checks:
        if doc_pattern.lower() not in content.lower():
            print(f"❌ Documentación faltante: {description}")
            all_passed = False
        else:
            print(f"✅ {description}")
    
    return all_passed

def main():
    """Función principal de verificación."""
    print("🎯 PROYECTO J - PRUEBAS DEL NUEVO DISEÑO VISUAL")
    print("=" * 60)
    
    tests = [
        ("Configuración del tema", verificar_configuracion_tema),
        ("CSS en app_encuestas.py", verificar_css_app_encuestas),
        ("CSS en app_front.py", verificar_css_app_front),
        ("Documentación del tema", verificar_documentacion)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Ejecutando: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASÓ")
            else:
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print(f"🎯 Resultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El nuevo diseño está correctamente implementado.")
        print("\n🎨 Características verificadas:")
        print("   ✅ Fondo azul claro (#C7DCE5) en área de contenido")
        print("   ✅ Fondo oscuro (#333333) en panel lateral")
        print("   ✅ Fondo general claro (#FBF7F2)")
        print("   ✅ Modo claro forzado")
        print("   ✅ CSS consistente en todas las aplicaciones")
        print("   ✅ Documentación actualizada")
        return 0
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar la implementación.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 