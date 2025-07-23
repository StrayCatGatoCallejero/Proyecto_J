"""
Tests para el módulo de I/O
"""
import pytest
import pandas as pd
from processing.io import cargar_archivo

def test_load_data_reads_csv():
    """Test que verifica que se puede cargar un archivo CSV"""
    # Crear un DataFrame de prueba
    test_data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    
    # Guardar como CSV temporal
    test_file = 'test_data.csv'
    test_data.to_csv(test_file, index=False)
    
    try:
        # Cargar usando la función correcta
        df, metadata = cargar_archivo(test_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'col1' in df.columns
        assert 'col2' in df.columns
        assert metadata['format'] == 'csv'
        
    finally:
        # Limpiar archivo temporal
        import os
        if os.path.exists(test_file):
            os.remove(test_file) 