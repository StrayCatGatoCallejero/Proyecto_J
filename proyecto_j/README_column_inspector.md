# column_inspector

Módulo para inspección automática de columnas en datasets. Detecta tipo de dato y sugiere análisis y visualizaciones con snippets de código.

## Uso

```python
from column_inspector import get_column_report, read_data

df = read_data('archivo.csv')
report = get_column_report(df)
print(report)
```

## Integración en Streamlit

- Botón "Inspeccionar columnas" en el panel de resumen.
- Tabla interactiva con sugerencias y botón "Ejecutar" para cada análisis.

## Tests

```bash
pytest tests/test_column_inspector.py
``` 