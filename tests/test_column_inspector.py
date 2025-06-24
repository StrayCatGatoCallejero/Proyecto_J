import pandas as pd
from column_inspector import detect_type, get_column_report

def test_detect_type_numeric():
    s = pd.Series([1, 2, 3, 4])
    assert detect_type(s) == 'numeric'

def test_detect_type_categorical():
    s = pd.Series(['a', 'b', 'a', 'c'])
    assert detect_type(s) == 'categorical'

def test_get_column_report_structure():
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    report = get_column_report(df)
    assert set(report.columns) == {'column', 'detected_type', 'suggested_analysis', 'code_snippet', 'output_type'} 