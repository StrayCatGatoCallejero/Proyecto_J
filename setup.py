from setuptools import setup, find_packages

setup(
    name="proyecto_j",
    version="2.0.0",
    description="Sistema de Análisis de Datos para Ciencias Sociales",
    author="Proyecto J Team",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "plotly",
        "scikit-learn",
        "scipy",
        "openpyxl",
        "pyyaml",
    ],
    python_requires=">=3.11",
) 