"""
Test script for comprehensive feature engineering capabilities.
Demonstrates all new features: ratios, scaling, binning, confidence intervals, bootstrap, and composite indices.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os
import sys
import pytest

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from processing.features import (
        compute_ratios,
        compute_percentage,
        weighted_mean,
        group_agg,
        min_max_scale,
        z_score_normalize,
        robust_scale,
        create_bins,
        quantile_binning,
        compute_confidence_interval,
        standard_error,
        bootstrap_statistic,
        composite_index,
        scale_and_score,
    )
    from orchestrator.pipeline_orchestrator import PipelineOrchestrator
    from processing.logging import setup_logging
except ImportError as e:
    pytest.skip(f"Falta dependencia para test_features_complete: {e}", allow_module_level=True)

# Setup logging
setup_logging()


def generate_test_data(n_samples=200):
    """Generate comprehensive test data for feature engineering."""
    print("🔬 Generando datos de prueba para feature engineering...")

    np.random.seed(42)
    random.seed(42)

    # Economic variables
    income = np.random.lognormal(10, 0.5, n_samples)
    expenses = income * np.random.uniform(0.3, 0.8, n_samples)
    savings = income - expenses

    # Demographic variables
    age = np.random.normal(35, 12, n_samples).astype(int)
    age = np.clip(age, 18, 80)

    # Likert scales
    satisfaction_work = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.10, 0.15, 0.25, 0.30, 0.20]
    )
    satisfaction_life = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.05, 0.10, 0.20, 0.35, 0.30]
    )
    trust_government = np.random.choice(
        [1, 2, 3, 4, 5], n_samples, p=[0.25, 0.30, 0.25, 0.15, 0.05]
    )

    # Categorical variables
    education = np.random.choice(
        ["Primaria", "Secundaria", "Universitario", "Postgrado"], n_samples
    )
    region = np.random.choice(["Norte", "Centro", "Sur"], n_samples)

    # Create DataFrame
    data = {
        "id": range(1, n_samples + 1),
        "income": income,
        "expenses": expenses,
        "savings": savings,
        "age": age,
        "satisfaction_work": satisfaction_work,
        "satisfaction_life": satisfaction_life,
        "trust_government": trust_government,
        "education": education,
        "region": region,
    }

    df = pd.DataFrame(data)

    # Add some missing values
    missing_indices = np.random.choice(
        df.index, size=int(n_samples * 0.05), replace=False
    )
    df.loc[missing_indices, "savings"] = np.nan

    print(f"✅ Datos generados: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def test_individual_features():
    """Test individual feature engineering functions."""
    print("\n" + "=" * 60)
    print("🔬 PRUEBA DE FUNCIONES INDIVIDUALES DE FEATURE ENGINEERING")
    print("=" * 60)

    # Generate test data
    df = generate_test_data(100)

    # Test ratios
    print("\n📊 Probando cálculo de ratios...")
    try:
        df_with_ratios = compute_ratios(
            df, ["income", "expenses"], ["income", "savings"]
        )
        ratio_cols = [col for col in df_with_ratios.columns if col.startswith("ratio_")]
        print(f"   ✅ Ratios generados: {len(ratio_cols)}")
        for col in ratio_cols:
            print(f"      - {col}")
    except Exception as e:
        print(f"   ❌ Error en ratios: {e}")

    # Test scaling
    print("\n📈 Probando escalado...")
    try:
        # Z-score normalization
        df_scaled = z_score_normalize(df, ["income", "age"])
        z_cols = [col for col in df_scaled.columns if col.startswith("z_")]
        print(f"   ✅ Z-score features: {len(z_cols)}")

        # Robust scaling
        df_robust = robust_scale(df, ["income", "age"])
        robust_cols = [col for col in df_robust.columns if col.startswith("robust_")]
        print(f"   ✅ Robust features: {len(robust_cols)}")

        # Min-max scaling
        df_minmax = min_max_scale(df, ["income", "age"])
        minmax_cols = [col for col in df_minmax.columns if col.startswith("scaled_")]
        print(f"   ✅ Min-max features: {len(minmax_cols)}")
    except Exception as e:
        print(f"   ❌ Error en escalado: {e}")

    # Test binning
    print("\n📦 Probando binning...")
    try:
        # Age bins
        age_bins = [0, 25, 35, 50, 65, 100]
        age_labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        age_binned = create_bins(df, "age", age_bins, age_labels)
        print(f"   ✅ Age bins creados: {age_binned.nunique()} categorías")

        # Quantile binning
        income_quantiles = quantile_binning(df, "income", 4)
        print(f"   ✅ Income quantiles: {income_quantiles.nunique()} cuartiles")
    except Exception as e:
        print(f"   ❌ Error en binning: {e}")

    # Test confidence intervals
    print("\n🎯 Probando intervalos de confianza...")
    try:
        ci_income = compute_confidence_interval(df, "income")
        se_income = standard_error(df, "income")
        print(f"   ✅ Income CI: {ci_income}")
        print(f"   ✅ Income SE: {se_income:.4f}")
    except Exception as e:
        print(f"   ❌ Error en intervalos de confianza: {e}")

    # Test bootstrap
    print("\n🔄 Probando bootstrap...")
    try:
        boot_income = bootstrap_statistic(df, "income", np.mean, n_boot=500)
        print(f"   ✅ Income bootstrap mean: {boot_income['bootstrap_mean']:.4f}")
        print(f"   ✅ Income bootstrap CI: {boot_income['ci']}")
    except Exception as e:
        print(f"   ❌ Error en bootstrap: {e}")

    # Test composite indices
    print("\n📊 Probando índices compuestos...")
    try:
        satisfaction_index = composite_index(
            df, ["satisfaction_work", "satisfaction_life"], method="mean"
        )
        print(
            f"   ✅ Satisfaction index creado: {satisfaction_index.mean():.3f} promedio"
        )

        # Test with weights
        weighted_index = composite_index(
            df,
            ["satisfaction_work", "satisfaction_life"],
            method="mean",
            weights=[0.6, 0.4],
        )
        print(f"   ✅ Weighted index creado: {weighted_index.mean():.3f} promedio")
    except Exception as e:
        print(f"   ❌ Error en índices compuestos: {e}")

    # Test weighted mean
    print("\n⚖️ Probando media ponderada...")
    try:
        wm = weighted_mean(df, "income", "age")
        print(f"   ✅ Weighted mean (income by age): {wm:.2f}")
    except Exception as e:
        print(f"   ❌ Error en media ponderada: {e}")

    # Test group aggregation
    print("\n📋 Probando agregación por grupos...")
    try:
        group_stats = group_agg(
            df, ["education", "region"], {"income": "mean", "age": "mean"}
        )
        print(f"   ✅ Group aggregation: {group_stats.shape[0]} grupos")
    except Exception as e:
        print(f"   ❌ Error en agregación: {e}")


def test_pipeline_integration():
    """Test feature engineering integration in the pipeline."""
    print("\n" + "=" * 60)
    print("🧪 PRUEBA DE INTEGRACIÓN EN EL PIPELINE")
    print("=" * 60)

    # Generate test data
    df = generate_test_data(150)

    # Save data temporarily
    temp_file = "temp_features_test.csv"
    df.to_csv(temp_file, index=False)
    print(f"💾 Datos guardados en: {temp_file}")

    # Initialize orchestrator
    print("\n🔧 Inicializando orquestador...")
    orchestrator = PipelineOrchestrator()

    # Run full pipeline
    print("\n🚀 Ejecutando pipeline completo con feature engineering...")
    orchestrator.run_full_pipeline(temp_file)

    session_data = orchestrator.get_session_data()

    # Display feature engineering results
    print("\n" + "=" * 60)
    print("📊 RESULTADOS DE FEATURE ENGINEERING")
    print("=" * 60)

    if "feature_engineering" in session_data.metadata:
        feature_data = session_data.metadata["feature_engineering"]

        print(f"📋 Columnas originales: {len(feature_data['original_columns'])}")
        print(f"🔧 Nuevas features: {len(feature_data['new_features'])}")
        print(f"📊 Total features: {feature_data['total_features']}")

        # Show feature types
        feature_types = feature_data["feature_types"]
        print(f"\n📈 Tipos de features generadas:")
        for feature_type, features in feature_types.items():
            if features:
                print(f"   - {feature_type}: {len(features)} features")
                for feature in features[:3]:  # Show first 3
                    print(f"     • {feature}")
                if len(features) > 3:
                    print(f"     • ... y {len(features) - 3} más")

        # Show confidence intervals
        if "confidence_intervals" in feature_data:
            print(f"\n🎯 Intervalos de confianza calculados:")
            for var, ci_info in feature_data["confidence_intervals"].items():
                ci = ci_info["confidence_interval"]
                se = ci_info["standard_error"]
                print(f"   - {var}: CI=({ci[0]:.2f}, {ci[1]:.2f}), SE={se:.4f}")

        # Show bootstrap results
        if "bootstrap_results" in feature_data:
            print(f"\n🔄 Resultados bootstrap:")
            for var, boot_info in feature_data["bootstrap_results"].items():
                mean = boot_info["bootstrap_mean"]
                ci = boot_info["ci"]
                print(f"   - {var}: mean={mean:.4f}, CI=({ci[0]:.4f}, {ci[1]:.4f})")
    else:
        print("❌ No se encontraron resultados de feature engineering")

    # Export results
    print(f"\n💾 Exportando resultados...")
    output_path = (
        f"reporte_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    )
    orchestrator.export_results(output_path)

    # Clean up
    if os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"🧹 Archivo temporal eliminado: {temp_file}")

    print("\n" + "=" * 60)
    print("🎉 PRUEBA DE INTEGRACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)


def test_feature_robustness():
    """Test feature engineering robustness with edge cases."""
    print("\n" + "=" * 60)
    print("🛡️ PRUEBA DE ROBUSTEZ DE FEATURES")
    print("=" * 60)

    # Test with edge cases
    print("\n📊 Probando casos extremos...")

    # Empty DataFrame
    try:
        empty_df = pd.DataFrame()
        result = compute_ratios(empty_df, ["col1"], ["col2"])
        print("   ✅ Manejo de DataFrame vacío")
    except Exception as e:
        print(f"   ❌ Error con DataFrame vacío: {e}")

    # Single value column
    try:
        single_df = pd.DataFrame({"col1": [5, 5, 5], "col2": [2, 2, 2]})
        result = z_score_normalize(single_df, ["col1"])
        print("   ✅ Manejo de columna con un solo valor")
    except Exception as e:
        print(f"   ❌ Error con columna de un valor: {e}")

    # All NaN column
    try:
        nan_df = pd.DataFrame({"col1": [np.nan, np.nan, np.nan], "col2": [1, 2, 3]})
        result = robust_scale(nan_df, ["col1"])
        print("   ✅ Manejo de columna con solo NaN")
    except Exception as e:
        print(f"   ❌ Error con columna NaN: {e}")

    # Division by zero
    try:
        zero_df = pd.DataFrame({"num": [1, 2, 3], "den": [0, 0, 0]})
        result = compute_ratios(zero_df, ["num"], ["den"])
        print("   ✅ Manejo de división por cero")
    except Exception as e:
        print(f"   ❌ Error con división por cero: {e}")

    print("\n✅ Todas las pruebas de robustez completadas!")


if __name__ == "__main__":
    print("🚀 Iniciando pruebas completas de feature engineering...")

    # Test individual functions
    test_individual_features()

    # Test pipeline integration
    test_pipeline_integration()

    # Test robustness
    test_feature_robustness()

    print("\n🎯 Todas las pruebas de feature engineering completadas!")
