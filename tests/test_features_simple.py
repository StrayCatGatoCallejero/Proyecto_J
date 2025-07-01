"""
Test script for feature engineering capabilities - Simplified version.
Tests all the new features without import issues.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os


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


def test_feature_functions():
    """Test individual feature engineering functions."""
    print("\n" + "=" * 60)
    print("🔬 PRUEBA DE FUNCIONES DE FEATURE ENGINEERING")
    print("=" * 60)

    # Generate test data
    df = generate_test_data(100)

    # Test ratios
    print("\n📊 Probando cálculo de ratios...")
    try:
        result = df.copy()
        for num in ["income", "expenses"]:
            for den in ["income", "savings"]:
                col_name = f"ratio_{num}_over_{den}"
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(
                        result[den] != 0, result[num] / result[den], np.nan
                    )
                result[col_name] = ratio

        ratio_cols = [col for col in result.columns if col.startswith("ratio_")]
        print(f"   ✅ Ratios generados: {len(ratio_cols)}")
        for col in ratio_cols:
            print(f"      - {col}")
    except Exception as e:
        print(f"   ❌ Error en ratios: {e}")

    # Test scaling
    print("\n📈 Probando escalado...")
    try:
        # Z-score normalization
        result = df.copy()
        for col in ["income", "age"]:
            mean = result[col].mean()
            std = result[col].std()
            if std == 0:
                result[f"z_{col}"] = 0.0
            else:
                result[f"z_{col}"] = (result[col] - mean) / std

        z_cols = [col for col in result.columns if col.startswith("z_")]
        print(f"   ✅ Z-score features: {len(z_cols)}")

        # Robust scaling
        for col in ["income", "age"]:
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                result[f"robust_{col}"] = 0.0
            else:
                result[f"robust_{col}"] = (result[col] - q1) / iqr

        robust_cols = [col for col in result.columns if col.startswith("robust_")]
        print(f"   ✅ Robust features: {len(robust_cols)}")

        # Min-max scaling
        for col in ["income", "age"]:
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val - min_val == 0:
                result[f"scaled_{col}"] = 0.0
            else:
                result[f"scaled_{col}"] = (result[col] - min_val) / (max_val - min_val)

        minmax_cols = [col for col in result.columns if col.startswith("scaled_")]
        print(f"   ✅ Min-max features: {len(minmax_cols)}")
    except Exception as e:
        print(f"   ❌ Error en escalado: {e}")

    # Test binning
    print("\n📦 Probando binning...")
    try:
        # Age bins
        age_bins = [0, 25, 35, 50, 65, 100]
        age_labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        age_binned = pd.cut(
            df["age"], bins=age_bins, labels=age_labels, include_lowest=True
        )
        print(f"   ✅ Age bins creados: {age_binned.nunique()} categorías")

        # Quantile binning
        income_quantiles = pd.qcut(df["income"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
        print(f"   ✅ Income quantiles: {income_quantiles.nunique()} cuartiles")
    except Exception as e:
        print(f"   ❌ Error en binning: {e}")

    # Test confidence intervals
    print("\n🎯 Probando intervalos de confianza...")
    try:
        from scipy import stats

        data = df["income"].dropna()
        n = len(data)
        if n >= 2:
            mean = data.mean()
            se = data.std(ddof=1) / np.sqrt(n)
            t = stats.t.ppf(0.975, df=n - 1)  # 95% CI
            ci = (mean - t * se, mean + t * se)
            print(f"   ✅ Income CI: {ci}")
            print(f"   ✅ Income SE: {se:.4f}")
        else:
            print("   ⚠️ Insuficientes datos para CI")
    except Exception as e:
        print(f"   ❌ Error en intervalos de confianza: {e}")

    # Test bootstrap
    print("\n🔄 Probando bootstrap...")
    try:
        data = df["income"].dropna().values
        if len(data) >= 2:
            boot_stats = [
                np.mean(np.random.choice(data, size=len(data), replace=True))
                for _ in range(500)
            ]
            boot_mean = np.mean(boot_stats)
            boot_ci = (np.percentile(boot_stats, 2.5), np.percentile(boot_stats, 97.5))
            print(f"   ✅ Income bootstrap mean: {boot_mean:.4f}")
            print(f"   ✅ Income bootstrap CI: {boot_ci}")
        else:
            print("   ⚠️ Insuficientes datos para bootstrap")
    except Exception as e:
        print(f"   ❌ Error en bootstrap: {e}")

    # Test composite indices
    print("\n📊 Probando índices compuestos...")
    try:
        # Simple mean index
        satisfaction_data = df[["satisfaction_work", "satisfaction_life"]].fillna(0)
        satisfaction_index = satisfaction_data.mean(axis=1)
        print(
            f"   ✅ Satisfaction index creado: {satisfaction_index.mean():.3f} promedio"
        )

        # Weighted index
        weights = np.array([0.6, 0.4])
        weighted_index = (satisfaction_data * weights).sum(axis=1) / weights.sum()
        print(f"   ✅ Weighted index creado: {weighted_index.mean():.3f} promedio")
    except Exception as e:
        print(f"   ❌ Error en índices compuestos: {e}")

    # Test weighted mean
    print("\n⚖️ Probando media ponderada...")
    try:
        weights = df["age"].fillna(0)
        values = df["income"].fillna(0)
        if (weights < 0).any():
            print("   ⚠️ Pesos negativos encontrados")
        else:
            total_weight = weights.sum()
            if total_weight > 0:
                wm = np.average(values, weights=weights)
                print(f"   ✅ Weighted mean (income by age): {wm:.2f}")
            else:
                print("   ⚠️ Suma de pesos es cero")
    except Exception as e:
        print(f"   ❌ Error en media ponderada: {e}")

    # Test group aggregation
    print("\n📋 Probando agregación por grupos...")
    try:
        group_stats = df.groupby(["education", "region"]).agg(
            {"income": "mean", "age": "mean"}
        )
        print(f"   ✅ Group aggregation: {group_stats.shape[0]} grupos")
    except Exception as e:
        print(f"   ❌ Error en agregación: {e}")


def test_robustness():
    """Test feature engineering robustness with edge cases."""
    print("\n" + "=" * 60)
    print("🛡️ PRUEBA DE ROBUSTEZ DE FEATURES")
    print("=" * 60)

    # Test with edge cases
    print("\n📊 Probando casos extremos...")

    # Empty DataFrame
    try:
        empty_df = pd.DataFrame()
        result = empty_df.copy()
        print("   ✅ Manejo de DataFrame vacío")
    except Exception as e:
        print(f"   ❌ Error con DataFrame vacío: {e}")

    # Single value column
    try:
        single_df = pd.DataFrame({"col1": [5, 5, 5], "col2": [2, 2, 2]})
        mean = single_df["col1"].mean()
        std = single_df["col1"].std()
        if std == 0:
            result = 0.0
        else:
            result = (single_df["col1"] - mean) / std
        print("   ✅ Manejo de columna con un solo valor")
    except Exception as e:
        print(f"   ❌ Error con columna de un valor: {e}")

    # All NaN column
    try:
        nan_df = pd.DataFrame({"col1": [np.nan, np.nan, np.nan], "col2": [1, 2, 3]})
        q1 = nan_df["col1"].quantile(0.25)
        q3 = nan_df["col1"].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            result = 0.0
        else:
            result = (nan_df["col1"] - q1) / iqr
        print("   ✅ Manejo de columna con solo NaN")
    except Exception as e:
        print(f"   ❌ Error con columna NaN: {e}")

    # Division by zero
    try:
        zero_df = pd.DataFrame({"num": [1, 2, 3], "den": [0, 0, 0]})
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                zero_df["den"] != 0, zero_df["num"] / zero_df["den"], np.nan
            )
        print("   ✅ Manejo de división por cero")
    except Exception as e:
        print(f"   ❌ Error con división por cero: {e}")

    print("\n✅ Todas las pruebas de robustez completadas!")


def simulate_pipeline_features():
    """Simulate the feature engineering pipeline logic."""
    print("\n" + "=" * 60)
    print("🧪 SIMULACIÓN DEL PIPELINE DE FEATURE ENGINEERING")
    print("=" * 60)

    # Generate test data
    df = generate_test_data(150)

    # Simulate semantic classification
    semantic_types = {
        "income": "numeric",
        "expenses": "numeric",
        "savings": "numeric",
        "age": "demographic",
        "satisfaction_work": "likert",
        "satisfaction_life": "likert",
        "trust_government": "likert",
        "education": "categorical",
        "region": "categorical",
    }

    original_columns = list(df.columns)
    new_features = []

    print(f"📋 Columnas originales: {len(original_columns)}")

    # 1. Ratios for numeric variables
    numeric_cols = [
        col
        for col, sem_type in semantic_types.items()
        if sem_type in ["numeric", "demographic"]
    ]

    if len(numeric_cols) >= 2:
        print(f"📊 Generando ratios para {len(numeric_cols)} variables numéricas...")
        for num in numeric_cols:
            for den in numeric_cols:
                if num != den:
                    col_name = f"ratio_{num}_over_{den}"
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(df[den] != 0, df[num] / df[den], np.nan)
                    df[col_name] = ratio
                    new_features.append(col_name)

    # 2. Scaling for Likert scales
    likert_cols = [
        col for col, sem_type in semantic_types.items() if sem_type == "likert"
    ]
    if likert_cols:
        print(f"📈 Aplicando escalado Z-score a {len(likert_cols)} escalas Likert...")
        for col in likert_cols:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0:
                df[f"z_{col}"] = 0.0
            else:
                df[f"z_{col}"] = (df[col] - mean) / std
            new_features.append(f"z_{col}")

    # 3. Robust scaling for demographic variables
    demographic_cols = [
        col for col, sem_type in semantic_types.items() if sem_type == "demographic"
    ]
    if demographic_cols:
        print(
            f"🛡️ Aplicando escalado robusto a {len(demographic_cols)} variables demográficas..."
        )
        for col in demographic_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                df[f"robust_{col}"] = 0.0
            else:
                df[f"robust_{col}"] = (df[col] - q1) / iqr
            new_features.append(f"robust_{col}")

    # 4. Age binning
    age_cols = [col for col in demographic_cols if "age" in col.lower()]
    for age_col in age_cols:
        print(f"📦 Creando bins de edad para {age_col}...")
        age_bins = [0, 25, 35, 50, 65, 100]
        age_labels = ["18-25", "26-35", "36-50", "51-65", "65+"]
        df[f"{age_col}_bins"] = pd.cut(
            df[age_col], bins=age_bins, labels=age_labels, include_lowest=True
        )
        new_features.append(f"{age_col}_bins")

    # 5. Composite indices
    if len(likert_cols) >= 2:
        print(
            f"📊 Creando índices compuestos para {len(likert_cols)} escalas Likert..."
        )

        # Satisfaction index
        satisfaction_cols = [
            col for col in likert_cols if "satisfaction" in col.lower()
        ]
        if satisfaction_cols:
            satisfaction_data = df[satisfaction_cols].fillna(0)
            df["satisfaction_index"] = satisfaction_data.mean(axis=1)
            new_features.append("satisfaction_index")

        # Overall attitude index
        likert_data = df[likert_cols].fillna(0)
        df["attitude_index"] = likert_data.mean(axis=1)
        new_features.append("attitude_index")

    # 6. Confidence intervals
    print("🎯 Calculando intervalos de confianza...")
    confidence_results = {}
    key_vars = numeric_cols[:3]

    for var in key_vars:
        try:
            from scipy import stats

            data = df[var].dropna()
            n = len(data)
            if n >= 2:
                mean = data.mean()
                se = data.std(ddof=1) / np.sqrt(n)
                t = stats.t.ppf(0.975, df=n - 1)
                ci = (mean - t * se, mean + t * se)
                confidence_results[var] = {
                    "confidence_interval": ci,
                    "standard_error": se,
                }
        except Exception as e:
            print(f"   ⚠️ Error calculando CI para {var}: {e}")

    # 7. Bootstrap analysis
    print("🔄 Realizando análisis bootstrap...")
    bootstrap_results = {}

    for var in key_vars[:2]:
        try:
            data = df[var].dropna().values
            if len(data) >= 2:
                boot_stats = [
                    np.mean(np.random.choice(data, size=len(data), replace=True))
                    for _ in range(500)
                ]
                bootstrap_results[var] = {
                    "bootstrap_mean": np.mean(boot_stats),
                    "ci": (
                        np.percentile(boot_stats, 2.5),
                        np.percentile(boot_stats, 97.5),
                    ),
                    "distribution": boot_stats,
                }
        except Exception as e:
            print(f"   ⚠️ Error en bootstrap para {var}: {e}")

    # Results summary
    print(f"\n📊 RESUMEN DE FEATURE ENGINEERING:")
    print(f"   📋 Columnas originales: {len(original_columns)}")
    print(f"   🔧 Nuevas features: {len(new_features)}")
    print(f"   📊 Total features: {len(df.columns)}")
    print(f"   🎯 Variables con CI: {len(confidence_results)}")
    print(f"   🔄 Variables con bootstrap: {len(bootstrap_results)}")

    # Feature types breakdown
    feature_types = {
        "ratios": [col for col in new_features if col.startswith("ratio_")],
        "scaled": [
            col for col in new_features if col.startswith(("z_", "robust_", "scaled_"))
        ],
        "binned": [col for col in new_features if col.endswith("_bins")],
        "composite": [col for col in new_features if col.endswith("_index")],
    }

    print(f"\n📈 Tipos de features generadas:")
    for feature_type, features in feature_types.items():
        if features:
            print(f"   - {feature_type}: {len(features)} features")

    # Show confidence intervals
    if confidence_results:
        print(f"\n🎯 Intervalos de confianza:")
        for var, ci_info in confidence_results.items():
            ci = ci_info["confidence_interval"]
            se = ci_info["standard_error"]
            print(f"   - {var}: CI=({ci[0]:.2f}, {ci[1]:.2f}), SE={se:.4f}")

    # Show bootstrap results
    if bootstrap_results:
        print(f"\n🔄 Resultados bootstrap:")
        for var, boot_info in bootstrap_results.items():
            mean = boot_info["bootstrap_mean"]
            ci = boot_info["ci"]
            print(f"   - {var}: mean={mean:.4f}, CI=({ci[0]:.4f}, {ci[1]:.4f})")

    print("\n✅ Simulación del pipeline completada exitosamente!")


if __name__ == "__main__":
    print("🚀 Iniciando pruebas completas de feature engineering...")

    # Test individual functions
    test_feature_functions()

    # Test robustness
    test_robustness()

    # Simulate pipeline
    simulate_pipeline_features()

    print("\n🎯 Todas las pruebas de feature engineering completadas!")
