import pandas as pd
import numpy as np

from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset limpio."""
    return pd.read_csv(path)


def split_reference_current(df: pd.DataFrame, test_size: float = 0.30):
    """
    Divide el dataset en datos históricos y datos actuales.
    Se usa una división simple para simular monitoreo.
    """
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    split_index = int(len(df) * (1 - test_size))

    reference_data = df.iloc[:split_index].copy()
    current_data = df.iloc[split_index:].copy()

    return reference_data, current_data


def calculate_psi(reference, current, buckets: int = 10) -> float:
    """
    Calcula Population Stability Index para variables numéricas.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    try:
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) <= 2:
            return np.nan

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)

        ref_perc = ref_counts / max(len(reference), 1)
        cur_perc = cur_counts / max(len(current), 1)

        ref_perc = np.where(ref_perc == 0, 0.0001, ref_perc)
        cur_perc = np.where(cur_perc == 0, 0.0001, cur_perc)

        psi = np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc))
        return round(float(psi), 4)

    except Exception:
        return np.nan


def calculate_jensen_shannon(reference, current, buckets: int = 10) -> float:
    """
    Calcula Jensen-Shannon divergence para variables numéricas.
    Compara la diferencia entre dos distribuciones.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    try:
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) <= 2:
            return np.nan

        ref_counts, _ = np.histogram(reference, bins=breakpoints)
        cur_counts, _ = np.histogram(current, bins=breakpoints)

        ref_dist = ref_counts / max(ref_counts.sum(), 1)
        cur_dist = cur_counts / max(cur_counts.sum(), 1)

        ref_dist = np.where(ref_dist == 0, 0.0001, ref_dist)
        cur_dist = np.where(cur_dist == 0, 0.0001, cur_dist)

        return round(float(jensenshannon(ref_dist, cur_dist)), 4)

    except Exception:
        return np.nan


def psi_status(psi_value):
    """Clasifica el nivel de drift según PSI."""
    if pd.isna(psi_value):
        return "No calculable"
    elif psi_value < 0.10:
        return "Sin drift"
    elif psi_value < 0.25:
        return "Drift moderado"
    else:
        return "Drift fuerte"


def ks_status(p_value):
    """Clasifica drift según p-value del KS test."""
    if pd.isna(p_value):
        return "No calculable"
    elif p_value < 0.05:
        return "Drift detectado"
    else:
        return "Sin drift"


def chi_status(p_value):
    """Clasifica drift según p-value del Chi-cuadrado."""
    if pd.isna(p_value):
        return "No calculable"
    elif p_value < 0.05:
        return "Drift detectado"
    else:
        return "Sin drift"


def chi_square_test(reference, current):
    """
    Calcula Chi-cuadrado para variables categóricas.
    """
    reference = pd.Series(reference).dropna()
    current = pd.Series(current).dropna()

    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()

    categories = list(set(ref_counts.index).union(set(cur_counts.index)))

    ref = [ref_counts.get(cat, 0) for cat in categories]
    cur = [cur_counts.get(cat, 0) for cat in categories]

    contingency_table = np.array([ref, cur])

    try:
        _, p_value, _, _ = chi2_contingency(contingency_table)
        return round(float(p_value), 4)
    except Exception:
        return np.nan


def generate_drift_report(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    target_col: str = None,
    max_categories: int = 20
) -> pd.DataFrame:
    """
    Genera tabla final con métricas de data drift.
    """
    results = []

    if target_col and target_col in reference_data.columns:
        columns = [col for col in reference_data.columns if col != target_col]
    else:
        columns = reference_data.columns.tolist()

    numeric_cols = reference_data[columns].select_dtypes(include=[np.number]).columns.tolist()

    categorical_cols = [
        col for col in columns
        if col not in numeric_cols
        and reference_data[col].nunique() <= max_categories
    ]

    for col in numeric_cols:
        ks_stat, ks_pvalue = ks_2samp(
            reference_data[col].dropna(),
            current_data[col].dropna()
        )

        psi_value = calculate_psi(reference_data[col], current_data[col])
        js_value = calculate_jensen_shannon(reference_data[col], current_data[col])

        results.append({
            "variable": col,
            "tipo": "numérica",
            "ks_pvalue": round(float(ks_pvalue), 4),
            "psi": psi_value,
            "jensen_shannon": js_value,
            "chi2_pvalue": np.nan,
            "estado_ks": ks_status(ks_pvalue),
            "estado_psi": psi_status(psi_value),
            "estado_chi2": "No aplica"
        })

    for col in categorical_cols:
        chi_pvalue = chi_square_test(reference_data[col], current_data[col])

        results.append({
            "variable": col,
            "tipo": "categórica",
            "ks_pvalue": np.nan,
            "psi": np.nan,
            "jensen_shannon": np.nan,
            "chi2_pvalue": chi_pvalue,
            "estado_ks": "No aplica",
            "estado_psi": "No aplica",
            "estado_chi2": chi_status(chi_pvalue)
        })

    return pd.DataFrame(results)


def generate_temporal_drift(reference_data, current_data, periods: int = 4):
    """
    Simula análisis temporal dividiendo los datos actuales en períodos.
    Calcula PSI promedio por período.
    """
    chunks = np.array_split(current_data, periods)
    temporal_results = []

    numeric_cols = reference_data.select_dtypes(include=[np.number]).columns.tolist()

    for i, chunk in enumerate(chunks, start=1):
        psi_values = []

        for col in numeric_cols:
            psi = calculate_psi(reference_data[col], chunk[col])
            if not pd.isna(psi):
                psi_values.append(psi)

        avg_psi = np.mean(psi_values) if psi_values else np.nan

        temporal_results.append({
            "periodo": f"Periodo {i}",
            "psi_promedio": round(float(avg_psi), 4) if not pd.isna(avg_psi) else np.nan
        })

    return pd.DataFrame(temporal_results)
