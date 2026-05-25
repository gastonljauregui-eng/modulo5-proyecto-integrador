import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("src")

from model_monitoring import load_data, split_reference_current

st.set_page_config(
    page_title="Monitoreo de Data Drift",
    layout="wide"
)

st.title("Monitoreo de Data Drift - Proyecto Integrador")

st.write("""
Esta aplicación permite monitorear cambios en la distribución de los datos
comparando una muestra histórica contra una muestra actual.
""")

# =========================
# Cargar reportes
# =========================

drift_report = pd.read_csv("reports/drift_report.csv")
temporal_drift = pd.read_csv("reports/temporal_drift.csv")

# =========================
# Tabla principal
# =========================

st.subheader("Tabla de métricas de drift")
st.dataframe(drift_report)

# =========================
# Resumen de alertas
# =========================

st.subheader("Resumen de alertas")

drift_fuerte = drift_report[drift_report["estado_psi"] == "Drift fuerte"]
drift_moderado = drift_report[drift_report["estado_psi"] == "Drift moderado"]

col1, col2, col3 = st.columns(3)

col1.metric("Variables con drift fuerte", len(drift_fuerte))
col2.metric("Variables con drift moderado", len(drift_moderado))
col3.metric("Variables monitoreadas", len(drift_report))

if len(drift_fuerte) > 0:
    st.error("Se detectó drift fuerte. Se recomienda revisar variables y considerar retraining.")
elif len(drift_moderado) > 0:
    st.warning("Se detectó drift moderado. Se recomienda continuar monitoreando.")
else:
    st.success("No se detectó drift significativo.")

# =========================
# PSI por variable
# =========================

st.subheader("PSI por variable numérica")

numeric_drift = drift_report.dropna(subset=["psi"])

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(numeric_drift["variable"], numeric_drift["psi"])

ax.axhline(0.10, linestyle="--", label="Drift moderado")
ax.axhline(0.25, linestyle="--", label="Drift fuerte")

ax.set_ylabel("PSI")
ax.set_xlabel("Variables")
ax.set_title("Population Stability Index")

ax.tick_params(axis="x", rotation=90)

ax.legend()

st.pyplot(fig)

# =========================
# Comparación distribuciones
# =========================

st.subheader("Comparación distribución histórica vs actual")

df = load_data("data/dataset_limpio.csv")

reference_data, current_data = split_reference_current(df)

numeric_columns = reference_data.select_dtypes(include="number").columns.tolist()

selected_variable = st.selectbox(
    "Seleccioná una variable",
    numeric_columns
)

fig2, ax2 = plt.subplots(figsize=(10, 5))

ax2.hist(
    reference_data[selected_variable].dropna(),
    bins=30,
    alpha=0.5,
    label="Histórico"
)

ax2.hist(
    current_data[selected_variable].dropna(),
    bins=30,
    alpha=0.5,
    label="Actual"
)

ax2.set_title(f"Distribución histórica vs actual - {selected_variable}")

ax2.legend()

st.pyplot(fig2)

# =========================
# Drift temporal
# =========================

st.subheader("Evolución temporal del drift")

st.dataframe(temporal_drift)

fig3, ax3 = plt.subplots(figsize=(10, 5))

ax3.plot(
    temporal_drift["periodo"],
    temporal_drift["psi_promedio"],
    marker="o"
)

ax3.axhline(0.10, linestyle="--", label="Drift moderado")
ax3.axhline(0.25, linestyle="--", label="Drift fuerte")

ax3.set_ylabel("PSI promedio")
ax3.set_xlabel("Periodo")
ax3.set_title("Evolución temporal del drift")

ax3.legend()

st.pyplot(fig3)

# =========================
# Recomendaciones
# =========================

st.subheader("Recomendaciones automáticas")

if len(drift_fuerte) > 0:
    st.error("""
    Se detectó drift fuerte en una o más variables.
    Se recomienda revisar las variables afectadas,
    validar la calidad de los datos y considerar retraining.
    """)
elif len(drift_moderado) > 0:
    st.warning("""
    Se detectó drift moderado.
    Se recomienda continuar monitoreando antes de reentrenar.
    """)
else:
    st.success("""
    No se detectó drift significativo.
    El modelo puede continuar operando normalmente.
    """)