# =============================================================================
# app.py — Insurance Company EDA Dashboard
# Especialización en Python for Analytics — Caso de Estudio N°3
# Autor : Jonatan Gabriel Carbajal Carmen
# Curso : Especialización en Python for Analytics
# Año   : 2026
# =============================================================================
# Descripción:
#   Aplicación interactiva construida con Streamlit para el Análisis
#   Exploratorio de Datos (EDA) del dataset InsuranceCompany.csv.
#   NO construye modelos predictivos; aplica estadística descriptiva,
#   visualización y Programación Orientada a Objetos (POO).
# =============================================================================

import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DE PÁGINA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance EDA Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores consistente en toda la app
PALETTE_MAIN   = "#2E86AB"   # azul principal
PALETTE_ACCENT = "#E84855"   # rojo acento (No renewal)
PALETTE_OK     = "#44BBA4"   # verde (Yes renewal)
PALETTE_WARN   = "#F4A261"   # naranja (advertencias / nulos)
PALETTE_NEU    = "#6C757D"   # gris neutro
PALETTE_SEQ    = "Blues"     # colormap secuencial
CUSTOM_COLORS  = [PALETTE_MAIN, PALETTE_ACCENT, PALETTE_OK,
                  PALETTE_WARN, "#A23B72", "#C5E1A5"]

# CSS mínimo para mejorar presentación
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0d1b2a; }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    /* Métricas */
    [data-testid="stMetric"] { background:#26c6da; border-radius:8px; padding:8px; }
    /* Títulos de sección */
    .section-title { font-size:1.3rem; font-weight:700;
                     border-left:4px solid #2E86AB;
                     padding-left:10px; margin-bottom:12px; }
    /* Insight box */
    .insight-box { background:#e8f4f8; border-left:4px solid #2E86AB;
                   padding:12px 16px; border-radius:4px; margin:8px 0; }
    /* Warning box */
    .warn-box { background:#fff3e0; border-left:4px solid #F4A261;
                padding:12px 16px; border-radius:4px; margin:8px 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PROGRAMACIÓN ORIENTADA A OBJETOS — Clase principal de análisis
# =============================================================================
class DataAnalyzer:
    """
    Encapsula toda la lógica de análisis exploratorio de datos (EDA).

    Atributos
    ---------
    df : pd.DataFrame
        Dataset cargado y pre-procesado.
    num_cols : list[str]
        Columnas numéricas (excluye 'id' y 'renewal').
    cat_cols : list[str]
        Columnas categóricas (incluye 'renewal' como variable objetivo).

    Métodos principales
    -------------------
    classify_variables()    → dict con listas de columnas por tipo
    descriptive_stats()     → pd.DataFrame con estadísticas extendidas
    missing_summary()       → pd.DataFrame con conteo y porcentaje de nulos
    plot_histogram()        → Figure de distribución de una variable numérica
    plot_bar_categorical()  → Figure de barras de una variable categórica
    plot_bivariate_num_cat()→ Figure boxplot numérico vs categórico
    plot_bivariate_cat_cat()→ Figure heatmap proporcional cat vs cat
    plot_correlation()      → Figure heatmap de correlación
    plot_renewal_summary()  → Figure resumen de hallazgos clave
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Inicializa el analizador y clasifica las columnas automáticamente."""
        self.df = df.copy()
        self._prepare()          # conversiones y limpieza tipológica
        vars_dict = self.classify_variables()
        self.num_cols = vars_dict["numeric"]
        self.cat_cols = vars_dict["categorical"]

    # ------------------------------------------------------------------
    # PREPARACIÓN INTERNA
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        """
        Conversión de tipos: 'renewal' se mantiene int (0/1) pero se usa
        como etiqueta categórica en visualizaciones.
        'sourcing_channel' y 'residence_area_type' se castean a string.
        """
        for col in ["sourcing_channel", "residence_area_type"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str)

    # ------------------------------------------------------------------
    # ÍTEM 2 — CLASIFICACIÓN DE VARIABLES (función personalizada)
    # ------------------------------------------------------------------
    def classify_variables(self) -> dict:
        """
        Clasifica las columnas del DataFrame en numéricas y categóricas.

        Criterio:
          - 'id' se excluye de ambos grupos (es solo identificador).
          - 'renewal' se incluye en categóricas (es la variable objetivo binaria).
          - Resto numérico (int/float) → numéricas.
          - Resto object/string → categóricas.

        Returns
        -------
        dict con claves 'numeric' y 'categorical'.
        """
        exclude = {"id"}
        target  = "renewal"

        numeric: list[str] = []
        categorical: list[str] = []

        for col in self.df.columns:
            if col in exclude:
                continue
            if col == target:
                categorical.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                numeric.append(col)
            else:
                categorical.append(col)

        return {"numeric": numeric, "categorical": categorical}

    # ------------------------------------------------------------------
    # ÍTEM 3 — ESTADÍSTICAS DESCRIPTIVAS
    # ------------------------------------------------------------------
    def descriptive_stats(self) -> pd.DataFrame:
        """
        Extiende .describe() con mediana, moda, skewness y kurtosis.

        Returns
        -------
        pd.DataFrame con estadísticas por columna numérica.
        """
        base = self.df[self.num_cols].describe().T
        base["median"]   = self.df[self.num_cols].median()
        base["mode"]     = self.df[self.num_cols].mode().iloc[0]
        base["skewness"] = self.df[self.num_cols].skew()
        base["kurtosis"] = self.df[self.num_cols].kurt()
        # Renombrar para claridad
        base.columns = [c.replace("%", "pct") for c in base.columns]
        return base.round(4)

    # ------------------------------------------------------------------
    # ÍTEM 4 — VALORES FALTANTES
    # ------------------------------------------------------------------
    def missing_summary(self) -> pd.DataFrame:
        """
        Resume valores nulos: conteo absoluto y porcentaje.

        Returns
        -------
        pd.DataFrame ordenado de mayor a menor porcentaje de nulos.
        """
        total   = self.df.isnull().sum()
        pct     = (total / len(self.df) * 100).round(2)
        summary = pd.DataFrame({
            "Columna"    : total.index,
            "Nulos"      : total.values,
            "Porcentaje (%)": pct.values,
        })
        summary = summary[summary["Nulos"] > 0].sort_values(
            "Porcentaje (%)", ascending=False
        ).reset_index(drop=True)
        return summary

    # ------------------------------------------------------------------
    # ÍTEM 5 — DISTRIBUCIÓN NUMÉRICA (Histograma)
    # ------------------------------------------------------------------
    def plot_histogram(
        self,
        col: str,
        bins: int = 40,
        show_kde: bool = True,
        color: str = PALETTE_MAIN,
    ) -> plt.Figure:
        """
        Histograma con línea KDE opcional para una columna numérica.

        Parámetros
        ----------
        col      : nombre de la columna.
        bins     : número de intervalos.
        show_kde : superponer curva de densidad.
        color    : color de las barras.
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_data = self.df[col].dropna()

        sns.histplot(plot_data, bins=bins, kde=show_kde, color=color,
                     edgecolor="white", linewidth=0.4, ax=ax)

        # Líneas de media y mediana
        mean_val   = plot_data.mean()
        median_val = plot_data.median()
        ax.axvline(mean_val,   color=PALETTE_ACCENT, linestyle="--",
                   linewidth=1.5, label=f"Media: {mean_val:,.2f}")
        ax.axvline(median_val, color=PALETTE_OK, linestyle="-.",
                   linewidth=1.5, label=f"Mediana: {median_val:,.2f}")

        ax.set_title(f"Distribución de {col}", fontsize=13, fontweight="bold")
        ax.set_xlabel(col, fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"
        ))
        sns.despine(ax=ax)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ÍTEM 6 — VARIABLES CATEGÓRICAS (Barras + Proporciones)
    # ------------------------------------------------------------------
    def plot_bar_categorical(
        self,
        col: str,
        show_pct: bool = True,
    ) -> plt.Figure:
        """
        Gráfico de barras con conteo y porcentaje para variable categórica.
        """
        counts = self.df[col].value_counts()
        pcts   = (counts / counts.sum() * 100).round(1)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # — Barras de conteo
        bars = axes[0].bar(
            counts.index.astype(str), counts.values,
            color=CUSTOM_COLORS[:len(counts)], edgecolor="white", linewidth=0.5
        )
        axes[0].set_title(f"{col} — Conteo", fontsize=12, fontweight="bold")
        axes[0].set_ylabel("Frecuencia", fontsize=10)
        axes[0].set_xlabel(col, fontsize=10)
        for bar, val in zip(bars, counts.values):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9
            )
        sns.despine(ax=axes[0])

        # — Gráfico de torta con proporciones
        axes[1].pie(
            pcts.values, labels=pcts.index.astype(str),
            autopct="%1.1f%%", startangle=90,
            colors=CUSTOM_COLORS[:len(pcts)],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        axes[1].set_title(f"{col} — Proporciones", fontsize=12, fontweight="bold")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ÍTEM 7 — BIVARIADO: NUMÉRICO vs CATEGÓRICO
    # ------------------------------------------------------------------
    def plot_bivariate_num_cat(
        self,
        num_col: str,
        cat_col: str,
    ) -> plt.Figure:
        """
        Boxplot + violin para comparar una variable numérica entre
        categorías de la variable categórica.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Etiquetas legibles si cat_col == 'renewal'
        plot_df = self.df.copy()
        if cat_col == "renewal":
            plot_df["renewal_label"] = plot_df["renewal"].map(
                {1: "Renueva (1)", 0: "No Renueva (0)"}
            )
            cat_col_plot = "renewal_label"
            palette_use  = {"Renueva (1)": PALETTE_OK, "No Renueva (0)": PALETTE_ACCENT}
        else:
            cat_col_plot = cat_col
            palette_use  = CUSTOM_COLORS[:plot_df[cat_col_plot].nunique()]

        # — Boxplot
        sns.boxplot(
            data=plot_df, x=cat_col_plot, y=num_col,
            palette=palette_use, ax=axes[0],
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.4},
        )
        axes[0].set_title(f"Boxplot: {num_col} vs {cat_col}", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(cat_col, fontsize=10)
        axes[0].set_ylabel(num_col, fontsize=10)
        axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        sns.despine(ax=axes[0])

        # — Violin plot
        sns.violinplot(
            data=plot_df, x=cat_col_plot, y=num_col,
            palette=palette_use, inner="quartile", ax=axes[1],
        )
        axes[1].set_title(f"Violin: {num_col} vs {cat_col}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(cat_col, fontsize=10)
        axes[1].set_ylabel(num_col, fontsize=10)
        axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        sns.despine(ax=axes[1])

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ÍTEM 8 — BIVARIADO: CATEGÓRICO vs CATEGÓRICO
    # ------------------------------------------------------------------
    def plot_bivariate_cat_cat(
        self,
        col_x: str,
        col_y: str,
    ) -> plt.Figure:
        """
        Heatmap de proporciones + gráfico de barras agrupadas para
        dos variables categóricas.
        """
        # tabla de contingencia normalizada por fila
        ct = pd.crosstab(
            self.df[col_x],
            self.df[col_y],
            normalize="index",
        ).round(3) * 100

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        # — Heatmap
        sns.heatmap(
            ct, annot=True, fmt=".1f", cmap="Blues",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "% dentro de fila"},
            ax=axes[0],
        )
        axes[0].set_title(f"Heatmap: {col_x} vs {col_y} (%)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel(col_y, fontsize=10)
        axes[0].set_ylabel(col_x, fontsize=10)

        # — Barras agrupadas (conteos absolutos)
        ct_abs = pd.crosstab(self.df[col_x], self.df[col_y])
        ct_abs.plot(
            kind="bar", ax=axes[1],
            color=CUSTOM_COLORS[:ct_abs.shape[1]],
            edgecolor="white", linewidth=0.5,
        )
        axes[1].set_title(f"Barras: {col_x} vs {col_y}", fontsize=12, fontweight="bold")
        axes[1].set_xlabel(col_x, fontsize=10)
        axes[1].set_ylabel("Conteo", fontsize=10)
        axes[1].tick_params(axis="x", rotation=0)
        axes[1].legend(title=col_y, fontsize=9)
        sns.despine(ax=axes[1])

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # CORRELACIÓN
    # ------------------------------------------------------------------
    def plot_correlation(self, method: str = "pearson") -> plt.Figure:
        """
        Heatmap de correlación entre variables numéricas.
        """
        corr = self.df[self.num_cols].corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))  # triángulo superior

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            cbar_kws={"shrink": 0.8},
            ax=ax,
        )
        ax.set_title(
            f"Matriz de Correlación ({method.capitalize()})",
            fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ÍTEM 10 — HALLAZGOS CLAVE
    # ------------------------------------------------------------------
    def plot_renewal_summary(self) -> plt.Figure:
        """
        Panel de 4 visualizaciones resumen orientadas al target 'renewal'.
        """
        df = self.df.copy()
        df["renewal_label"] = df["renewal"].map({1: "Renueva", 0: "No Renueva"})

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle("Panel de Hallazgos Clave — Renovación de Póliza",
                     fontsize=14, fontweight="bold", y=1.01)

        pal = {"Renueva": PALETTE_OK, "No Renueva": PALETTE_ACCENT}

        # 1) Tasa de renovación
        renewal_counts = df["renewal_label"].value_counts()
        axes[0, 0].pie(
            renewal_counts.values,
            labels=renewal_counts.index,
            autopct="%1.1f%%",
            colors=[PALETTE_OK, PALETTE_ACCENT],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        axes[0, 0].set_title("Tasa de Renovación", fontsize=12, fontweight="bold")

        # 2) Income por grupo de renovación
        sns.boxplot(
            data=df, x="renewal_label", y="Income",
            palette=pal, ax=axes[0, 1],
            flierprops={"marker": "o", "markersize": 2, "alpha": 0.3},
        )
        axes[0, 1].set_title("Ingreso vs Renovación", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("")
        axes[0, 1].yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K")
        )
        sns.despine(ax=axes[0, 1])

        # 3) Canal de captación vs renovación
        ct = pd.crosstab(df["sourcing_channel"], df["renewal_label"], normalize="index") * 100
        ct[["Renueva", "No Renueva"]].plot(
            kind="bar", ax=axes[1, 0],
            color=[PALETTE_OK, PALETTE_ACCENT],
            edgecolor="white", linewidth=0.5,
        )
        axes[1, 0].set_title("Canal de Captación vs Renovación (%)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Canal", fontsize=10)
        axes[1, 0].set_ylabel("% dentro del canal", fontsize=10)
        axes[1, 0].tick_params(axis="x", rotation=0)
        axes[1, 0].legend(fontsize=9)
        sns.despine(ax=axes[1, 0])

        # 4) application_underwriting_score por renovación
        for label, color in pal.items():
            subset = df[df["renewal_label"] == label]["application_underwriting_score"].dropna()
            axes[1, 1].hist(subset, bins=40, alpha=0.6, color=color,
                            label=label, edgecolor="white", linewidth=0.3)
        axes[1, 1].set_title("Score de Suscripción vs Renovación", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("application_underwriting_score", fontsize=10)
        axes[1, 1].set_ylabel("Frecuencia", fontsize=10)
        axes[1, 1].legend(fontsize=9)
        sns.despine(ax=axes[1, 1])

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # ESTADÍSTICAS GRUPALES (helper)
    # ------------------------------------------------------------------
    def group_stats(self, num_col: str, cat_col: str) -> pd.DataFrame:
        """
        Estadísticas descriptivas de num_col agrupadas por cat_col.
        """
        return (
            self.df.groupby(cat_col)[num_col]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .round(2)
            .reset_index()
            .rename(columns={
                "count": "N", "mean": "Media",
                "median": "Mediana", "std": "Desv.Est.",
                "min": "Mín.", "max": "Máx.",
            })
        )


# =============================================================================
# FUNCIONES DE RENDERIZADO POR MÓDULO
# =============================================================================

def render_home() -> None:
    """Módulo 1: Presentación del proyecto."""
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("## 🛡️")
    with col_title:
        st.title("Insurance Company — EDA Dashboard")
        st.markdown("*Análisis Exploratorio de Datos interactivo y profesional*")

    st.markdown("---")

    col_info, col_tech = st.columns(2)

    with col_info:
        st.markdown('<div class="section-title">📋 Datos del Proyecto</div>',
                    unsafe_allow_html=True)
        st.markdown("""
| Campo | Detalle |
|---|---|
| **Autor** | Jonatan Gabriel Carbajal Carmen|
| **Curso** | Especialización en Python for Analytics |
| **Institución** | DMC Institute |
| **Docente** | MSc. Carlos Carrillo Villavicencio |
| **Año** | 2025 |
        """)

        st.markdown('<div class="section-title">🎯 Objetivo del Análisis</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        Aplicar de forma integrada los conceptos del curso para explorar el dataset
        **InsuranceCompany.csv**, identificar patrones, distribuciones y relaciones
        entre variables, con foco en la variable objetivo **`renewal`** (renovación
        de póliza), sin construir modelos predictivos.
        """)

    with col_tech:
        st.markdown('<div class="section-title">📊 Sobre el Dataset</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        El dataset contiene información de **~79,853 clientes** de una compañía
        de seguros. Cada fila representa una póliza con sus características
        demográficas, financieras y de comportamiento de pago.

        | Variable clave | Descripción |
        |---|---|
        | `income` | Ingreso mensual del cliente |
        | `renewal` | ¿Renovó la póliza? (1=Sí, 0=No) |
        | `sourcing_channel` | Canal de captación (A–E) |
        | `application_underwriting_score` | Score de riesgo |
        | `premium` | Valor de la prima del seguro |
        """)

        st.markdown('<div class="section-title">🛠️ Tecnologías Utilizadas</div>',
                    unsafe_allow_html=True)

        tech_cols = st.columns(3)
        techs = [
            ("🐍", "Python 3.11"),
            ("📊", "Streamlit"),
            ("🐼", "Pandas"),
            ("🔢", "NumPy"),
            ("📈", "Matplotlib"),
            ("🎨", "Seaborn"),
        ]
        for i, (icon, name) in enumerate(techs):
            with tech_cols[i % 3]:
                st.markdown(
                    f"<div style='text-align:center;padding:8px;"
                    f"background:#109dfa;border-radius:8px;margin:4px;'>"
                    f"{icon}<br><small><b>{name}</b></small></div>",
                    unsafe_allow_html=True
                )

    st.info(
        "👈 **Usa el menú lateral** para navegar entre módulos. "
        "Comienza por **Carga de Dataset** para subir el archivo CSV."
    )


def render_upload() -> pd.DataFrame | None:
    """Módulo 2: Carga y validación del dataset."""
    st.header("📂 Carga del Dataset")
    st.markdown(
        "Sube el archivo **InsuranceCompany.csv** para iniciar el análisis. "
        "Ningún análisis se ejecuta sin un archivo válido cargado."
    )

    uploaded = st.file_uploader(
        label="Selecciona el archivo CSV",
        type=["csv"],
        help="Formato esperado: InsuranceCompany.csv con separador coma.",
    )

    if uploaded is None:
        st.warning("⚠️ No se ha cargado ningún archivo. Por favor sube el CSV.")
        return None

    # ── Validación del archivo
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {e}")
        return None

    # Verificar columnas mínimas esperadas
    expected_cols = {
        "id", "perc_premium_paid_by_cash_credit", "age_in_days", "Income",
        "Count_3-6_months_late", "Count_6-12_months_late",
        "Count_more_than_12_months_late", "application_underwriting_score",
        "no_of_premiums_paid", "sourcing_channel", "residence_area_type",
        "premium", "renewal",
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        st.error(f"❌ Columnas faltantes en el archivo: {missing_cols}")
        return None

    st.success(f"✅ Archivo cargado correctamente: **{uploaded.name}**")

    # ── Métricas de dimensión
    st.markdown("### Dimensiones del Dataset")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📄 Filas", f"{df.shape[0]:,}")
    m2.metric("📋 Columnas", df.shape[1])
    m3.metric("🔢 Variables Num.", int(df.select_dtypes(include="number").shape[1]))
    m4.metric("🔤 Variables Cat.", int(df.select_dtypes(exclude="number").shape[1]))

    # ── Vista previa
    st.markdown("### Vista Previa (primeras 5 filas)")
    st.dataframe(df.head(), use_container_width=True)

    # ── Tipos de datos rápidos
    st.markdown("### Tipos de Datos por Columna")
    dtype_df = pd.DataFrame({
        "Columna"   : df.dtypes.index,
        "Tipo"      : df.dtypes.values.astype(str),
        "Nulos"     : df.isnull().sum().values,
        "No Nulos"  : df.notnull().sum().values,
        "Único(s)"  : [df[c].nunique() for c in df.columns],
    })
    st.dataframe(dtype_df, use_container_width=True)

    return df


def render_eda(analyzer: DataAnalyzer) -> None:
    """Módulo 3: EDA completo organizado en tabs."""
    st.header("🔍 Análisis Exploratorio de Datos (EDA)")

    tabs = st.tabs([
        "📌 Info General",
        "📊 Distribuciones",
        "🔗 Bivariado",
        "🎛️ Análisis Dinámico",
        "💡 Hallazgos Clave",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 — Ítems 1, 2, 3, 4
    # ══════════════════════════════════════════════════════════════
    with tabs[0]:

        # ── ÍTEM 1: Información general
        st.markdown('<div class="section-title">Ítem 1 — Información General del Dataset</div>',
                    unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Resumen `.info()` equivalente:**")
            info_df = pd.DataFrame({
                "Columna"   : analyzer.df.dtypes.index,
                "Dtype"     : analyzer.df.dtypes.values.astype(str),
                "No Nulos"  : analyzer.df.notnull().sum().values,
                "Nulos"     : analyzer.df.isnull().sum().values,
            })
            st.dataframe(info_df, use_container_width=True, height=400)
        with col_b:
            st.markdown("**Tipos de datos:**")
            dtype_counts = analyzer.df.dtypes.astype(str).value_counts()
            fig_dt, ax_dt = plt.subplots(figsize=(5, 3))
            ax_dt.barh(dtype_counts.index, dtype_counts.values,
                       color=[PALETTE_MAIN, PALETTE_ACCENT, PALETTE_OK][:len(dtype_counts)])
            ax_dt.set_xlabel("Cantidad de columnas")
            ax_dt.set_title("Distribución de Tipos de Dato", fontweight="bold")
            for i, v in enumerate(dtype_counts.values):
                ax_dt.text(v + 0.05, i, str(v), va="center", fontsize=10)
            sns.despine(ax=ax_dt)
            plt.tight_layout()
            st.pyplot(fig_dt)
            plt.close(fig_dt)

            st.markdown("**Conteo de valores nulos por columna:**")
            nulls = analyzer.df.isnull().sum()
            nulls = nulls[nulls > 0]
            if len(nulls) > 0:
                st.dataframe(
                    nulls.rename("Nulos").reset_index().rename(columns={"index": "Columna"}),
                    use_container_width=True,
                )
            else:
                st.success("✅ No hay valores nulos en el dataset.")

        st.markdown("---")

        # ── ÍTEM 2: Clasificación de variables
        st.markdown('<div class="section-title">Ítem 2 — Clasificación de Variables</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Se utiliza la **función personalizada** `DataAnalyzer.classify_variables()` "
            "para clasificar automáticamente las columnas según su tipo de dato."
        )

        col_num, col_cat = st.columns(2)
        with col_num:
            st.markdown(f"**🔢 Variables Numéricas ({len(analyzer.num_cols)})**")
            num_df = pd.DataFrame({
                "Variable": analyzer.num_cols,
                "Dtype"   : [str(analyzer.df[c].dtype) for c in analyzer.num_cols],
                "Únicos"  : [analyzer.df[c].nunique() for c in analyzer.num_cols],
            })
            st.dataframe(num_df, use_container_width=True)
        with col_cat:
            st.markdown(f"**🔤 Variables Categóricas ({len(analyzer.cat_cols)})**")
            cat_df = pd.DataFrame({
                "Variable"  : analyzer.cat_cols,
                "Dtype"     : [str(analyzer.df[c].dtype) for c in analyzer.cat_cols],
                "Categorías": [analyzer.df[c].nunique() for c in analyzer.cat_cols],
                "Valores"   : [str(list(analyzer.df[c].unique()[:5])) for c in analyzer.cat_cols],
            })
            st.dataframe(cat_df, use_container_width=True)

        st.markdown("---")

        # ── ÍTEM 3: Estadísticas descriptivas
        st.markdown('<div class="section-title">Ítem 3 — Estadísticas Descriptivas</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Se extiende `.describe()` con **mediana, moda, skewness y kurtosis** "
            "para un análisis más completo de cada variable numérica."
        )
        desc = analyzer.descriptive_stats()
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

        with st.expander("📖 ¿Cómo interpretar estos valores?"):
            st.markdown("""
- **mean / median**: Si difieren mucho, la distribución está sesgada.
- **std**: Dispersión; cuanto mayor, más variados los datos.
- **skewness**: >0 → cola derecha; <0 → cola izquierda; ~0 → simétrica.
- **kurtosis**: >3 → colas pesadas (outliers probables); <3 → distribución aplanada.
- **min / max**: Detecta rangos extremos o posibles errores de captura.
            """)

        st.markdown("---")

        # ── ÍTEM 4: Valores faltantes
        st.markdown('<div class="section-title">Ítem 4 — Análisis de Valores Faltantes</div>',
                    unsafe_allow_html=True)
        missing = analyzer.missing_summary()

        if missing.empty:
            st.success("✅ El dataset no presenta valores faltantes.")
        else:
            col_m1, col_m2 = st.columns([1, 2])
            with col_m1:
                st.dataframe(missing, use_container_width=True)
            with col_m2:
                fig_miss, ax_miss = plt.subplots(figsize=(7, 3.5))
                colors_miss = [PALETTE_WARN if p > 1 else PALETTE_MAIN
                               for p in missing["Porcentaje (%)"]]
                bars_miss = ax_miss.barh(
                    missing["Columna"], missing["Porcentaje (%)"],
                    color=colors_miss, edgecolor="white"
                )
                ax_miss.set_xlabel("% de Valores Nulos")
                ax_miss.set_title("Porcentaje de Nulos por Columna", fontweight="bold")
                for bar, pct in zip(bars_miss, missing["Porcentaje (%)"]):
                    ax_miss.text(
                        bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                        f"{pct}%", va="center", fontsize=9
                    )
                sns.despine(ax=ax_miss)
                plt.tight_layout()
                st.pyplot(fig_miss)
                plt.close(fig_miss)

            st.markdown(
                '<div class="warn-box">⚠️ <b>Discusión:</b> Las columnas '
                '<code>Count_*_months_late</code> presentan ~0.12% de nulos (97 registros) '
                'y <code>application_underwriting_score</code> tiene ~3.7% (2,974 registros). '
                'Dado el volumen total (79,853 filas) se recomienda imputar con la mediana '
                'o tratar los nulos como categoría separada antes de modelar.</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════════════
    # TAB 2 — Ítems 5, 6
    # ══════════════════════════════════════════════════════════════
    with tabs[1]:

        # ── ÍTEM 5: Distribución de variables numéricas
        st.markdown('<div class="section-title">Ítem 5 — Distribución de Variables Numéricas</div>',
                    unsafe_allow_html=True)

        col_sel, col_opt = st.columns([2, 1])
        with col_sel:
            num_selected = st.selectbox(
                "Selecciona una variable numérica",
                options=analyzer.num_cols,
                key="hist_select",
            )
        with col_opt:
            bins_val  = st.slider("Número de bins", 10, 100, 40, key="hist_bins")
            show_kde  = st.checkbox("Mostrar curva KDE", value=True, key="hist_kde")

        fig_hist = analyzer.plot_histogram(num_selected, bins=bins_val, show_kde=show_kde)
        st.pyplot(fig_hist)
        plt.close(fig_hist)

        # Interpretación automática con f-strings
        col_val  = analyzer.df[num_selected].dropna()
        skew_val = col_val.skew()
        skew_txt = (
            "sesgada a la derecha (cola derecha prolongada)" if skew_val > 0.5
            else "sesgada a la izquierda (cola izquierda prolongada)" if skew_val < -0.5
            else "aproximadamente simétrica"
        )
        st.markdown(
            f'<div class="insight-box">'
            f'📊 <b>{num_selected}</b>: '
            f'Media = <b>{col_val.mean():,.2f}</b> | '
            f'Mediana = <b>{col_val.median():,.2f}</b> | '
            f'Desv. Est. = <b>{col_val.std():,.2f}</b> | '
            f'Skewness = <b>{skew_val:.3f}</b> → distribución {skew_txt}.'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown("---")

        # Opción: mostrar todas las distribuciones
        if st.checkbox("Ver histogramas de todas las variables numéricas", key="all_hists"):
            n_cols = 3
            cols_grid = st.columns(n_cols)
            for idx, col in enumerate(analyzer.num_cols):
                with cols_grid[idx % n_cols]:
                    fig_g = analyzer.plot_histogram(col, bins=30, show_kde=True)
                    st.pyplot(fig_g)
                    plt.close(fig_g)

        st.markdown("---")

        # ── ÍTEM 6: Variables categóricas
        st.markdown('<div class="section-title">Ítem 6 — Análisis de Variables Categóricas</div>',
                    unsafe_allow_html=True)

        cat_selected = st.selectbox(
            "Selecciona una variable categórica",
            options=analyzer.cat_cols,
            key="cat_select",
        )
        fig_bar = analyzer.plot_bar_categorical(cat_selected)
        st.pyplot(fig_bar)
        plt.close(fig_bar)

        # Tabla de proporciones con f-strings
        counts_cat  = analyzer.df[cat_selected].value_counts()
        pcts_cat    = (counts_cat / counts_cat.sum() * 100).round(2)
        prop_df     = pd.DataFrame({
            "Categoría": counts_cat.index.astype(str),
            "Conteo"   : counts_cat.values,
            "Proporción (%)": pcts_cat.values,
        })
        st.dataframe(prop_df, use_container_width=True)

        # Correlación
        st.markdown("---")
        st.markdown('<div class="section-title">Matriz de Correlación</div>',
                    unsafe_allow_html=True)
        corr_method = st.selectbox(
            "Método de correlación", ["pearson", "spearman", "kendall"], key="corr_method"
        )
        fig_corr = analyzer.plot_correlation(method=corr_method)
        st.pyplot(fig_corr)
        plt.close(fig_corr)
        st.markdown(
            '<div class="insight-box">💡 La correlación de Spearman es más robusta '
            'ante distribuciones no normales y outliers, como los presentes en '
            '<code>Income</code> y <code>premium</code>.</div>',
            unsafe_allow_html=True
        )

    # ══════════════════════════════════════════════════════════════
    # TAB 3 — Ítems 7, 8
    # ══════════════════════════════════════════════════════════════
    with tabs[2]:

        # ── ÍTEM 7: Bivariado numérico vs categórico
        st.markdown('<div class="section-title">Ítem 7 — Bivariado: Numérico vs Categórico</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Compara la distribución de una variable numérica entre los grupos "
            "de una variable categórica mediante **boxplot** y **violin plot**."
        )

        col7a, col7b = st.columns(2)
        with col7a:
            num7 = st.selectbox(
                "Variable Numérica", analyzer.num_cols,
                index=analyzer.num_cols.index("Income") if "Income" in analyzer.num_cols else 0,
                key="biv_num7",
            )
        with col7b:
            cat7 = st.selectbox(
                "Variable Categórica", analyzer.cat_cols,
                index=analyzer.cat_cols.index("renewal") if "renewal" in analyzer.cat_cols else 0,
                key="biv_cat7",
            )

        fig7 = analyzer.plot_bivariate_num_cat(num7, cat7)
        st.pyplot(fig7)
        plt.close(fig7)

        # Tabla de estadísticas grupales
        st.markdown("**Estadísticas por grupo:**")
        grp_stats = analyzer.group_stats(num7, cat7)
        st.dataframe(grp_stats, use_container_width=True)

        st.markdown("---")

        # ── ÍTEM 8: Bivariado categórico vs categórico
        st.markdown('<div class="section-title">Ítem 8 — Bivariado: Categórico vs Categórico</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Analiza la relación entre dos variables categóricas mediante "
            "un **heatmap de proporciones** y un **gráfico de barras agrupadas**."
        )

        col8a, col8b = st.columns(2)
        with col8a:
            cat8x = st.selectbox(
                "Variable categórica (eje X)",
                [c for c in analyzer.cat_cols if c != "renewal"],
                key="cat8x",
            )
        with col8b:
            cat8y = st.selectbox(
                "Variable categórica (eje Y)",
                analyzer.cat_cols,
                index=analyzer.cat_cols.index("renewal") if "renewal" in analyzer.cat_cols else 0,
                key="cat8y",
            )

        fig8 = analyzer.plot_bivariate_cat_cat(cat8x, cat8y)
        st.pyplot(fig8)
        plt.close(fig8)

        ct_show = pd.crosstab(
            analyzer.df[cat8x], analyzer.df[cat8y], margins=True, margins_name="Total"
        )
        with st.expander("Ver tabla de contingencia completa"):
            st.dataframe(ct_show, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4 — Ítem 9: Análisis Dinámico
    # ══════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown('<div class="section-title">Ítem 9 — Análisis Dinámico por Parámetros</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Configura el análisis según las columnas y filtros que elijas. "
            "Usa los controles para explorar segmentos específicos del dataset."
        )

        # ── Filtros laterales visibles dentro del tab
        fcol1, fcol2, fcol3 = st.columns(3)

        with fcol1:
            area_filter = st.multiselect(
                "Filtrar por Área de Residencia",
                options=sorted(analyzer.df["residence_area_type"].unique()),
                default=sorted(analyzer.df["residence_area_type"].unique()),
                key="dyn_area",
            )
        with fcol2:
            channel_filter = st.multiselect(
                "Filtrar por Canal de Captación",
                options=sorted(analyzer.df["sourcing_channel"].unique()),
                default=sorted(analyzer.df["sourcing_channel"].unique()),
                key="dyn_channel",
            )
        with fcol3:
            renewal_filter = st.multiselect(
                "Filtrar por Renovación",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda x: "Renueva (1)" if x == 1 else "No Renueva (0)",
                key="dyn_renewal",
            )

        # Slider de rango de ingreso
        inc_min = int(analyzer.df["Income"].min())
        inc_max = int(analyzer.df["Income"].max())
        income_range = st.slider(
            "Rango de Ingreso (Income)",
            min_value=inc_min, max_value=inc_max,
            value=(inc_min, inc_max),
            step=1000,
            key="dyn_income",
        )

        # Aplicar filtros
        mask = (
            analyzer.df["residence_area_type"].isin(area_filter)
            & analyzer.df["sourcing_channel"].isin(channel_filter)
            & analyzer.df["renewal"].isin(renewal_filter)
            & analyzer.df["Income"].between(*income_range)
        )
        df_filtered = analyzer.df[mask]

        st.markdown(
            f'<div class="insight-box">📌 Registros filtrados: '
            f'<b>{len(df_filtered):,}</b> de {len(analyzer.df):,} '
            f'({len(df_filtered)/len(analyzer.df)*100:.1f}%)</div>',
            unsafe_allow_html=True
        )

        if df_filtered.empty:
            st.warning("⚠️ El filtro no retornó datos. Ajusta los parámetros.")
        else:
            # Selección de columnas para scatter / comparación
            st.markdown("#### Configurar Visualización")
            dcol1, dcol2 = st.columns(2)
            with dcol1:
                x_col = st.selectbox(
                    "Eje X (variable numérica)", analyzer.num_cols,
                    index=analyzer.num_cols.index("Income") if "Income" in analyzer.num_cols else 0,
                    key="dyn_x",
                )
            with dcol2:
                y_col = st.selectbox(
                    "Eje Y (variable numérica)", analyzer.num_cols,
                    index=analyzer.num_cols.index("premium") if "premium" in analyzer.num_cols else 1,
                    key="dyn_y",
                )

            # Scatter plot dinámico con color por renewal
            fig_dyn, ax_dyn = plt.subplots(figsize=(10, 5))
            for renew_val, color, label in [(1, PALETTE_OK, "Renueva"),
                                             (0, PALETTE_ACCENT, "No Renueva")]:
                subset = df_filtered[df_filtered["renewal"] == renew_val]
                ax_dyn.scatter(
                    subset[x_col], subset[y_col],
                    c=color, label=label, alpha=0.35, s=10, edgecolors="none"
                )
            ax_dyn.set_xlabel(x_col, fontsize=11)
            ax_dyn.set_ylabel(y_col, fontsize=11)
            ax_dyn.set_title(
                f"Scatter: {x_col} vs {y_col} — segmento filtrado",
                fontsize=13, fontweight="bold"
            )
            ax_dyn.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax_dyn.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax_dyn.legend(fontsize=10)
            sns.despine(ax=ax_dyn)
            plt.tight_layout()
            st.pyplot(fig_dyn)
            plt.close(fig_dyn)

            # Estadísticas del segmento filtrado
            with st.expander("📋 Ver estadísticas del segmento filtrado"):
                st.dataframe(
                    df_filtered[analyzer.num_cols].describe().T.round(2),
                    use_container_width=True,
                )

    # ══════════════════════════════════════════════════════════════
    # TAB 5 — Ítem 10: Hallazgos Clave
    # ══════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown('<div class="section-title">Ítem 10 — Hallazgos Clave del EDA</div>',
                    unsafe_allow_html=True)

        # Panel visual de resumen
        fig_summary = analyzer.plot_renewal_summary()
        st.pyplot(fig_summary)
        plt.close(fig_summary)

        st.markdown("---")
        st.markdown("### 📌 5 Conclusiones Finales")

        conclusiones = {
            "1. Alta tasa de renovación — pero con riesgo concentrado": (
                f"El **{analyzer.df['renewal'].mean()*100:.1f}%** de los clientes renueva "
                "su póliza. Sin embargo, el 6.3% de no renovación se concentra en segmentos "
                "específicos (canales B y D, área rural), lo que sugiere una estrategia de "
                "retención focalizada."
            ),
            "2. El ingreso no es el único diferenciador": (
                "Los clientes que no renuevan presentan un ingreso ligeramente menor en "
                "promedio, pero la superposición de distribuciones es alta. Esto indica que "
                "el ingreso por sí solo **no es suficiente** para predecir la renovación; "
                "el canal de captación y el score de suscripción son factores adicionales clave."
            ),
            "3. El score de suscripción discrimina mejor que el ingreso": (
                "Los no renovadores concentran sus scores entre 95 y 98, mientras que los "
                "renovadores se agrupan entre 98 y 100. Un **umbral de score ~98** podría "
                "servir como señal de alerta temprana para activar campañas de retención."
            ),
            "4. Canal de captación impacta la lealtad del cliente": (
                "El canal **A** muestra la mayor proporción de renovación, mientras que el "
                "canal **D** tiene el mayor porcentaje de pérdida de clientes. La estrategia "
                "de adquisición impacta directamente la retención a largo plazo."
            ),
            "5. Pagos tardíos son señal de alarma, pero poco frecuentes": (
                "La mayoría de clientes (>80%) tiene 0 pagos tardíos en todos los rangos. "
                "Sin embargo, quienes acumulan más de 3 pagos tardíos en cualquier rango "
                "presentan menores tasas de renovación. Su baja frecuencia (~3%) los hace "
                "candidatos ideales para intervenciones preventivas personalizadas."
            ),
        }

        for titulo, texto in conclusiones.items():
            with st.expander(f"**{titulo}**", expanded=True):
                st.markdown(texto)

        # Tabla de métricas clave
        st.markdown("### 📊 Métricas Resumen del Dataset")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric(
            "Tasa de Renovación",
            f"{analyzer.df['renewal'].mean()*100:.1f}%"
        )
        mc2.metric(
            "Ingreso Mediano",
            f"${analyzer.df['Income'].median():,.0f}"
        )
        mc3.metric(
            "Prima Mediana",
            f"${analyzer.df['premium'].median():,.0f}"
        )
        mc4.metric(
            "Score Prom. Suscripción",
            f"{analyzer.df['application_underwriting_score'].mean():.2f}"
        )


def render_conclusions() -> None:
    """Módulo 4: Conclusiones y reflexión final."""
    st.header("📝 Conclusiones y Reflexión Final")
    st.markdown("""
    Este proyecto integra los conceptos fundamentales del curso en una herramienta
    analítica real y desplegable en la nube. Las 5 conclusiones del Ítem 10 proveen
    una base sólida para la toma de decisiones sin necesidad de modelos predictivos.

    El EDA reveló que la retención de clientes en esta aseguradora no depende de un
    único factor, sino de la **combinación** de canal de captación, score de suscripción
    y comportamiento de pago. Esta es precisamente la riqueza del análisis exploratorio:
    construir conocimiento antes de modelar.

    > *"In data science, the model is only as good as the data it receives.
    > Understanding your data is the most important step."*
    """)

    st.info(
        "💼 **Entregables**: "
        "[GitHub Repository](#) · [App desplegada en Streamlit Cloud](#)"
    )


# =============================================================================
# PUNTO DE ENTRADA PRINCIPAL
# =============================================================================
def main() -> None:
    """
    Función principal: configura el sidebar y enruta a cada módulo.
    El dataset se mantiene en st.session_state para persistencia entre tabs.
    """

    # ── Sidebar — Menú principal
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#00b4d8;'>🛡️ Insurance EDA</h2>",
            unsafe_allow_html=True
        )
        st.markdown("*Especialización Python for Analytics*")
        st.markdown("---")

        modulo = st.selectbox(
            "📁 Módulo",
            options=[
                "🏠 Home",
                "📂 Carga del Dataset",
                "🔍 EDA",
                "📝 Conclusiones",
            ],
            key="modulo_select",
        )

        st.markdown("---")
        st.markdown("**Dataset requerido:**")
        st.code("InsuranceCompany.csv", language="text")
        st.markdown("**Versión:** 1.0.0")
        st.markdown("**Curso:** Python for Analytics")

    # ── Gestión de estado del DataFrame cargado
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "analyzer" not in st.session_state:
        st.session_state["analyzer"] = None

    # ── Enrutamiento por módulo
    if modulo == "🏠 Home":
        render_home()

    elif modulo == "📂 Carga del Dataset":
        df = render_upload()
        if df is not None:
            st.session_state["df"] = df
            st.session_state["analyzer"] = DataAnalyzer(df)

    elif modulo == "🔍 EDA":
        if st.session_state["analyzer"] is None:
            st.warning(
                "⚠️ Debes cargar el dataset primero. "
                "Ve al módulo **Carga del Dataset**."
            )
        else:
            render_eda(st.session_state["analyzer"])

    elif modulo == "📝 Conclusiones":
        render_conclusions()


# ── Ejecución
if __name__ == "__main__":
    main()
