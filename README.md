# 🛡️ Insurance Company — EDA Dashboard

> **Caso de Estudio N°3 — Especialización en Python for Analytics**  
> MSc. Carlos Carrillo Villavicencio · DMC Institute · 2026

---

## 📋 Descripción del Proyecto

Aplicación interactiva construida con **Streamlit** para el **Análisis Exploratorio de Datos (EDA)** del dataset `InsuranceCompany.csv`.

El objetivo **NO es construir modelos predictivos**, sino aplicar de forma integrada los conceptos del curso: variables y tipos de datos, funciones, f-strings, POO, NumPy, Pandas, Matplotlib, Seaborn y estadística descriptiva.

---

## 🎯 Variable Objetivo

`renewal` — Indicador binario de si el cliente **renovó su póliza de seguro** (1 = Sí, 0 = No).  
La tasa de renovación en el dataset es ~**93.7%**.

---

## 📊 Módulos de la Aplicación

| Módulo | Descripción |
|--------|-------------|
| 🏠 **Home** | Presentación del proyecto, autor, tecnologías y descripción del dataset |
| 📂 **Carga del Dataset** | `st.file_uploader()`, validación, vista previa y dimensiones |
| 🔍 **EDA** | 10 ítems de análisis organizados en 5 tabs con widgets interactivos |
| 📝 **Conclusiones** | 5 conclusiones basadas en el EDA, reflexión final |

### Ítems del EDA

1. Información general del dataset (`.info()`, tipos, nulos)
2. Clasificación de variables con función personalizada (POO)
3. Estadísticas descriptivas extendidas (media, mediana, skewness, kurtosis)
4. Análisis de valores faltantes con visualización
5. Distribución de variables numéricas (histogramas + KDE)
6. Análisis de variables categóricas (barras + proporciones)
7. Bivariado: Numérico vs Categórico (boxplot + violin)
8. Bivariado: Categórico vs Categórico (heatmap + barras agrupadas)
9. Análisis dinámico con filtros interactivos (multiselect, slider, scatter)
10. Hallazgos clave (panel resumen + 5 conclusiones expandibles)

---

## 🛠️ Tecnologías Utilizadas

- **Python 3.11**
- **Streamlit** — interfaz web interactiva
- **Pandas** — manipulación y análisis de datos
- **NumPy** — operaciones vectorizadas
- **Matplotlib** — visualizaciones base
- **Seaborn** — visualizaciones estadísticas

---

## 🏗️ Arquitectura (POO)

```python
class DataAnalyzer:
    """Encapsula toda la lógica de EDA."""
    
    def classify_variables(self) -> dict         # Ítem 2
    def descriptive_stats(self) -> pd.DataFrame  # Ítem 3
    def missing_summary(self) -> pd.DataFrame    # Ítem 4
    def plot_histogram(...)    -> plt.Figure      # Ítem 5
    def plot_bar_categorical(...)-> plt.Figure    # Ítem 6
    def plot_bivariate_num_cat(...)-> plt.Figure  # Ítem 7
    def plot_bivariate_cat_cat(...)-> plt.Figure  # Ítem 8
    def plot_correlation(...)  -> plt.Figure      # correlación
    def plot_renewal_summary() -> plt.Figure      # Ítem 10
    def group_stats(...)       -> pd.DataFrame    # estadísticas grupales
```

---

---
################################################################
## ☁️ Despliegue en Streamlit Cloud
################################################################

### Requisitos previos
- Cuenta en [GitHub](https://github.com)
- Cuenta en [Streamlit Cloud](https://streamlit.io/cloud)

### Pasos

1. **Subir el repositorio a GitHub** con todos los archivos:
   ```
   insurance-eda-dashboard/
   ├── app.py
   ├── requirements.txt
   ├── README.md
   └── InsuranceCompany.csv
   ```

2. **Ir a** [share.streamlit.io](https://share.streamlit.io)

3. **Hacer click en "New app"**

4. **Configurar:**
   - Repository: `https://github.com/JoGCarC/practica_fin`
   - Branch: `main`
   - Main file path: `app.py`

5. **Hacer click en "Deploy!"**

6. En ~2 minutos la app estará disponible en:  
   `https://practicafin.streamlit.app/`

---

## 📁 Estructura del Repositorio

```
insurance-eda-dashboard/
│
├── app.py                  # Aplicación principal Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md               # Este archivo
├── InsuranceCompany.csv    # Dataset
└── assets/                 # (opcional) capturas de pantalla
    ├── screenshot_home.png
    ├── screenshot_eda.png
    └── screenshot_findings.png
```

---

## 📦 Dataset — InsuranceCompany.csv

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `id` | int | Identificador único |
| `perc_premium_paid_by_cash_credit` | float | % prima pagada en efectivo/crédito |
| `age_in_days` | int | Edad del cliente en días |
| `Income` | int | Ingreso mensual |
| `Count_3-6_months_late` | float | Pagos demorados 3-6 meses |
| `Count_6-12_months_late` | float | Pagos demorados 6-12 meses |
| `Count_more_than_12_months_late` | float | Pagos demorados >12 meses |
| `application_underwriting_score` | float | Score de riesgo |
| `no_of_premiums_paid` | int | Total de primas pagadas |
| `sourcing_channel` | str | Canal de captación (A–E) |
| `residence_area_type` | str | Área residencia (Urban/Rural) |
| `premium` | int | Valor de la prima |
| `renewal` | int | Renovación: 1=Sí, 0=No |

**Dimensiones:** 79,853 filas × 13 columnas  
**Valores nulos:** `Count_*_months_late` (~0.12%) y `application_underwriting_score` (~3.7%)

---

## 🔗 Links Relevantes

- 🌐 **App desplegada:** [Ver en Streamlit Cloud](#)
- 💻 **Repositorio:** [GitHub](#)
- 📚 **Streamlit Docs:** https://docs.streamlit.io
- 🐼 **Pandas Docs:** https://pandas.pydata.org/docs/

---

## Captura de Aplicación

-  **Interfaz de la app:** (https://github.com/JoGCarC/practica_fin/blob/main/c1.png?raw=true)


## 👤 Autor

**[Jonatan Gabriel Carbajal Carmen]**  
Especialización en Python for Analytics — DMC Institute  
📧 jonatancarbajal19@gmail.com
🔗 [LinkedIn](#) | [GitHub](#)

---

*Proyecto desarrollado como entregable del Caso de Estudio N°3.*
