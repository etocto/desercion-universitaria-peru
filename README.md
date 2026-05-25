# University Dropout Prediction in Peru (2020–2025)
### Ensemble Machine Learning on Census-Level Enrollment Data · A SHAP-Based Analysis

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Journal: Expert Systems with Applications](https://img.shields.io/badge/Journal-Expert%20Systems%20with%20Applications-red.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications)
[![Institution: UPeU](https://img.shields.io/badge/Institution-Universidad%20Peruana%20Unión-blue.svg)](https://upeu.edu.pe)

---

## Overview

This repository contains the complete reproducible pipeline for the paper:

> **Tocto Cano, E. (2026).** *Predicting university dropout in Peru using ensemble machine learning on census-level enrollment data (2020–2025): A SHAP-based interpretability analysis with out-of-cohort validation.* Expert Systems with Applications. Universidad Peruana Unión.

### Key Results

| Metric | Value |
|--------|-------|
| Dataset | 3,553,322 unique students · 17,361,785 enrollment records |
| Best model | Random Forest (AUC = 0.8855 ± 0.0419) |
| Hold-out AUC (cohort 2020) | **0.8980** |
| CV–HO gap | 0.0126 (near-zero overfitting) |
| Validation type | 5-fold Time-Series CV + Out-of-cohort validation |
| COVID cohort OR | **2.307** (p < 0.001) |
| Semesters enrolled OR | **0.589** (p < 0.001) |

---

## Repository Structure

```
├── paper1_desercion_pipeline_v6.py   # Main pipeline (final version)
├── regenerar_figuras_en.py           # Regenerate all figures in English
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── .gitignore                        # Excludes data and results
└── resultados_paper1/                # Generated results (see .gitignore)
    ├── tabla2_resultados_cv.csv
    ├── tabla3_delong_test.csv
    ├── tabla3b_logit_coef.csv
    ├── tabla4_cox_ph.csv
    ├── tabla5_fairness.csv
    ├── shap_importancia_global.csv
    ├── holdout_resultados.csv
    ├── figures_en/                   # Figures in English
    │   ├── fig1_dataset_overview.png
    │   ├── fig2_roc_curves_cv.png
    │   ├── fig2b_roc_holdout_cohort2020.png
    │   ├── fig3_model_comparison_table.png
    │   ├── fig4_delong_comparison.png
    │   ├── fig5_odds_ratios_forest.png
    │   ├── fig6_shap_importance.png
    │   ├── fig7_fairness_subgroups.png
    │   ├── fig8_threshold_optimization.png
    │   └── fig9_graphical_abstract.png
    └── modelos_v6.pkl
```

---

## Methodology

### Data
- **Source:** MINEDU (Ministerio de Educación del Perú) administrative records
- **Period:** 2020–2025 (12 semesters)
- **Coverage:** All Peruvian universities (census-level)
- **Files:** 12 enrollment files + 4 graduate files

### Dropout Definition (Triple Criterion)
1. Student appeared in ≥1 enrollment semester
2. Disappeared for ≥2 consecutive semesters
3. Not found in any graduate file

### Models
| Model | Type | Optimization |
|-------|------|-------------|
| Logistic Regression | Baseline | Optuna (50 trials) |
| Random Forest | Ensemble | Optuna (60 trials) |
| XGBoost | Gradient Boosting | Optuna (100 trials) |
| MLP | Neural Network | Fixed architecture |

### Validation Strategy
- **Development:** Cohorts 2021–2025 (5-fold Time-Series CV)
- **Hold-out:** Cohort 2020 (out-of-cohort validation, 4–5 years follow-up)
- **Balancing:** SMOTE (adaptive, inside each fold)
- **Threshold:** Optimal t* per model (maximizes F1)

### Features (9 optimal, selected by RFECV)
| Feature | Description |
|---------|-------------|
| `n_semestres` | Number of semesters enrolled |
| `EDAD_INGRESO` | Age at first enrollment |
| `COHORTE_COVID` | Enrolled during COVID-19 (2020–2021) |
| `ES_PRIVADA` | Private institution |
| `LICENCIADA` | Licensed by SUNEDU |
| `MACROREGION` | Geographic macro-region |
| `NIVEL_ACADEMICO` | Academic level |
| `AREA_CONOCIMIENTO` | Knowledge area (5 categories) |
| `brecha` | Enrollment gap (semester absence) |

---

## Key Findings

1. **`n_semestres` dominates** (SHAP 0.31 — 6× more important than next variable): Early persistence is the strongest predictor of retention.

2. **COVID-19 impact** (OR = 1.748, p < 0.001): Students enrolled during the pandemic have 75% higher dropout risk.

3. **Private institutions** (OR = 0.398, p < 0.001): 60% lower dropout risk compared to public institutions.

4. **SUNEDU licensing paradox** (OR = 1.701, p < 0.001): Licensed institutions show higher reported dropout — likely because unlicensed institutions closed, removing their students from records.

5. **Algorithmic fairness**: LICENCIADA gap = 0.176 and COHORTE_COVID gap = 0.167 indicate structural heterogeneity — scientific findings for educational policy, not model errors.

---

## Installation

```bash
git clone https://github.com/[tu-usuario]/desercion-universitaria-peru.git
cd desercion-universitaria-peru

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

### Run the complete pipeline
```bash
python paper1_desercion_pipeline_v6.py
```

### Regenerate figures in English
```bash
python regenerar_figuras_en.py
```

**Important:** The data files (CSV) are not included in this repository due to size constraints (~7 GB total). Place the MINEDU enrollment and graduate files in the project root directory before running.

---

## Data Files Required

```
matriculado_2020_I.csv   matriculado_2020_II.csv
matriculado_2021_I.csv   matriculado_2021_II.csv
matriculado_2022_I.csv   matriculado_2022_II.csv
matriculado_2023_I.csv   matriculado_2023_II.csv
matriculado_2024_I.csv   matriculado_2024_II.csv
matriculado_2025_I.csv   matriculado_2025_II.csv
egresado_2022.csv        egresado_2023.csv
egresado_2024.csv        egresado_2025.csv
```

Encoding: `latin-1` · Separator: `|`

---

## Citation

```bibtex
@article{tocto2026dropout,
  title   = {Predicting university dropout in Peru using ensemble machine
             learning on census-level enrollment data (2020--2025):
             A SHAP-based interpretability analysis with out-of-cohort validation},
  author  = {Tocto Cano, Esteban},
  journal = {Expert Systems with Applications},
  year    = {2026},
  note    = {Universidad Peruana Uni\'on}
}
```

---

## Author

**Esteban Tocto Cano**
- Institution: Universidad Peruana Unión — Ingeniería de Sistemas
- Target journal: Expert Systems with Applications (IF 7.5, Q1, Elsevier)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
