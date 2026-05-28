"""
=============================================================================
REGENERAR_FIGURAS_EN.PY
Lee los CSVs y PKLs ya generados por el pipeline v6 y produce todos
los paneles en inglés, SIN número ni título de figura.

Autor : Esteban Tocto Cano — Universidad Peruana Unión
Uso   : python regenerar_figuras_en.py
Salida: resultados_paper1/figures_en/
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
import joblib
from pathlib import Path

warnings.filterwarnings('ignore')

# ── dependencias opcionales ──────────────────────────────────────────────────
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("  ⚠  shap not installed — SHAP figures will be skipped")

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("  ⚠  lifelines not installed — Kaplan-Meier figure will be skipped")

from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

# =============================================================================
# PATHS
# =============================================================================
BASE_PATH  = Path(__file__).parent
OUT_PATH   = BASE_PATH / 'resultados_paper1'
EN_PATH    = OUT_PATH  / 'figures_en'
EN_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ESTILO GLOBAL — conforme a revistas Q1 (sin títulos de figura)
# =============================================================================
COLORS = {
    'blue'  : '#378ADD',
    'green' : '#1D9E75',
    'orange': '#D85A30',
    'purple': '#8C4FA8',
    'gray'  : '#888888',
}

MODEL_COLORS = {
    'Logit'       : COLORS['blue'],
    'RandomForest': COLORS['green'],
    'XGBoost'     : COLORS['orange'],
    'MLP'         : COLORS['purple'],
}

FEATURE_NAMES_EN = {
    'EDAD_INGRESO'      : 'Age at enrollment',
    'COHORTE_COVID'     : 'COVID-19 cohort',
    'ES_PRIVADA'        : 'Private institution',
    'LICENCIADA'        : 'SUNEDU licensed',
    'MACROREGION'       : 'Macro-region',
    'NIVEL_ACADEMICO'   : 'Academic level',
    'AREA_CONOCIMIENTO' : 'Knowledge area',
    'n_semestres'       : 'Semesters enrolled',
    'brecha'            : 'Enrollment gap (semesters)',
}

plt.rcParams.update({
    'font.family'      : 'DejaVu Serif',
    'font.size'        : 11,
    'axes.labelsize'   : 11,
    'legend.fontsize'  : 10,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
    'grid.linestyle'   : '--',
    'savefig.dpi'      : 300,
    'savefig.bbox'     : 'tight',
    'savefig.facecolor': 'white',
})

# =============================================================================
# HELPERS
# =============================================================================

def save_fig(fig, name: str):
    """Guarda la figura y cierra."""
    path = EN_PATH / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✔  {name}.png")


def load_csv(filename: str) -> pd.DataFrame | None:
    path = OUT_PATH / filename
    if path.exists():
        df = pd.read_csv(path)
        print(f"  ✔  {filename}  ({len(df)} rows)")
        return df
    print(f"  ✘  {filename}  NOT FOUND — skipping related figure")
    return None


def load_pkl(filename: str):
    path = OUT_PATH / filename
    if path.exists():
        obj = joblib.load(path)
        print(f"  ✔  {filename}  (loaded)")
        return obj
    print(f"  ✘  {filename}  NOT FOUND")
    return None


# =============================================================================
# CARGA DE DATOS
# =============================================================================
print("\n" + "=" * 60)
print("  REGENERATING FIGURES IN ENGLISH")
print("  Reading results from pipeline v6 …")
print("=" * 60 + "\n")

df_cv       = load_csv('tabla2_resultados_cv.csv')
df_delong   = load_csv('tabla3_delong_test.csv')
df_cox      = load_csv('tabla4_cox_ph.csv')
df_fairness = load_csv('tabla5_fairness_subgrupos.csv')
df_shap_imp = load_csv('shap_importancia_global.csv')

# Datos de predicción para ROC / calibración
holdout_preds = load_pkl('holdout_predictions.pkl')   # dict: {'Logit': (y_true, y_prob), ...}
shap_data     = load_pkl('shap_data.pkl')             # dict: {'values': np.array, 'X': pd.DataFrame}
survival_data = load_pkl('survival_data.pkl')         # pd.DataFrame con duration, event, group

# =============================================================================
# FIGURA 1 — CV Performance Comparison (bar chart)
# =============================================================================
if df_cv is not None:
    # Columnas reales: AUC_mean, F1_mean, MCC_mean, Brier_mean, Recall_mean
    METRIC_MAP = {
        'AUC'   : 'AUC_mean',
        'F1'    : 'F1_mean',
        'Recall': 'Recall_mean',
        'MCC'   : 'MCC_mean',
        'Brier' : 'Brier_mean',
    }
    STD_MAP = {
        'AUC'   : 'AUC_std',
        'F1'    : 'F1_std',
        'Recall': 'Recall_std',
        'MCC'   : 'MCC_std',
        'Brier' : 'Brier_std',
    }
    available = [(label, col) for label, col in METRIC_MAP.items()
                 if col in df_cv.columns]

    model_col = 'Modelo' if 'Modelo' in df_cv.columns else df_cv.columns[0]
    models    = df_cv[model_col].tolist()

    fig, axes = plt.subplots(1, len(available),
                             figsize=(3.8 * len(available), 4.8), sharey=False)
    if len(available) == 1:
        axes = [axes]

    for ax, (label, col) in zip(axes, available):
        vals    = df_cv[col].values
        std_col = STD_MAP.get(label)
        stds    = df_cv[std_col].values if std_col in df_cv.columns else None
        colors  = [MODEL_COLORS.get(m, COLORS['gray']) for m in models]

        bars = ax.bar(models, vals, yerr=stds, color=colors, width=0.55,
                      edgecolor='white', linewidth=0.8,
                      error_kw=dict(ecolor='#444', capsize=4, lw=1.1))

        ax.set_ylabel(label)
        ax.set_xticklabels(models, rotation=20, ha='right')

        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (max(vals) * 0.012),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

        y_min = max(0, min(vals) - 0.06)
        ax.set_ylim(y_min, min(1.0, max(vals) + 0.12))

    fig.tight_layout()
    save_fig(fig, 'cv_performance_comparison')


# =============================================================================
# FIGURA 2 — ROC Curves (Hold-out)
# =============================================================================
if holdout_preds is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, label='Random classifier')

    for model_name, (y_true, y_prob) in holdout_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        color       = MODEL_COLORS.get(model_name, COLORS['gray'])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{model_name} (AUC = {roc_auc:.4f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    save_fig(fig, 'roc_curves_holdout')
else:
    print("  ⚠  holdout_predictions.pkl not found — ROC figure skipped")
    print("     (re-run the main pipeline or save y_true/y_prob per model)")


# =============================================================================
# FIGURA 3 — Calibration Plot
# =============================================================================
if holdout_preds is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5, label='Perfect calibration')

    for model_name, (y_true, y_prob) in holdout_preds.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
        color               = MODEL_COLORS.get(model_name, COLORS['gray'])
        ax.plot(mean_pred, frac_pos, 'o-', color=color, lw=1.8,
                markersize=5, label=model_name)

    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_fig(fig, 'calibration_plot')


# =============================================================================
# FIGURA 4 — Kaplan-Meier Survival Curves
# =============================================================================
if HAS_LIFELINES and survival_data is not None:
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(6, 5))

    group_col    = 'group'     if 'group'    in survival_data.columns else survival_data.columns[2]
    duration_col = 'duration'  if 'duration' in survival_data.columns else survival_data.columns[0]
    event_col    = 'event'     if 'event'    in survival_data.columns else survival_data.columns[1]

    groups = survival_data[group_col].unique()
    palette = [COLORS['blue'], COLORS['orange'], COLORS['green'], COLORS['purple']]

    for i, grp in enumerate(groups):
        mask = survival_data[group_col] == grp
        kmf.fit(
            survival_data.loc[mask, duration_col],
            event_observed=survival_data.loc[mask, event_col],
            label=str(grp)
        )
        kmf.plot_survival_function(ax=ax, ci_show=True, color=palette[i % len(palette)])

    # Log-rank test entre los dos primeros grupos (si hay exactamente 2)
    if len(groups) == 2:
        g0 = survival_data[group_col] == groups[0]
        g1 = survival_data[group_col] == groups[1]
        lr = logrank_test(
            survival_data.loc[g0, duration_col], survival_data.loc[g1, duration_col],
            event_observed_A=survival_data.loc[g0, event_col],
            event_observed_B=survival_data.loc[g1, event_col]
        )
        ax.text(0.97, 0.97, f'Log-rank p {lr.p_value:.2e}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Semesters enrolled')
    ax.set_ylabel('Retention probability')
    ax.set_ylim([0, 1.05])
    ax.legend(title='Institution type', framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, 'kaplan_meier')
elif not HAS_LIFELINES:
    pass  # mensaje ya impreso arriba
else:
    print("  ⚠  survival_data.pkl not found — Kaplan-Meier figure skipped")


# =============================================================================
# FIGURA 5 — Cox Proportional Hazards (Forest Plot)
# =============================================================================
if df_cox is not None:
    # Detectar columnas automáticamente
    col_map = {}
    for c in df_cox.columns:
        cl = c.lower()
        if 'coef' in cl or 'hr' in cl or 'hazard' in cl:
            col_map.setdefault('hr', c)
        elif 'lower' in cl or '0.025' in cl or 'ci_l' in cl:
            col_map.setdefault('lower', c)
        elif 'upper' in cl or '0.975' in cl or 'ci_u' in cl:
            col_map.setdefault('upper', c)
        elif 'covariate' in cl or 'feature' in cl or 'variable' in cl:
            col_map.setdefault('var', c)

    if len(col_map) >= 3:
        # Traducir nombres de variables si están disponibles
        if 'var' in col_map:
            df_cox['label'] = df_cox[col_map['var']].map(
                lambda x: FEATURE_NAMES_EN.get(x, x))
        else:
            df_cox['label'] = df_cox.index.map(
                lambda x: FEATURE_NAMES_EN.get(x, x))

        hr_col  = col_map['hr']
        lo_col  = col_map['lower']
        hi_col  = col_map['upper']

        # Ordenar por HR descendente
        df_cox = df_cox.sort_values(hr_col, ascending=True).reset_index(drop=True)

        n   = len(df_cox)
        fig, ax = plt.subplots(figsize=(7, max(3, n * 0.45 + 1)))

        y_pos   = np.arange(n)
        hrs     = df_cox[hr_col].values
        lowers  = hrs - df_cox[lo_col].values
        uppers  = df_cox[hi_col].values - hrs
        colors  = [COLORS['orange'] if h > 1 else COLORS['blue'] for h in hrs]

        ax.barh(y_pos, hrs, xerr=[lowers, uppers],
                color=colors, alpha=0.75, height=0.55,
                error_kw=dict(ecolor='#444', capsize=4, lw=1.2))
        ax.axvline(x=1, color='black', linestyle='--', lw=0.9, alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_cox['label'].tolist())
        ax.set_xlabel('Hazard Ratio (95% CI)')

        fig.tight_layout()
        save_fig(fig, 'cox_hazard_ratios')
    else:
        print("  ⚠  Could not detect HR/CI columns in tabla4_cox_ph.csv — skipping")


## =============================================================================
# FIGURA 6a — SHAP Beeswarm
# =============================================================================
if HAS_SHAP and shap_data is not None:
    shap_values = shap_data.get('values')
    X_shap      = shap_data.get('X')

    if shap_values is not None and X_shap is not None:
        # Traducir nombres de columnas
        if hasattr(shap_values, 'feature_names') and shap_values.feature_names:
            shap_values.feature_names = [
                FEATURE_NAMES_EN.get(f, f) for f in shap_values.feature_names]

        # Seleccionar clase 1 si es multiclase (3D)
        sv = shap_values[:, :, 1] if (hasattr(shap_values, 'shape')
                                       and len(shap_values.shape) == 3) else shap_values

        # beeswarm no acepta ax= — genera su propia figura
        shap.plots.beeswarm(sv, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 5)
        plt.xlabel('SHAP value (impact on dropout probability)')
        fig.tight_layout()
        save_fig(fig, 'shap_beeswarm')
    else:
        print("  ⚠  shap_data.pkl missing 'values' or 'X' keys — skipping beeswarm")

elif not HAS_SHAP:
    pass
else:
    print("  ⚠  shap_data.pkl not found — SHAP figures skipped")

# =============================================================================
# FIGURA 6b — SHAP Feature Importance Bar (desde CSV si no hay pkl)
# =============================================================================
if df_shap_imp is not None:
    # Columnas reales: 'Unnamed: 0' = feature, '0' = importance
    df_shap_imp.columns = ['feature', 'importance']
    df_si = (df_shap_imp
             .sort_values('importance', ascending=True)
             .tail(10)
             .copy())
    df_si['label'] = df_si['feature'].map(lambda x: FEATURE_NAMES_EN.get(x, x))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.barh(df_si['label'], df_si['importance'],
                   color=COLORS['blue'], alpha=0.82, height=0.6)
    ax.set_xlabel('Mean |SHAP value|')
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(False)

    for bar, val in zip(bars, df_si['importance']):
        ax.text(bar.get_width() + ax.get_xlim()[1] * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)

    fig.tight_layout()
    save_fig(fig, 'shap_importance_bar')


# =============================================================================
# FIGURA 7 — Fairness / Subgroup Analysis
# =============================================================================
if df_fairness is not None:
    # Detectar columnas
    group_col  = None
    metric_col = None
    for c in df_fairness.columns:
        cl = c.lower()
        if 'group' in cl or 'subgroup' in cl or 'segment' in cl or 'subgrupo' in cl:
            group_col = c
        elif 'auc' in cl or 'f1' in cl or 'recall' in cl or 'precision' in cl:
            metric_col = c

    if group_col and metric_col:
        df_f = df_fairness.sort_values(metric_col, ascending=True)

        fig, ax = plt.subplots(figsize=(6.5, max(3, len(df_f) * 0.5 + 1)))
        colors_f = [COLORS['green'] if v >= df_f[metric_col].mean()
                    else COLORS['orange'] for v in df_f[metric_col]]

        ax.barh(df_f[group_col].astype(str), df_f[metric_col],
                color=colors_f, alpha=0.80, height=0.55)
        ax.axvline(df_f[metric_col].mean(), color='black',
                   linestyle='--', lw=1, alpha=0.7, label='Overall mean')

        ax.set_xlabel(metric_col.upper())
        ax.legend(framealpha=0.9)
        fig.tight_layout()
        save_fig(fig, 'fairness_subgroups')
    else:
        print("  ⚠  Could not detect group/metric columns in tabla5_fairness_subgrupos.csv")


# =============================================================================
# FIGURA 8 — DeLong Test Heatmap
# =============================================================================
if df_delong is not None:
    # Columnas reales: Modelo_A, Modelo_B, AUC_A, AUC_B, z, p_value, significancia
    # Construir matriz simétrica de p-values por pivote
    all_models = sorted(set(df_delong['Modelo_A'].tolist() +
                            df_delong['Modelo_B'].tolist()))
    n = len(all_models)
    idx = {m: i for i, m in enumerate(all_models)}

    matrix = np.full((n, n), np.nan)
    np.fill_diagonal(matrix, 1.0)   # diagonal = 1 (mismo modelo)

    for _, row in df_delong.iterrows():
        i, j = idx[row['Modelo_A']], idx[row['Modelo_B']]
        matrix[i, j] = row['p_value']
        matrix[j, i] = row['p_value']   # simétrico

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    cmap = plt.cm.RdYlGn_r
    im   = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=0.5, aspect='auto')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('p-value (DeLong test)')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(all_models, rotation=30, ha='right')
    ax.set_yticklabels(all_models)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_color = 'white' if val < 0.1 else 'black'
                label = f'{val:.3f}' if val > 0 else '<0.001'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=9, color=txt_color, fontweight='bold')

    ax.set_xlabel('Model')
    ax.set_ylabel('Model')
    fig.tight_layout()
    save_fig(fig, 'delong_pvalue_heatmap')


# =============================================================================
# RESUMEN
# =============================================================================
generated = sorted(EN_PATH.glob('*.png'))
print("\n" + "=" * 60)
print(f"  DONE — {len(generated)} figures saved to:")
print(f"  {EN_PATH}")
print("=" * 60)
for f in generated:
    size_kb = f.stat().st_size / 1024
    print(f"    {f.name:<45} {size_kb:>6.1f} KB")