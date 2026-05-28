"""
=============================================================================
REGENERAR_FIGURAS_EN_V2.PY — versión final limpia
Lee los CSVs y PKLs del pipeline v6 y genera todas las figuras
en inglés, sin número ni título de figura.

Prerequisito: correr primero generar_holdout_pkl_v2.py

Uso   : python regenerar_figuras_en_v2.py
Salida: resultados_paper1/figures_en/
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False

# =============================================================================
# PATHS
# =============================================================================
BASE_PATH = Path(__file__).parent
OUT_PATH  = BASE_PATH / 'resultados_paper1'
EN_PATH   = OUT_PATH  / 'figures_en'
EN_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# COLORES Y ESTILO
# =============================================================================
MODEL_COLORS = {
    'Logit'       : '#378ADD',
    'RandomForest': '#1D9E75',
    'XGBoost'     : '#D85A30',
    'MLP'         : '#8C4FA8',
}
C = {
    'blue'  : '#378ADD',
    'green' : '#1D9E75',
    'orange': '#D85A30',
    'purple': '#8C4FA8',
    'gray'  : '#888888',
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
def save_fig(fig, name):
    fig.savefig(EN_PATH / f'{name}.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  ✔  {name}.png')

def load_csv(name):
    p = OUT_PATH / name
    if p.exists():
        df = pd.read_csv(p)
        print(f'  ✔  {name}  ({len(df)} rows)')
        return df
    print(f'  ✘  {name}  NOT FOUND')
    return None

def load_pkl(name):
    p = OUT_PATH / name
    if p.exists():
        obj = joblib.load(p)
        print(f'  ✔  {name}  (loaded)')
        return obj
    print(f'  ✘  {name}  NOT FOUND')
    return None

# =============================================================================
# CARGA
# =============================================================================
print('\n' + '=' * 60)
print('  REGENERATING FIGURES IN ENGLISH  (v2 — final)')
print('=' * 60 + '\n')

df_cv        = load_csv('tabla2_resultados_cv.csv')
df_delong    = load_csv('tabla3_delong_test.csv')
df_shap_imp  = load_csv('shap_importancia_global.csv')
holdout_preds = load_pkl('holdout_predictions.pkl')
shap_data    = load_pkl('shap_data.pkl')
survival_data = load_pkl('survival_data.pkl')

# =============================================================================
# FIG 1 — CV Performance (barras con error bars)
# =============================================================================
if df_cv is not None:
    METRIC_MAP = [('AUC','AUC_mean','AUC_std'),
                  ('F1' ,'F1_mean' ,'F1_std'),
                  ('Recall','Recall_mean','Recall_std'),
                  ('MCC','MCC_mean','MCC_std'),
                  ('Brier','Brier_mean','Brier_std')]
    available = [(l,m,s) for l,m,s in METRIC_MAP if m in df_cv.columns]
    model_col = 'Modelo' if 'Modelo' in df_cv.columns else df_cv.columns[0]
    models    = df_cv[model_col].tolist()

    fig, axes = plt.subplots(1, len(available),
                             figsize=(3.8*len(available), 4.8))
    if len(available) == 1:
        axes = [axes]

    for ax, (label, mcol, scol) in zip(axes, available):
        vals  = df_cv[mcol].values
        stds  = df_cv[scol].values if scol in df_cv.columns else None
        cols  = [MODEL_COLORS.get(m, C['gray']) for m in models]
        bars  = ax.bar(models, vals, yerr=stds, color=cols, width=0.55,
                       edgecolor='white', linewidth=0.8,
                       error_kw=dict(ecolor='#444', capsize=4, lw=1.1))
        ax.set_ylabel(label)
        ax.set_xticklabels(models, rotation=20, ha='right')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.012,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)
        ax.set_ylim(max(0, min(vals)-0.06), min(1.0, max(vals)+0.12))

    fig.tight_layout()
    save_fig(fig, 'cv_performance_comparison')

# =============================================================================
# FIG 2 — ROC Curves (Hold-out)
# =============================================================================
if holdout_preds is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0,1],[0,1],'k--', lw=0.8, alpha=0.5, label='Random classifier')
    for name, (y_true, y_prob) in holdout_preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=MODEL_COLORS.get(name, C['gray']),
                lw=2, label=f'{name} (AUC = {roc_auc:.4f})')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()
    save_fig(fig, 'roc_curves_holdout')

# =============================================================================
# FIG 3 — Calibration Plot
# =============================================================================
if holdout_preds is not None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0,1],[0,1],'k--', lw=0.8, alpha=0.5, label='Perfect calibration')
    for name, (y_true, y_prob) in holdout_preds.items():
        frac, mean_p = calibration_curve(y_true, y_prob,
                                          n_bins=10, strategy='uniform')
        ax.plot(mean_p, frac, 'o-', color=MODEL_COLORS.get(name, C['gray']),
                lw=1.8, markersize=5, label=name)
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    fig.tight_layout()
    save_fig(fig, 'calibration_plot')

# =============================================================================
# FIG 4 — Kaplan-Meier
# =============================================================================
if HAS_LIFELINES and survival_data is not None:
    df_s = survival_data if isinstance(survival_data, pd.DataFrame) \
           else pd.DataFrame(survival_data)
    dur_col   = 'duration'
    evt_col   = 'event'
    grp_col   = 'group'
    palette   = [C['blue'], C['orange'], C['green'], C['purple']]
    groups    = df_s[grp_col].unique()

    fig, ax = plt.subplots(figsize=(6, 5))
    kmf = KaplanMeierFitter()
    for i, grp in enumerate(groups):
        mask = df_s[grp_col] == grp
        kmf.fit(df_s.loc[mask, dur_col],
                event_observed=df_s.loc[mask, evt_col],
                label=str(grp))
        kmf.plot_survival_function(ax=ax, ci_show=True,
                                    color=palette[i % len(palette)])

    if len(groups) == 2:
        g0 = df_s[grp_col] == groups[0]
        g1 = df_s[grp_col] == groups[1]
        lr = logrank_test(
            df_s.loc[g0, dur_col], df_s.loc[g1, dur_col],
            event_observed_A=df_s.loc[g0, evt_col],
            event_observed_B=df_s.loc[g1, evt_col])
        p_str = f'{lr.p_value:.2e}' if lr.p_value >= 1e-300 else '< 1e-300'
        ax.text(0.97, 0.97, f'Log-rank p = {p_str}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel('Semesters enrolled')
    ax.set_ylabel('Retention probability')
    ax.set_ylim([0, 1.05])
    ax.legend(title='Institution type', framealpha=0.9)
    fig.tight_layout()
    save_fig(fig, 'kaplan_meier')
elif not HAS_LIFELINES:
    print('  ⚠  lifelines not installed — pip install lifelines')

# =============================================================================
# FIG 5 — DeLong p-value Heatmap
# =============================================================================
if df_delong is not None:
    all_models = sorted(set(df_delong['Modelo_A'].tolist() +
                            df_delong['Modelo_B'].tolist()))
    n   = len(all_models)
    idx = {m: i for i, m in enumerate(all_models)}
    mat = np.full((n, n), np.nan)
    np.fill_diagonal(mat, 1.0)

    for _, row in df_delong.iterrows():
        i, j = idx[row['Modelo_A']], idx[row['Modelo_B']]
        mat[i, j] = row['p_value']
        mat[j, i] = row['p_value']

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im   = ax.imshow(mat, cmap=plt.cm.RdYlGn_r, vmin=0, vmax=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('p-value (DeLong test)')
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(all_models, rotation=30, ha='right')
    ax.set_yticklabels(all_models)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                txt = '<0.001' if v < 0.001 else f'{v:.3f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=9,
                        color='white' if v < 0.1 else 'black', fontweight='bold')
    ax.set_xlabel('Model'); ax.set_ylabel('Model')
    fig.tight_layout()
    save_fig(fig, 'delong_pvalue_heatmap')

# =============================================================================
# FIG 6 — SHAP Importance Bar (desde CSV)
# =============================================================================
if df_shap_imp is not None:
    df_si = df_shap_imp.copy()
    df_si.columns = ['feature', 'importance']
    df_si = (df_si.sort_values('importance', ascending=True)
                  .tail(10).copy())
    df_si['label'] = df_si['feature'].map(
        lambda x: FEATURE_NAMES_EN.get(x, x))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.barh(df_si['label'], df_si['importance'],
                   color=C['blue'], alpha=0.82, height=0.6)
    ax.set_xlabel('Mean |SHAP value|')
    ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(False)
    for bar, val in zip(bars, df_si['importance']):
        ax.text(bar.get_width() + ax.get_xlim()[1]*0.01,
                bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    fig.tight_layout()
    save_fig(fig, 'shap_importance_bar')

# =============================================================================
# FIG 7 — SHAP Beeswarm
# =============================================================================
if HAS_SHAP and shap_data is not None:
    sv = shap_data.get('values')
    if sv is not None:
        # Traducir nombres de features
        if hasattr(sv, 'feature_names') and sv.feature_names:
            sv.feature_names = [FEATURE_NAMES_EN.get(f, f)
                                  for f in sv.feature_names]
        # Seleccionar clase 1 si multiclase (shape 3D)
        sv_plot = (sv[:, :, 1]
                   if hasattr(sv, 'shape') and len(sv.shape) == 3
                   else sv)
        # beeswarm no acepta ax= — usa gcf()
        shap.plots.beeswarm(sv_plot, max_display=10, show=False)
        fig = plt.gcf()
        fig.set_size_inches(7, 5)
        plt.xlabel('SHAP value (impact on dropout probability)')
        fig.tight_layout()
        save_fig(fig, 'shap_beeswarm')

# =============================================================================
# RESUMEN FINAL
# =============================================================================
generated = sorted(EN_PATH.glob('*.png'))
print('\n' + '=' * 60)
print(f'  DONE — {len(generated)} figures in:')
print(f'  {EN_PATH}')
print('=' * 60)
for f in generated:
    print(f'    {f.name:<45} {f.stat().st_size/1024:>6.1f} KB')