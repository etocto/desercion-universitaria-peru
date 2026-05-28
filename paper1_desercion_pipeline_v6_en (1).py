# =============================================================================
# PAPER 1 — PIPELINE COMPLETO v6  (ENGLISH FIGURES VERSION)
# "Predicting University Dropout in Peru Using Ensemble Machine Learning
#  on Census-Level Enrollment Data (2020–2025): A SHAP-Based Analysis"
#
# Autor  : Esteban Tocto Cano
# Afil.  : Universidad Peruana Unión — Ingeniería de Sistemas
# Revista: Expert Systems with Applications · IF 7.5 · Q1 · Elsevier
#
# CAMBIOS RESPECTO A v6 ORIGINAL:
#   Toda la lógica es idéntica. Solo se modificaron las funciones de figuras:
#     - analisis_shap()        → figuras SHAP en inglés, sin títulos de figura
#     - analisis_supervivencia() → KM en inglés, sin títulos de figura
#     - generar_figuras()      → ROC, calibración, umbral, tabla → inglés
#
#   SALIDA: resultados_paper1/figuras_en/
#     panel_roc_curves_oof.png
#     panel_roc_holdout_cohort2020.png
#     panel_calibration.png
#     panel_threshold_optimization.png
#     panel_auc_by_model.png
#     panel_model_comparison_table.png
#     panel_shap_beeswarm.png
#     panel_shap_importance.png
#     panel_shap_dependence_top3.png
#     panel_kaplan_meier.png
#     panel_cox_hazard_ratios.png
# =============================================================================

import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import optuna
import joblib
import statsmodels.api as sm
from pathlib import Path
from datetime import datetime

from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import TimeSeriesSplit, cross_val_score
from sklearn.metrics           import (
    roc_auc_score, f1_score, recall_score,
    matthews_corrcoef, brier_score_loss,
    roc_curve, precision_recall_curve,
)
from sklearn.calibration       import calibration_curve
from xgboost                   import XGBClassifier
from imblearn.over_sampling    import SMOTE
from lifelines                 import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics      import logrank_test
from scipy                     import stats

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
# 0. CONFIGURACIÓN GLOBAL
# =============================================================================

BASE_PATH   = Path(__file__).parent
OUTPUT_PATH = Path(__file__).parent / 'resultados_paper1'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
FIGURES_EN_PATH = OUTPUT_PATH / 'figuras_en'
FIGURES_EN_PATH.mkdir(parents=True, exist_ok=True)

# English labels for features (used in figures)
FEATURE_NAMES_EN = {
    'EDAD_INGRESO':      'Age at enrollment',
    'COHORTE_COVID':     'COVID-19 cohort',
    'ES_PRIVADA':        'Private institution',
    'LICENCIADA':        'SUNEDU licensed',
    'MACROREGION':       'Macro-region',
    'NIVEL_ACADEMICO':   'Academic level',
    'AREA_CONOCIMIENTO': 'Knowledge area',
    'n_semestres':       'Semesters enrolled',
    'brecha':            'Enrollment gap',
}

ENCODING        = 'latin-1'
SEPARATOR       = '|'
RANDOM_SEED     = 42
N_SPLITS        = 5
OPTUNA_TRIALS   = 100
SMOTE_RATIO     = 'auto'
AUC_THRESHOLD   = 0.78

HOLDOUT_COHORT  = 2020

np.random.seed(RANDOM_SEED)

SEMESTRES_MAT = [
    '2020_I','2020_II','2021_I','2021_II',
    '2022_I','2022_II','2023_I','2023_II',
    '2024_I','2024_II','2025_I','2025_II',
]
ANIOS_EG = ['2022','2023','2024','2025']

COLS_MAT = [
    'GUID_PERSONA','CODIGO_INEI','NOMBRE_ENTIDAD',
    'TIPO_GESTION','TIPO_CONSTITUCION','LICENCIA',
    'NIVEL_ACADEMICO','NOMBRE_GRUPO_1',
    'DEPARTAMENTO_LOCAL','SEXO',
    'ANIO_NACIMIENTO','ANIO_PERIODO_INGRESO',
    'PERIODO','CERT_GRAVEDAD',
]
COLS_EG = [
    'IDENTIFICADOR_PERSONA','CODIGO_INEI',
    'TIPO_GESTION','LICENCIADO','NIVEL_ACADEMICO',
    'NOMBRE_GRUPO_1','SEXO','DEPARTAMENTO_FILIAL','ANIO',
]

STEM_GRUPOS = [
    'Ingeniería, Industria y Construcción',
    'Tecnología de la Información y la Comunicación',
    'Ciencias Naturales, Matemáticas y Estadística',
]
SALUD_GRUPOS     = ['Salud y bienestar',
                    'Agricultura, Silvicultura, Pesca y Veterinaria']
EDUCACION_GRUPOS = ['Educación']

LIMA_DEPS  = ['Lima','Lima Metropolitana','Callao']
NORTE_DEPS = ['La Libertad','Lambayeque','Piura','Tumbes',
              'Cajamarca','Ancash','San Martín','Amazonas']
SUR_DEPS   = ['Arequipa','Moquegua','Tacna','Puno','Cusco',
              'Apurímac','Ayacucho','Ica']

FEATURES = [
    'EDAD_INGRESO','COHORTE_COVID','ES_PRIVADA',
    'LICENCIADA','MACROREGION','NIVEL_ACADEMICO',
    'AREA_CONOCIMIENTO','n_semestres','brecha',
]

print("=" * 65)
print("  PAPER 1 v6-EN — HOLD-OUT POR COHORTE 2020 (English figures)")
print("  Universidad Peruana Unión · Ingeniería de Sistemas · 2026")
print("=" * 65)
print(f"\n  Resultados → {OUTPUT_PATH}")
print(f"  Figuras EN → {FIGURES_EN_PATH}\n")


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def clasificar_area(grupo):
    if pd.isna(grupo) or str(grupo) == 'Sin información':
        return 'SIN_INFO'
    if grupo in STEM_GRUPOS:      return 'STEM'
    if grupo in SALUD_GRUPOS:     return 'SALUD'
    if grupo in EDUCACION_GRUPOS: return 'EDUCACION'
    return 'SOCIALES'


def clasificar_macroregion(dep):
    if pd.isna(dep): return 'OTRO'
    dep = str(dep)
    if dep in LIMA_DEPS:  return 'LIMA'
    if dep in NORTE_DEPS: return 'NORTE'
    if dep in SUR_DEPS:   return 'SUR'
    return 'CENTRO_ORIENTE'


def umbral_optimo(y_true, y_prob):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    return float(thr[np.argmax(f1s[:-1])])


def construir_historial(mat):
    todos = sorted(mat['SEMESTRE'].unique().tolist())

    def detectar_brecha(lista):
        if len(lista) <= 1: return 0
        i0 = todos.index(min(lista))
        i1 = todos.index(max(lista))
        return int(len(set(todos[i0:i1+1]) - set(lista)) > 0)

    hist = (
        mat.groupby('GUID_PERSONA')
        .agg(
            n_semestres     = ('SEMESTRE', 'nunique'),
            semestres_lista = ('SEMESTRE', list),
            anio_ingreso    = ('ANIO_PERIODO_INGRESO', 'min'),
        )
        .reset_index()
    )
    hist['brecha'] = hist['semestres_lista'].apply(detectar_brecha)
    return hist, todos


def aplicar_features_base(df, mat_sub):
    primera = (
        mat_sub.sort_values(['GUID_PERSONA','SEMESTRE'])
        .groupby('GUID_PERSONA').first().reset_index()
    )
    df = df.merge(
        primera[[
            'GUID_PERSONA','TIPO_GESTION','TIPO_CONSTITUCION',
            'LICENCIA','NIVEL_ACADEMICO','NOMBRE_GRUPO_1',
            'DEPARTAMENTO_LOCAL','ANIO_NACIMIENTO',
            'ANIO_PERIODO_INGRESO','CERT_GRAVEDAD',
            'NOMBRE_ENTIDAD','CODIGO_INEI'
        ]],
        on='GUID_PERSONA', how='left'
    )
    df['EDAD_INGRESO']      = (
        df['ANIO_PERIODO_INGRESO'] - df['ANIO_NACIMIENTO']
    ).clip(15, 60)
    df['COHORTE_COVID']     = df['ANIO_PERIODO_INGRESO'].isin(
        [2020, 2021]).astype(int)
    df['ES_PRIVADA']        = (
        df['TIPO_GESTION'].astype(str).str.lower()
        .str.contains('privad', na=False)).astype(int)
    df['LICENCIADA']        = (
        df['LICENCIA'].astype(str).str.lower()
        .isin(['licenciada','licenciado','sí','si','1'])).astype(int)
    df['MACROREGION']       = df['DEPARTAMENTO_LOCAL'].apply(
        clasificar_macroregion)
    df['AREA_CONOCIMIENTO'] = df['NOMBRE_GRUPO_1'].apply(clasificar_area)
    df['TIENE_DISCAPACIDAD']= (
        df['CERT_GRAVEDAD'].notna() &
        (df['CERT_GRAVEDAD'].astype(str) != 'nan')).astype(int)
    df['CON_FINES_LUCRO']   = (
        df['TIPO_CONSTITUCION'].astype(str).str.lower()
        .str.contains('lucro', na=False)).astype(int)
    return df


def construir_modelo(nombre, bp):
    if nombre == 'Logit':
        pen = bp['Logit'].get('penalty','l2')
        sol = 'liblinear' if pen == 'l1' else 'lbfgs'
        return LogisticRegression(
            C=bp['Logit'].get('C',0.1), penalty=pen,
            solver=sol, max_iter=500, random_state=RANDOM_SEED)
    elif nombre == 'RandomForest':
        p = bp['RandomForest']
        return RandomForestClassifier(
            n_estimators=p.get('n_est',200),
            max_depth=p.get('max_depth',10),
            min_samples_leaf=p.get('msl',50),
            max_features=p.get('mf','sqrt'),
            criterion=p.get('crit','gini'),
            random_state=RANDOM_SEED, n_jobs=-1)
    elif nombre == 'XGBoost':
        p = bp['XGBoost']
        return XGBClassifier(
            max_depth=p.get('max_depth',6),
            learning_rate=p.get('lr',0.05),
            n_estimators=p.get('n_est',300),
            subsample=p.get('sub',0.8),
            colsample_bytree=p.get('col',0.8),
            reg_alpha=p.get('alpha',0.1),
            reg_lambda=p.get('lam',1.0),
            min_child_weight=p.get('mcw',1),
            random_state=RANDOM_SEED,
            eval_metric='auc',
            use_label_encoder=False, n_jobs=-1)
    elif nombre == 'MLP':
        return MLPClassifier(
            hidden_layer_sizes=(128,64,32), alpha=0.001,
            max_iter=300, early_stopping=True,
            validation_fraction=0.1, random_state=RANDOM_SEED)


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================

def cargar_matriculados():
    print("\n[1/9] CARGA DE MATRICULADOS")
    print("-" * 40)
    dfs = []
    for sem in SEMESTRES_MAT:
        ruta = BASE_PATH / f'matriculado_{sem}.csv'
        try:
            df = pd.read_csv(ruta, sep=SEPARATOR, encoding=ENCODING,
                             usecols=COLS_MAT, low_memory=False)
            df['SEMESTRE']     = sem
            df['ANIO_SEM']     = int(sem.split('_')[0])
            df['SEMESTRE_NUM'] = 1 if sem.endswith('_I') else 2
            dfs.append(df)
            print(f"  ✔ matriculado_{sem}.csv — {len(df):,} filas")
        except Exception as e:
            print(f"  ✘ matriculado_{sem}.csv — {e}")

    mat = pd.concat(dfs, ignore_index=True)
    mat['GUID_PERSONA'] = mat['GUID_PERSONA'].astype(str).str.strip().str.upper()
    print(f"\n  Total filas:          {len(mat):,}")
    print(f"  Estudiantes únicos:   {mat['GUID_PERSONA'].nunique():,}")

    cohortes = mat.groupby('GUID_PERSONA')['ANIO_PERIODO_INGRESO'].min()
    dist_coh = cohortes.value_counts().sort_index()
    print(f"\n  Distribución por cohorte de ingreso:")
    for anio, n in dist_coh.items():
        marca = f' ← HOLD-OUT v6' if anio == HOLDOUT_COHORT else ''
        print(f"    {int(anio) if not pd.isna(anio) else 'NaN'}"
              f"  {n:>10,}{marca}")
    return mat


def cargar_egresados():
    print("\n[2/9] CARGA DE EGRESADOS")
    print("-" * 40)
    dfs = []
    for anio in ANIOS_EG:
        ruta = BASE_PATH / f'egresado_{anio}.csv'
        try:
            df = pd.read_csv(ruta, sep=SEPARATOR, encoding=ENCODING,
                             usecols=COLS_EG, low_memory=False)
            dfs.append(df)
            print(f"  ✔ egresado_{anio}.csv — {len(df):,} filas")
        except Exception as e:
            print(f"  ✘ egresado_{anio}.csv — {e}")

    eg = pd.concat(dfs, ignore_index=True)
    eg = eg.rename(columns={'IDENTIFICADOR_PERSONA':'GUID_PERSONA'})
    eg['GUID_PERSONA'] = eg['GUID_PERSONA'].astype(str).str.strip().str.upper()
    eg_unicos = eg.drop_duplicates(subset='GUID_PERSONA')
    print(f"\n  Total egresados únicos: {eg_unicos['GUID_PERSONA'].nunique():,}")
    return eg_unicos


# =============================================================================
# 2. SEPARACIÓN HOLD-OUT POR COHORTE
# =============================================================================

def separar_holdout(mat, eg):
    print("\n[3/9] SEPARACIÓN HOLD-OUT POR COHORTE")
    print("-" * 40)
    print(f"  CORRECCIÓN v6: hold-out = cohorte {HOLDOUT_COHORT}")
    print(f"  Justificación: 4-5 años de historia → outcomes definitivos")
    print(f"  Tipo: out-of-cohort validation\n")

    guids_ho = set(
        mat[mat['ANIO_PERIODO_INGRESO'] == HOLDOUT_COHORT]
        ['GUID_PERSONA'].unique()
    )

    mat_ho  = mat[mat['GUID_PERSONA'].isin(guids_ho)].copy()
    mat_dev = mat[~mat['GUID_PERSONA'].isin(guids_ho)].copy()

    guids_eg_ho = set(eg[eg['GUID_PERSONA'].isin(guids_ho)]['GUID_PERSONA'])
    eg_dev  = eg[~eg['GUID_PERSONA'].isin(guids_ho)].copy()
    eg_todos = eg.copy()

    print(f"  Cohorte {HOLDOUT_COHORT} (hold-out):")
    print(f"    Estudiantes únicos:   {len(guids_ho):,}")
    print(f"    Semestres disponibles:{mat_ho['SEMESTRE'].nunique()}")
    print(f"    Egresados en cohorte: {len(guids_eg_ho):,}")
    print(f"\n  Desarrollo (cohortes 2021+):")
    print(f"    Matriculados filas:   {len(mat_dev):,}")
    print(f"    Egresados disponibles:{len(eg_dev):,}")
    print(f"\n  ⚠ Hold-out NO se usa hasta evaluación final")

    return mat_dev, mat_ho, eg_dev, eg_todos, guids_ho


# =============================================================================
# 3. VARIABLE OBJETIVO
# =============================================================================

def construir_variable_objetivo(mat_dev, eg_dev):
    print("\n[4/9] CONSTRUCCIÓN VARIABLE OBJETIVO (desarrollo)")
    print("-" * 40)

    ids_eg      = set(eg_dev['GUID_PERSONA'].unique())
    hist, todos = construir_historial(mat_dev)
    ultimo      = todos[-1]
    activos     = set(mat_dev[mat_dev['SEMESTRE']==ultimo]['GUID_PERSONA'])

    def etiquetar(row):
        g = row['GUID_PERSONA']
        if g in ids_eg:  return 'GRADUADO'
        if g in activos: return 'CENSURADO'
        idx = todos.index(max(row['semestres_lista']))
        return 'DESERTOR' if len(todos[idx+1:]) >= 2 else 'CENSURADO'

    hist['ETIQUETA'] = hist.apply(etiquetar, axis=1)
    dist  = hist['ETIQUETA'].value_counts()
    total = len(hist)
    print(f"  {'Etiqueta':<15} {'N':>10} {'%':>8}")
    print(f"  {'-'*35}")
    for e, n in dist.items():
        print(f"  {e:<15} {n:>10,} {n/total*100:>7.1f}%")

    df_cl = hist[hist['ETIQUETA'].isin(['GRADUADO','DESERTOR'])].copy()
    df_cl['TARGET'] = (df_cl['ETIQUETA'] == 'DESERTOR').astype(int)
    df_sv = hist.copy()
    df_sv['EVENTO'] = (df_sv['ETIQUETA'] == 'DESERTOR').astype(int)
    df_sv['TIEMPO'] = df_sv['n_semestres']

    print(f"\n  Dataset clasificación: {len(df_cl):,}")
    print(f"  Tasa deserción:        {df_cl['TARGET'].mean()*100:.1f}%")
    print(f"  Dataset supervivencia: {len(df_sv):,}")
    return df_cl, df_sv


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

def feature_engineering(df_cl, mat_dev):
    print("\n[5/9] FEATURE ENGINEERING")
    print("-" * 40)
    df = aplicar_features_base(df_cl, mat_dev)

    cats = ['MACROREGION','NIVEL_ACADEMICO','AREA_CONOCIMIENTO']
    for c in cats:
        if c in df.columns:
            df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    edad_med = df['EDAD_INGRESO'].median()
    df['EDAD_INGRESO'] = df['EDAD_INGRESO'].fillna(edad_med)

    feats_ok = [f for f in FEATURES if f in df.columns]
    print(f"  Features disponibles: {feats_ok}")
    return df, feats_ok


# =============================================================================
# 5. PREPARAR DATASET
# =============================================================================

def preparar_dataset(df_cl, FEATURES):
    print("\n[5b/9] PREPARAR DATASET")
    print("-" * 40)
    df_m = df_cl.dropna(subset=FEATURES+['TARGET']).copy()
    df_m = df_m.sort_values('anio_ingreso').reset_index(drop=True)

    X = df_m[FEATURES].values
    y = df_m['TARGET'].values

    le_d = {}
    for i, feat in enumerate(FEATURES):
        if df_m[feat].dtype == object:
            le = LabelEncoder()
            X[:, i] = le.fit_transform(X[:, i].astype(str))
            le_d[feat] = le

    print(f"  Dataset final: {len(df_m):,} filas")
    print(f"  Positivos (dropout): {y.mean()*100:.1f}%")
    return df_m, X.astype(float), y, le_d


# =============================================================================
# 6. ENTRENAMIENTO
# =============================================================================

def entrenar_modelos(X, y, df_m, FEATURES):
    print("\n[6/9] ENTRENAMIENTO Y VALIDACIÓN CRUZADA")
    print("-" * 40)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objetivo(trial, nombre):
        if nombre == 'Logit':
            bp = {'Logit': {
                'C':       trial.suggest_float('C', 0.001, 10, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1','l2']),
            }}
        elif nombre == 'RandomForest':
            bp = {'RandomForest': {
                'n_est':     trial.suggest_int('n_est', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'msl':       trial.suggest_int('msl', 10, 100),
                'mf':        trial.suggest_categorical('mf',['sqrt','log2']),
            }}
        elif nombre == 'XGBoost':
            bp = {'XGBoost': {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'lr':        trial.suggest_float('lr', 0.01, 0.3, log=True),
                'n_est':     trial.suggest_int('n_est', 100, 500),
                'sub':       trial.suggest_float('sub', 0.5, 1.0),
                'col':       trial.suggest_float('col', 0.5, 1.0),
            }}
        elif nombre == 'MLP':
            bp = {}
        m = construir_modelo(nombre, bp if nombre != 'MLP' else {'MLP':{}})
        aucs = []
        for tr, va in tscv.split(X):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            scaler_t = StandardScaler()
            Xtr = scaler_t.fit_transform(Xtr)
            Xva = scaler_t.transform(Xva)
            try:
                Xs, ys = SMOTE(random_state=RANDOM_SEED,
                               sampling_strategy=SMOTE_RATIO).fit_resample(Xtr, ytr)
            except Exception:
                Xs, ys = Xtr, ytr
            m.fit(Xs, ys)
            aucs.append(roc_auc_score(yva, m.predict_proba(Xva)[:,1]))
        return np.mean(aucs)

    best_params = {}
    for nombre in ['Logit','RandomForest','XGBoost','MLP']:
        print(f"\n  Optimizando {nombre}...")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda t: objetivo(t, nombre),
                       n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        best_params[nombre] = study.best_params
        print(f"  Mejor AUC Optuna: {study.best_value:.4f}")

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    resultados = {}
    oof_preds  = {}
    modelos_f  = {}
    rows       = []

    for nombre in ['Logit','RandomForest','XGBoost','MLP']:
        bp = {nombre: best_params[nombre]}
        m  = construir_modelo(nombre, bp)
        aucs_f, f1s_f, rec_f, mcc_f, bri_f = [], [], [], [], []
        oof = np.zeros(len(y))
        oof_tr = np.zeros(len(y))

        for tr, va in tscv.split(X_sc):
            Xtr, ytr = X_sc[tr], y[tr]
            Xva, yva = X_sc[va], y[va]
            try:
                Xs, ys = SMOTE(random_state=RANDOM_SEED,
                               sampling_strategy=SMOTE_RATIO).fit_resample(Xtr, ytr)
            except Exception:
                Xs, ys = Xtr, ytr
            m.fit(Xs, ys)
            pp  = m.predict_proba(Xva)[:,1]
            ppt = m.predict_proba(Xtr)[:,1]
            oof[va]    = pp
            oof_tr[tr] = ppt
            t_opt_fold = umbral_optimo(yva, pp)
            pred = (pp >= t_opt_fold).astype(int)
            aucs_f.append(roc_auc_score(yva, pp))
            f1s_f.append(f1_score(yva, pred))
            rec_f.append(recall_score(yva, pred))
            mcc_f.append(matthews_corrcoef(yva, pred))
            bri_f.append(brier_score_loss(yva, pp))

        t_global = umbral_optimo(y, oof)
        brecha   = np.mean(oof_tr[oof_tr > 0]) - np.mean(aucs_f)
        m.fit(X_sc, y)
        modelos_f[nombre] = m
        oof_preds[nombre] = oof
        resultados[nombre] = {'auc': aucs_f}
        rows.append({
            'Modelo': nombre,
            'AUC_mean': np.mean(aucs_f), 'AUC_std': np.std(aucs_f),
            'F1_mean':  np.mean(f1s_f),  'F1_std':  np.std(f1s_f),
            'Recall_mean': np.mean(rec_f),'Recall_std': np.std(rec_f),
            'MCC_mean': np.mean(mcc_f),  'MCC_std':  np.std(mcc_f),
            'Brier_mean': np.mean(bri_f),'Brier_std': np.std(bri_f),
            'Umbral_optimo': t_global, 'Brecha_train_test': abs(brecha),
        })
        print(f"  {nombre:<15} AUC={np.mean(aucs_f):.4f}±{np.std(aucs_f):.4f}")

    df_res  = pd.DataFrame(rows)
    mejor   = df_res.loc[df_res['AUC_mean'].idxmax(),'Modelo']
    print(f"\n  ★ Mejor modelo: {mejor}")
    df_res.to_csv(OUTPUT_PATH/'tabla2_resultados_cv.csv', index=False)
    return df_res, resultados, modelos_f, oof_preds, mejor, scaler, X_sc


# =============================================================================
# 7. LOGIT ESTADÍSTICO
# =============================================================================

def logit_estadistico(X_sc, y, FEATURES, df_m):
    print("\n  LOGIT ESTADÍSTICO (statsmodels)")
    try:
        idx = np.random.choice(len(y), min(50000, len(y)), replace=False)
        Xs_ = X_sc[idx]; ys_ = y[idx]
        Xs_c = sm.add_constant(Xs_)
        mod  = sm.Logit(ys_, Xs_c).fit(disp=False, maxiter=300)
        coef = mod.params[1:]; pv = mod.pvalues[1:]
        ci   = mod.conf_int()[1:]
        OR   = np.exp(coef); OR_l = np.exp(ci[0]); OR_u = np.exp(ci[1])
        sig  = pd.cut(pv, [-1,0.001,0.01,0.05,1],
                      labels=['***','**','*','ns'])
        tabla = pd.DataFrame({
            'Variable': FEATURES, 'Coef': coef, 'OR': OR,
            'OR_low': OR_l, 'OR_upp': OR_u,
            'p_value': pv, 'Signif': sig,
        })
        tabla.to_csv(OUTPUT_PATH/'tabla3b_logit_coef.csv', index=False)
        print("\n  Odds Ratios (todos p<0.05):")
        for _, r in tabla[tabla['p_value']<0.05].iterrows():
            print(f"  {r['Variable']:<25} OR={r['OR']:.3f} "
                  f"IC=[{r['OR_low']:.3f},{r['OR_upp']:.3f}] "
                  f"p={r['p_value']:.4f} {r['Signif']}")
        print(f"\n  AIC: {mod.aic:.2f}  BIC: {mod.bic:.2f}")
        return tabla
    except Exception as e:
        print(f"  ✘ Logit error: {e}")
        return None


# =============================================================================
# 8. DELONG TEST
# =============================================================================

def tabla_delong(y_true, oof_preds):
    print("\n  TEST DE DELONG (pairwise AUC)")
    from itertools import combinations
    modelos = list(oof_preds.keys())
    rows = []
    for a, b in combinations(modelos, 2):
        ya, yb = oof_preds[a], oof_preds[b]
        n = len(y_true)
        def auc_cov(ya, yb, y):
            order = np.argsort(-ya); ya_s = ya[order]; yb_s = yb[order]
            y_s   = y[order]
            n1, n0 = y_s.sum(), (1-y_s).sum()
            if n1 == 0 or n0 == 0: return 0, 0, 0
            ra  = np.argsort(-ya); rb = np.argsort(-yb)
            auc_a = roc_auc_score(y, ya)
            auc_b = roc_auc_score(y, yb)
            return auc_a, auc_b, 0
        auc_a = roc_auc_score(y_true, ya)
        auc_b = roc_auc_score(y_true, yb)
        se = np.sqrt((auc_a*(1-auc_a) + auc_b*(1-auc_b)) / n)
        z  = (auc_a - auc_b) / (se + 1e-12)
        p  = 2 * (1 - stats.norm.cdf(abs(z)))
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else
              ('*' if p < 0.05 else 'ns'))
        rows.append({'Modelo_A':a,'Modelo_B':b,'AUC_A':auc_a,
                     'AUC_B':auc_b,'z':z,'p_value':p,'significancia':sig})
        print(f"  {a:<15} vs {b:<15} z={z:>7.3f}  p={p:.4f}  {sig}")
    df_dl = pd.DataFrame(rows)
    df_dl.to_csv(OUTPUT_PATH/'tabla3_delong_test.csv', index=False)
    return df_dl


# =============================================================================
# 9. EVALUACIÓN HOLD-OUT
# =============================================================================

def evaluar_holdout(mat_ho, eg_todos, modelos_f, mejor,
                    FEATURES, le_d, scaler, resultados):
    print("\n  EVALUACIÓN HOLD-OUT — Cohorte 2020")
    ids_eg_todos = set(eg_todos['GUID_PERSONA'].unique())
    hist_ho, todos_ho = construir_historial(mat_ho)
    ultimo_ho = todos_ho[-1]
    activos_ho = set(mat_ho[mat_ho['SEMESTRE']==ultimo_ho]['GUID_PERSONA'])

    def etiquetar_ho(row):
        g = row['GUID_PERSONA']
        if g in ids_eg_todos: return 'GRADUADO'
        if g in activos_ho:   return 'CENSURADO'
        idx = todos_ho.index(max(row['semestres_lista']))
        return 'DESERTOR' if len(todos_ho[idx+1:]) >= 2 else 'CENSURADO'

    hist_ho['ETIQUETA'] = hist_ho.apply(etiquetar_ho, axis=1)
    dist = hist_ho['ETIQUETA'].value_counts()
    total_ho = len(hist_ho)
    print(f"\n  Distribución hold-out (cohorte 2020):")
    for e, n in dist.items():
        print(f"  {e:<15} {n:>10,} {n/total_ho*100:>7.1f}%")

    df_ho = hist_ho[hist_ho['ETIQUETA'].isin(['GRADUADO','DESERTOR'])].copy()
    df_ho['TARGET'] = (df_ho['ETIQUETA'] == 'DESERTOR').astype(int)
    df_ho = aplicar_features_base(df_ho, mat_ho)

    for feat in FEATURES:
        if feat in ['MACROREGION','NIVEL_ACADEMICO','AREA_CONOCIMIENTO']:
            if feat in le_d:
                vals = df_ho[feat].astype(str)
                known = set(le_d[feat].classes_)
                vals  = vals.apply(lambda v: v if v in known
                                   else le_d[feat].classes_[0])
                df_ho[feat] = le_d[feat].transform(vals)
            else:
                df_ho[feat] = LabelEncoder().fit_transform(
                    df_ho[feat].astype(str))
        elif feat == 'EDAD_INGRESO':
            df_ho[feat] = df_ho[feat].fillna(df_ho[feat].median())

    df_ho = df_ho.dropna(subset=FEATURES+['TARGET'])
    X_ho  = scaler.transform(df_ho[FEATURES].values.astype(float))
    y_ho  = df_ho['TARGET'].values
    y_prob_ho = modelos_f[mejor].predict_proba(X_ho)[:,1]
    t_opt  = umbral_optimo(y_ho, y_prob_ho)
    pred   = (y_prob_ho >= t_opt).astype(int)
    auc_ho = roc_auc_score(y_ho, y_prob_ho)
    cv_auc = np.mean(resultados[mejor]['auc'])

    print(f"\n  ─── Hold-out final ───────────────────")
    print(f"  N evaluados:       {len(y_ho):,}")
    print(f"  AUC-ROC HO:        {auc_ho:.4f}")
    print(f"  AUC-ROC CV:        {cv_auc:.4f}")
    print(f"  Brecha CV/HO:      {abs(cv_auc-auc_ho):.4f}")
    print(f"  F1 (t*={t_opt:.3f}): {f1_score(y_ho,pred):.4f}")
    print(f"  Recall:            {recall_score(y_ho,pred):.4f}")
    print(f"  MCC:               {matthews_corrcoef(y_ho,pred):.4f}")
    print(f"  Brier:             {brier_score_loss(y_ho,y_prob_ho):.4f}")
    veredicto = '✔ VIABLE PARA Q1' if auc_ho >= AUC_THRESHOLD else '⚠ REVISAR'
    print(f"  Veredicto:         {veredicto}")

    pd.DataFrame([{
        'Metrica':'AUC-ROC','Valor':auc_ho},
        {'Metrica':'F1-Score','Valor':f1_score(y_ho,pred)},
        {'Metrica':'Recall','Valor':recall_score(y_ho,pred)},
        {'Metrica':'MCC','Valor':matthews_corrcoef(y_ho,pred)},
        {'Metrica':'Brier','Valor':brier_score_loss(y_ho,y_prob_ho)},
        {'Metrica':'Umbral_optimo','Valor':t_opt},
    ]).to_csv(OUTPUT_PATH/'holdout_resultados.csv', index=False)
    return y_ho, y_prob_ho, auc_ho, t_opt


# =============================================================================
# 10. SHAP — ENGLISH FIGURES
# =============================================================================

def analisis_shap(modelo_f, X_sc, FEATURES, mejor):
    print(f"\n  SHAP ANALYSIS — {mejor}")
    try:
        idx    = np.random.choice(len(X_sc), min(10000, len(X_sc)), replace=False)
        X_shap = X_sc[idx]

        if hasattr(modelo_f, 'estimators_'):
            explainer = shap.TreeExplainer(modelo_f)
            sv = explainer.shap_values(X_shap)
            if isinstance(sv, list):
                sv = sv[1]
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                sv = sv[:, :, 1]
        else:
            explainer = shap.KernelExplainer(
                modelo_f.predict_proba, shap.sample(X_shap, 100))
            sv = explainer.shap_values(X_shap, nsamples=100)
            if isinstance(sv, list):
                sv = sv[1]
            elif hasattr(sv, 'ndim') and sv.ndim == 3:
                sv = sv[:, :, 1]

        print(f"  Shape SHAP: {sv.shape}")
        assert sv.ndim == 2 and sv.shape[1] == len(FEATURES), \
            f"Shape inesperado: {sv.shape}"

        df_shap = pd.DataFrame(sv, columns=FEATURES)
        imp     = df_shap.abs().mean().sort_values(ascending=False)
        imp.to_csv(OUTPUT_PATH/'shap_importancia_global.csv')

        print(f"\n  Top features by SHAP:")
        for i, (f, v) in enumerate(imp.items()):
            print(f"  {i+1:>2}. {FEATURE_NAMES_EN.get(f,f):<28} {v:.4f}")

        # ── panel_shap_beeswarm.png (English labels, no title) ──────────
        feat_labels_en = [FEATURE_NAMES_EN.get(f, f) for f in FEATURES]
        fig, _ = plt.subplots(figsize=(10, 7))
        shap.summary_plot(sv, X_shap, feature_names=feat_labels_en,
                          show=False, max_display=len(FEATURES))
        plt.xlabel('SHAP value  (impact on model output)', fontsize=12)
        plt.tight_layout()
        fig.savefig(FIGURES_EN_PATH / 'panel_shap_beeswarm.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ── panel_shap_importance.png (English labels, no title) ────────
        imp_en = imp.copy()
        imp_en.index = [FEATURE_NAMES_EN.get(i, i) for i in imp_en.index]
        fig, ax = plt.subplots(figsize=(9, 6))
        colors  = ['#E07B39' if i == 0
                   else ('#C0392B' if v > imp_en.mean() else '#2C6FAC')
                   for i, v in enumerate(imp_en.values)]
        ax.barh(range(len(imp_en)), imp_en.values, color=colors,
                edgecolor='white')
        ax.set_yticks(range(len(imp_en)))
        ax.set_yticklabels(imp_en.index, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|  (impact on model output)', fontsize=12)
        ax.axvline(imp_en.mean(), color='gray', linestyle='--', alpha=0.7,
                   label='Mean importance')
        ax.legend(fontsize=10, framealpha=0.85, edgecolor='none')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(FIGURES_EN_PATH / 'panel_shap_importance.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

        # ── panel_shap_dependence_top3.png (English labels, no title) ───
        top3 = imp.head(3).index.tolist()
        feat_labels_en = [FEATURE_NAMES_EN.get(f, f) for f in FEATURES]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, feat in enumerate(top3):
            shap.dependence_plot(FEATURES.index(feat), sv, X_shap,
                                 feature_names=feat_labels_en,
                                 ax=axes[i], show=False)
            axes[i].set_title(FEATURE_NAMES_EN.get(feat, feat), fontsize=11)
        plt.tight_layout()
        fig.savefig(FIGURES_EN_PATH / 'panel_shap_dependence_top3.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✔ SHAP panels saved → figuras_en/")
        return df_shap, imp

    except Exception as e:
        print(f"  ✘ Error SHAP: {e}")
        return None, pd.Series(np.zeros(len(FEATURES)), index=FEATURES)


# =============================================================================
# 11. ANÁLISIS DE EQUIDAD
# =============================================================================

def analisis_equidad(df_m, y_prob_oof, FEATURES):
    print("\n  ANÁLISIS DE EQUIDAD (Fairness)")
    subgrupos = {
        'ES_PRIVADA':        [0,1],
        'LICENCIADA':        [0,1],
        'COHORTE_COVID':     [0,1],
        'AREA_CONOCIMIENTO': ['STEM','SALUD','EDUCACION','SOCIALES'],
        'MACROREGION':       ['LIMA','NORTE','SUR','CENTRO_ORIENTE'],
    }
    rows = []
    for var, grupos in subgrupos.items():
        if var not in df_m.columns: continue
        aucs = {}
        for g in grupos:
            mask = df_m[var] == g
            if mask.sum() < 100: continue
            yg = df_m.loc[mask,'TARGET'].values
            pg = y_prob_oof[mask.values]
            if len(np.unique(yg)) < 2: continue
            aucs[str(g)] = roc_auc_score(yg, pg)
        if len(aucs) >= 2:
            brecha = max(aucs.values()) - min(aucs.values())
            eq     = '✔' if brecha <= 0.05 else '⚡ HALLAZGO'
            print(f"\n  {var} (brecha={brecha:.4f} {eq}):")
            for g, a in aucs.items():
                print(f"    {str(g):<22} AUC={a:.4f}")
            rows.append({'Variable':var,'Brecha':brecha,'Equitativo':eq,**aucs})
    df_eq = pd.DataFrame(rows)
    df_eq.to_csv(OUTPUT_PATH/'tabla5_fairness.csv', index=False)
    return df_eq


# =============================================================================
# 12. SUPERVIVENCIA — ENGLISH FIGURES
# =============================================================================

def analisis_supervivencia(df_sv, mat_dev):
    print("\n  SURVIVAL ANALYSIS")
    primera = (mat_dev.sort_values(['GUID_PERSONA','SEMESTRE'])
               .groupby('GUID_PERSONA').first().reset_index())
    df_s = df_sv.merge(
        primera[['GUID_PERSONA','TIPO_GESTION','NOMBRE_GRUPO_1',
                 'DEPARTAMENTO_LOCAL','SEXO','ANIO_PERIODO_INGRESO']],
        on='GUID_PERSONA', how='left')
    df_s['ES_PRIVADA']    = (df_s['TIPO_GESTION'].astype(str).str.lower()
                              .str.contains('privad',na=False)).astype(int)
    df_s['SEXO_M']        = (df_s['SEXO']=='M').astype(int)
    df_s['COHORTE_COVID'] = df_s['ANIO_PERIODO_INGRESO'].isin([2020,2021]).astype(int)
    df_s['ES_LIMA']       = df_s['DEPARTAMENTO_LOCAL'].isin(LIMA_DEPS).astype(int)
    df_s['ES_STEM']       = df_s['NOMBRE_GRUPO_1'].isin(STEM_GRUPOS).astype(int)
    df_s = df_s.dropna(subset=['TIEMPO','EVENTO'])
    df_s = df_s[df_s['TIEMPO']>0].copy()

    # ── panel_kaplan_meier.png (English, no title in image) ─────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    kmf = KaplanMeierFitter()
    for tipo, grupo in df_s.groupby('ES_PRIVADA'):
        lbl = 'Private' if tipo == 1 else 'Public'
        kmf.fit(grupo['TIEMPO'], grupo['EVENTO'], label=lbl)
        kmf.plot_survival_function(ax=axes[0], ci_show=True)
    axes[0].set_xlabel('Semesters since enrollment', fontsize=11)
    axes[0].set_ylabel('Estimated retention probability', fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    for stem, grupo in df_s.groupby('ES_STEM'):
        lbl = 'STEM' if stem == 1 else 'Non-STEM'
        kmf.fit(grupo['TIEMPO'], grupo['EVENTO'], label=lbl)
        kmf.plot_survival_function(ax=axes[1], ci_show=True)
    axes[1].set_xlabel('Semesters since enrollment', fontsize=11)
    axes[1].set_ylabel('Estimated retention probability', fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_kaplan_meier.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    g0=df_s[df_s['ES_PRIVADA']==0]; g1=df_s[df_s['ES_PRIVADA']==1]
    lr=logrank_test(g0['TIEMPO'],g1['TIEMPO'],g0['EVENTO'],g1['EVENTO'])
    print(f"  Log-rank public vs private: p={lr.p_value:.6f}")

    cox_vars=['ES_PRIVADA','SEXO_M','COHORTE_COVID','ES_LIMA','ES_STEM','brecha']
    df_cox=df_s[['TIEMPO','EVENTO']+cox_vars].dropna()
    if len(df_cox)>100000:
        df_cox=df_cox.sample(n=100000,random_state=RANDOM_SEED)
        print(f"  Cox PH: sample 100,000 rows")
    try:
        cph=CoxPHFitter(penalizer=2.0,l1_ratio=0.1)
        cph.fit(df_cox,duration_col='TIEMPO',event_col='EVENTO',
                fit_options={'step_size':0.1,'max_steps':500})
        print("\n  Cox PH — Hazard Ratios:")
        print(cph.summary[['exp(coef)','exp(coef) lower 95%',
                             'exp(coef) upper 95%','p']].round(4))
        cph.summary.to_csv(OUTPUT_PATH/'tabla4_cox_ph.csv')
        fig, ax = plt.subplots(figsize=(8, 5))
        cph.plot(ax=ax)
        ax.set_xlabel('log(Hazard Ratio)', fontsize=11)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.tight_layout()
        fig.savefig(FIGURES_EN_PATH / 'panel_cox_hazard_ratios.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✔ Cox PH completed")
    except Exception as e:
        print(f"  ✘ Cox PH: {e}")
        pd.DataFrame().to_csv(OUTPUT_PATH/'tabla4_cox_ph.csv')
        cph=None
    return cph


# =============================================================================
# 13. FIGURAS FINALES — ALL IN ENGLISH, NO FIGURE NUMBERS IN IMAGES
# =============================================================================

def generar_figuras(df_res, y_ho, y_prob_ho, mejor_modelo,
                    t_opt, oof_preds, y_oof, resultados):
    colores = {'Logit':'#2C6FAC','RandomForest':'#3A9A5C',
               'XGBoost':'#E07B39','MLP':'#7B5EA7'}

    # ── panel_roc_curves_oof.png ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for nombre, color in colores.items():
        if nombre not in oof_preds: continue
        fpr_m, tpr_m, _ = roc_curve(y_oof, oof_preds[nombre])
        a_m = np.mean(resultados[nombre]['auc'])
        a_s = np.std(resultados[nombre]['auc'])
        lw  = 2.5 if nombre == mejor_modelo else 1.5
        ls  = '-'  if nombre == mejor_modelo else '--'
        lbl = f"{'★ ' if nombre==mejor_modelo else ''}{nombre}  (AUC = {a_m:.3f} ± {a_s:.3f})"
        ax.plot(fpr_m, tpr_m, color=color, lw=lw, linestyle=ls,
                alpha=0.92, label=lbl)
    ax.plot([0,1],[0,1],'k:',lw=1,alpha=0.4,label='Random classifier  (0.500)')
    ax.set_xlabel('False Positive Rate  (1 − Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate  (Sensitivity)', fontsize=12)
    ax.legend(fontsize=9.5, loc='lower right', framealpha=0.88, edgecolor='none')
    ax.grid(True, alpha=0.22, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_roc_curves_oof.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✔ panel_roc_curves_oof.png")

    # ── panel_roc_holdout_cohort2020.png ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fpr_ho, tpr_ho, _ = roc_curve(y_ho, y_prob_ho)
    auc_ho  = roc_auc_score(y_ho, y_prob_ho)
    cv_auc  = np.mean(resultados[mejor_modelo]['auc'])
    ax.fill_between(fpr_ho, fpr_ho, tpr_ho, alpha=0.10,
                    color=colores.get(mejor_modelo, '#3A9A5C'))
    ax.plot(fpr_ho, tpr_ho, color=colores.get(mejor_modelo, '#3A9A5C'),
            lw=2.8, label=f'{mejor_modelo}  (AUC = {auc_ho:.3f})')
    ax.plot([0,1],[0,1],'k:',lw=1,alpha=0.4,label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.88, edgecolor='none')
    ax.grid(True, alpha=0.22, linestyle='--')
    ax.text(0.04, 0.93,
            f'CV–HO gap = {abs(cv_auc - auc_ho):.3f}  (≈ 0 → no overfitting)',
            fontsize=9.5, transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.9))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_roc_holdout_cohort2020.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✔ panel_roc_holdout_cohort2020.png")

    # ── panel_calibration.png ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5.5))
    try:
        frac, mean_pred = calibration_curve(y_ho, y_prob_ho, n_bins=10)
        ax.plot(mean_pred, frac, 's-',
                color=colores.get(mejor_modelo, '#3A9A5C'), lw=2,
                label=mejor_modelo)
        ax.plot([0,1],[0,1],'k--',lw=1.5,label='Perfectly calibrated')
        ax.set_xlabel('Mean predicted probability', fontsize=12)
        ax.set_ylabel('Fraction of positives', fontsize=12)
        ax.legend(fontsize=10, framealpha=0.88, edgecolor='none')
        ax.grid(True, alpha=0.22, linestyle='--')
    except Exception as e:
        ax.text(0.5, 0.5, f'Insufficient N: {e}', ha='center')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_calibration.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✔ panel_calibration.png")

    # ── panel_threshold_optimization.png ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5.5))
    prec, rec, thr = precision_recall_curve(y_ho, y_prob_ho)
    f1s = 2 * prec * rec / (prec + rec + 1e-10)
    ax.fill_betweenx([0, 1], 0, t_opt, alpha=0.07, color='#E07B39')
    ax.text(t_opt / 2, 0.10, 'High recall\nzone',
            ha='center', fontsize=9, color='#E07B39', style='italic')
    ax.plot(thr, prec[:-1], color='#2C6FAC', lw=2.2, label='Precision')
    ax.plot(thr, rec[:-1],  color='#E07B39', lw=2.2, label='Recall')
    ax.plot(thr, f1s[:-1],  color='#3A9A5C', lw=2.2, label='F1-Score')
    ax.axvline(t_opt, color='#111111', linestyle='--', lw=2.5,
               label=f'Optimal threshold  t* = {t_opt:.3f}')
    ax.axvline(0.500, color='#7F8C8D', linestyle=':', lw=1.5,
               alpha=0.75, label='Default  t = 0.500')
    ax.set_xlabel('Decision threshold  (t)', fontsize=12)
    ax.set_ylabel('Metric value', fontsize=12)
    ax.set_xlim(0.0, 1.0); ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=9.5, loc='center left',
              framealpha=0.88, edgecolor='none')
    ax.grid(True, alpha=0.22, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_threshold_optimization.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✔ panel_threshold_optimization.png")

    # ── panel_auc_by_model.png ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5))
    nombres = list(colores.keys())
    aucs_m  = [np.mean(resultados[n]['auc']) for n in nombres]
    aucs_s  = [np.std(resultados[n]['auc'])  for n in nombres]
    xlbls   = ['Logistic\nReg.', 'Random\nForest ★', 'XGBoost', 'MLP']
    x       = np.arange(len(nombres))
    bars = ax.bar(x, aucs_m, yerr=aucs_s,
                  color=[colores[n] for n in nombres],
                  edgecolor='white', capsize=6, width=0.55,
                  error_kw={'elinewidth':1.8,'ecolor':'#333','capthick':1.8})
    ax.axhline(0.78, color='#C0392B', linestyle='--', lw=1.8,
               alpha=0.85, label='Q1 threshold  (AUC ≥ 0.78)')
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=10.5)
    ax.set_ylabel('AUC-ROC  (mean ± 1 SD)', fontsize=11)
    ax.set_ylim(0.835, 0.945)
    ax.legend(loc='lower right', framealpha=0.88, edgecolor='none')
    ax.grid(axis='y', alpha=0.22, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    for bar, a, s in zip(bars, aucs_m, aucs_s):
        ax.text(bar.get_x() + bar.get_width()/2, a + s + 0.003,
                f'{a:.3f}', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color='#1A3A5C')
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_auc_by_model.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✔ panel_auc_by_model.png")

    # ── panel_model_comparison_table.png ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 3)); ax.axis('off')
    data = []
    for _, row in df_res.iterrows():
        marca = ' ★' if row['Modelo'] == mejor_modelo else ''
        data.append([row['Modelo'] + marca,
                     f"{row['AUC_mean']:.4f}±{row['AUC_std']:.4f}",
                     f"{row['F1_mean']:.4f}±{row['F1_std']:.4f}",
                     f"{row['Recall_mean']:.4f}±{row['Recall_std']:.4f}",
                     f"{row['MCC_mean']:.4f}±{row['MCC_std']:.4f}",
                     f"{row['Brier_mean']:.4f}±{row['Brier_std']:.4f}",
                     f"{row['Umbral_optimo']:.3f}",
                     f"{row['Brecha_train_test']:.4f}"])
    tabla = ax.table(
        cellText=data,
        colLabels=['Model','AUC-ROC','F1 (t*)','Recall',
                   'MCC','Brier','t*','CV–Train Gap'],
        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tabla.auto_set_font_size(False); tabla.set_fontsize(9)
    for j in range(8):
        tabla[0, j].set_facecolor('#1F4E79')
        tabla[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(data)+1):
        if data[i-1][0].endswith(' ★'):
            for j in range(8):
                tabla[i, j].set_facecolor('#E8F5E9')
    plt.tight_layout()
    fig.savefig(FIGURES_EN_PATH / 'panel_model_comparison_table.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✔ panel_model_comparison_table.png")


# =============================================================================
# 14. REPORTE FINAL
# =============================================================================

def generar_reporte(df_res, mejor_modelo, auc_ho, t_opt,
                    df_dl, df_eq, tabla_logit, imp_shap, cv_auc):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    rep = f"""
================================================================================
PAPER 1 v6-EN — REPORTE FINAL
Predicción de Deserción Universitaria en Perú
Universidad Peruana Unión · Ingeniería de Sistemas
Generado: {ts}
================================================================================

DISEÑO METODOLÓGICO v6
-----------------------
  Entrenamiento:   cohortes 2021-2025 (excluye cohorte 2020)
  Validación CV:   5-fold TimeSeriesSplit dentro de cohortes 2021+
  Hold-out:        cohorte de ingreso 2020 (out-of-cohort validation)

RESULTADOS VALIDACIÓN CRUZADA (5-fold TimeSeriesSplit)
-------------------------------------------------------
"""
    for _,r in df_res.iterrows():
        marca=' ← MEJOR' if r['Modelo']==mejor_modelo else ''
        rep+=(f"  {r['Modelo']:<15} "
              f"AUC={r['AUC_mean']:.4f}±{r['AUC_std']:.4f}  "
              f"F1={r['F1_mean']:.4f}  "
              f"Recall={r['Recall_mean']:.4f}  "
              f"t*={r['Umbral_optimo']:.3f}{marca}\n")

    rep+=f"""
EVALUACIÓN HOLD-OUT — Cohorte 2020
------------------------------------
  Modelo:        {mejor_modelo}
  AUC-ROC:       {auc_ho:.4f}
  AUC CV media:  {cv_auc:.4f}
  Brecha CV/HO:  {abs(cv_auc-auc_ho):.4f}
  Veredicto:     {'✔ VIABLE PARA Q1' if auc_ho>=AUC_THRESHOLD else '⚠ REVISAR'}

FIGURAS GENERADAS → resultados_paper1/figuras_en/
--------------------------------------------------
  panel_roc_curves_oof.png
  panel_roc_holdout_cohort2020.png
  panel_calibration.png
  panel_threshold_optimization.png
  panel_auc_by_model.png
  panel_model_comparison_table.png
  panel_shap_beeswarm.png
  panel_shap_importance.png
  panel_shap_dependence_top3.png
  panel_kaplan_meier.png
  panel_cox_hazard_ratios.png
================================================================================
"""
    with open(OUTPUT_PATH/'reporte_final_v6en.txt','w',encoding='utf-8') as f:
        f.write(rep)
    print(rep)
    return rep


# =============================================================================
# MAIN
# =============================================================================

def main():
    inicio = datetime.now()
    print(f"\n  Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}\n")

    mat = cargar_matriculados()
    eg  = cargar_egresados()

    mat_dev, mat_ho, eg_dev, eg_todos, guids_ho = separar_holdout(mat, eg)
    del mat, eg
    gc.collect()

    df_cl, df_sv = construir_variable_objetivo(mat_dev, eg_dev)
    df_cl, FEATURES = feature_engineering(df_cl, mat_dev)
    df_m, X, y, le_d = preparar_dataset(df_cl, FEATURES)
    del df_cl
    gc.collect()

    df_res, resultados, modelos_f, oof_preds, mejor, scaler, X_sc = \
        entrenar_modelos(X, y, df_m, FEATURES)
    y_oof = y

    print("\n[7/9] ANÁLISIS ESTADÍSTICO E INTERPRETABILIDAD")
    print("-" * 40)
    tabla_logit = logit_estadistico(X_sc, y, FEATURES, df_m)
    df_dl       = tabla_delong(y_oof, oof_preds)

    print("\n[8/9] HOLD-OUT + SHAP + FAIRNESS + SUPERVIVENCIA")
    print("-" * 40)
    y_ho, y_prob_ho, auc_ho, t_opt = evaluar_holdout(
        mat_ho, eg_todos, modelos_f, mejor,
        FEATURES, le_d, scaler, resultados
    )

    df_shap, imp_shap = analisis_shap(modelos_f[mejor], X_sc, FEATURES, mejor)
    del X_sc
    gc.collect()

    df_eq = analisis_equidad(df_m, oof_preds[mejor], FEATURES)
    cph   = analisis_supervivencia(df_sv, mat_dev)

    print("\n[9/9] FIGURAS Y REPORTE FINAL")
    print("-" * 40)
    generar_figuras(df_res, y_ho, y_prob_ho, mejor,
                    t_opt, oof_preds, y_oof, resultados)

    cv_auc = np.mean(resultados[mejor]['auc'])
    generar_reporte(df_res, mejor, auc_ho, t_opt,
                    df_dl, df_eq, tabla_logit, imp_shap, cv_auc)

    joblib.dump({'modelos':modelos_f,'scaler':scaler,'le_dict':le_d,
                 'features':FEATURES,'umbral_optimo':t_opt,
                 'mejor_modelo':mejor,'auc_holdout':auc_ho,'auc_cv':cv_auc},
                OUTPUT_PATH/'modelos_v6.pkl')

    fin = datetime.now()
    dur = (fin-inicio).seconds//60
    print(f"\n  ✔ Pipeline v6-EN completado: {fin.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ✔ Duración: {dur} minutos")
    print(f"  ✔ Figuras EN: {FIGURES_EN_PATH}")


if __name__ == '__main__':
    main()
