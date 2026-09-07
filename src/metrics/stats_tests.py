# src/metrics/stats_tests.py
import numpy as np
from scipy.stats import wilcoxon, levene, mannwhitneyu

def paired_wilcoxon(df, solver_a, solver_b, group_col, value_col):
    """Test de Wilcoxon pareado por semilla, para cada valor de group_col (p.ej. N)."""
    results = {}
    for g, sub in df.groupby(group_col):
        a = sub[sub.Solver == solver_a].sort_values('seed')[value_col].values
        b = sub[sub.Solver == solver_b].sort_values('seed')[value_col].values
        stat, p = wilcoxon(a, b)
        results[g] = {"mean_a": a.mean(), "mean_b": b.mean(), "p_value": p}
    return results

def bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=0):
    rng = np.random.default_rng(seed)
    boots = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return lo, hi

def variance_reduction_test(sample_a, sample_b, n_boot=10000, alpha=0.05, seed=0):
    """Contrasta H0: var(sample_a) == var(sample_b) frente a H1: var(sample_a) != var(sample_b),
    usando el test de Levene centrado en la mediana (equivalente a Brown-Forsythe, robusto a
    no-normalidad y outliers - mas apropiado que un F-test clasico para un GAP de optimizacion,
    que tipicamente no es gaussiano).

    Ademas devuelve un bootstrap CI95 para std(sample_a) - std(sample_b) y para el ratio
    std(sample_a) / std(sample_b), para cuantificar la magnitud del efecto con incertidumbre
    (a diferencia de Levene, que solo da un p-valor de existencia del efecto).

    Parameters:
    sample_a, sample_b (array-like): observaciones independientes de la metrica de interes
        (p.ej. Optimization GAP %) bajo dos condiciones (p.ej. init_type='random' vs 'tqa'),
        sobre la MISMA instancia fija (mismo N, mismo Q), variando solo la semilla de
        inicializacion. No necesitan estar pareadas ni tener el mismo tamano.

    Returns:
    dict con std_a, std_b, std_reduction_pct (positivo si b es mas estable que a),
    levene_stat, levene_p_value, std_diff_ci95, std_ratio_ci95.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)

    std_a, std_b = a.std(ddof=1), b.std(ddof=1)
    stat, p = levene(a, b, center='median')

    rng = np.random.default_rng(seed)
    diff_boots = np.empty(n_boot)
    ratio_boots = np.empty(n_boot)
    for i in range(n_boot):
        ba = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        sa, sb = ba.std(ddof=1), bb.std(ddof=1)
        diff_boots[i] = sa - sb
        ratio_boots[i] = sa / sb if sb > 0 else np.nan

    diff_lo, diff_hi = np.percentile(diff_boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    ratio_boots = ratio_boots[~np.isnan(ratio_boots)]
    if len(ratio_boots) > 0:
        ratio_lo, ratio_hi = np.percentile(ratio_boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    else:
        ratio_lo, ratio_hi = np.nan, np.nan

    return {
        "n_a": len(a), "n_b": len(b),
        "mean_a": a.mean(), "mean_b": b.mean(),
        "std_a": std_a, "std_b": std_b,
        "std_reduction_pct": 100.0 * (std_a - std_b) / std_a if std_a > 0 else np.nan,
        "levene_stat": stat, "levene_p_value": p,
        "std_diff_ci95_lo": diff_lo, "std_diff_ci95_hi": diff_hi,
        "std_ratio_ci95_lo": ratio_lo, "std_ratio_ci95_hi": ratio_hi,
    }

def location_shift_test(sample_a, sample_b, n_boot=10000, alpha=0.05, seed=0):
    """Contrasta H0: la distribucion de sample_a y sample_b tienen la misma mediana,
    frente a H1: difieren, usando Mann-Whitney U (no parametrico, para muestras
    independientes no necesariamente del mismo tamano ni pareadas).

    Complementa a variance_reduction_test: mientras ese test mira la DISPERSION,
    este mira si hay un desplazamiento sistematico en la TENDENCIA CENTRAL (p.ej.
    menos evaluaciones de COBYLA hasta convergencia con una condicion que con otra).

    Returns:
    dict con median_a, median_b, median_diff_pct (positivo si b es menor que a),
    mannwhitney_stat, mannwhitney_p_value, median_diff_ci95.
    """
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)

    median_a, median_b = np.median(a), np.median(b)
    stat, p = mannwhitneyu(a, b, alternative='two-sided')

    rng = np.random.default_rng(seed)
    diff_boots = np.empty(n_boot)
    for i in range(n_boot):
        ba = rng.choice(a, size=len(a), replace=True)
        bb = rng.choice(b, size=len(b), replace=True)
        diff_boots[i] = np.median(ba) - np.median(bb)
    diff_lo, diff_hi = np.percentile(diff_boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "n_a": len(a), "n_b": len(b),
        "median_a": median_a, "median_b": median_b,
        "median_reduction_pct": 100.0 * (median_a - median_b) / median_a if median_a > 0 else np.nan,
        "mannwhitney_stat": stat, "mannwhitney_p_value": p,
        "median_diff_ci95_lo": diff_lo, "median_diff_ci95_hi": diff_hi,
    }