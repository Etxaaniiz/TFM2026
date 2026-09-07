# src/metrics/run_stats_tests.py
"""Calcula los p-valores definitivos (Wilcoxon pareado + bootstrap CI) para:
  - GAP (%): Simulated Annealing vs XY-QAOA Regularized
  - Sharpe OOS: Gurobi vs XY-QAOA Regularized
para los 8 valores de N de la fase 'N_scaling'.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.stats_tests import paired_wilcoxon, bootstrap_ci

RESULTS_PATH = PROJECT_ROOT / "output/results/results.csv"
OUT_PATH = PROJECT_ROOT / "output/results/stats_tests_results.csv"
VAR_OUT_PATH = PROJECT_ROOT / "output/results/stats_summary.csv"


def main():
    df = pd.read_csv(RESULTS_PATH)
    df = df[df.experiment_phase == "N_scaling"]

    comparisons = [
        ("GAP", "Simulated Annealing", "XY-QAOA Regularized", "Optimization GAP (%)"),
        ("Sharpe_OOS", "Gurobi", "XY-QAOA Regularized", "Sharpe Ratio Out-of-Sample"),
        # Efecto puro de TQA (H2): mismo mixer XY y alpha=0.0 en ambos solvers,
        # la unica diferencia es init_type='random' vs init_type='tqa'.
        ("GAP_TQA_effect", "XY-QAOA", "XY-QAOA Regularized", "Optimization GAP (%)"),
    ]

    rows = []
    for label, solver_a, solver_b, value_col in comparisons:
        results = paired_wilcoxon(df, solver_a, solver_b, "N", value_col)
        for n, res in sorted(results.items()):
            sub = df[df.N == n]
            a = sub[sub.Solver == solver_a].sort_values("seed")[value_col].values
            b = sub[sub.Solver == solver_b].sort_values("seed")[value_col].values
            diff = a - b
            ci_lo, ci_hi = bootstrap_ci(diff)
            rows.append({
                "metric": label,
                "solver_a": solver_a,
                "solver_b": solver_b,
                "N": n,
                "mean_a": res["mean_a"],
                "mean_b": res["mean_b"],
                "mean_diff": diff.mean(),
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "p_value": res["p_value"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    pd.set_option("display.width", 140)
    print(out.to_string(index=False))
    print(f"\nGuardado en {OUT_PATH}")

    # Reduccion de variabilidad atribuible a TQA (std del GAP entre semillas, por N)
    var_rows = []
    for n, sub in df.groupby("N"):
        std_random = sub[sub.Solver == "XY-QAOA"]["Optimization GAP (%)"].std()
        std_tqa = sub[sub.Solver == "XY-QAOA Regularized"]["Optimization GAP (%)"].std()
        reduction_pct = 100 * (std_random - std_tqa) / std_random
        var_rows.append({
            "N": n,
            "std_GAP_random_init": std_random,
            "std_GAP_tqa_init": std_tqa,
            "variability_reduction_pct": reduction_pct,
        })
    var_out = pd.DataFrame(var_rows).sort_values("N")
    var_out.to_csv(VAR_OUT_PATH, index=False)
    print("\nReduccion de variabilidad (std GAP) atribuible a TQA init, por N:")
    print(var_out.to_string(index=False))
    print(f"Reduccion media across N: {var_out.variability_reduction_pct.mean():.1f}%")
    print(f"Guardado en {VAR_OUT_PATH}")


if __name__ == "__main__":
    main()
