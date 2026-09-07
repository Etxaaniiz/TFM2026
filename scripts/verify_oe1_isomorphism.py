"""
verify_oe1_isomorphism.py
=========================
Verificación de OE1: Equivalencia exacta QUBO ↔ Ising para N ≤ 14.

Para cada configuración binaria x ∈ {0,1}^N se calcula:
  - E_QUBO(x)  = x^T Q x
  - E_Ising(x) = h^T s + s^T J s + offset   (s_i = 2*x_i - 1)

y se comprueba que son idénticas para todas las 2^N configuraciones.
El resultado relevante es la discrepancia máxima entre ambas representaciones.
"""

import os
import sys
import numpy as np
import dimod

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.portfolio.portfolio_model import build_qubo, qubo_to_ising

# ── Configuración del experimento ─────────────────────────────────────────────
Ns = [6, 8, 10, 12, 14]   # tamaños cubiertos por la verificación bruta
seeds = [42, 43, 44]       # sub-instancias con distintos activos
lambda_val = 0.5

# ── Carga de datos reales (idéntica a run_benchmarks.py) ─────────────────────
def load_regime_data(processed_dir):
    import pandas as pd
    mu_is_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Estable.csv"))
    cov_is_df = pd.read_csv(os.path.join(processed_dir, "covariance_Estable.csv"), index_col=0)
    tickers = [t for t in mu_is_df['Ticker'] if t in cov_is_df.columns]
    mu_is = mu_is_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_is = cov_is_df.loc[tickers, tickers]
    return tickers, mu_is, cov_is


def energy_ising(x_bin, h, J, offset):
    """Calcula la energía Ising de una configuración binaria x ∈ {0,1}^N."""
    N = len(x_bin)
    s = 2.0 * x_bin - 1.0   # {0,1} → {-1,+1}
    e = offset
    for i, hi in h.items():
        e += hi * s[i]
    for (i, j), Jij in J.items():
        e += Jij * s[i] * s[j]
    return e


def verify_instance(N, K, mu_arr, cov_arr):
    """
    Verifica la equivalencia QUBO ↔ Ising para todas las 2^N configuraciones.
    Devuelve el error máximo absoluto.
    """
    Q = build_qubo(mu_arr, cov_arr, K, lambda_val=lambda_val)
    h, J, offset = qubo_to_ising(Q)

    max_error = 0.0
    for idx in range(2 ** N):
        # Decodificar bit-string (MSB primero)
        x_bin = np.array([int(b) for b in bin(idx)[2:].zfill(N)], dtype=float)

        # Energía QUBO directa
        e_qubo = float(x_bin @ Q @ x_bin)

        # Energía Ising convertida
        e_ising = energy_ising(x_bin, h, J, offset)

        err = abs(e_qubo - e_ising)
        if err > max_error:
            max_error = err

    return max_error


def main():
    print("=" * 70)
    print("VERIFICACION OE1 -- Equivalencia QUBO <-> Ising (fuerza bruta)")
    print("=" * 70)

    processed_dir = os.path.join(project_root, "data", "processed")
    all_tickers, mu_all, cov_all = load_regime_data(processed_dir)

    results = []
    all_passed = True

    for N in Ns:
        K = N // 2
        for seed in seeds:
            rng = np.random.RandomState(seed)
            idx_sel = rng.choice(len(all_tickers), size=N, replace=False)
            selected = [all_tickers[i] for i in idx_sel]

            mu_arr = mu_all.loc[selected].values
            cov_arr = cov_all.loc[selected, selected].values

            max_err = verify_instance(N, K, mu_arr, cov_arr)

            passed = max_err < 1e-9
            if not passed:
                all_passed = False

            tag = "PASS" if passed else "FAIL"
            print(f"  N={N:2d}  K={K:2d}  seed={seed}  "
                  f"max_error={max_err:.4e}   {tag}")

            results.append({
                "N": N, "K": K, "seed": seed,
                "max_abs_error": max_err, "passed": passed
            })

    print()
    print("-" * 70)
    if all_passed:
        print("RESULTADO GLOBAL: PASS -- La conversion QUBO <-> Ising es exacta")
        print("  para todos los tamanhos verificados (N in {6,8,10,12,14}).")
        print("  Discrepancia maxima absoluta: < 1e-9 (error de redondeo flotante)")
    else:
        failed = [r for r in results if not r['passed']]
        print(f"RESULTADO GLOBAL: FAIL -- {len(failed)} instancias con error > 1e-9.")
        for f in failed:
            print(f"  N={f['N']}  K={f['K']}  seed={f['seed']}  error={f['max_abs_error']:.4e}")
    print("=" * 70)

    # Guardar CSV de resultados
    import csv
    out_path = os.path.join(project_root, "output", "results", "oe1_verification.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["N", "K", "seed", "max_abs_error", "passed"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Tabla de resultados guardada en: {out_path}")


if __name__ == "__main__":
    main()
