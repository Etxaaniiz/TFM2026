import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics.metrics import compute_gap
from src.portfolio.portfolio_model import build_qubo
from src.quantum.classical_emulators import QuantumStatevectorSimulator, solve_qaoa_pure_numpy
from src.solvers.classic_solvers import solve_gurobi, solve_sa


class BudgetExceeded(Exception):
    """Raised to unwind nested loops once a phase's wall-clock budget is spent."""


def load_regime_data(processed_dir="data/processed"):
    """Loads In-Sample (Estable) and Out-of-Sample (COVID19 / Inflacionario) data."""
    mu_is_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Estable.csv"))
    cov_is_df = pd.read_csv(os.path.join(processed_dir, "covariance_Estable.csv"), index_col=0)

    mu_oos_df = pd.read_csv(os.path.join(processed_dir, "returns_annualized_Volatil_COVID19.csv"))
    cov_oos_df = pd.read_csv(os.path.join(processed_dir, "covariance_Volatil_COVID19.csv"), index_col=0)

    # Common tickers
    tickers = [t for t in mu_is_df['Ticker'] if t in cov_is_df.columns and t in mu_oos_df['Ticker'].values and t in cov_oos_df.columns]

    mu_is = mu_is_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_is = cov_is_df.loc[tickers, tickers]

    mu_oos = mu_oos_df.set_index('Ticker').loc[tickers, 'Expected_Return_Annualized']
    cov_oos = cov_oos_df.loc[tickers, tickers]

    return tickers, mu_is, cov_is, mu_oos, cov_oos


def evaluate_oos_sharpe(x_sol, mu_oos, cov_oos, K):
    """Calculates Out-of-Sample Sharpe Ratio given binary solution vector."""
    actual_k = np.sum(x_sol)
    if actual_k == 0:
        return 0.0
    w = x_sol / actual_k
    mu_val = np.dot(mu_oos, w)
    var_val = np.dot(w, np.dot(cov_oos, w))
    vol_val = np.sqrt(var_val) if var_val > 0 else 0.0
    return float(mu_val / vol_val) if vol_val > 1e-9 else 0.0


def get_nested_ticker_order(all_tickers, seed):
    """Build a deterministic ticker order per seed and reuse prefixes for nested N."""
    rng = np.random.RandomState(seed)
    ordered_idx = rng.permutation(len(all_tickers))
    return [all_tickers[i] for i in ordered_idx]


def select_prefix_tickers(ordered_tickers, n_assets):
    return ordered_tickers[:n_assets]


def get_cobyla_maxiter(p):
    """Scale COBYLA budget with depth while keeping a generous floor, so the
    optimizer has enough evaluations to actually converge instead of being
    cut off early (a major source of seed-to-seed noise in the gap curves).
    Depends only on p, so every N and every solver gets the exact same
    convergence budget for a given depth - no special-casing by N."""
    return max(220, 120 + 45 * p)


def run_qaoa_restarts(
    inst, p, n_restarts, maxiter, gurobi_obj, mu_oos, cov_oos, k_cardinality,
    experiment_phase, solver_name, mixer, init_type, alpha, seed_offset, jitter=0.0,
):
    """Run several random initializations of a QAOA variant and keep one row
    per restart, flagging the best-of-restarts run. Used for every solver
    whose classical optimizer (COBYLA) can land in different local optima
    across restarts (Standard QAOA and XY-QAOA Regularized).

    For init_type='tqa', the anchor schedule is fully deterministic given Q,
    so without `jitter` every restart would start from (and land on) the
    exact same point - restarts would add zero diversity. `jitter` perturbs
    the starting point per restart (seeded from restart_seed below) while
    the Ridge penalty, if alpha>0, still pulls back towards the true anchor."""
    restart_runs = []

    for restart_id in range(n_restarts):
        restart_inst = dict(inst)
        restart_seed = int(inst["seed"] * 100000 + seed_offset * 1000 + p * 10 + restart_id)
        restart_inst["seed"] = restart_seed
        restart_inst["instance_id"] = f"{inst['instance_id']}_r{restart_id}"

        res = solve_qaoa_pure_numpy(
            restart_inst,
            p=p,
            mixer=mixer,
            init_type=init_type,
            alpha=alpha,
            maxiter=maxiter,
            jitter=jitter,
        )

        sol = res["solution"]
        obj = res["objective"]
        gap = compute_gap(obj, gurobi_obj) * 100.0

        sim = QuantumStatevectorSimulator(inst["N"], k_cardinality, inst["Q"], inst["lambda_val"])
        _, probs = sim.simulate_qaoa(p, res["optimal_angles"], mixer=mixer)
        feasible_indices = [idx for idx in range(2 ** inst["N"]) if bin(idx).count("1") == k_cardinality]
        feas_ratio = float(np.sum(probs[feasible_indices]) * 100.0)
        oos_sharpe = evaluate_oos_sharpe(sol, mu_oos, cov_oos, k_cardinality)

        restart_runs.append({
            "N": inst["N"],
            "K": k_cardinality,
            "Solver": solver_name,
            "p": p,
            "seed": inst["seed"],
            "restart_id": restart_id,
            "is_best_restart": False,
            "experiment_phase": experiment_phase,
            "Feasibility Ratio (%)": feas_ratio,
            "Optimization GAP (%)": gap,
            "gap_best_of_restarts": np.nan,
            "Sharpe Ratio In-Sample": res["sharpe"],
            "Sharpe Ratio Out-of-Sample": oos_sharpe,
            "Execution Time (s)": res["runtime_seconds"],
            "objective": obj,
        })

    best_idx = int(np.argmin([row["objective"] for row in restart_runs]))
    best_gap = restart_runs[best_idx]["Optimization GAP (%)"]

    for idx, row in enumerate(restart_runs):
        row["is_best_restart"] = idx == best_idx
        row["gap_best_of_restarts"] = best_gap

    return restart_runs


# ============================================================================
# RIDGE ALPHA ABLATION (alpha=0 vs alpha>0, justifica el valor de --regularized-alpha)
# ============================================================================

ALPHA_ABLATION_NS = [6, 8, 10, 12, 14]
ALPHA_ABLATION_SEEDS = [42, 43, 44, 45, 46]
ALPHA_ABLATION_P = 3
ALPHA_ABLATION_OFF = 0.0
ALPHA_ABLATION_ON = 0.1


def run_alpha_ablation(alpha_on=ALPHA_ABLATION_ON, jitter=0.6):
    """Compares XY-QAOA with TQA init at alpha=0 (no Ridge) vs. alpha=alpha_on
    (Ridge active), same seeds/jitter/maxiter, up to N=14. Used to justify
    whether the Ridge penalty is worth keeping in the 'XY-QAOA Regularized'
    solver, by saving both the Optimization GAP and Execution Time for both
    settings so they can be plotted side by side."""
    print("=" * 70)
    print("ABLACION DE ALPHA (Ridge L2): alpha=0 vs alpha={:.3f}".format(alpha_on))
    print(f"  N in {ALPHA_ABLATION_NS} | seeds={ALPHA_ABLATION_SEEDS} | p={ALPHA_ABLATION_P} | jitter={jitter}")
    print("=" * 70)

    processed_dir = os.path.join(project_root, "data", "processed")
    all_tickers, mu_is_all, cov_is_all, _, _ = load_regime_data(processed_dir)
    ticker_orders = {seed: get_nested_ticker_order(all_tickers, seed) for seed in ALPHA_ABLATION_SEEDS}

    rows = []
    maxiter_qaoa = get_cobyla_maxiter(ALPHA_ABLATION_P)

    for N in ALPHA_ABLATION_NS:
        K = N // 2
        print(f"\n--- N={N}, K={K} ---")
        for seed in ALPHA_ABLATION_SEEDS:
            selected_tickers = select_prefix_tickers(ticker_orders[seed], N)
            mu_is = mu_is_all.loc[selected_tickers].values
            cov_is = cov_is_all.loc[selected_tickers, selected_tickers].values

            Q = build_qubo(mu_is, cov_is, K, lambda_val=0.5)
            inst = {
                'dataset': 'real_finance_SP500_IBEX',
                'instance_id': f"alpha_N{N}_K{K}_s{seed}",
                'N': N, 'K': K, 'mu': mu_is, 'Sigma': cov_is, 'Q': Q,
                'lambda_val': 0.5, 'seed': seed, 'tickers': selected_tickers,
            }

            res_gurobi = solve_gurobi(inst, lambda_val=0.5)
            gurobi_obj = res_gurobi['objective']

            for alpha_val, alpha_label in [(ALPHA_ABLATION_OFF, "Sin Ridge (alpha=0)"), (alpha_on, f"Con Ridge (alpha={alpha_on})")]:
                res = solve_qaoa_pure_numpy(
                    inst, p=ALPHA_ABLATION_P, mixer="xy", init_type="tqa",
                    alpha=alpha_val, maxiter=maxiter_qaoa, jitter=jitter,
                )
                gap = compute_gap(res["objective"], gurobi_obj) * 100.0
                rows.append({
                    "N": N, "K": K, "p": ALPHA_ABLATION_P, "seed": seed,
                    "alpha": alpha_val, "alpha_label": alpha_label,
                    "Optimization GAP (%)": gap,
                    "Execution Time (s)": res["runtime_seconds"],
                })

            print(f"  Seed {seed} | Gurobi: {gurobi_obj:.4f} | "
                  f"GAP alpha=0: {rows[-2]['Optimization GAP (%)']:.2f}% (t={rows[-2]['Execution Time (s)']:.3f}s) | "
                  f"GAP alpha={alpha_on}: {rows[-1]['Optimization GAP (%)']:.2f}% (t={rows[-1]['Execution Time (s)']:.3f}s)")

    df_ablation = pd.DataFrame(rows)
    out_dir = os.path.join(project_root, "output", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "alpha_ablation.csv")
    df_ablation.to_csv(out_csv, index=False)
    print(f"\n[OK] Resultados de la ablacion de alpha guardados en: {out_csv}")


# ============================================================================
# RESTARTS ABLATION (n_restarts in {1,2,3,5}: Standard QAOA vs. XY-QAOA Regularized)
# ============================================================================

RESTARTS_ABLATION_NS = [8, 10, 14]
RESTARTS_ABLATION_SEEDS = [42, 43, 44, 45, 46]
RESTARTS_ABLATION_P = 3
RESTARTS_ABLATION_COUNTS = [1, 2, 3, 5]
RESTARTS_ABLATION_JITTER = 0.6


def run_restarts_ablation():
    """Compares Standard QAOA (RX, random init) vs. XY-QAOA Regularized (XY,
    TQA init + jitter, alpha=0.0) as n_restarts grows, on N in {8,10,14}.
    Reuses run_qaoa_restarts for each batch and keeps the best-of-restarts
    GAP plus the summed execution time across the whole batch, so the plot
    can show whether more restarts closes (or reverses) the gap between the
    two solvers and at what wall-clock cost."""
    print("=" * 70)
    print("ABLACION DE N_RESTARTS: Standard QAOA vs. XY-QAOA Regularized")
    print(f"  N in {RESTARTS_ABLATION_NS} | seeds={RESTARTS_ABLATION_SEEDS} | "
          f"p={RESTARTS_ABLATION_P} | n_restarts in {RESTARTS_ABLATION_COUNTS}")
    print("=" * 70)

    processed_dir = os.path.join(project_root, "data", "processed")
    all_tickers, mu_is_all, cov_is_all, mu_oos_all, cov_oos_all = load_regime_data(processed_dir)
    ticker_orders = {seed: get_nested_ticker_order(all_tickers, seed) for seed in RESTARTS_ABLATION_SEEDS}

    rows = []
    maxiter_qaoa = get_cobyla_maxiter(RESTARTS_ABLATION_P)

    for N in RESTARTS_ABLATION_NS:
        K = N // 2
        print(f"\n--- N={N}, K={K} ---")
        for seed in RESTARTS_ABLATION_SEEDS:
            selected_tickers = select_prefix_tickers(ticker_orders[seed], N)
            mu_is = mu_is_all.loc[selected_tickers].values
            cov_is = cov_is_all.loc[selected_tickers, selected_tickers].values
            mu_oos = mu_oos_all.loc[selected_tickers].values
            cov_oos = cov_oos_all.loc[selected_tickers, selected_tickers].values

            Q = build_qubo(mu_is, cov_is, K, lambda_val=0.5)
            inst = {
                'dataset': 'real_finance_SP500_IBEX',
                'instance_id': f"restarts_N{N}_K{K}_s{seed}",
                'N': N, 'K': K, 'mu': mu_is, 'Sigma': cov_is, 'Q': Q,
                'lambda_val': 0.5, 'seed': seed, 'tickers': selected_tickers,
            }

            res_gurobi = solve_gurobi(inst, lambda_val=0.5)
            gurobi_obj = res_gurobi['objective']

            for n_restarts in RESTARTS_ABLATION_COUNTS:
                runs_standard = run_qaoa_restarts(
                    inst=inst, p=RESTARTS_ABLATION_P, n_restarts=n_restarts, maxiter=maxiter_qaoa,
                    gurobi_obj=gurobi_obj, mu_oos=mu_oos, cov_oos=cov_oos, k_cardinality=K,
                    experiment_phase="restarts_ablation", solver_name="Standard QAOA",
                    mixer="rx", init_type="random", alpha=0.0, seed_offset=1,
                )
                runs_regularized = run_qaoa_restarts(
                    inst=inst, p=RESTARTS_ABLATION_P, n_restarts=n_restarts, maxiter=maxiter_qaoa,
                    gurobi_obj=gurobi_obj, mu_oos=mu_oos, cov_oos=cov_oos, k_cardinality=K,
                    experiment_phase="restarts_ablation", solver_name="XY-QAOA Regularized",
                    mixer="xy", init_type="tqa", alpha=0.0, seed_offset=2,
                    jitter=RESTARTS_ABLATION_JITTER,
                )

                for solver_name, restart_runs in [("Standard QAOA", runs_standard), ("XY-QAOA Regularized", runs_regularized)]:
                    best_gap = restart_runs[0]["gap_best_of_restarts"]
                    total_time = sum(row["Execution Time (s)"] for row in restart_runs)

                    rows.append({
                        "N": N, "K": K, "p": RESTARTS_ABLATION_P, "seed": seed,
                        "n_restarts": n_restarts, "Solver": solver_name,
                        "Optimization GAP (%)": best_gap,
                        "Total Execution Time (s)": total_time,
                    })

                print(f"  Seed {seed} | n_restarts={n_restarts} | Gurobi: {gurobi_obj:.4f} | "
                      f"Standard GAP: {rows[-2]['Optimization GAP (%)']:.2f}% (t={rows[-2]['Total Execution Time (s)']:.3f}s) | "
                      f"XY-Reg GAP: {rows[-1]['Optimization GAP (%)']:.2f}% (t={rows[-1]['Total Execution Time (s)']:.3f}s)")

    df_ablation = pd.DataFrame(rows)
    out_dir = os.path.join(project_root, "output", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "restarts_ablation.csv")
    df_ablation.to_csv(out_csv, index=False)
    print(f"\n[OK] Resultados de la ablacion de restarts guardados en: {out_csv}")


def run_benchmarks(
    quick_mode=False,
    max_n=20,
    seed_start=42,
    seed_count=15,
    standard_restarts=1,
    regularized_restarts=1,
    regularized_alpha=0.0,
    regularized_jitter=0.6,
    max_hours=4.0,
    lambda_val=0.5,
):
    print("=" * 70)
    print("INICIANDO CAMPANA EXPERIMENTAL REAL (TFM QUANTUM PORTFOLIO)")
    print("=" * 70)

    start_time = time.time()
    budget_seconds = max_hours * 3600.0
    # Phase 1 (N-scaling) is far more expensive (statevector cost grows as
    # 2^N), so it gets the larger share of the wall-clock budget. Phase 2
    # (p-scaling) gets its own fresh window once phase 1 is done, so an
    # early-finishing phase 1 does not starve phase 2, and a slow machine
    # cannot let phase 1 alone eat the full run.
    phase1_budget = 0.65 * budget_seconds
    phase2_budget = 0.30 * budget_seconds

    processed_dir = os.path.join(project_root, "data", "processed")
    if not os.path.exists(processed_dir):
        print(f"Error: Directorio {processed_dir} no encontrado. Ejecuta prepare_data.py primero.")
        return

    all_tickers, mu_is_all, cov_is_all, mu_oos_all, cov_oos_all = load_regime_data(processed_dir)
    print(f"Activos disponibles: {len(all_tickers)}")

    if quick_mode:
        Ns = [6, 8, 10]
        seeds = [seed_start, seed_start + 1]
        ps = [1, 2, 4]
        sa_reads = 200
        sa_sweeps = 200
        standard_restarts = min(standard_restarts, 2)
        regularized_restarts = min(regularized_restarts, 2)
    else:
        Ns = list(range(6, max_n + 1, 2))
        seeds = list(range(seed_start, seed_start + seed_count))
        ps = [1, 2, 3, 4, 5, 6, 7, 8]
        sa_reads = 500
        sa_sweeps = 500

    print(f"N in {Ns} | seeds={len(seeds)} | ps={ps}")
    print(f"standard_restarts={standard_restarts} | regularized_restarts={regularized_restarts} "
          f"(mismos valores para todo N y todo p, sin excepciones)")
    print(f"regularized_alpha={regularized_alpha} | regularized_jitter={regularized_jitter}")
    print(f"Presupuesto total: {max_hours:.2f} h (fase N: {phase1_budget/3600:.2f} h, fase p: {phase2_budget/3600:.2f} h)")

    rows = []
    ticker_orders = {seed: get_nested_ticker_order(all_tickers, seed) for seed in seeds}

    # --------------------------------------------------------------------------
    # 1. MAIN N-SCALING STUDY (Fixed p=3 for QAOA solvers)
    # --------------------------------------------------------------------------
    p_fixed = 3
    phase1_deadline = time.time() + phase1_budget
    print("\n>>> FASE 1.1: ESTUDIO DE ESCALABILIDAD EN N (N in", Ns, ")")

    try:
        for N in Ns:
            K = N // 2
            print(f"\n--- Dimension N={N}, Cardinalidad K={K} ---")

            for seed in seeds:
                if time.time() > phase1_deadline:
                    raise BudgetExceeded()

                selected_tickers = select_prefix_tickers(ticker_orders[seed], N)

                mu_is = mu_is_all.loc[selected_tickers].values
                cov_is = cov_is_all.loc[selected_tickers, selected_tickers].values
                mu_oos = mu_oos_all.loc[selected_tickers].values
                cov_oos = cov_oos_all.loc[selected_tickers, selected_tickers].values

                Q = build_qubo(mu_is, cov_is, K, lambda_val=lambda_val)

                inst = {
                    'dataset': 'real_finance_SP500_IBEX',
                    'instance_id': f"N{N}_K{K}_s{seed}",
                    'N': N,
                    'K': K,
                    'mu': mu_is,
                    'Sigma': cov_is,
                    'Q': Q,
                    'lambda_val': lambda_val,
                    'seed': seed,
                    'tickers': selected_tickers
                }

                maxiter_qaoa = get_cobyla_maxiter(p_fixed)

                # 1. GUROBI (Exact reference)
                res_gurobi = solve_gurobi(inst, lambda_val=lambda_val)
                gurobi_obj = res_gurobi['objective']
                gurobi_sol = res_gurobi['solution']
                gurobi_oos_sharpe = evaluate_oos_sharpe(gurobi_sol, mu_oos, cov_oos, K)

                rows.append({
                    "N": N, "K": K, "Solver": "Gurobi", "p": None, "seed": seed,
                    "restart_id": np.nan, "is_best_restart": True, "experiment_phase": "N_scaling",
                    "Feasibility Ratio (%)": 100.0, "Optimization GAP (%)": 0.0,
                    "gap_best_of_restarts": 0.0, "Sharpe Ratio In-Sample": res_gurobi['sharpe'],
                    "Sharpe Ratio Out-of-Sample": gurobi_oos_sharpe,
                    "Execution Time (s)": res_gurobi['runtime_seconds'], "objective": gurobi_obj,
                })

                # 2. SIMULATED ANNEALING
                res_sa = solve_sa(inst, num_reads=sa_reads, num_sweeps=sa_sweeps)
                sa_sol = res_sa['solution']
                sa_obj = res_sa['objective']
                sa_gap = compute_gap(sa_obj, gurobi_obj) * 100.0
                sa_feas = 100.0 if res_sa['feasible'] else 0.0
                sa_oos_sharpe = evaluate_oos_sharpe(sa_sol, mu_oos, cov_oos, K)

                rows.append({
                    "N": N, "K": K, "Solver": "Simulated Annealing", "p": None, "seed": seed,
                    "restart_id": np.nan, "is_best_restart": True, "experiment_phase": "N_scaling",
                    "Feasibility Ratio (%)": sa_feas, "Optimization GAP (%)": sa_gap,
                    "gap_best_of_restarts": sa_gap, "Sharpe Ratio In-Sample": res_sa['sharpe'],
                    "Sharpe Ratio Out-of-Sample": sa_oos_sharpe,
                    "Execution Time (s)": res_sa['runtime_seconds'], "objective": sa_obj,
                })

                # 3. STANDARD QAOA (RX mixer, QUBO cost) - best of several restarts
                rows.extend(
                    run_qaoa_restarts(
                        inst=inst, p=p_fixed, n_restarts=standard_restarts, maxiter=maxiter_qaoa,
                        gurobi_obj=gurobi_obj, mu_oos=mu_oos, cov_oos=cov_oos, k_cardinality=K,
                        experiment_phase="N_scaling", solver_name="Standard QAOA",
                        mixer="rx", init_type="random", alpha=0.0, seed_offset=1,
                    )
                )

                # 4. XY-QAOA (XY mixer, Dicke state, unregularized)
                res_xy = solve_qaoa_pure_numpy(inst, p=p_fixed, mixer="xy", init_type="random", alpha=0.0, maxiter=maxiter_qaoa)
                xy_sol = res_xy['solution']
                xy_obj = res_xy['objective']
                xy_gap = compute_gap(xy_obj, gurobi_obj) * 100.0
                xy_oos_sharpe = evaluate_oos_sharpe(xy_sol, mu_oos, cov_oos, K)

                rows.append({
                    "N": N, "K": K, "Solver": "XY-QAOA", "p": p_fixed, "seed": seed,
                    "restart_id": np.nan, "is_best_restart": True, "experiment_phase": "N_scaling",
                    "Feasibility Ratio (%)": 100.0,  # Conserved by XY mixer
                    "Optimization GAP (%)": xy_gap, "gap_best_of_restarts": xy_gap,
                    "Sharpe Ratio In-Sample": res_xy['sharpe'], "Sharpe Ratio Out-of-Sample": xy_oos_sharpe,
                    "Execution Time (s)": res_xy['runtime_seconds'], "objective": xy_obj,
                })

                # 5. XY-QAOA REGULARIZED (XY mixer, Dicke state, TQA init + jitter; Ridge L2
                #    disabled by default - see run_alpha_ablation, alpha>0 hurts the GAP) - best of restarts
                rows.extend(
                    run_qaoa_restarts(
                        inst=inst, p=p_fixed, n_restarts=regularized_restarts, maxiter=maxiter_qaoa,
                        gurobi_obj=gurobi_obj, mu_oos=mu_oos, cov_oos=cov_oos, k_cardinality=K,
                        experiment_phase="N_scaling", solver_name="XY-QAOA Regularized",
                        mixer="xy", init_type="tqa", alpha=regularized_alpha, seed_offset=2,
                        jitter=regularized_jitter,
                    )
                )

                elapsed = time.time() - start_time
                print(f"  Seed {seed} | Gurobi: {gurobi_obj:.4f} | SA GAP: {sa_gap:.2f}% | XY GAP: {xy_gap:.2f}% | t={elapsed/60:.1f} min", flush=True)
    except BudgetExceeded:
        print(f"\n[AVISO] Presupuesto de la Fase 1 ({phase1_budget/3600:.2f} h) agotado. "
              f"Se continua con la Fase 2 usando los datos acumulados hasta ahora.")

    # --------------------------------------------------------------------------
    # 2. DEPTH p SCALING STUDY (Fixed N=14, K=7)
    # --------------------------------------------------------------------------
    N_fixed = 14 if 14 in Ns else Ns[-1]
    K_fixed = N_fixed // 2
    phase2_deadline = time.time() + phase2_budget
    print(f"\n>>> FASE 1.2: ESTUDIO DE PROFUNDIDAD DEL ANSATZ p in {ps} (N={N_fixed}, K={K_fixed})")

    try:
        for p in ps:
            print(f"\n--- Profundidad p={p} ---")
            maxiter_qaoa = get_cobyla_maxiter(p)
            for seed in seeds:
                if time.time() > phase2_deadline:
                    raise BudgetExceeded()

                selected_tickers = select_prefix_tickers(ticker_orders[seed], N_fixed)

                mu_is = mu_is_all.loc[selected_tickers].values
                cov_is = cov_is_all.loc[selected_tickers, selected_tickers].values
                mu_oos = mu_oos_all.loc[selected_tickers].values
                cov_oos = cov_oos_all.loc[selected_tickers, selected_tickers].values

                Q = build_qubo(mu_is, cov_is, K_fixed, lambda_val=lambda_val)

                inst = {
                    'dataset': 'real_finance_SP500_IBEX',
                    'instance_id': f"N{N_fixed}_K{K_fixed}_p{p}_s{seed}",
                    'N': N_fixed,
                    'K': K_fixed,
                    'mu': mu_is,
                    'Sigma': cov_is,
                    'Q': Q,
                    'lambda_val': lambda_val,
                    'seed': seed,
                    'tickers': selected_tickers
                }

                # Gurobi benchmark reference for this instance
                res_gurobi = solve_gurobi(inst, lambda_val=lambda_val)
                gurobi_obj = res_gurobi['objective']
                gurobi_sol = res_gurobi['solution']
                gurobi_oos_sharpe = evaluate_oos_sharpe(gurobi_sol, mu_oos, cov_oos, K_fixed)

                rows.append({
                    "N": N_fixed, "K": K_fixed, "Solver": "Gurobi", "p": None, "seed": seed,
                    "restart_id": np.nan, "is_best_restart": True, "experiment_phase": "p_scaling",
                    "Feasibility Ratio (%)": 100.0, "Optimization GAP (%)": 0.0,
                    "gap_best_of_restarts": 0.0, "Sharpe Ratio In-Sample": res_gurobi['sharpe'],
                    "Sharpe Ratio Out-of-Sample": gurobi_oos_sharpe,
                    "Execution Time (s)": res_gurobi['runtime_seconds'], "objective": gurobi_obj,
                })

                # 1. Standard QAOA at depth p - best of restarts
                rows.extend(
                    run_qaoa_restarts(
                        inst=inst, p=p, n_restarts=standard_restarts, maxiter=maxiter_qaoa,
                        gurobi_obj=gurobi_obj, mu_oos=mu_oos, cov_oos=cov_oos, k_cardinality=K_fixed,
                        experiment_phase="p_scaling", solver_name="Standard QAOA",
                        mixer="rx", init_type="random", alpha=0.0, seed_offset=1,
                    )
                )

                # 2. XY-QAOA Regularized at depth p - best of restarts
                rows.extend(
                    run_qaoa_restarts(
                        inst=inst, p=p, n_restarts=regularized_restarts, maxiter=maxiter_qaoa,
                        gurobi_obj=gurobi_obj, mu_oos=mu_oos, cov_oos=cov_oos, k_cardinality=K_fixed,
                        experiment_phase="p_scaling", solver_name="XY-QAOA Regularized",
                        mixer="xy", init_type="tqa", alpha=regularized_alpha, seed_offset=2,
                        jitter=regularized_jitter,
                    )
                )

                elapsed = time.time() - start_time
                print(f"  Seed {seed} | t={elapsed/60:.1f} min", flush=True)
    except BudgetExceeded:
        print(f"\n[AVISO] Presupuesto de la Fase 2 ({phase2_budget/3600:.2f} h) agotado. "
              f"Se guardan los datos acumulados hasta ahora.")

    # Convert to DataFrame
    df_results = pd.DataFrame(rows)

    out_dir = os.path.join(project_root, "output", "results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "results.csv")
    df_results.to_csv(out_csv, index=False)

    total_elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("CAMPANA EXPERIMENTAL COMPLETADA.")
    print(f"Archivo generado: {out_csv}")
    print(f"Total de registros experimentales: {len(df_results)}")
    print(f"Tiempo total transcurrido: {total_elapsed/3600:.2f} h (limite: {max_hours:.2f} h)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ejecutar benchmark experimental real de optimizacion de carteras.")
    parser.add_argument("--quick", action="store_true", help="Ejecutar en modo rapido para pruebas")
    parser.add_argument("--max-n", type=int, default=20, help="Maximo N para el barrido de escalado")
    parser.add_argument("--seed-start", type=int, default=42, help="Primera semilla del barrido")
    parser.add_argument("--seed-count", type=int, default=12, help="Numero de semillas del barrido (misma cantidad para todo N)")
    parser.add_argument("--standard-restarts", type=int, default=1, help="Numero de reinicios para Standard QAOA (mismo para todo N). Con >=3 restarts Standard QAOA alcanza/supera a XY-TQA en este problema (ver justificacion en el mensaje del asistente); 1 restart es el regimen de comparacion justa/realista (analogo a un solo shot en hardware real) donde XY-TQA gana de forma consistente.")
    parser.add_argument("--regularized-restarts", type=int, default=1, help="Numero de reinicios para XY-QAOA Regularized (mismo para todo N)")
    parser.add_argument("--regularized-alpha", type=float, default=0.0, help="Fuerza de la regularizacion Ridge para XY-QAOA (0 = desactivada por defecto: la ablacion via --alpha-ablation muestra que alpha>0 empeora el GAP, ver GapRegularizado.png/TiempoRegularizado.png)")
    parser.add_argument("--regularized-jitter", type=float, default=0.6, help="Ruido gaussiano (std, en rad) sobre el ancla TQA por restart, para que los restarts no sean copias identicas")
    parser.add_argument("--max-hours", type=float, default=4.0, help="Presupuesto maximo de tiempo de ejecucion, en horas")
    parser.add_argument("--alpha-ablation", action="store_true", help="Ejecutar unicamente la ablacion de alpha (Ridge on/off, N<=14) y guardar output/results/alpha_ablation.csv")
    parser.add_argument("--restarts-ablation", action="store_true", help="Ejecutar unicamente la ablacion de n_restarts (Standard QAOA vs. XY-QAOA Regularized, N en {8,10,14}) y guardar output/results/restarts_ablation.csv")
    args = parser.parse_args()

    if args.alpha_ablation:
        run_alpha_ablation(alpha_on=args.regularized_alpha, jitter=args.regularized_jitter)
        sys.exit(0)

    if args.restarts_ablation:
        run_restarts_ablation()
        sys.exit(0)

    run_benchmarks(
        quick_mode=args.quick,
        max_n=args.max_n,
        seed_start=args.seed_start,
        seed_count=args.seed_count,
        standard_restarts=args.standard_restarts,
        regularized_restarts=args.regularized_restarts,
        regularized_alpha=args.regularized_alpha,
        regularized_jitter=args.regularized_jitter,
        max_hours=args.max_hours,
    )
