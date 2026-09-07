import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def setup_plot_style():
    # Apply seaborn-v0_8-whitegrid style
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        # Fallback if the style is not available in older matplotlib versions
        sns.set_style('whitegrid')
        
    # Custom aesthetic adjustments for premium look
    sns.set_theme(style="whitegrid", rc={
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans', 'Liberation Sans'],
        'grid.color': '#E2E8F0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'axes.edgecolor': '#94A3B8',
        'axes.linewidth': 0.8,
        'xtick.color': '#64748B',
        'ytick.color': '#64748B'
    })
    
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })


# "XY-QAOA Regularized" is the Solver value stored in results.csv/results16.csv,
# kept as-is so filters against existing data keep working. Ridge (alpha) is
# disabled by default (see run_alpha_ablation / GapRegularizado.png), so the
# display label used on every plot below says what the solver actually does:
# TQA-anchored initialization + jitter, not Ridge regularization.
DISPLAY_NAME = {
    "XY-QAOA Regularized": "XY-QAOA (TQA)",
}


def filter_analysis_frame(df, phase_name):
    """Keep only the requested phase and the selected best restart rows."""
    filtered = df.copy()

    if 'experiment_phase' in filtered.columns:
        filtered = filtered[filtered['experiment_phase'] == phase_name]

    if 'is_best_restart' in filtered.columns:
        filtered = filtered[filtered['is_best_restart'].fillna(True)]

    return filtered

def main():
    setup_plot_style()
    
    # Path setup
    csv_path = os.path.join(project_root, "output", "results", "results.csv")
    output_dir = os.path.join(project_root, "output", "figures_tfm", "6.Resultados")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} does not exist. Please run scripts/run_benchmarks.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    # Define a consistent color palette
    colors = {
        "Gurobi": "#1E293B",                 # Deep slate
        "Simulated Annealing": "#64748B",    # Slate gray
        "Standard QAOA": "#EA580C",          # Orange/red
        "XY-QAOA": "#0284C7",                # Sky blue
        "XY-QAOA Regularized": "#10B981"     # Emerald green
    }
    
    # ==========================================
    # 1. OPTIMIZATION GAP (%) VS N
    # ==========================================
    print("Generating gap_vs_N.png...")
    plt.figure(figsize=(7.5, 4.5))
    
    # Filter for the N-scaling phase and collapse Standard QAOA to its best restart.
    df_n = filter_analysis_frame(df, 'N_scaling')
    
    # Group by Solver and N to get the mean gap
    df_gap = df_n.groupby(['Solver', 'N'])['Optimization GAP (%)'].mean().reset_index()
    
    # Filter for specified solvers
    target_solvers_gap = ["Simulated Annealing", "Standard QAOA", "XY-QAOA Regularized"]
    df_gap_filtered = df_gap[df_gap['Solver'].isin(target_solvers_gap)]
    
    for solver in target_solvers_gap:
        solver_data = df_gap_filtered[df_gap_filtered['Solver'] == solver]
        plt.plot(
            solver_data['N'], 
            solver_data['Optimization GAP (%)'], 
            marker='o', 
            label=DISPLAY_NAME.get(solver, solver), 
            color=colors.get(solver), 
            linewidth=1.75
        )
        
    plt.title("Gap de Optimización Medio vs. Número de Activos ($N$)", weight='bold', pad=12)
    plt.xlabel("Número de Activos ($N$)")
    plt.ylabel("Optimization GAP (%)")
    plt.xticks(sorted(df_gap_filtered['N'].unique()))
    plt.legend(title="Solver", frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gap_vs_N.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 2. SHARPE RATIO OUT-OF-SAMPLE VS N
    # ==========================================
    print("Generating sharpe_vs_N.png...")
    plt.figure(figsize=(7.5, 4.5))

    # This figure blends two runs: N=6..16 from results16.csv, N=18..20 from
    # results.csv, since results16.csv only covers up to N=16.
    csv16_path = os.path.join(project_root, "output", "results", "results16.csv")
    if os.path.exists(csv16_path):
        df16_n = filter_analysis_frame(pd.read_csv(csv16_path), 'N_scaling')
        df_sharpe_source = pd.concat([
            df16_n[df16_n['N'] <= 16],
            df_n[df_n['N'] > 16],
        ], ignore_index=True)
    else:
        print(f"  [AVISO] {csv16_path} no encontrado, usando solo results.csv")
        df_sharpe_source = df_n

    # Group by Solver and N to get mean Sharpe ratios
    df_sharpe = df_sharpe_source.groupby(['Solver', 'N'])[['Sharpe Ratio In-Sample', 'Sharpe Ratio Out-of-Sample']].mean().reset_index()
    
    # Get Gurobi data
    gurobi_data = df_sharpe[df_sharpe['Solver'] == "Gurobi"]
    # Get XY-QAOA Regularized data
    reg_data = df_sharpe[df_sharpe['Solver'] == "XY-QAOA Regularized"]
    
    # Plot Gurobi In-Sample (black dotted line)
    plt.plot(
        gurobi_data['N'], 
        gurobi_data['Sharpe Ratio In-Sample'], 
        color='black', 
        linestyle=':', 
        marker='o', 
        label="Gurobi In-Sample"
    )
    
    # Plot Gurobi Out-of-Sample (black solid line)
    plt.plot(
        gurobi_data['N'], 
        gurobi_data['Sharpe Ratio Out-of-Sample'], 
        color='black', 
        linestyle='-', 
        marker='x', 
        label="Gurobi Out-of-Sample"
    )
    
    # Plot XY-QAOA Regularized Out-of-Sample (thick solid green line)
    plt.plot(
        reg_data['N'], 
        reg_data['Sharpe Ratio Out-of-Sample'], 
        color='#10B981', 
        linestyle='-', 
        linewidth=2.5, 
        marker='s', 
        label="XY-QAOA TQA (Out-of-Sample)"
    )
    
    plt.title("Rendimiento Financiero (Ratio de Sharpe) vs. Número de Activos ($N$)", weight='bold', pad=12)
    plt.xlabel("Número de Activos ($N$)")
    plt.ylabel("Ratio de Sharpe (Anualizado)")
    plt.xticks(sorted(gurobi_data['N'].unique()))
    plt.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sharpe_vs_N.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 3. RUNTIME VS N (LOG SCALE)
    # ==========================================
    print("Generating tiempo_vs_N.png...")
    plt.figure(figsize=(7.5, 4.5))
    
    df_time = df_n.groupby(['Solver', 'N'])['Execution Time (s)'].mean().reset_index()
    
    # We include Gurobi, Simulated Annealing, Standard QAOA and XY-QAOA Regularized
    target_solvers_time = ["Gurobi", "Simulated Annealing", "Standard QAOA", "XY-QAOA Regularized"]
    df_time_filtered = df_time[df_time['Solver'].isin(target_solvers_time)]
    
    for solver in target_solvers_time:
        solver_data = df_time_filtered[df_time_filtered['Solver'] == solver]
        plt.plot(
            solver_data['N'], 
            solver_data['Execution Time (s)'], 
            marker='o', 
            label=DISPLAY_NAME.get(solver, solver), 
            color=colors.get(solver), 
            linewidth=1.75
        )
        
    plt.yscale('log')
    plt.title("Complejidad Temporal: Tiempo de Ejecución vs. Número de Activos ($N$)", weight='bold', pad=12)
    plt.xlabel("Número de Activos ($N$)")
    plt.ylabel("Tiempo de Ejecución (segundos) - Escala Log")
    plt.xticks(sorted(df_time_filtered['N'].unique()))
    plt.legend(title="Solver", frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tiempo_vs_N.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 4. FEASIBILITY RATIO VS N
    # ==========================================
    print("Generating factibilidad_vs_N.png...")
    plt.figure(figsize=(7.5, 4.5))
    
    df_feas = df_n.groupby(['Solver', 'N'])['Feasibility Ratio (%)'].mean().reset_index()
    
    # Filter for Standard QAOA vs XY-QAOA (or XY-QAOA Regularized)
    # We plot Standard QAOA, XY-QAOA, and XY-QAOA Regularized to demonstrate the mixer difference
    target_solvers_feas = ["Standard QAOA", "XY-QAOA"]
    df_feas_filtered = df_feas[df_feas['Solver'].isin(target_solvers_feas)]
    
    for solver in target_solvers_feas:
        solver_data = df_feas_filtered[df_feas_filtered['Solver'] == solver]
        plt.plot(
            solver_data['N'], 
            solver_data['Feasibility Ratio (%)'], 
            marker='o', 
            label=DISPLAY_NAME.get(solver, solver), 
            color=colors.get(solver), 
            linewidth=2.0
        )
        
    plt.title("Tasa de Viabilidad Empírica de las Soluciones vs. Número de Activos ($N$)", weight='bold', pad=12)
    plt.xlabel("Número de Activos ($N$)")
    plt.ylabel("Feasibility Ratio (%)")
    plt.ylim(0, 110)
    plt.xticks(sorted(df_feas_filtered['N'].unique()))
    plt.legend(title="Algoritmo", frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "factibilidad_vs_N.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 5. GAP VS DEPTH (p) FOR N=14
    # ==========================================
    print("Generating gap_vs_p.png...")
    plt.figure(figsize=(7.5, 4.5))
    
    # Filter for the p-scaling phase and collapse Standard QAOA to its best restart.
    df_p = filter_analysis_frame(df, 'p_scaling')
    df_p = df_p[(df_p['N'] == 14) & df_p['p'].notna()]
    
    df_gap_p = df_p.groupby(['Solver', 'p'])['Optimization GAP (%)'].mean().reset_index()
    
    target_solvers_p = ["Standard QAOA", "XY-QAOA Regularized"]
    df_gap_p_filtered = df_gap_p[df_gap_p['Solver'].isin(target_solvers_p)]
    
    for solver in target_solvers_p:
        solver_data = df_gap_p_filtered[df_gap_p_filtered['Solver'] == solver]
        plt.plot(
            solver_data['p'], 
            solver_data['Optimization GAP (%)'], 
            marker='s', 
            label=DISPLAY_NAME.get(solver, solver), 
            color=colors.get(solver), 
            linewidth=1.75
        )
        
    plt.title("Impacto de la Profundidad del Ansatz ($p$) en el GAP ($N=14$)", weight='bold', pad=12)
    plt.xlabel("Profundidad del Circuito ($p$)")
    plt.ylabel("Optimization GAP (%)")
    plt.xticks(sorted(df_gap_p_filtered['p'].unique()))
    plt.legend(title="Ansatz / Algoritmo", frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gap_vs_p.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 5b. RUNTIME VS DEPTH (p) FOR N=14
    # ==========================================
    print("Generating tiempo_vs_p.png...")
    plt.figure(figsize=(7.5, 4.5))
    
    df_time_p = df_p.groupby(['Solver', 'p'])['Execution Time (s)'].mean().reset_index()
    
    target_solvers_p = ["Standard QAOA", "XY-QAOA Regularized"]
    df_time_p_filtered = df_time_p[df_time_p['Solver'].isin(target_solvers_p)]
    
    for solver in target_solvers_p:
        solver_data = df_time_p_filtered[df_time_p_filtered['Solver'] == solver]
        plt.plot(
            solver_data['p'], 
            solver_data['Execution Time (s)'], 
            marker='o', 
            label=DISPLAY_NAME.get(solver, solver), 
            color=colors.get(solver), 
            linewidth=1.75
        )
        
    plt.title("Evaluación del Coste Temporal del Bucle Variacional vs. Profundidad ($p$)", weight='bold', pad=12)
    plt.xlabel("Profundidad del Circuito ($p$)")
    plt.ylabel("Tiempo de Ejecución Neto (segundos)")
    plt.xticks(sorted(df_time_p_filtered['p'].unique()))
    plt.legend(title="Ansatz / Algoritmo", frameon=True, facecolor='white', edgecolor='#E2E8F0')
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tiempo_vs_p.png"), dpi=300)
    plt.close()
    
    # ==========================================
    # 6. RADAR CHART (SPIDER CHART)
    # ==========================================
    print("Generating radar_chart_performance.png...")
    
    # Aggregated metrics by solver over the N-scaling dataset
    df_agg = df_n.groupby('Solver').mean(numeric_only=True).reset_index()
    
    # Select solvers
    target_solvers_radar = ["Gurobi", "Simulated Annealing", "Standard QAOA", "XY-QAOA Regularized"]
    df_agg = df_agg[df_agg['Solver'].isin(target_solvers_radar)]
    
    # Prepare metrics:
    # 1. Viabilidad = Feasibility Ratio
    # 2. Precision = 100 - GAP
    # 3. Robustez = Sharpe Ratio Out-of-Sample
    # 4. Eficiencia Temporal = Inverse Log Execution Time (higher is faster/more efficient)
    
    df_agg['Viabilidad'] = df_agg['Feasibility Ratio (%)']
    df_agg['Precisión'] = 100.0 - df_agg['Optimization GAP (%)']
    df_agg['Robustez'] = df_agg['Sharpe Ratio Out-of-Sample']
    df_agg['Eficiencia Temporal'] = -np.log10(df_agg['Execution Time (s)'])
    
    metrics = ['Viabilidad', 'Precisión', 'Robustez', 'Eficiencia Temporal']
    
    # Manual normalization of the metrics to a common [0.1, 1.0] scale to prevent polygon collapsing at 0
    # and keep it visually intuitive (Gurobi will have high marks on precision, efficiency; XY-QAOA Reg on robustness, feasibility)
    normalized_data = {}
    for metric in metrics:
        min_val = df_agg[metric].min()
        max_val = df_agg[metric].max()
        # Scale between 0.1 and 1.0 to avoid complete zero points in the radar chart
        if max_val - min_val > 1e-8:
            df_agg[metric + '_norm'] = 0.1 + 0.9 * (df_agg[metric] - min_val) / (max_val - min_val)
        else:
            df_agg[metric + '_norm'] = 1.0
            
    # Set up radar chart parameters
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    # The radar chart requires the loop to be closed, so append the start value to the end
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    
    for solver in target_solvers_radar:
        solver_row = df_agg[df_agg['Solver'] == solver]
        if solver_row.empty:
            continue
            
        values = [solver_row[m + '_norm'].values[0] for m in metrics]
        values += values[:1] # Close the polygon
        
        ax.plot(angles, values, color=colors.get(solver), linewidth=2.0, label=DISPLAY_NAME.get(solver, solver))
        ax.fill(angles, values, color=colors.get(solver), alpha=0.15)
        
    # Set the labels for each category
    ax.set_theta_offset(np.pi / 2) # Put the first category at the top
    ax.set_theta_direction(-1)     # Draw clockwise
    
    plt.xticks(angles[:-1], metrics, color='#475569', size=10, weight='bold')
    
    # Adjust y-labels (radial ticks)
    ax.set_rscale('linear')
    ax.set_ylim(0, 1.1)
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="#94A3B8", size=8)
    
    plt.title("Comparativa de Criterios de Rendimiento (Gráfico de Radar)", weight='bold', pad=20)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True, facecolor='white', edgecolor='#E2E8F0')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "radar_chart_performance.png"), dpi=300)
    plt.close()
    
    print("\nAll 7 plots successfully generated and saved to output/figures_tfm/6.Resultados/")

if __name__ == "__main__":
    main()
