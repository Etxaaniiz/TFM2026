import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── project root ──────────────────────────────────────────────────────────────
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def setup_thesis_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        pass

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'figure.dpi': 300,
        'savefig.bbox': 'tight'
    })


# ============================================================================
# SECTION 2 FIGURES
# ============================================================================

def generate_barren_plateaus(output_dir):
    print("Generando barren_plateaus.png...")
    fig = plt.figure(figsize=(12, 5.5))

    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)

    Z1 = X**2 + Y**2

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X, Y, Z1, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True)
    ax1.set_title("A. Paisaje Optimizable", weight='bold', pad=15)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.set_zticklabels([])
    ax1.grid(True, linestyle='--', alpha=0.3)
    ax1.view_init(elev=25, azim=45)

    Z2 = -np.exp(-(X**2 + Y**2) / 0.15)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(X, Y, Z2, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True)
    ax2.set_title("B. Meseta Estéril / Barren Plateau", weight='bold', pad=15)
    ax2.set_zlim(-1.1, 0.1)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_zticklabels([])
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.view_init(elev=25, azim=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "barren_plateaus.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado barren_plateaus.png")


def generate_xy_vs_qaoa_feasibility(output_dir):
    print("Generando xy_vs_qaoa_feasibility.png...")
    fig = plt.figure(figsize=(12, 5.5))

    vertices = {
        (0, 0, 0): "000",
        (1, 0, 0): "100",
        (0, 1, 0): "010",
        (0, 0, 1): "001",
        (1, 1, 0): "110",
        (1, 0, 1): "101",
        (0, 1, 1): "011",
        (1, 1, 1): "111"
    }

    valid_vertices = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    edges = [
        ((0,0,0), (1,0,0)), ((0,0,0), (0,1,0)), ((0,0,0), (0,0,1)),
        ((1,0,0), (1,1,0)), ((1,0,0), (1,0,1)),
        ((0,1,0), (1,1,0)), ((0,1,0), (0,1,1)),
        ((0,0,1), (1,0,1)), ((0,0,1), (0,1,1)),
        ((1,1,1), (1,1,0)), ((1,1,1), (1,0,1)), ((1,1,1), (0,1,1))
    ]

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_title("A. QAOA Estándar: Espacio Completo $2^N$", weight='bold', pad=15)
    for edge in edges:
        p1, p2 = edge
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#CBD5E1', linestyle='--', linewidth=1.2)

    for vertex, label in vertices.items():
        if vertex in valid_vertices:
            ax1.scatter(vertex[0], vertex[1], vertex[2], color='#10B981', s=150, depthshade=False, edgecolor='black', zorder=10)
            ax1.text(vertex[0] + 0.05, vertex[1] + 0.05, vertex[2] + 0.05, f"$|{label}\\rangle$", color='#047857', weight='bold', fontsize=9)
        else:
            ax1.scatter(vertex[0], vertex[1], vertex[2], color='#EF4444', s=90, depthshade=False, alpha=0.6, zorder=5)
            ax1.text(vertex[0] + 0.05, vertex[1] + 0.05, vertex[2] + 0.05, f"$|{label}\\rangle$", color='#B91C1C', alpha=0.7, fontsize=8)

    ax1.set_axis_off()
    ax1.view_init(elev=20, azim=30)
    ax1.set_xlim(-0.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_zlim(-0.2, 1.2)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.set_title(r"B. XY-QAOA: Subespacio de Dicke $\binom{N}{K}$", weight='bold', pad=15)
    poly = Poly3DCollection([valid_vertices], alpha=0.25, facecolor='#10B981', edgecolor='#059669', linewidth=1.5)
    ax2.add_collection3d(poly)
    for vertex in valid_vertices:
        label = vertices[vertex]
        ax2.scatter(vertex[0], vertex[1], vertex[2], color='#10B981', s=150, depthshade=False, edgecolor='black', zorder=10)
        ax2.text(vertex[0] + 0.05, vertex[1] + 0.05, vertex[2] + 0.05, f"$|{label}\\rangle$", color='#047857', weight='bold', fontsize=9)
    for edge in edges:
        p1, p2 = edge
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='#CBD5E1', linestyle=':', linewidth=0.8, alpha=0.4)

    ax2.set_axis_off()
    ax2.view_init(elev=20, azim=30)
    ax2.set_xlim(-0.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_zlim(-0.2, 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "xy_vs_qaoa_feasibility.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado xy_vs_qaoa_feasibility.png")


# ============================================================================
# SECTION 3 FIGURES
# ============================================================================

def apply_cost_layer(state, E_diag, gamma):
    return state * np.exp(-1j * gamma * E_diag)


def apply_rx_mixer_layer(state, beta, N):
    state = state.reshape([2] * N)
    cos_b = np.cos(beta)
    sin_b = np.sin(beta)
    for j in range(N):
        state = cos_b * state - 1j * sin_b * np.roll(state, shift=1, axis=j)
    return state.flatten()


def run_emulator_qaoa_rx_p1(N, Q, gamma, beta):
    state = np.ones(2**N, dtype=complex) / np.sqrt(2**N)
    E_diag = np.zeros(2**N)
    for i in range(2**N):
        x = np.array([int(b) for b in bin(i)[2:].zfill(N)])
        E_diag[i] = x.T @ Q @ x
    state = apply_cost_layer(state, E_diag, gamma)
    state = apply_rx_mixer_layer(state, beta, N)
    probs = np.abs(state) ** 2
    expected_energy = np.sum(E_diag * probs)
    return expected_energy


def generate_plot_3_1_flowchart(output_dir):
    print("Generando diagrama_flujo_mapeo.png...")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')

    box_style_1 = dict(boxstyle="round,pad=0.8", facecolor="#E0F2FE", edgecolor="#0284C7", lw=2)
    box_style_2 = dict(boxstyle="round,pad=0.8", facecolor="#DCFCE7", edgecolor="#15803D", lw=2)
    box_style_3 = dict(boxstyle="round,pad=0.8", facecolor="#F3E8FF", edgecolor="#7E22CE", lw=2)

    text1 = r"Datos de Entrada\n\nMatriz de covarianza $\Sigma$\nVector de retornos esperados $\mu$"
    text2 = r"Formulación Matemática QUBO\n\nFunción objetivo con restricciones\nincorporadas (Penalización $A$)"
    text3 = r"Hamiltoniano de Ising\n\nRed de espines formada por cúbits\nAcoplamientos $J_{ij}$ y Campos $h_i$"

    ax.text(2.0, 5.0, text1, ha='center', va='center', bbox=box_style_1, fontsize=11, weight='bold', color='#1E293B')
    ax.text(6.0, 5.0, text2, ha='center', va='center', bbox=box_style_2, fontsize=11, weight='bold', color='#1E293B')
    ax.text(10.0, 5.0, text3, ha='center', va='center', bbox=box_style_3, fontsize=11, weight='bold', color='#1E293B')

    ax.add_patch(FancyArrowPatch((3.6, 5.0), (4.4, 5.0), arrowstyle='-|>', mutation_scale=20, color='#64748B', linewidth=3.0))
    ax.add_patch(FancyArrowPatch((7.6, 5.0), (8.4, 5.0), arrowstyle='-|>', mutation_scale=20, color='#64748B', linewidth=3.0))

    ax.set_xlim(0, 12)
    ax.set_ylim(3, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diagrama_flujo_mapeo.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado diagrama_flujo_mapeo.png")


def generate_plot_3_2_venn(output_dir):
    print("Generando espacio_dicke_combinatorio.png...")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')

    large_circle = plt.Circle((4.0, 4.0), 3.2, facecolor='#FEE2E2', edgecolor='#EF4444', alpha=0.3, linestyle='--', linewidth=2.0)
    ax.add_patch(large_circle)
    ax.text(4.0, 6.7, "Espacio Completo de Hilbert ($2^N$)\nEstados no factibles explorados por QAOA Estándar",
            ha='center', va='center', fontsize=11, color='#991B1B', weight='bold')

    small_circle = plt.Circle((4.0, 4.0), 0.9, facecolor='#10B981', edgecolor='#047857', alpha=0.75, linewidth=2.0)
    ax.add_patch(small_circle)

    np.random.seed(1337)
    for _ in range(160):
        r = np.random.uniform(0.95, 3.1)
        theta = np.random.uniform(0, 2*np.pi)
        x = 4.0 + r * np.cos(theta)
        y = 4.0 + r * np.sin(theta)
        ax.scatter(x, y, color='#94A3B8', s=15, alpha=0.6, edgecolors='none')

    for _ in range(18):
        r = np.random.uniform(0, 0.8)
        theta = np.random.uniform(0, 2*np.pi)
        x = 4.0 + r * np.cos(theta)
        y = 4.0 + r * np.sin(theta)
        ax.scatter(x, y, color='white', s=18, alpha=0.9, edgecolors='none', zorder=5)

    ax.annotate("Subespacio de Dicke $\\binom{N}{K}$\nConfiguraciones válidas (XY-QAOA)",
                xy=(4.2, 4.2), xytext=(5.8, 1.8),
                arrowprops=dict(facecolor='#047857', edgecolor='#047857', lw=1.5, arrowstyle="->", shrinkB=5,
                                connectionstyle="arc3,rad=-0.1"),
                fontsize=11, color='#047857', weight='bold', ha='center')

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "espacio_dicke_combinatorio.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado espacio_dicke_combinatorio.png")


def generate_plot_3_3_circuit(output_dir):
    print("Generando topologia_circuito_xy.png...")
    qc = QuantumCircuit(4)

    init_gate = Gate("Init: Estado de Dicke |D_K^N>", 4, [])
    qc.append(init_gate, [0, 1, 2, 3])
    qc.barrier()

    cost_gate = Gate("Coste: U_C(gamma)", 4, [])
    qc.append(cost_gate, [0, 1, 2, 3])

    mixer_gate = Gate("Mixer: U_M^{XY}(beta) (XX+YY)", 2, [])
    qc.append(mixer_gate, [0, 1])
    qc.append(mixer_gate, [1, 2])
    qc.append(mixer_gate, [2, 3])
    qc.barrier()

    cost_gate_p2 = Gate("Coste: U_C(gamma_2)", 4, [])
    mixer_gate_p2 = Gate("Mixer: U_M^{XY}(beta_2) (XX+YY)", 2, [])
    qc.append(cost_gate_p2, [0, 1, 2, 3])
    qc.append(mixer_gate_p2, [0, 1])
    qc.append(mixer_gate_p2, [1, 2])
    qc.append(mixer_gate_p2, [2, 3])
    qc.barrier()

    qc.measure_all()
    fig = qc.draw(output='mpl', style={'backgroundcolor': '#FFFFFF'})
    fig.savefig(os.path.join(output_dir, "topologia_circuito_xy.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  [OK] Guardado topologia_circuito_xy.png")


def generate_plot_3_4_regularization(output_dir):
    print("Generando paisaje_regularizacion_ridge.png...")

    N = 4
    Q = np.array([
        [-0.15,  0.10, -0.05,  0.22],
        [ 0.10, -0.25,  0.08, -0.12],
        [-0.05,  0.08, -0.18,  0.05],
        [ 0.22, -0.12,  0.05, -0.10]
    ])

    gamma_vals = np.linspace(0, np.pi, 30)
    beta_vals = np.linspace(0, np.pi, 30)
    Gamma, Beta = np.meshgrid(gamma_vals, beta_vals)

    Z_sin_reg = np.zeros((30, 30))
    for i, g in enumerate(gamma_vals):
        for j, b in enumerate(beta_vals):
            Z_sin_reg[j, i] = run_emulator_qaoa_rx_p1(N, Q, g, b)

    gamma_tqa, beta_tqa = 1.0, 1.0
    alpha = 5.0
    Z_con_reg = Z_sin_reg + alpha * ((Gamma - gamma_tqa)**2 + (Beta - beta_tqa)**2)

    fig = plt.figure(figsize=(13, 5.5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(Gamma, Beta, Z_sin_reg, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
    ax1.set_title("A. Paisaje de Coste Sin Regularización (Rugoso)", weight='bold', pad=15)
    ax1.set_xlabel(r'$\gamma$ (Coste)')
    ax1.set_ylabel(r'$\beta$ (Mezclador)')
    ax1.set_zlabel('Coste Esperado')
    ax1.view_init(elev=30, azim=135)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.plot_surface(Gamma, Beta, Z_con_reg, cmap='plasma', edgecolor='none', alpha=0.9, antialiased=True)
    ax2.set_title(r"B. Paisaje Con Regularización Ridge $L_2$ (Suave)", weight='bold', pad=15)
    ax2.set_xlabel(r'$\gamma$ (Coste)')
    ax2.set_ylabel(r'$\beta$ (Mezclador)')
    ax2.set_zlabel('Coste Esperado')
    ax2.view_init(elev=30, azim=135)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "paisaje_regularizacion_ridge.png"), dpi=300)
    plt.close()
    print("  [OK] Guardado paisaje_regularizacion_ridge.png")


def main():
    setup_thesis_style()
    section2_dir = os.path.join(project_root, "output", "figures")
    section3_dir = os.path.join(project_root, "output", "figures_tfm", "3.Desarrollo")
    os.makedirs(section2_dir, exist_ok=True)
    os.makedirs(section3_dir, exist_ok=True)

    generate_barren_plateaus(section2_dir)
    generate_xy_vs_qaoa_feasibility(section2_dir)
    generate_plot_3_1_flowchart(section3_dir)
    generate_plot_3_2_venn(section3_dir)
    generate_plot_3_3_circuit(section3_dir)
    generate_plot_3_4_regularization(section3_dir)

    print("\nGeneración de figuras completada con éxito.")


if __name__ == "__main__":
    main()
