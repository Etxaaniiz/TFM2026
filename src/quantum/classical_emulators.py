import numpy as np
import time
from scipy.optimize import minimize
from typing import Dict, Any, Tuple, Union, List
from src.metrics.metrics import calculate_portfolio_metrics, calculate_qubo_energy

class QuantumStatevectorSimulator:
    """
    A pure NumPy implementation of a statevector simulator for QAOA (RX mixer) 
    and XY-QAOA (XY mixer) without depending on external quantum libraries.
    
    Optimized for fast classical simulation up to ~16 qubits.
    """
    def __init__(self, N: int, K: int, Q: np.ndarray, lambda_val: float = 0.5):
        """
        Parameters:
        N (int): Number of qubits (assets).
        K (int): Cardinality constraint.
        Q (np.ndarray): Symmetric QUBO matrix of size N x N.
        lambda_val (float): Risk aversion parameter.
        """
        self.N = N
        self.K = K
        self.Q = Q
        self.lambda_val = lambda_val
        self.num_states = 2 ** N
        
        # Precompute diagonal of Hamiltonian energies (E_diag) to avoid O(2^N) loops during optimization
        self.E_diag = np.zeros(self.num_states)
        for i in range(self.num_states):
            x = np.array([int(b) for b in bin(i)[2:].zfill(N)])
            self.E_diag[i] = x.T @ self.Q @ x
            
        # Cache for Dicke state |D_K^N> to avoid recomputation
        self._dicke_state = None

    def get_dicke_state(self) -> np.ndarray:
        """
        Prepares a Dicke State |D_K^N> of Hamming weight K.
        This is the initial state for XY-QAOA.
        """
        if self._dicke_state is not None:
            return self._dicke_state.copy()
            
        state = np.zeros(self.num_states, dtype=complex)
        indices = [idx for idx in range(self.num_states) if bin(idx).count('1') == self.K]
        
        if len(indices) == 0:
            raise ValueError(f"No states found with Hamming weight {self.K} for N={self.N}")
            
        val = 1.0 / np.sqrt(len(indices))
        for idx in indices:
            state[idx] = val
            
        self._dicke_state = state
        return state.copy()

    def get_superposition_state(self) -> np.ndarray:
        """
        Prepares a uniform superposition state.
        This is the initial state for standard RX-QAOA.
        """
        return np.ones(self.num_states, dtype=complex) / np.sqrt(self.num_states)

    def apply_cost_layer(self, state: np.ndarray, gamma: float) -> np.ndarray:
        """
        Applies the cost operator: U_C(gamma) = e^{-i * gamma * H_C}
        """
        return state * np.exp(-1j * gamma * self.E_diag)

    def apply_rx_mixer_layer(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Applies the RX mixer: U_M(beta) = e^{-i * beta * X_j} to all qubits.
        Uses array reshaping and rolls to simulate single-qubit X operations.
        """
        state = state.reshape([2] * self.N)
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)
        
        for j in range(self.N):
            # Roll along axis j, applying the rotation
            state = cos_b * state - 1j * sin_b * np.roll(state, shift=1, axis=j)
            
        return state.flatten()

    def apply_xxyy_gate(self, state: np.ndarray, q1: int, q2: int, beta: float) -> np.ndarray:
        """
        Applies the two-qubit XXYY gate with parameter 4*beta to qubits q1 and q2.
        XXYY(theta) = exp(-i * theta/4 * (XX + YY))
        """
        state = state.reshape([2] * self.N)
        # Swap target axes to the front (axes 0 and 1) for easy indexing
        state = np.swapaxes(state, q1, 0)
        state = np.swapaxes(state, q2, 1)
        
        c = np.cos(2.0 * beta)
        s = np.sin(2.0 * beta)
        
        # XXYY matrix couples only |01> and |10>
        s01 = state[0, 1].copy()
        s10 = state[1, 0].copy()
        
        state[0, 1] = c * s01 - 1j * s * s10
        state[1, 0] = c * s10 - 1j * s * s01
        
        # Swap back to original order
        state = np.swapaxes(state, q2, 1)
        state = np.swapaxes(state, q1, 0)
        return state.flatten()

    def apply_xy_mixer_layer(self, state: np.ndarray, beta: float) -> np.ndarray:
        """
        Applies the ring-topology XY mixer layer.
        """
        # Couples even-odd pairs: (0,1), (2,3), ..., (2*i, 2*i + 1)
        for i in range(self.N // 2):
            state = self.apply_xxyy_gate(state, 2 * i, 2 * i + 1, beta)
            
        # Couples odd-even pairs: (1,2), (3,4), ..., (2*i + 1, 2*i + 2)
        for i in range((self.N - 2 + self.N % 2) // 2):
            state = self.apply_xxyy_gate(state, 2 * i + 1, 2 * i + 2, beta)
            
        # Couples the boundaries to close the ring: (N - 1, 0)
        state = self.apply_xxyy_gate(state, self.N - 1, 0, beta)
        return state

    def simulate_qaoa(self, p: int, params: np.ndarray, mixer: str = "xy") -> Tuple[float, np.ndarray]:
        """
        Simulates the full QAOA circuit and calculates the expected energy value.
        
        Parameters:
        p (int): Number of layers (depth).
        params (np.ndarray): 1D array of angles of shape (2*p,), containing gamma (first p) and beta (last p).
        mixer (str): 'rx' for Standard QAOA, 'xy' for XY-QAOA.
        
        Returns:
        Tuple[float, np.ndarray]: Expected energy value and final statevector probabilities.
        """
        # 1. Initialize Statevector
        if mixer == "xy":
            state = self.get_dicke_state()
        elif mixer == "rx":
            state = self.get_superposition_state()
        else:
            raise ValueError(f"Unknown mixer type: {mixer}")
            
        gamma = params[:p]
        beta = params[p:]
        
        # 2. Alternating layers
        for i in range(p):
            state = self.apply_cost_layer(state, gamma[i])
            if mixer == "xy":
                state = self.apply_xy_mixer_layer(state, beta[i])
            elif mixer == "rx":
                state = self.apply_rx_mixer_layer(state, beta[i])
                
        # 3. Calculate measurements probabilities
        probs = np.abs(state) ** 2
        expected_energy = np.sum(self.E_diag * probs)
        return float(expected_energy), probs

def solve_qaoa_pure_numpy(
    instance: Dict[str, Any],
    p: int = 3,
    mixer: str = "xy",
    init_type: str = "random",
    alpha: float = 0.0,
    maxiter: int = 100,
    jitter: float = 0.0
) -> Dict[str, Any]:
    """
    A standalone solver function that optimizes QAOA RX, XY-QAOA, and Regularized XY-QAOA
    using the pure NumPy Statevector simulator.

    Parameters:
    instance (dict): The portfolio problem instance containing N, K, mu, Sigma, Q.
    p (int): Number of layers.
    mixer (str): 'rx' (Standard QAOA) or 'xy' (XY-QAOA / Regularized).
    init_type (str): 'random' or 'tqa' (Trotterized Quantum Annealing schedules).
    alpha (float): Ridge L2 regularization multiplier (used with 'tqa' initialization).
    maxiter (int): Maximum classic optimization evaluations.
    jitter (float): Std-dev of Gaussian noise added to the TQA anchor before starting
        COBYLA (only used with init_type='tqa'). The anchor search itself is fully
        deterministic given Q, so without this, every restart of a 'tqa'-initialized
        solver starts from (and converges to) the exact same point - restarts add no
        diversity at all. jitter, seeded from instance['seed'], gives each restart a
        distinct starting point near the anchor while the Ridge penalty (if alpha>0)
        still pulls optimization back towards the unperturbed anchor itself.

    Returns:
    dict: Detailed solver results compatible with classic_solvers output format.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    # Initialize simulator
    sim = QuantumStatevectorSimulator(N, K, Q, lambda_val)
    
    # Measure time
    start_time = time.perf_counter()
    
    # 1. Parameter Initialization (TQA Anchor or Random)
    tqa_anchor = None
    if init_type == "tqa":
        # Linear annealing schedule parameters: gamma increases, beta decreases
        dt_vals = np.linspace(0.1, 1.0, 10)
        best_e = float('inf')
        best_sched = None
        for dt in dt_vals:
            t = (np.arange(1, p + 1) - 0.5) / p
            gamma_t = t * dt
            beta_t = (1.0 - t) * dt
            sched = np.concatenate((gamma_t, beta_t))
            e, _ = sim.simulate_qaoa(p, sched, mixer=mixer)
            if e < best_e:
                best_e = e
                best_sched = sched
        tqa_anchor = best_sched
        if jitter > 0.0:
            rng = np.random.RandomState(instance.get('seed', 42))
            init_point = tqa_anchor + rng.normal(0.0, jitter, size=tqa_anchor.shape)
        else:
            init_point = tqa_anchor.copy()
    else:
        # Random initial parameters in [0, pi/2]
        np.random.seed(instance.get('seed', 42))
        init_point = np.random.rand(2 * p) * np.pi / 2.0

    # 2. Define Classical Cost Wrapper (with optional L2 Ridge Regularization)
    history = []
    def classical_objective(params):
        expected_energy, _ = sim.simulate_qaoa(p, params, mixer=mixer)
        
        # Apply L2 Ridge penalty if alpha > 0 and we have a TQA anchor
        if alpha > 0.0 and tqa_anchor is not None:
            penalty = alpha * np.sum((params - tqa_anchor) ** 2)
            total_cost = expected_energy + penalty
        else:
            total_cost = expected_energy
            
        history.append(expected_energy)
        return total_cost

    # 3. Classical Optimization Loop (COBYLA)
    res = minimize(
        classical_objective,
        init_point,
        method='COBYLA',
        options={'maxiter': maxiter}
    )
    
    # 4. Final measurement with optimal angles
    opt_energy, probs = sim.simulate_qaoa(p, res.x, mixer=mixer)
    
    # Decode best state (state index with highest probability)
    best_idx = np.argmax(probs)
    x_sol = np.array([int(b) for b in bin(best_idx)[2:].zfill(N)])
    
    end_time = time.perf_counter()
    runtime = end_time - start_time
    
    # 5. Compute metrics
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": f"qaoa_{mixer}" + ("_regularized" if alpha > 0.0 else ""),
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": p,
        "alpha": alpha,
        "objective": metrics['objective'],
        "energy": opt_energy + offset,
        "gap": 0.0,  # Will be calculated comparison-wise later
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol,
        "optimal_angles": res.x,
        "history": history
    }
    
    return results
