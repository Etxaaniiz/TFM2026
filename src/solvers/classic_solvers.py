import time
import numpy as np
import dimod
import neal
from typing import Dict, Any, Tuple
from src.utils.utils import get_memory_usage
from src.metrics.metrics import calculate_portfolio_metrics, calculate_qubo_energy

# Try importing gurobipy, but handle cases where it's not installed or licensed
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

def solve_gurobi(instance: Dict[str, Any], lambda_val: float = 0.5) -> Dict[str, Any]:
    """
    Solves the exact cardinality-constrained portfolio optimization model using Gurobi.
    
    Parameters:
    instance (dict): The problem instance containing N, K, mu, Sigma.
    lambda_val (float): Risk aversion parameter (default: 0.5).
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("Gurobi is not installed or not available in the current environment.")
        
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    
    # Measure memory and time
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # Initialize Gurobi environment and model
    # Set output flag to 0 to keep optimization output silent
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    
    model = gp.Model("Portfolio_Markowitz", env=env)
    
    # Variables: x_i in {0, 1}
    x = model.addVars(N, vtype=GRB.BINARY, name="x")
    
    # Cardinality constraint: sum(x_i) == K
    model.addConstr(gp.quicksum(x[i] for i in range(N)) == K, name="cardinality")
    
    # Objective: min lambda * w^T * Sigma * w - (1 - lambda) * mu^T * w
    # w_i = x_i / K
    # obj = lambda / K^2 * sum_{i,j} Sigma_ij * x_i * x_j - (1 - lambda) / K * sum_i mu_i * x_i
    obj = gp.QuadExpr()
    for i in range(N):
        obj += - (1.0 - lambda_val) / K * mu[i] * x[i]
        for j in range(N):
            obj += lambda_val / (K ** 2) * Sigma[i, j] * x[i] * x[j]
            
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # Retrieve solution
    if model.Status == GRB.OPTIMAL:
        x_sol = np.array([round(x[i].X) for i in range(N)])
    else:
        # Fallback to zeros if not solved
        x_sol = np.zeros(N)
        print(f"Warning: Gurobi optimization did not reach optimal status (Status: {model.Status}).")
        
    # Free Gurobi model resources
    model.close()
    env.close()
    
    # Compute metrics
    runtime = end_time - start_time
    mem_used = mem_after  # RSS memory usage after execution
    
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, instance['Q']) + instance.get('offset', 0.0)
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "gurobi",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": None,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0, # Gurobi is the reference exact solver, so GAP = 0.0
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol
    }
    
    return results

def solve_exact(instance: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solves the QUBO formulation using dimod.ExactSolver (brute force).
    Warning: Do not run for N > 20 due to exponential time complexity.
    
    Parameters:
    instance (dict): The problem instance containing N, K, mu, Sigma, Q, lambda_val.
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # Convert numpy matrix Q to upper-triangular dict for dimod
    Q_dict = {}
    for i in range(N):
        Q_dict[(i, i)] = Q[i, i]
        for j in range(i + 1, N):
            Q_dict[(i, j)] = Q[i, j] + Q[j, i]
            
    sampler = dimod.ExactSolver()
    response = sampler.sample_qubo(Q_dict)
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # Retrieve best sample
    best_sample = response.first.sample
    x_sol = np.array([best_sample[i] for i in range(N)])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "exact",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": None,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0, # Will be computed post-solve in comparison to Gurobi
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol
    }
    
    return results

def solve_sa(instance: Dict[str, Any], num_reads: int = 1000, num_sweeps: int = 1000) -> Dict[str, Any]:
    """
    Solves the QUBO formulation using Simulated Annealing (dwave-neal).
    
    Parameters:
    instance (dict): The problem instance containing N, K, mu, Sigma, Q, lambda_val.
    num_reads (int): Number of annealing runs (default: 1000).
    num_sweeps (int): Number of sweeps per run (default: 1000).
    
    Returns:
    dict: Solver results including portfolio metrics and runtime.
    """
    N = instance['N']
    K = instance['K']
    mu = instance['mu']
    Sigma = instance['Sigma']
    Q = instance['Q']
    lambda_val = instance.get('lambda_val', 0.5)
    offset = instance.get('offset', 0.0)
    
    mem_before = get_memory_usage()
    start_time = time.perf_counter()
    
    # Convert numpy matrix Q to upper-triangular dict for dimod
    Q_dict = {}
    for i in range(N):
        Q_dict[(i, i)] = Q[i, i]
        for j in range(i + 1, N):
            Q_dict[(i, j)] = Q[i, j] + Q[j, i]
            
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q_dict, num_reads=num_reads, num_sweeps=num_sweeps)
    
    end_time = time.perf_counter()
    mem_after = get_memory_usage()
    
    # Retrieve best sample
    best_sample = response.first.sample
    x_sol = np.array([best_sample[i] for i in range(N)])
    
    runtime = end_time - start_time
    mem_used = mem_after
    
    metrics = calculate_portfolio_metrics(x_sol, mu, Sigma, K, lambda_val)
    energy = calculate_qubo_energy(x_sol, Q) + offset
    
    tickers_list = instance.get('tickers', [])
    selected_tickers = ",".join([tickers_list[i] for i in range(N) if x_sol[i] == 1]) if tickers_list else ""
    num_selected = int(np.sum(x_sol))
    
    results = {
        "dataset": instance['dataset'],
        "solver": "simulated_annealing",
        "N": N,
        "K": K,
        "instance_id": instance['instance_id'],
        "seed": instance['seed'],
        "p": None,
        "objective": metrics['objective'],
        "energy": energy,
        "gap": 0.0, # Will be computed post-solve in comparison to Gurobi
        "sharpe": metrics['sharpe'],
        "expected_return": metrics['expected_return'],
        "volatility": metrics['volatility'],
        "feasible": metrics['feasible'],
        "runtime_seconds": runtime,
        "memory_mb": mem_used,
        "selected_tickers": selected_tickers,
        "num_selected": num_selected,
        "solution": x_sol
    }
    
    return results
