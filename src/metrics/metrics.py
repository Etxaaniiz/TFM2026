import numpy as np
import pandas as pd
from typing import Dict, Any, Union

def calculate_portfolio_metrics(
    x: np.ndarray, 
    mu: np.ndarray, 
    Sigma: np.ndarray, 
    K: int, 
    lambda_val: float
) -> Dict[str, float]:
    """
    Computes portfolio financial metrics for a binary asset selection vector x.
    
    Parameters:
    x (np.ndarray): Binary selection vector of shape (N,).
    mu (np.ndarray): Expected returns vector of shape (N,).
    Sigma (np.ndarray): Covariance matrix of shape (N, N).
    K (int): Cardinality constraint.
    lambda_val (float): Risk aversion parameter.
    
    Returns:
    dict: Portfolio metrics: objective, expected_return, volatility, sharpe, feasible.
    """
    # 1. Check feasibility: sum(x) == K
    actual_k = np.sum(x)
    feasible = bool(np.isclose(actual_k, K))
    
    # Weights are equiponderated for selected assets
    # Even if actual_k != K (infeasible), we divide by actual_k to avoid division by zero
    # but the strict constraint evaluates w_i = x_i / K.
    # Let's compute weights using the target K for the mathematical objective value.
    w = x / K
    
    # Financial metrics
    expected_return = np.dot(mu, w)
    variance = np.dot(w, np.dot(Sigma, w))
    volatility = np.sqrt(variance) if variance > 0 else 0.0
    
    # Sharpe ratio (risk-free rate = 0)
    sharpe = expected_return / volatility if volatility > 1e-9 else 0.0
    
    # Objective value: lambda * w^T * Sigma * w - (1 - lambda) * mu^T * w
    objective = lambda_val * variance - (1.0 - lambda_val) * expected_return
    
    return {
        "objective": float(objective),
        "expected_return": float(expected_return),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "feasible": feasible
    }

def calculate_qubo_energy(x: np.ndarray, Q: np.ndarray) -> float:
    """
    Calculates the energy of a binary configuration x under the QUBO matrix Q.
    E(x) = x^T * Q * x
    
    Parameters:
    x (np.ndarray): Binary selection vector of shape (N,).
    Q (np.ndarray): Symmetric QUBO matrix of shape (N, N).
    
    Returns:
    float: The QUBO energy.
    """
    return float(np.dot(x, np.dot(Q, x)))

def compute_gap(objective: float, gurobi_obj: Union[float, None]) -> float:
    """
    Computes the optimization GAP relative to the Gurobi exact solution.
    
    Parameters:
    objective (float): Objective value of the solver solution.
    gurobi_obj (float or None): Exact optimal objective value from Gurobi.
    
    Returns:
    float: Relative GAP (%). Returns 0.0 if Gurobi solution is not available or if both are zero.
    """
    if gurobi_obj is None:
        return 0.0
    
    # Relative difference: (Obj - Obj_gurobi) / (|Obj_gurobi| + 1e-9)
    # Since we are minimizing: Obj >= Obj_gurobi
    # If the solver finds a better solution than Gurobi (due to numeric precision), gap is 0.
    if objective <= gurobi_obj:
        return 0.0
        
    denom = abs(gurobi_obj)
    if denom < 1e-9:
        denom = 1e-9
        
    gap = (objective - gurobi_obj) / denom
    return float(gap)
