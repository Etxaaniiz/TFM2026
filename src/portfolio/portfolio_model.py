import numpy as np
import pandas as pd
import dimod
from typing import Tuple, Union, Dict

def build_qubo(
    mu: Union[pd.Series, np.ndarray], 
    Sigma: Union[pd.DataFrame, np.ndarray], 
    K: int, 
    lambda_val: float = 0.5, 
    penalty: float = None
) -> np.ndarray:
    """
    Formulates the portfolio optimization problem as a QUBO matrix Q.
    
    Objective: min lambda * w^T * Sigma * w - (1 - lambda) * mu^T * w
    Subject to: sum(x_i) = K, and w_i = x_i / K, x_i in {0, 1}.
    
    Parameters:
    mu (pd.Series or np.ndarray): Annualized expected returns.
    Sigma (pd.DataFrame or np.ndarray): Annualized covariance matrix.
    K (int): Number of assets to select in the portfolio.
    lambda_val (float): Risk aversion parameter (default: 0.5).
    penalty (float, optional): Penalty multiplier P. If None, calculated heuristically.
    
    Returns:
    np.ndarray: Symmetric QUBO matrix Q of size N x N.
    """
    # Convert to numpy arrays if pandas types are passed
    mu_arr = mu.values if isinstance(mu, pd.Series) else np.array(mu)
    Sigma_arr = Sigma.values if isinstance(Sigma, pd.DataFrame) else np.array(Sigma)
    
    N = len(mu_arr)
    if N <= 0:
        raise ValueError("The number of assets must be greater than zero.")
    if K <= 0 or K > N:
        raise ValueError(f"Cardinality K must be between 1 and N ({N}).")
        
    # 1. Build unpenalized QUBO matrix Q0 (symmetric)
    Q0 = np.zeros((N, N))
    for i in range(N):
        # Diagonal elements: x_i^2 = x_i
        Q0[i, i] = lambda_val * Sigma_arr[i, i] / (K ** 2) - (1.0 - lambda_val) * mu_arr[i] / K
        for j in range(i + 1, N):
            # Off-diagonal elements (symmetric weight distribution)
            val = lambda_val * Sigma_arr[i, j] / (K ** 2)
            Q0[i, j] = val / 2.0
            Q0[j, i] = val / 2.0
            
    # 2. Determine penalty coefficient P if not provided
    if penalty is None or penalty < 0:
        P = 10.0 * np.max(np.abs(Q0))
    else:
        P = penalty
        
    # 3. Add penalty term: P * (sum(x_i) - K)^2
    # P * (sum(x_i) - K)^2 = P * (sum(x_i) + sum_{i!=j} x_i x_j - 2*K*sum(x_i) + K^2)
    # Diagonal term coefficient: P * (1 - 2*K)
    # Off-diagonal term coefficient: P / 2 (symmetric)
    Q = Q0.copy()
    for i in range(N):
        Q[i, i] += P * (1.0 - 2.0 * K)
        for j in range(i + 1, N):
            Q[i, j] += P
            Q[j, i] += P
            
    return Q

def qubo_to_ising(Q: np.ndarray) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Converts a symmetric QUBO matrix Q into Ising parameters (h, J) and energy offset.
    
    Parameters:
    Q (np.ndarray): Symmetric QUBO matrix of size N x N.
    
    Returns:
    Tuple[dict, dict, float]:
        - h (dict): Linear coefficients for the Ising model.
        - J (dict): Quadratic coefficients for the Ising model.
        - offset (float): Energy offset.
    """
    N = Q.shape[0]
    
    # Construct dimod QUBO dictionary
    # keys are (i, j) for i <= j.
    # Note: dimod's qubo_to_ising expects the upper triangular part (or symmetric where we combine off-diagonals)
    Q_dict = {}
    for i in range(N):
        # Diagonal
        Q_dict[(i, i)] = Q[i, i]
        for j in range(i + 1, N):
            # Sum of Q[i,j] and Q[j,i] is 2 * Q[i,j] since Q is symmetric
            Q_dict[(i, j)] = Q[i, j] + Q[j, i]
            
    h, J, offset = dimod.qubo_to_ising(Q_dict)
    return h, J, offset
