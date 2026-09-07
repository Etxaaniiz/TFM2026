import os
import sys

# Automatically resolve and set working directory to project root
utils_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(utils_dir, "..", ".."))
os.chdir(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import time
import yaml
import psutil
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Union

def get_memory_usage() -> float:
    """
    Returns the current memory RSS usage of the process in MB.
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)

def load_config(config_path: str = "configs/default_config.yaml") -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def save_instance(instance: Dict[str, Any], filepath: str) -> None:
    """
    Saves an experimental instance dict using pickle.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(instance, f)

def load_instance(filepath: str) -> Dict[str, Any]:
    """
    Loads an experimental instance dict using pickle.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Instance file not found: {filepath}")
        
    with open(filepath, "rb") as f:
        instance = pickle.load(f)
    return instance

def append_results_to_csv(result_dict: Dict[str, Any], csv_path: str = "results/results.csv") -> None:
    """
    Appends a solver result dictionary to a consolidated CSV file.
    Converts solution vector to a comma-separated string representation.
    """
    res_copy = result_dict.copy()
    if 'solution' in res_copy:
        if isinstance(res_copy['solution'], np.ndarray):
            res_copy['solution'] = ",".join(map(str, res_copy['solution'].tolist()))
        elif isinstance(res_copy['solution'], list):
            res_copy['solution'] = ",".join(map(str, res_copy['solution']))
            
    # Ensure parents directories exist
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    df = pd.DataFrame([res_copy])
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', header=True, index=False)

def create_directory_structure() -> None:
    """
    Ensures all folders required by the repository structure exist.
    """
    dirs = [
        "data/raw",
        "data/processed",
        "data/instances",
        "configs",
        "results",
        "figures",
        "tables",
        "tests",
        "docs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
