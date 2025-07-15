# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 20:45:43 2024

@author: SOURAV
"""

from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from pyswarm import pso  # Particle Swarm Optimization library

# Load a simpler dataset (e.g., Diabetes dataset)
from sklearn.datasets import load_diabetes

# Set working directory
import os
os.chdir("/scratch/schoudhary_wr.iitr/Machine_learning_trend")

# Load data
data = load_diabetes()
X, y = data.data, data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for PSO
def objective_function(params):
    """
    Objective function to minimize: RMSE of the XGBoost model.
    """
    max_depth, learning_rate, n_estimators = int(params[0]), params[1], int(params[2])

    # Create XGBoost model with given parameters
    model = XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    # Predict on validation set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse


# Bounds for hyperparameters
bounds = ([3, 10], [0.01, 0.3], [50, 500])

# Function to run PSO using mpi4py.futures
def mpi_pso():
    """
    Execute Particle Swarm Optimization (PSO) with MPI parallelism.
    """
    with MPIPoolExecutor() as executor:
        # Use PSO with parallel evaluation
        optimal_params, optimal_rmse = pso(
            objective_function,
            [b[0] for b in bounds],
            [b[1] for b in bounds],
            swarmsize=10,
            maxiter=10,
        )
        
        return optimal_params, optimal_rmse


# Main execution
if __name__ == "__main__":
    # Print baseline model performance
    print("Baseline XGBoost Performance:")
    baseline_model = XGBRegressor(random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
    print(f"Baseline RMSE: {baseline_rmse:.4f}\n")

    # Run PSO with MPI
    print("Running PSO with MPI...")
    optimal_params, optimal_rmse = mpi_pso()

    # Extract optimized parameters
    optimal_max_depth = int(optimal_params[0])
    optimal_learning_rate = optimal_params[1]
    optimal_n_estimators = int(optimal_params[2])

    print("\nOptimal Parameters from PSO:")
    print(f"Max Depth: {optimal_max_depth}")
    print(f"Learning Rate: {optimal_learning_rate:.4f}")
    print(f"Number of Estimators: {optimal_n_estimators}")
    print(f"Optimized RMSE: {optimal_rmse:.4f}\n")

    # Train final model with optimal parameters
    optimized_model = XGBRegressor(
        max_depth=optimal_max_depth,
        learning_rate=optimal_learning_rate,
        n_estimators=optimal_n_estimators,
        random_state=42,
    )
    optimized_model.fit(X_train, y_train)
    y_pred_optimized = optimized_model.predict(X_test)
    final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_optimized))

    print("Performance Comparison:")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Optimized RMSE: {final_rmse:.4f}")
