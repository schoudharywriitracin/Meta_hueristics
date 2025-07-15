# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 22:20:05 2024

@author: SOURAV
"""

from mpi4py import MPI
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import csv
import HHO as hho
import functions
import time

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Rank of the current process
size = comm.Get_size()  # Total number of processes

# General parameters
PopulationSize = 500
Iterations = 50
FuncAgain = 1
Export = True
ExportToFile = "hho_xgb-" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".csv"

# Load datasets (only on rank 0)
if rank == 0:
    df_wnd = pd.read_csv('F:/Trend_analysis/Nasa_Power/WND.csv', parse_dates=['Date'])
    df_sol = pd.read_csv('F:/Trend_analysis/Nasa_Power/SOL.csv', parse_dates=['Date'])
    df_hmd = pd.read_csv('F:/Trend_analysis/Nasa_Power/HMD.csv', parse_dates=['Date'])
    df_temp = pd.read_csv('F:/Trend_analysis/Nasa_Power/TMP.csv', parse_dates=['Date'])
    df_pcp = pd.read_csv('F:/Trend_analysis/Result/Final_MERRA_13_datasets_stn_num.csv', parse_dates=['Date'])

    df_pcp_long = df_pcp.melt(id_vars=['Date'], var_name='Station', value_name='PCP')
    df_pcp_long_filtered = df_pcp_long[
        (df_pcp_long['Date'] >= '1984-01-01') & (df_pcp_long['Date'] <= '2021-12-31')
    ].reset_index(drop=True)
    df_pcp_long_filtered['Station'] = df_pcp_long_filtered['Station'].astype(str)

    combined_df = pd.merge(df_wnd[['Date', 'Station', 'WND']], df_sol[['Date', 'Station', 'SLR']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_hmd[['Date', 'Station', 'HMD']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_temp[['Date', 'Station', 'TMPmax', 'TMPmin']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_pcp_long_filtered, on=['Date'], how='outer')  
    combined_df.rename(columns={'Station_x': 'Station'}, inplace=True)
    combined_df.drop(columns = ['Station_y'])
    
    station_mapping = {
        1: "Ambajogai",
        2: "Bodhegaon",
        3: "Ranjni",
        4: "Bhavarwadi",
        5: "Bhagur",
        6: "Georai",
        7: "Bhandardara",
        8: "Kotul",
        9: "Maljune",
        10: "Dawarwadi",
        11: "Nasik",
        12: "Gangpur Dam",
        13: "Deogaon rangari"
    }

   # if combined_df['Station'].dtype != 'object':
        #combined_df['Station'] = combined_df['Station'].map(station_mapping)

    #station_dfs = {station_mapping[int(station)]: group for station, group in combined_df.groupby('Station')}
    station_dfs = {station: group for station, group in combined_df.groupby('Station')}


# Broadcast station data to all nodes
station_dfs = comm.bcast(station_dfs, root=0)

# Distribute each station to a specific node
stations = list(station_dfs.items())
assigned_station = stations[rank] if rank < len(stations) else None

# Define the list of functions
func = [functions.F1, functions.F2, functions.F3, functions.F4, functions.F5, functions.F6, functions.F7, functions.F8, functions.F9, functions.F10,
        functions.F11, functions.F12, functions.F13, functions.F14, functions.F15, functions.F16, functions.F17, functions.F18, functions.F19, functions.F20,
        functions.F21, functions.F22, functions.F23]

# Define objective function for HHO
def objective_function(params):
    n_estimators, learning_rate, max_depth, subsample, colsample_bytree = params
    n_estimators = int(n_estimators)
    learning_rate = np.float64(learning_rate)
    subsample = np.float64(subsample)
    colsample_bytree = np.float64(colsample_bytree)

    # Clamp hyperparameters to reasonable ranges
    learning_rate = np.clip(learning_rate, 0.01, 5)
    subsample = np.clip(subsample, 0, 1)
    colsample_bytree = np.clip(colsample_bytree, 0, 1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure no invalid values in training and validation data
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(y_train)):
        print("Training data contains invalid values. Replacing with 0.0.")
    if not np.all(np.isfinite(X_val)) or not np.all(np.isfinite(y_val)):
        print("Validation data contains invalid values. Replacing with 0.0.")

    # Train XGBoost model
    try:
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict and validate predictions
        y_pred = model.predict(X_val)
        if not np.all(np.isfinite(y_pred)):
            print("Predictions contain invalid values (NaN or infinity). Replacing with 0.0.")
            y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # Evaluate MSE
        mse = mean_squared_error(y_val, y_pred)
        return mse
    except ValueError as e:
        print(f"Error during model training or prediction: {e}")
        return float('inf')  # Return a high value to indicate failure

# Process the assigned station if available
if assigned_station:
    stn_id, station_df = assigned_station
    print(f"Processing station: {stn_id} on rank {rank}")

    X = station_df[['WND', 'SLR', 'HMD', 'TMPmin', 'TMPmax']].values
    y = station_df['PCP'].values

    # Ensure no non-finite values in the data
    if not np.all(np.isfinite(X)):
        print(f"Non-finite values found in features (X) for station {stn_id}. Replacing with 0.0.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and infinities with 0.0

    if not np.all(np.isfinite(y)):
        print(f"Non-finite values found in target (y) for station {stn_id}. Replacing with 0.0.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaN and infinities with 0.0

    for j in range(len(func)):  # Loop through all functions in the func list
        if func[j]:  # If the function is marked as True
            for k in range(FuncAgain):  # Repeat for the specified number of times

                func_details = functions.getFunctionDetails(j)  # Get function details

                lb = [50, 0.01, 3, 0.6, 0.6]  # Lower bounds for hyperparameters
                ub = [500, 5, 10, 1.0, 1.0]   # Upper bounds for valid parameters

                dim = len(lb)  # Dimensionality of the search space

                # Run HHO optimization
                best_params = hho.HHO(
                    objective_function,  # Objective function
                    lb,                  # Lower bounds
                    ub,                  # Upper bounds
                    dim,                 # Dimensions
                    PopulationSize,      # Number of hawks
                    Iterations           # Number of iterations
                )

                # Extract optimized parameters
                opt_n_estimators = int(best_params.optimizer[0])
                opt_learning_rate = np.float64(best_params.optimizer[1])
                opt_max_depth = int(best_params.optimizer[2])
                opt_subsample = np.float64(best_params.optimizer[3])
                opt_colsample_bytree = np.float64(best_params.optimizer[4])
                opt_colsample_bytree = np.clip(opt_colsample_bytree, 0, 1)

                opt_n_estimators = int(best_params.optimizer[0])
                opt_learning_rate = np.clip(np.float64(best_params.optimizer[1]), 0.01, 5)
                opt_max_depth = max(1, int(best_params.optimizer[2]))  # Ensure max_depth is at least 1
                opt_subsample = np.clip(np.float64(best_params.optimizer[3]), 0, 1)
                opt_colsample_bytree = np.clip(np.float64(best_params.optimizer[4]), 0, 1)

                # Train final model with optimized hyperparameters
                final_model = xgb.XGBRegressor(
                    n_estimators=opt_n_estimators,
                    learning_rate=opt_learning_rate,                               
                    max_depth=opt_max_depth,
                    subsample=opt_subsample,
                    colsample_bytree=opt_colsample_bytree,
                    random_state=42
                )
                final_model.fit(X, y)

                # Save results
                if Export:
                    with open(ExportToFile, 'a', newline='\n') as out:
                        writer = csv.writer(out, delimiter=',')
                        if out.tell() == 0:
                            header = ["Station", "n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "Function"]
                            writer.writerow(header)

                        writer.writerow([
                            stn_id,
                            opt_n_estimators,
                            opt_learning_rate,
                            opt_max_depth,
                            opt_subsample,
                            opt_colsample_bytree,
                            func_details[0]  # Add function name to the results
                        ])

# Finalize MPI
comm.Barrier()
if rank == 0:
    print("Processing complete!")
