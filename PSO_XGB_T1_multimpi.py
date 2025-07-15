from mpi4py.futures import MPICommExecutor
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import math
import pandas as pd
import multiprocessing
import time

# Start timing the process
st_time = time.time()

# Constants
POPULATION = 20
MAX_ITER = 100
CONVERGENCE = 1e-6
PERSONAL_C = 1.5
SOCIAL_C = 1.5
V_MAX = 0.1
HYPERPARAM_BOUNDS = {
    'n_estimators': (10, 500),
    'learning_rate': (0.01, 0.3),
    'max_depth': (3, 10),
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0)
}

class Particle:
    def __init__(self):
        self.pos = {key: np.random.uniform(*HYPERPARAM_BOUNDS[key]) for key in HYPERPARAM_BOUNDS.keys()}
        self.velocity = {key: np.random.uniform(-V_MAX, V_MAX) for key in HYPERPARAM_BOUNDS.keys()}
        self.best_pos = self.pos.copy()
        self.best_cost = float('inf')

class Swarm:
    def __init__(self, size):
        self.particles = [Particle() for _ in range(size)]
        self.best_pos = None
        self.best_cost = float('inf')

# Cost function
def cost_function(params, X_train, y_train, X_val, y_val):
    params = {k: int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in params.items()}
    model = xgb.XGBRegressor(
        n_estimators=params['n_estimators'],
        learning_rate=params['learning_rate'],
        max_depth=params['max_depth'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds)

# Function to evaluate and update a single particle
def evaluate_particle(particle, global_best_pos, X_train, y_train, X_val, y_val, inertia_weight):
    cost = cost_function(particle.pos, X_train, y_train, X_val, y_val)

    # Update personal best
    if cost < particle.best_cost:
        particle.best_cost = cost
        particle.best_pos = particle.pos.copy()

    # Update velocity and position
    for key in particle.pos.keys():
        r1, r2 = np.random.rand(), np.random.rand()
        cognitive = PERSONAL_C * r1 * (particle.best_pos[key] - particle.pos[key])
        social = SOCIAL_C * r2 * (global_best_pos[key] - particle.pos[key])
        particle.velocity[key] = inertia_weight * particle.velocity[key] + cognitive + social

        # Limit velocity
        particle.velocity[key] = np.clip(particle.velocity[key], -V_MAX, V_MAX)

        # Update position
        particle.pos[key] += particle.velocity[key]
        particle.pos[key] = np.clip(particle.pos[key], *HYPERPARAM_BOUNDS[key])

    return particle

# Parallel Particle Swarm Optimization
def particle_swarm_optimization_parallel(X_train, y_train, X_val, y_val):
    swarm = Swarm(POPULATION)
    inertia_weight = 0.5 + (np.random.rand() / 2)
    curr_iter = 0

    while curr_iter < MAX_ITER:
        global_best = None

        with MPICommExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            if executor is not None:  # Ensure only active workers participate
                futures = [
                    executor.submit(
                        evaluate_particle,
                        particle,
                        swarm.best_pos or {key: np.random.uniform(*HYPERPARAM_BOUNDS[key]) for key in HYPERPARAM_BOUNDS.keys()},
                        X_train,
                        y_train,
                        X_val,
                        y_val,
                        inertia_weight
                    )
                    for particle in swarm.particles
                ]

                # Collect updated particles
                updated_particles = [future.result() for future in futures]

        # Update swarm with updated particles
        swarm.particles = updated_particles

        # Update global best
        for particle in swarm.particles:
            if particle.best_cost < swarm.best_cost:
                swarm.best_cost = particle.best_cost
                swarm.best_pos = particle.best_pos.copy()

        # Check for convergence
        if swarm.best_cost < CONVERGENCE:
            print(f"Converged after {curr_iter} iterations.")
            break

        curr_iter += 1

    print(f"Best parameters: {swarm.best_pos}")
    print(f"Best MSE: {swarm.best_cost}")

    # Evaluate additional metrics
    best_params = {k: int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in swarm.best_pos.items()}
    best_model = xgb.XGBRegressor(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    final_preds = best_model.predict(X_val)

    mse = mean_squared_error(y_val, final_preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_val, final_preds)
    rrmse = rmse / np.mean(y_val)
    print('-' * 100)
    print(f"Final RMSE: {rmse}")
    print(f"Final RRMSE: {rrmse}")
    print(f"Final MAE: {mae}")

if __name__ == "__main__":
    # Example data preparation
    # Replace this with your actual station data loading
    df_wnd = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/WND.csv', parse_dates=['Date'])
    df_sol = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/SOL.csv', parse_dates=['Date'])
    df_hmd = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/HMD.csv', parse_dates=['Date'])
    df_temp = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/TMP.csv', parse_dates=['Date'])
    df_pcp = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/Final_MERRA_13_datasets_stn_num.csv', parse_dates=['Date'])

    df_pcp_long = df_pcp.melt(id_vars=['Date'], var_name='Station', value_name='PCP')
    df_pcp_long_filtered = df_pcp_long[(df_pcp_long['Date'] >= '1984-01-01') & (df_pcp_long['Date'] <= '2021-12-31')].reset_index(drop=True)
    df_pcp_long_filtered['Station'] = df_pcp_long_filtered['Station'].astype(str)

   # Merge datasets
    combined_df = pd.merge(df_wnd[['Date', 'Station', 'WND']], df_sol[['Date', 'Station', 'SLR']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_hmd[['Date', 'Station', 'HMD']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_temp[['Date', 'Station', 'TMPmax', 'TMPmin']], on=['Date', 'Station'], how='outer')
    combined_df['Station'] = combined_df['Station'].astype(str)
    combined_df = pd.merge(combined_df, df_pcp_long_filtered, on=['Date', 'Station'], how='outer')

   # Map station numbers to names
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

    if combined_df['Station'].dtype != 'object':
        combined_df['Station'] = combined_df['Station'].map(station_mapping)

    station_dfs = {station_mapping[int(station)]: group for station, group in combined_df.groupby('Station')}

    for stn_id, station_df in station_dfs.items():
        print(f"Processing station ID: {stn_id}")

        X = station_df[['WND', 'SLR', 'HMD', 'TMPmin', 'TMPmax']].values
        y = station_df['PCP'].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        particle_swarm_optimization_parallel(X_train, y_train, X_val, y_val)
# End timing the process
en_time = time.time()
print('Time taken', en_time - st_time)
