import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=50, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)

# Distributed Training Function
def train_ddp(rank, world_size, station_dfs, output_file):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)

    if rank == 0:
        print(f"Starting Distributed Training with {world_size} processes")

    results = []

    for stn_name, station_df in station_dfs.items():
        if rank == 0:
            print(f"\nProcessing Station: {stn_name}")
            print('-' * 50)

        X = station_df[['WND', 'SLR', 'HMD', 'TMPmin', 'TMPmax']].values
        y = station_df['PCP'].values
        X = X.reshape(X.shape[0], 1, X.shape[1])  # Reshape for LSTM

        # Convert to Tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Use KFold Cross Validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mse_list, rmse_list, rrmse_list, mae_list, mape_list = [], [], [], [], []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X_tensor[train_index], X_tensor[test_index]
            y_train, y_test = y_tensor[train_index], y_tensor[test_index]

            # Create Dataset and Distributed Sampler
            train_dataset = TensorDataset(X_train, y_train)
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

            # Initialize Model and Optimizer
            model = LSTMModel().to(device)
            model = DDP(model, device_ids=[rank])
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.MSELoss()

            # Train the Model
            model.train()
            for epoch in range(20):
                train_sampler.set_epoch(epoch)
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(X_batch).squeeze()
                    loss = loss_fn(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

            # Evaluate the Model
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test.to(device)).squeeze().cpu().numpy()
                y_true = y_test.numpy()

            # Calculate Metrics
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            rrmse = rmse / np.mean(y_true)
            mae = mean_absolute_error(y_true, y_pred)
            mape = mean_absolute_percentage_error(y_true, y_pred)

            mse_list.append(mse)
            rmse_list.append(rmse)
            rrmse_list.append(rrmse)
            mae_list.append(mae)
            mape_list.append(mape)

        # Average Metrics
        mean_mse = np.mean(mse_list)
        mean_rmse = np.mean(rmse_list)
        mean_rrmse = np.mean(rrmse_list)
        mean_mae = np.mean(mae_list)
        mean_mape = np.mean(mape_list)

        if rank == 0:
            results.append({
                'Station': stn_name,
                'MSE': mean_mse,
                'RMSE': mean_rmse,
                'RRMSE': mean_rrmse,
                'MAE': mean_mae,
                'MAPE': mean_mape
            })
            print(f"Results for Station: {stn_name}")
            print(f"RMSE: {mean_rmse:.4f}, RRMSE: {mean_rrmse:.4f}, MAE: {mean_mae:.4f}, MSE: {mean_mse:.4f}, MAPE: {mean_mape:.4f}")
            print('-' * 50)

    if rank == 0:
        pd.DataFrame(results).to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    dist.destroy_process_group()

# Main Function
def main():
    world_size = torch.cuda.device_count()  # Number of GPUs
    output_file = "LSTM_MPI_Results.csv"

    # Load Data (replace with your file paths)
    df_wnd = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/WND.csv')
    df_sol = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/SOL.csv')
    df_hmd = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/HMD.csv')
    df_temp = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/TMP.csv')
    df_pcp = pd.read_csv('/scratch/schoudhary_wr.iitr/Machine_learning_trend/Final_MERRA_13_datasets_stn_num.csv')
    
    # Preprocess and combine data
    df_pcp_long = df_pcp.melt(id_vars=['Date'], var_name='Station', value_name='PCP')
    df_pcp_long_filtered = df_pcp_long[(df_pcp_long['Date'] >= '1984-01-01') & (df_pcp_long['Date'] <= '2021-12-31')].reset_index(drop=True)
    df_pcp_long_filtered['Station'] = df_pcp_long_filtered['Station'].astype(str)

    combined_df = pd.merge(df_wnd[['Date', 'Station', 'WND']], df_sol[['Date', 'Station', 'SLR']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_hmd[['Date', 'Station', 'HMD']], on=['Date', 'Station'], how='outer')
    combined_df = pd.merge(combined_df, df_temp[['Date', 'Station', 'TMPmax', 'TMPmin']], on=['Date', 'Station'], how='outer')
    combined_df['Station'] = combined_df['Station'].astype(str)
    combined_df = pd.merge(combined_df, df_pcp_long_filtered, on=['Date'], how='outer')

    # Mapping station numbers to names
    station_mapping = {
        1: "Ambajogai", 2: "Bodhegaon", 3: "Ranjni", 4: "Bhavarwadi", 5: "Bhagur",
        6: "Georai", 7: "Bhandardara", 8: "Kotul", 9: "Maljune", 10: "Dawarwadi",
        11: "Nasik", 12: "Gangpur Dam", 13: "Deogaon rangari"
    }

    combined_df.rename(columns={'Station_x': 'Station'}, inplace=True)
    combined_df.drop(columns=['Station_y'], inplace=True)
    if combined_df['Station'].dtype != 'object':
        combined_df['Station'] = combined_df['Station'].map(station_mapping)

    # Filter data for a specific station
    station_dfs = {station: group for station, group in combined_df.groupby('Station')}

    mp.spawn(train_ddp, args=(world_size, station_dfs, output_file), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
