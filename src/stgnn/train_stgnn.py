from __future__ import annotations

import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from src.stgnn.model import STGNN

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SEQUENCES_PATH = DATA_DIR / "temporal_sequences.npy"
ADJACENCY_PATH = DATA_DIR / "adjacency_matrix.npy"
MODEL_SAVE_PATH = MODELS_DIR / "stgnn_model.pt"

def train_model():
    logging.info("Starting ST-GNN model training...")
    
    # 1. Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)
    
    # 2. Load data
    if not SEQUENCES_PATH.exists() or not ADJACENCY_PATH.exists():
        logging.error("Required dataset assets (sequences or adjacency) not found.")
        return
        
    data = np.load(SEQUENCES_PATH, allow_pickle=True).item()
    X = data["X"]
    Y = data["Y"]
    
    adj_matrix = np.load(ADJACENCY_PATH)
    
    num_samples, time_steps, num_roads, in_features = X.shape
    logging.info("Loaded dataset: %d samples, %d roads, %d steps, %d features", num_samples, num_roads, time_steps, in_features)
    
    # 3. Prepare Graph edge index
    edge_index_np = np.argwhere(adj_matrix == 1.0).T
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    logging.info("Graph edge index built: shape %s (connected roads)", edge_index.shape)
    
    # 4. Train-Validation Split (80% Train, 20% Val chronologically)
    split_idx = int(num_samples * 0.8)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    Y_train, Y_val = Y[:split_idx], Y[split_idx:]
    
    # Convert to PyTorch datasets
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 5. Initialize Model
    model = STGNN(num_nodes=num_roads, in_features=in_features, gru_hidden=64, gcn_hidden=64, dropout=0.2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Early stopping config
    epochs = 100
    patience = 10
    best_val_loss = float("inf")
    patience_counter = 0
    
    MODELS_DIR.mkdir(exist_ok=True)
    
    # 6. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            out = model(batch_x, edge_index)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation epoch
        model.eval()
        val_loss = 0.0
        mae = 0.0
        rmse = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out = model(batch_x, edge_index)
                
                loss = criterion(out, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                
                # Metrics
                mae += torch.mean(torch.abs(out - batch_y)).item() * batch_x.size(0)
                rmse += torch.mean((out - batch_y) ** 2).item() * batch_x.size(0)
                
        val_loss /= len(val_dataset)
        mae /= len(val_dataset)
        rmse = np.sqrt(rmse / len(val_dataset))
        
        logging.info("Epoch %03d/%d | Train Loss: %.5f | Val Loss: %.5f | MAE: %.4f | RMSE: %.4f", 
                     epoch, epochs, train_loss, val_loss, mae, rmse)
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "mae": mae,
                "rmse": rmse,
                "num_nodes": num_roads,
                "in_features": in_features
            }, MODEL_SAVE_PATH)
            logging.info("--> Saved new best model checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping triggered at epoch %d. Best Val Loss: %.5f", epoch, best_val_loss)
                break
                
    logging.info("Training completed. Best model saved at %s", MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_model()
