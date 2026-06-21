from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network (ST-GNN) for Traffic Forecasting.
    
    Architecture:
    1. Temporal Block: GRU models temporal history (1 hour window -> 4 steps of 15 min).
    2. Graph Block: 2-layer Graph Convolution Network (GCN) models spatial congestion propagation.
    """
    def __init__(self, num_nodes: int, in_features: int, gru_hidden: int = 64, gcn_hidden: int = 64, dropout: float = 0.2):
        super(STGNN, self).__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.gru_hidden = gru_hidden
        self.gcn_hidden = gcn_hidden
        self.dropout = dropout
        
        # 1. Temporal GRU block
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True
        )
        
        # 2. Spatial Graph block
        self.gcn1 = GCNConv(gru_hidden, gcn_hidden)
        self.gcn2 = GCNConv(gcn_hidden, 1)
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, num_nodes, in_features)
            edge_index: Graph edge index representation of shape (2, num_edges)
            
        Returns:
            predicted_congestion: Tensor of shape (batch_size, num_nodes) containing predicted congestion scores.
        """
        batch_size, time_steps, num_nodes, in_features = x.shape
        
        # 1. Temporal block: Reshape to pass each node sequence independently through the GRU
        # (batch_size, time_steps, num_nodes, in_features) -> (batch_size, num_nodes, time_steps, in_features)
        x_transposed = x.transpose(1, 2)
        # Reshape to (batch_size * num_nodes, time_steps, in_features)
        x_reshaped = x_transposed.reshape(batch_size * num_nodes, time_steps, in_features)
        
        # Pass to GRU
        gru_out, _ = self.gru(x_reshaped) # Out shape: (batch_size * num_nodes, time_steps, gru_hidden)
        
        # Take the output of the last timestep (t)
        h_temporal = gru_out[:, -1, :] # Shape: (batch_size * num_nodes, gru_hidden)
        
        # Reshape back to (batch_size, num_nodes, gru_hidden)
        h_graph = h_temporal.view(batch_size, num_nodes, self.gru_hidden)
        
        # 2. Spatial GCN block: Loop over batch to process static adjacency
        gcn_out = []
        for b in range(batch_size):
            h = h_graph[b] # Shape: (num_nodes, gru_hidden)
            h = self.gcn1(h, edge_index)
            h = torch.relu(h)
            h = self.drop(h)
            h = self.gcn2(h, edge_index) # Shape: (num_nodes, 1)
            gcn_out.append(h.squeeze(-1)) # Shape: (num_nodes,)
            
        # Stack to (batch_size, num_nodes)
        predicted_congestion = torch.stack(gcn_out, dim=0)
        return predicted_congestion
