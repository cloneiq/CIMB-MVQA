import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class CausalGraphNetwork(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout

        # Node type embedding
        self.type_embed = nn.Embedding(4, hidden_dim)

        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            # The input dimension of the first layer is hidden_dim, the rest are hidden_dim
            in_dim = hidden_dim if i == 0 else hidden_dim
            # The last layer merges the multi-head attention results back to hidden_dim
            out_dim = hidden_dim // heads if i < num_layers - 1 else hidden_dim
            concat = i < num_layers - 1

            # Use GATv2Conv for stronger representation ability
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=heads,
                    concat=concat,
                    dropout=dropout,
                    edge_dim=1  # Edge feature dimension, used to represent causal strength
                )
            )

        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.weak_weight = nn.Parameter(torch.tensor(0.3))  # Initialized as 0.3

    def _build_graph(self, batch_size, device):
        edges = [
            # Strong edges (main causal paths)
            (0, 1), (1, 0),  # q <-> p
            (0, 2), (2, 0),  # q <-> e
            (0, 3), (3, 0),  # q <-> qv

            # Weak edges (secondary information paths)
            (1, 2), (2, 1),  # p <-> e
            (1, 3), (3, 1),  # p <-> qv
            (2, 3), (3, 2),  # e <-> qv
        ]

        # Create edge indices for each sample's graph
        edge_index_list = []
        edge_attr_list = []

        for b in range(batch_size):
            # Calculate the offset of nodes
            offset = b * 4  # Each sample has 4 nodes

            # Add edges with offset
            batch_edges = [(src + offset, dst + offset) for src, dst in edges]
            edge_index = torch.tensor(batch_edges, device=device).t()
            edge_index_list.append(edge_index)

            # Create edge attributes (causal strength)
            # The first 6 edges are strong edges, value 1.0; the last 6 edges are weak edges, value 0.3
            edge_attr = torch.cat([
                torch.ones(6, 1, device=device),  # Strong edges
                torch.ones(6, 1, device=device) * torch.sigmoid(self.weak_weight)  # Weak edges
            ])
            edge_attr_list.append(edge_attr)

        # Merge edges of all batches
        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)

        return edge_index, edge_attr

    def forward(self, q, p, e, qv):
        batch_size = q.size(0)
        device = q.device

        # Prepare node features for each sample in the batch
        x_list = []
        for b in range(batch_size):
            # For each sample, create 4 nodes: [q, p, e, qv]
            sample_nodes = [q[b:b + 1], p[b:b + 1], e[b:b + 1], qv[b:b + 1]]
            x_list.extend(sample_nodes)

        # Merge all node features
        x = torch.cat(x_list, dim=0)  # [batch_size*4, hidden_dim]

        # Add node type embedding
        node_types = torch.arange(4, device=device).repeat(batch_size)
        type_embeddings = self.type_embed(node_types)
        x = x + type_embeddings

        # Build graph structure
        edge_index, edge_attr = self._build_graph(batch_size, device)

        # GAT message passing
        for i, gat_layer in enumerate(self.gat_layers):
            x = x + gat_layer(x, edge_index, edge_attr=edge_attr)
            if i < self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.view(batch_size, 4, self.hidden_dim)
        return self.output_layer(x)
