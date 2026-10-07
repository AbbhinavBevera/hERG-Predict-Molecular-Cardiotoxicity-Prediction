"""
Graph Neural Network (GNN) implementation for Molecular Graph representation and hERG inhibition prediction.
Uses pure PyTorch message-passing graph convolutions (GCN / GraphConv architecture),
ensuring 100% reproducibility across environments without external C++ wheel dependencies.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

# Atomic feature configurations
ALLOWED_ATOMS = [6, 7, 8, 16, 9, 17, 35, 53, 15]  # C, N, O, S, F, Cl, Br, I, P
ALLOWED_DEGREES = [0, 1, 2, 3, 4, 5]
ALLOWED_CHARGES = [-2, -1, 0, 1, 2]
ALLOWED_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ALLOWED_NUM_HS = [0, 1, 2, 3, 4]


def one_hot_encode(value: Any, allowed_values: List[Any], include_unknown: bool = True) -> List[float]:
    vec = [1.0 if value == v else 0.0 for v in allowed_values]
    if include_unknown:
        vec.append(1.0 if value not in allowed_values else 0.0)
    return vec


def atom_to_feature_vector(atom: Chem.Atom) -> List[float]:
    features = []
    # Atomic number
    features.extend(one_hot_encode(atom.GetAtomicNum(), ALLOWED_ATOMS, include_unknown=True))
    # Degree
    features.extend(one_hot_encode(atom.GetDegree(), ALLOWED_DEGREES, include_unknown=True))
    # Formal charge
    features.extend(one_hot_encode(atom.GetFormalCharge(), ALLOWED_CHARGES, include_unknown=True))
    # Hybridization
    features.extend(one_hot_encode(atom.GetHybridization(), ALLOWED_HYBRIDIZATIONS, include_unknown=True))
    # Num Hydrogens
    features.extend(one_hot_encode(atom.GetTotalNumHs(), ALLOWED_NUM_HS, include_unknown=True))
    # Aromaticity
    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    # Is in ring
    features.append(1.0 if atom.IsInRing() else 0.0)
    return features


NODE_FEATURE_DIM = (
    len(ALLOWED_ATOMS) + 1 +
    len(ALLOWED_DEGREES) + 1 +
    len(ALLOWED_CHARGES) + 1 +
    len(ALLOWED_HYBRIDIZATIONS) + 1 +
    len(ALLOWED_NUM_HS) + 1 +
    1 + 1
)  # 10 + 7 + 6 + 6 + 6 + 1 + 1 = 37 dims


def mol_to_graph_data(mol: Chem.Mol) -> Optional[Dict[str, torch.Tensor]]:
    """
    Converts an RDKit Mol to graph tensors:
    - x: (num_atoms, NODE_FEATURE_DIM)
    - edge_index: (2, num_directed_edges)
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Compute node features
    atom_features = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_features, dtype=torch.float32)

    # Compute undirected edge index
    edge_list = []
    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edge_list.append((u, v))
        edge_list.append((v, u))  # undirected message passing

    if not edge_list:
        # Isolated atoms (e.g. single atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    return {"x": x, "edge_index": edge_index, "num_atoms": mol.GetNumAtoms()}


def collate_graph_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Batches a list of individual graphs into a single disconnected block graph.
    """
    batch = [g for g in batch if g is not None]
    if not batch:
        return {}

    all_x = []
    all_edge_index = []
    batch_index = []
    labels = []
    node_offset = 0

    for i, g in enumerate(batch):
        x = g["x"]
        edge_index = g["edge_index"]
        num_nodes = x.size(0)

        all_x.append(x)
        if edge_index.numel() > 0:
            shifted_edges = edge_index + node_offset
            all_edge_index.append(shifted_edges)

        batch_index.append(torch.full((num_nodes,), i, dtype=torch.long))
        if "y" in g:
            labels.append(g["y"])
        node_offset += num_nodes

    batched_x = torch.cat(all_x, dim=0)
    batched_batch = torch.cat(batch_index, dim=0)

    if all_edge_index:
        batched_edge_index = torch.cat(all_edge_index, dim=1)
    else:
        batched_edge_index = torch.empty((2, 0), dtype=torch.long)

    res = {
        "x": batched_x,
        "edge_index": batched_edge_index,
        "batch": batched_batch,
        "num_graphs": len(batch)
    }

    if labels:
        res["y"] = torch.tensor(labels, dtype=torch.float32)

    return res


class GraphConvLayer(nn.Module):
    """
    Message passing layer with self-loop connection and normalized neighborhood aggregation:
    H^{(l+1)} = ReLU( W_neigh * AGG({H_u : u in N(v)}) + W_self * H_v )
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear_neigh = nn.Linear(in_features, out_features, bias=False)
        self.linear_self = nn.Linear(in_features, out_features, bias=True)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        out = self.linear_self(x)

        if edge_index.numel() > 0:
            src, dst = edge_index[0], edge_index[1]
            src_messages = x[src]  # (E, F)

            # Degree-normalized aggregation
            deg = torch.zeros(num_nodes, device=x.device, dtype=x.dtype)
            deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            deg = torch.clamp(deg, min=1.0)
            norm_factor = 1.0 / deg[dst].unsqueeze(1)

            weighted_msg = src_messages * norm_factor
            agg = torch.zeros(num_nodes, x.size(1), device=x.device, dtype=x.dtype)
            agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), weighted_msg)

            out = out + self.linear_neigh(agg)

        return self.norm(F.relu(out))


class MolecularGNN(nn.Module):
    """
    Molecular Graph Neural Network for hERG Inhibition Classification.
    """
    def __init__(
        self,
        in_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.25
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Multi-pooling readout (mean + max pooling = 2 * hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Binary logit
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, num_graphs: int) -> torch.Tensor:
        h = F.relu(self.input_proj(x))

        for conv in self.conv_layers:
            h = h + self.dropout(conv(h, edge_index))  # Residual connection

        # Global pooling per graph in batch
        # Mean pooling
        mean_pool = torch.zeros(num_graphs, h.size(1), device=h.device, dtype=h.dtype)
        counts = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
        mean_pool.scatter_add_(0, batch.unsqueeze(1).expand(-1, h.size(1)), h)
        counts.scatter_add_(0, batch.unsqueeze(1), torch.ones_like(batch.unsqueeze(1), dtype=h.dtype))
        mean_pool = mean_pool / torch.clamp(counts, min=1.0)

        # Max pooling
        max_pool = torch.full((num_graphs, h.size(1)), -1e9, device=h.device, dtype=h.dtype)
        # Using scatter_reduce with max
        max_pool = max_pool.scatter_reduce(0, batch.unsqueeze(1).expand(-1, h.size(1)), h, reduce="amax", include_self=False)
        # Handle empty/default
        max_pool = torch.where(max_pool == -1e9, torch.zeros_like(max_pool), max_pool)

        graph_embedding = torch.cat([mean_pool, max_pool], dim=1)
        logits = self.classifier(graph_embedding).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_molecule(self, mol: Chem.Mol) -> Tuple[float, int]:
        """
        Inference helper for a single molecule. Returns (probability, predicted_class).
        """
        self.eval()
        graph = mol_to_graph_data(mol)
        if graph is None:
            return 0.5, 0

        batched = collate_graph_batch([graph])
        logits = self.forward(batched["x"], batched["edge_index"], batched["batch"], batched["num_graphs"])
        prob = float(torch.sigmoid(logits).item())
        pred_class = 1 if prob >= 0.5 else 0
        return prob, pred_class
