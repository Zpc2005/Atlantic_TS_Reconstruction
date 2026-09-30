"""Small mask-aware causal STGNN v2 and controlled ablations."""
from __future__ import annotations
import torch
from torch import nn


class MaskAwareStgnnV2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, use_graph: bool) -> None:
        super().__init__(); self.use_graph = use_graph
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        width = hidden_size * 2 if use_graph else hidden_size
        self.head = nn.Sequential(nn.Linear(width, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))

    def forward(self, inputs: torch.Tensor, target_index: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, channels = inputs.shape
        encoded, _ = self.gru(inputs.permute(0, 2, 1, 3).reshape(batch * nodes, steps, channels))
        local = encoded[:, -1].reshape(batch, nodes, -1); index = target_index[:, None, None].expand(-1, 1, local.shape[-1])
        target_local = local.gather(1, index).squeeze(1)
        if not self.use_graph: return self.head(target_local).squeeze(1)
        if adjacency is None: raise ValueError("full and no-missingness models require adjacency")
        graph = torch.einsum("ij,bjh->bih", adjacency, local).gather(1, index).squeeze(1)
        return self.head(torch.cat((target_local, graph), dim=1)).squeeze(1)
