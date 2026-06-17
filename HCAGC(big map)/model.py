import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    # Small two-head MLP used by the clustering baselines.
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.mlp1 = nn.Linear(dims[0], dims[1])
        self.mlp2 = nn.Linear(dims[0], dims[1])

    def forward(self, x):
        # Produce two normalized views for contrastive learning.
        out1 = self.mlp1(x)
        out2 = self.mlp2(x)

        out1 = F.normalize(out1, dim=1, p=2)
        out2 = F.normalize(out2, dim=1, p=2)
        return out1, out2
