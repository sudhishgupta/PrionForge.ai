import torch
import torch.nn as nn

class SimpleBinaryModel_onlyEmbeds(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)   # no activation here; we'll use BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.net(x)
    

class SimpleBinaryModel_onlyQN(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1)   # no activation here; we'll use BCEWithLogitsLoss
        )
        
    def forward(self, x):
        return self.net(x)

    
class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # branch for the 1024‑dim input
        self.branch1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # branch for the 2‑dim input is identity (no layers)
        # after concatenation:
        self.post = nn.Sequential(
            nn.Linear(64 + 2, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)   # linear activation; we'll use BCEWithLogitsLoss
        )

    def forward(self, x1024, x2):
        h1 = self.branch1(x1024)          # → [batch,64]
        h = torch.cat([h1, x2], dim=1)    # → [batch, 66]
        out = self.post(h)                # → [batch,1]
        return out