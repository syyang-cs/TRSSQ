import torch.nn as nn
import torch

class Pair_loss(nn.Module):
    def __init__(self):
        super(Pair_loss, self).__init__()
        self.mse = nn.MSELoss()
        self.theta = 1
        print(self.theta)

    def forward(self, p1, s1, p2, s2):
        lmse = self.mse(p1, s1) + self.mse(p2, s2)
        lrank = torch.mean(nn.functional.relu(-(p1 - p2) * (s1 - s2)))
        loss = lmse / 2 + self.theta * lrank
        # loss = lmse / 2
        return loss