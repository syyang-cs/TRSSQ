import torch.nn as nn

class Pair_loss(nn.Module):
    def __init__(self):
        super(Pair_loss, self).__init__()
        self.mse=nn.MSELoss()
        self.relu=nn.ReLU()

    def forward(self, p1, s1, p2, s2):
        lmse=self.mse(p1,s1)+self.mse(p2,s2)
        lrank=self.relu((p1-s1)*(s2-p2)).mean()
        loss=lmse+lrank
        return loss