import torch.nn as nn
import torch
import torch.nn.functional as F
from .layers import SoftmaxAttention, replace_masked


# import Pair_loss
def get_mask(sequences_batch):
    mask = torch.ones_like(sequences_batch)
    mask[sequences_batch[:, :] == 0] = 0.0
    return mask[:, :, 1], mask


class RSSE(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers):
        super(RSSE, self).__init__()
        # self.dropout=dropout
        self.embedding_dim = embedding_dim
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bi = 2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder1 = nn.GRU(input_size=embedding_dim,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)
        self.encoder2 = nn.GRU(input_size=4 * self.bi * self.num_hiddens,
                               hidden_size=num_hiddens,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=True)

        self.attention = SoftmaxAttention()

        self.mq = nn.BatchNorm1d(4 * self.bi * self.num_hiddens)
        self.mt = nn.BatchNorm1d(4 * self.bi * self.num_hiddens)

        self.mq2 = nn.BatchNorm1d(4 * self.bi * self.num_hiddens)
        self.mt2 = nn.BatchNorm1d(4 * self.bi * self.num_hiddens)
        self.other = nn.BatchNorm1d(2 * self.bi * self.num_hiddens)

        self.classification = nn.Sequential(nn.Linear(10 * self.bi * self.num_hiddens, self.num_hiddens),
                                            nn.Sigmoid(),
                                            # nn.ReLU(),
                                            nn.Linear(self.num_hiddens, self.num_hiddens // 2),
                                            # nn.ReLU(),
                                            nn.Sigmoid(),
                                            nn.Linear(self.num_hiddens // 2, 1))

    def forward(self, output_t, output_q):
        t_mask, t_m = get_mask(output_t)
        t_mask, t_m = t_mask.to(self.device), t_m.to(self.device)
        q_mask, q_m = get_mask(output_q)
        q_mask, q_m = q_mask.to(self.device), q_m.to(self.device)
        output_t, (h_t) = self.encoder1(output_t)
        output_q, (h_q) = self.encoder1(output_q)


        t_aligned, q_aligned = self.attention(output_t, t_mask, output_q, q_mask)

        output_q = torch.cat([output_q, q_aligned, output_q - q_aligned, output_q + q_aligned], dim=-1)
        output_t = torch.cat([output_t, t_aligned, output_t - t_aligned, output_t + t_aligned], dim=-1)

        output_q = self.mq(output_q.transpose(1, 2)).transpose(1, 2)
        output_t = self.mt(output_t.transpose(1, 2)).transpose(1, 2)

        q_compare, (_) = self.encoder2(output_q)
        t_compare, (_) = self.encoder2(output_t)


        q_avg_pool = torch.sum(q_compare * q_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(q_mask, dim=1,
                                                                                                   keepdim=True)
        t_avg_pool = torch.sum(t_compare * t_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(t_mask, dim=1,
                                                                                                   keepdim=True)
        q_max_pool, _ = replace_masked(q_compare, q_mask, -1e7).max(dim=1)
        t_max_pool, _ = replace_masked(t_compare, t_mask, -1e7).max(dim=1)

        ave_combined=torch.cat([q_avg_pool,t_avg_pool,q_avg_pool-t_avg_pool,q_avg_pool+t_avg_pool],dim=1)
        max_combined = torch.cat([q_max_pool, t_max_pool, q_max_pool - t_max_pool, q_max_pool + t_max_pool], dim=1)
        other_combined=torch.cat([q_max_pool - q_avg_pool, t_max_pool - t_avg_pool], dim=1)

        ave_combined=self.mq2(ave_combined)
        max_combined=self.mt2(max_combined)
        other_combined=self.other(other_combined)

        q_t_combined = torch.cat(
            [ave_combined,max_combined,other_combined], dim=1)



        outs = self.classification(q_t_combined)
        return outs

    def infer(self, batch, device):
        t, q, y = batch
        t, q, y = t.type(torch.float), q.type(torch.float), y.type(torch.float)
        t = t.to(device, non_blocking=True)
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        return y, self(t, q)


class Siamese_RSSE(nn.Module):
    def __init__(self, embedding_dim, num_hiddens, num_layers):
        super(Siamese_RSSE, self).__init__()
        self.model = RSSE(
            embedding_dim,
            num_hiddens,
            num_layers
        )
    def forward(self, q, t1, t2):
        p1 = self.model(t1, q)
        p2 = self.model(t2, q)

        return p1, p2
    def infer(self, batch, device):
        q, t1, s1, t2, s2 = batch
        q, t1, s1, t2, s2 = q.type(torch.float), t1.type(torch.float), s1.type(torch.float), t2.type(torch.float), s2.type(torch.float)
        q = q.to(device)
        t1 = t1.to(device)
        s1 = s1.to(device)
        t2 = t2.to(device)
        s2 = s2.to(device)

        return s1, s2, self(q, t1, t2)
