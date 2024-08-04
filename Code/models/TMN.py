
import torch.nn.functional as F
import torch

from .layers import SoftmaxAttention,replace_masked
def get_mask(sequences_batch):
    mask=torch.ones_like(sequences_batch)
    mask[sequences_batch[:, :] == 0] = 0.0
    return mask[:,:,1],mask



import torch.nn as nn
import torch


class TMN_net(nn.Module):
    def __init__(self, config):
        super(TMN_net, self).__init__()
        self.attention = SoftmaxAttention()
        self.g_embedding = nn.Linear(2, config.num_hiddens * 2)


        embedding_dim=2
        self.mlp_ele = torch.nn.Linear(2, config.num_hiddens).cuda()
        # bidirectional设为True即得到双向循环神经网络
        self.data_encoder = nn.LSTM(input_size=config.num_hiddens,
                                    hidden_size=config.num_hiddens,
                                    num_layers=config.num_layers,
                                    batch_first=True,
                                    bidirectional=True)

        # bidirectional设为True即得到双向循环神经网络
        self.query_encoder = nn.LSTM(input_size=config.num_hiddens,
                                     hidden_size=config.num_hiddens,
                                     num_layers=config.num_layers,
                                     batch_first=True,
                                     bidirectional=True)
        self.L1=nn.Linear(config.num_hiddens*4, config.num_hiddens)
        self.L2 = nn.Linear(config.num_hiddens * 4, config.num_hiddens)
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flat = nn.Flatten()
        self.nonLeaky = torch.nn.LeakyReLU(0.1)
    def forward(self, t, q):


        t_mask, t_m = get_mask(t)
        t_mask, t_m = t_mask.to(self.device), t_m.to(self.device)
        q_mask, q_m = get_mask(q)
        q_mask, q_m = q_mask.to(self.device), q_m.to(self.device)

        t = self.nonLeaky(self.g_embedding(t))
        q = self.nonLeaky(self.g_embedding(q))

        t_aligned, q_aligned = self.attention(t, t_mask, q, q_mask)

        t_combine = torch.cat([t,t-t_aligned], dim=-1)
        q_combine = torch.cat([q, q-q_aligned], dim=-1)

        outputs_t, semantic_t = self.data_encoder(self.L1(t_combine))  # output, (h, c)
        outputs_q, semantic_q = self.query_encoder(self.L1(q_combine))

        semantic_t = torch.transpose(semantic_t[0], 0, 1)
        semantic_q = torch.transpose(semantic_q[0], 0, 1)
        out_t = self.flat(semantic_t)
        out_q = self.flat(semantic_q)


        outs = nn.functional.pairwise_distance(out_t, out_q, p=2)
        # outs = self.sigmoid(outs)
        return outs.unsqueeze(1)

    def infer(self, batch, device):
        t, q, y = batch
        t, q, y = t.type(torch.float), q.type(torch.float), y.type(torch.float)
        t = t.to(device)
        q = q.to(device)
        y=y.to(device)
        return y, self(t, q)


class TMN(nn.Module):
    def __init__(self, config):
        super(TMN, self).__init__()
        self.model=TMN_net(
            config
        )
    def forward(self,q,t1,t2):
        p1 = self.model(t1,q)
        p2 = self.model(t2, q)

        return p1,p2
    def infer(self, batch, device):
        q,t1,s1,t2,s2=batch
        q, t1, s1, t2, s2= q.type(torch.float), t1.type(torch.float), s1.type(torch.float),t2.type(torch.float), s2.type(torch.float)
        q = q.to(device)
        t1 = t1.to(device)
        s1 = s1.to(device)
        t2 = t2.to(device)
        s2 = s2.to(device)

        return s1,s2, self(q,t1,t2)