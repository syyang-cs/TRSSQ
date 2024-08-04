import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import copy

class T3S_model(nn.Module):
    def __init__(self,config):
        super(T3S_model, self).__init__()
        self.postion_embedding = Positional_Encoding(config.num_hiddens, config.data_max_len, 0, config.device)
        self.g_embedding=nn.Linear(2,config.num_hiddens)


        # self.encoder_q = Encoder(config.num_hiddens, 16, 128, 0)
        # self.encoders_q = nn.ModuleList([
        #     copy.deepcopy(self.encoder_q)
        #     # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        #     for _ in range(2)])

        self.encoder_t = Encoder(config.num_hiddens, 16, config.num_hiddens*config.num_layers, 0)
        self.encoders_t = nn.ModuleList([
            copy.deepcopy(self.encoder_t)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(2)])

        self.t_encoder = nn.LSTM(input_size=config.embedding_dim,
                                    hidden_size=config.num_hiddens,
                                    num_layers=1,
                                    batch_first=True,
                                    bidirectional=False)
        # self.q_encoder = nn.LSTM(input_size=config.embedding_dim,
        #                             hidden_size=config.num_hiddens,
        #                             num_layers=config.num_layers,
        #                             batch_first=True,
        #                             bidirectional=True)

        self.sigmoid=nn.Sigmoid()
        self.flat = nn.Flatten()
        self.para = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.para.to(config.device)
        # self.L1=nn.Linear(config.num_hiddens*config.num_layers*2,config.num_hiddens)
    def forward(self, t, t_g, q, q_g):

        _,t= self.t_encoder(t)
        _,q= self.t_encoder(q)

        t = t[0].permute(1,0,2)
        q = q[0].permute(1,0,2)


        t = self.flat(t)
        q = self.flat(q)
        # t=self.L1(t)
        # q=self.L1(q)

        #
        #
        t_g = self.sigmoid(self.g_embedding(t_g))
        q_g = self.sigmoid(self.g_embedding(q_g))
        #
        t_g = self.postion_embedding(t_g)
        q_g = self.postion_embedding(q_g)

        #
        for encoder in self.encoders_t:
            t_g = encoder(t_g)
            q_g = encoder(q_g)

        # for encoder in self.encoders_t:
        #
        #
        # # for encoder in self.encoders_q:
        #
        #
        #
        t_g=torch.mean(t_g,dim=1)
        q_g=torch.mean(q_g,dim=1)


        # t_g=self.L2(t_g)
        # q_g=self.L2(q_g)


        t = self.para*t_g+(1-self.para)*t
        q = self.para*q_g+(1-self.para)*q
        outs = nn.functional.pairwise_distance(t, q, p=2)

        return outs.unsqueeze(1)

    def infer(self, batch, device):
        t, q, y, t_l, q_l = batch
        t, q, y, t_l, q_l = t.type(torch.float), q.type(torch.float), y.type(torch.float), t_l.type(torch.float), q_l.type(torch.float)
        t = t.to(device)
        q = q.to(device)
        t_l = t_l.to(device)
        q_l = q_l.to(device)
        y = y.to(device)
        return y, self(t, t_l, q, q_l)

class T3S(nn.Module):
    def __init__(self,config):
        super(T3S, self).__init__()
        self.model=T3S_model(config)
    def forward(self,q,t1,t2,g1,g2,gq):
        p1 = self.model(t1,g1,q,gq)
        p2 = self.model(t2,g2,q,gq)

        return p1,p2
    def infer(self, batch, device):
        q, s1, s2, t1, t2, g1, g2, gq = batch

        q, s1, s2, t1, t2, g1, g2, gq=q.type(torch.float), s1.type(torch.float), s2.type(torch.float), t1.type(torch.float), t2.type(torch.float), g1.type(torch.float), g2.type(torch.float), gq.type(torch.float)
        q = q.to(device)
        s1 = s1.to(device)
        s2 = s2.to(device)
        t1 = t1.to(device)
        t2 = t2.to(device)
        g1 = g1.to(device)
        g2 = g2.to(device)
        gq = gq.to(device)


        return s1,s2, self(q,t1,t2,g1,g2,gq)

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)

    def forward(self, x):
        out = self.attention(x)
        return out




class Positional_Encoding(nn.Module):
    def __init__(self, embed, max_len, dropout,device ):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(max_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.pe=self.pe.to(self.device)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe[:x.size(1), :], requires_grad=False).unsqueeze(dim=0)
        # out = self.dropout(out)
        return out

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        # scale = K.size(-1) ** -0.5  # 缩放因子
        # context = self.attention(Q, K, V, scale)
        context = self.attention(Q, K, V)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        # out = self.dropout(out)
        # out = out + x  # 残差连接
        # out = self.layer_norm(out)
        return out
class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context
class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out