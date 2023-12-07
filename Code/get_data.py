import numpy as np
import pandas as pd
import os
import torch
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence
import random
import copy

def get_data(config):
    # pass    print('Loading Data!!!')
    base_path = config.data_path
    ## scalability
    # base_path='/root/copy/subtra/data/chengdu'

    print('Loading Data!!!')
    q_idx = '_{}_{}'.format(config.query_min_len, config.query_max_len)
    d_idx = '_{}_{}'.format(config.data_min_len, config.data_max_len)
    score_table_path = base_path + '/{}_{}_'.format(config.query_min_len, config.data_min_len) + config.metric
    t_table_path = os.path.join(base_path, 'tra_Frame_td') + d_idx + '_N'
    q_table_path = os.path.join(base_path, 'tra_Frame_tq') + q_idx + '_N'

    score_table = pd.read_csv(score_table_path)
    score_table.columns = ['que_index', 'tra_idx', 's', 'e', 'score']

    if config.task=='tra':

        score_table['t'] = score_table['e'] - score_table['s']
        score_table = score_table[score_table['t'] > 2].drop(['t'], axis=1)

    t_table = pd.read_pickle(t_table_path)
    q_table = pd.read_pickle(q_table_path)

    print('Loading Data  finish!!')
    if config.metric == 'edr':
        score_table['score'] = score_table['score'] / config.query_max_len
    return score_table, t_table, q_table


class get_Dataset():
    def __init__(self, score_table, t_table, q_table, task='subtra'):  # 设置初始信息
        '''

        :param score_table:
        :param t_table: trajectory ( id , trajectory list)
        :param q_table: qurey (id , qurey list)
        '''
        self.score_table, self.t_table, self.q_table = score_table, t_table, q_table
        self.task = task

    def __len__(self):  # 返回长度
        return len(self.score_table)

    def __getitem__(self, item):  # 根据item返回数据
        score = self.score_table.iloc[item]['score']

        tra_index = self.score_table.iloc[item]['tra_idx']
        que_index = self.score_table.iloc[item]['que_index']
        if self.task == 'tra':
            s = self.score_table.iloc[item]['s']
            e = self.score_table.iloc[item]['e']
            tra_list = self.t_table.iloc[int(tra_index)]['tra'][int(s):int(e)]
        else:
            tra_list = self.t_table.iloc[int(tra_index)]['tra']
        que_list = self.q_table.iloc[int(que_index)]['tra']

        return score, tra_list, que_list


class get_Dataset_pair():
    def __init__(self, score_table, t_table, q_table, task='subtra'):  # 设置初始信息
        '''

        :param score_table:
        :param t_table: trajectory ( id , trajectory list)
        :param q_table: qurey (id , qurey list)
        '''
        self.score_table, self.t_table, self.q_table = score_table, t_table, q_table
        self.task = task

    def __len__(self):  # 返回长度
        return len(self.score_table)

    def __getitem__(self, item):  # 根据item返回数据
        score1 = self.score_table.iloc[item]['score1']
        score2 = self.score_table.iloc[item]['score2']
        tra_index1 = self.score_table.iloc[item]['tra_idx1']
        tra_index2 = self.score_table.iloc[item]['tra_idx2']
        que_index = self.score_table.iloc[item]['que_index']

        if self.task == 'tra':

            s1, e1 = self.score_table.iloc[item]['s1'], self.score_table.iloc[item]['e1']
            s2, e2 = self.score_table.iloc[item]['s2'], self.score_table.iloc[item]['e2']

            tra1_list = self.t_table.iloc[int(tra_index1)]['tra'][int(s1):int(e1)]
            tra2_list = self.t_table.iloc[int(tra_index2)]['tra'][int(s2):int(e2)]
        else:
            tra1_list = self.t_table.iloc[int(tra_index1)]['tra']
            tra2_list = self.t_table.iloc[int(tra_index2)]['tra']
        que_list = self.q_table.iloc[int(que_index)]['tra']

        return score1, tra1_list, score2, tra2_list, que_list




def collate_fn(data):
    score = []
    tra = []
    que = []
    for i in data:
        score.append(torch.tensor(i[0]))
        tra.append(torch.tensor(i[1]))
        que.append(torch.tensor(i[2]))

    tra = pad_sequence(tra, batch_first=True)
    que = pad_sequence(que, batch_first=True)
    return tra, que, torch.tensor(score, dtype=torch.float32).view(-1, 1)







def collate_fn_pair(data):
    '''
    score1, tra1_list, score2, tra2_list, que_list
    :param data:
    :return:
    '''
    score1 = []
    tra1 = []
    score2 = []
    tra2 = []
    que = []
    for i in data:
        score1.append(torch.tensor(i[0]))
        tra1.append(torch.tensor(i[1]))
        score2.append(torch.tensor(i[2]))
        tra2.append(torch.tensor(i[3]))
        que.append(torch.tensor(i[4]))

    que = pad_sequence(que, batch_first=True)
    tra1 = pad_sequence(tra1, batch_first=True)
    tra2 = pad_sequence(tra2, batch_first=True)
    return que, tra1, torch.tensor(score1, dtype=torch.float32).view(-1, 1), tra2, torch.tensor(score2,
                                                                                                dtype=torch.float32).view(
        -1, 1)

def get_mask(sequences_batch):
    mask = torch.ones_like(sequences_batch)
    mask[sequences_batch[:, :] == 0] = 0.0
    return mask[:, :, 1]




def split_data_pair(score_table,config,T=0):
    '''

    Args:
        score_table:
        train_rate:
        val_rate:

    Returns:

    '''
    random.seed(2)


    score_table = score_table[score_table['tra_idx'] < score_table['tra_idx'].max() - 1]
    temp = score_table.groupby('que_index')['score'].count().reset_index()
    que_list = temp['que_index'].tolist()
    score_table = score_table[score_table['que_index'].isin(que_list)]


    random.shuffle(que_list)
    dataset_len = len(que_list)

    train_index = que_list[:int(dataset_len * config.train_rate)]
    val_index = que_list[int(dataset_len * config.train_rate):int(dataset_len * (config.val_rate + config.train_rate))]
    test_index = que_list[int(dataset_len * (config.val_rate + config.train_rate)):]

    train_score_table = score_table[score_table['que_index'].isin(train_index)].reset_index(drop=True)
    val_score_table = score_table[score_table['que_index'].isin(val_index)].reset_index(drop=True)
    test_score_table = score_table[score_table['que_index'].isin(test_index)].reset_index(drop=True)
    # val_score_table_copy=copy.deepcopy(val_score_table)
    # trajectory similariy

    print('len_train: {} len_val: {} len_test: {}'.format(len(train_score_table), len(val_score_table),
                                                          len(test_score_table)))
    print('preparation pair')

    if T == 0:
        print(config.pair_p)
        train_score_table = pre_pair(train_score_table, k=config.pair_p)
        val_score_table = pre_pair(val_score_table, k=config.pair_p)


    print('preparation pair ok! len_pair_train:{}'.format(len(train_score_table)))
    print('preparation pair ok! len_pair_val:{}'.format(len(val_score_table)))
    return train_score_table, val_score_table, test_score_table




def fun(x, k):


    len_x = len(x)
    x = x.sort_values(by='score', ascending=True)

    t=int(len_x*k)
    # if len_x<400:
    #     t=int(len_x*0.2)
    if t % 2 != 0:
        t = t + 1





    x1 = x[:3*t]  #
    x1=shuffle(x1,random_state=42)
    x2=x[3*t:]
    x2 = shuffle(x2,random_state=43)  # 打乱数据


    p1=x1[:t]['idx'].values.tolist()
    p2=x1[t:2*t]['idx'].values.tolist()
    xx=pd.concat([x1[2*t:],x2[:t]])
    xx=shuffle(xx,random_state=44)

    temp = xx['idx'].values.reshape(2, -1)
    p1 = p1 + temp[0].tolist()
    p2 = p2 + temp[1].tolist()


    return (p1, p2)


score_list0 = []
score_list1 = []


def to_df(x, score):
    global score_list0
    global score_list1
    for i in range(0, len(x['pair'][0])):
        score_list0.append(score.loc[x['pair'][0][i]].values)
        score_list1.append(score.loc[x['pair'][1][i]].values)


def pre_pair(table, k=0.05):
    global score_list0
    global score_list1
    score_list0 = []
    score_list1 = []
    table['idx'] = table.index
    # table.drop(['s', 'e'], axis=1, inplace=True)
    # table.sort_values(by='que_index', inplace=True)
    pair_table = table.groupby('que_index')['idx', 'score'].apply(lambda x: fun(x, k)).reset_index()
    pair_table.columns = ['que_index', 'pair']
    pair_table.apply(lambda x: to_df(x, table), axis=1)
    p0 = pd.DataFrame(score_list0).reset_index(drop=True)
    p1 = pd.DataFrame(score_list1).reset_index(drop=True)
    p0.columns = ['que_index', 'tra_idx1', 's1', 'e1', 'score1', 'tra_idx2']
    p0['tra_idx2'] = p1[1]
    p0['s2'] = p1[2]
    p0['e2'] = p1[3]
    p0['score2'] = p1[4]
    return p0


