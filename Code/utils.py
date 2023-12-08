import random
import time

import numpy as np
import pandas as pd
import os
import torch.nn as nn
from torch import optim
import torch
from sklearn.metrics import  ndcg_score,mean_squared_error
from tqdm import tqdm
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models import Pair_loss

def my_metrics(y, y_p):
    return mean_squared_error(y, y_p)



def train_model(model, train_dataloader, val_dataloader, train_config):
    model.to(device)
    loss_fn = Pair_loss.Pair_loss()
    # loss_fn = eval(train_config.loss_func)
    loss_fn.to(device)



    parameters = filter(lambda p: p.requires_grad, model.parameters())
    lr = 0.001
    # if config.model=='NeuTraj':
    #     lr=0.04
    optimizer = optim.Adam(parameters, lr=lr, weight_decay=1e-4)  # neutraj learning rate 0.0005
    Earlystopper = EarlyStopping(train_config,train_config.patience, train_config.model_path, )
    epochs = train_config.epoch
    print('=' * 10 + train_config.model + '=' * 10)
    print('Plan to train {} epoches \n'.format(epochs) + time.strftime("%m_%d_%H_%M", time.localtime()))
    for epoch in range(epochs):

        # mini-batch for training
        loss_list = []
        model.train()

        for step, batch in enumerate(train_dataloader):
            s1, s2, (p1, p2) = model.infer(batch, device)
            the_loss = loss_fn(s1, p1, s2, p2)

            optimizer.zero_grad()
            the_loss.backward()
            optimizer.step()
            # if config.model == "NeuTraj":
            #     model.model.spatial_memory_update(batch)
            loss_list.append(the_loss.cpu().detach().numpy())

            if step % 100 == 0:
                mae_score = my_metrics(s1.cpu().detach().numpy().flatten(), p1.cpu().detach().numpy().flatten())
                print('In epoch:{:03d}|batch:{:04d}, train_loss:{:4f}, train_mse:{:.4f}'.format(epoch,
                                                                                                step,
                                                                                                np.mean(loss_list),
                                                                                                mae_score))

        # -------------------------5. validation --------------------------------------#
        model.eval()
        # mini-batch for validation
        loss_list = []
        # 查看验证集总体表现

        mean_loss = []
        mean_mae = []
        for step, batch in enumerate(val_dataloader):
            s1, s2, (p1, p2) = model.infer(batch, device)
            the_loss = loss_fn(s1, p1, s2, p2)
            loss_list.append(the_loss.cpu().detach().numpy())

            if step % 100 == 0:
                mae_score = my_metrics(s1.cpu().detach().numpy().flatten(), p1.cpu().detach().numpy().flatten())
                print('In epoch:{:03d}|batch:{:04d}, test_loss:{:4f}, val_mse_score:{:.4f}'.format(epoch,
                                                                                                   step,
                                                                                                   np.mean(
                                                                                                       loss_list),
                                                                                                   mae_score))
                mean_loss.append(np.mean(loss_list))
                mean_mae.append(mae_score)
        mean_loss = np.mean(mean_loss)
        mean_mae = np.mean(mean_mae)
        print('\033[31m val_mean_loss：{:05f},val_mean_mse：{:05f}, \033[0m'.format(
            mean_loss, mean_mae
        ))
        # -------------------------6. Save models --------------------------------------#
        Earlystopper(mean_loss, model)
        if Earlystopper.early_stop:
            print("Early stopping")
            break
    # #     cleanup()
    Earlystopper.save_model()
    return model





def top_n_recall(table, n1, n2):
    '''

    :param table:
    :param n1: top_n
    :param n2: hit n
    :return:
    '''

    def _fun(x, n1, n2):
        x1 = set(x[x['pre_rank'] <= n1]['tra_idx'].tolist())
        x2 = set(x[x['label_rank'] <= n2]['tra_idx'].tolist())

        return len(x1 & x2) / len(x2)

    return table[['que_index', 'label_rank', 'pre_rank', 'tra_idx']].groupby('que_index').apply(
        lambda x: _fun(x, n1, n2)
    ).mean()


def HRK(table, k):
    def _fun(x, k):
        x1 = set(x[x['pre_rank'] <= k]['tra_idx'].tolist())
        x2 = set(x[x['label_rank'] <= k]['tra_idx'].tolist())

        return len(x1 & x2) / len(x1)

    return table[['que_index', 'label_rank', 'pre_rank', 'tra_idx']].groupby('que_index').apply(
        lambda x: _fun(x, k)
    ).mean()


def AUC(table):
    def _fun(x):
        label = list(x['label_rank'])
        pre = list(x['pre_rank'])
        return _AUCSingle(label, pre)

    return table[['que_index', 'label_rank', 'pre_rank']].groupby('que_index').apply(
        lambda x: _fun(x)
    ).mean()


def NDCG_K(table, k):
    def _fun(x, k):
        idx = x[x['label_rank'] <= k]['tra_idx'].values
        t = x[x['tra_idx'].isin(idx)]['pre_rank'] + 1
        t = sum([math.log2(i + 1) for i in range(1, len(t) + 1)]) / sum(np.log2(t))

        return t

    return table[['que_index', 'tra_idx', 'label_rank', 'pre_rank']].groupby('que_index').apply(
        lambda x: _fun(x, k)
    ).mean()


def NDCG(table):
    def _fun(x):
        label = list(x['label_rank'])
        pre = list(x['pre_rank'])
        return ndcg_score([label], [pre])

    return table[['que_index', 'label_rank', 'pre_rank']].groupby('que_index').apply(
        lambda x: _fun(x)
    ).mean()







def evaluate(model, eva_data, test_dataloader,config):
    model = model.to(device)
    model.eval()
    result = []
    score = []
    print(time.strftime("%m_%d_%H_%M_%S", time.localtime()))
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            # y, pre_y = model.infer(batch, device)
            y, pre_y = model.model.infer(batch, device)
            result.append(pre_y.detach().cpu().numpy())
            score.append(y.detach().cpu().numpy())
        print(time.strftime("%m_%d_%H_%M_%S", time.localtime()))
        result = np.concatenate(result, axis=0)
        score = np.concatenate(score, axis=0)
        eva_data['pre'] = np.array(result)
        eva_data['score'] = np.array(score)
        eva_data['label_rank'] = eva_data.groupby('que_index')['score'].rank(ascending=True, method='first')
        eva_data['pre_rank'] = eva_data.groupby('que_index')['pre'].rank(ascending=True, method='first')
        eva_data.to_csv(config.result_file +config.model+'_' + config.metric+'_'+config.dataset, index=None)
    return eva_data




def result_csv(filename):
    all_result = os.listdir(filename)
    lll = ['name', 'NDCG20', 'NDCG50', 'NDCG100', 'HR20', 'HR50', 'HR100', '50_n_20', '100_n_20', 'mse', 'mae']
    a = []
    for i in all_result:
        print(i)
        s_r = [i]
        eva_data = pd.read_csv(os.path.join(filename, i))
        eva_data['label_rank'] = eva_data.groupby('que_index')['score'].rank(ascending=True, method='first')
        eva_data['pre_rank'] = eva_data.groupby('que_index')['pre'].rank(ascending=True, method='first')

        NDCG_score = [NDCG_K(eva_data, 20), NDCG_K(eva_data, 50), NDCG_K(eva_data, 100)]
        s_r = s_r + NDCG_score

        HR_score = [HRK(eva_data, 20), HRK(eva_data, 50), HRK(eva_data, 100)]
        s_r = s_r + HR_score

        Rcall = [top_n_recall(eva_data, 50, 20), top_n_recall(eva_data, 100, 20), Mse(eva_data), Mae(eva_data)]
        s_r = s_r + Rcall
        a.append(s_r)
    a = pd.DataFrame(a)
    a.columns = lll
    a.to_csv('result')


def analizeResult(eva_data):
    metrics_score = []
    # eva_data = pd.read_csv(filename)
    result_list=[]
    columns_list=[]
    eva_data['label_rank'] = eva_data.groupby('que_index')['score'].rank(ascending=True, method='first')
    eva_data['pre_rank'] = eva_data.groupby('que_index')['pre'].rank(ascending=True, method='first')

    NDCG50 = NDCG_K(eva_data, 50)
    print('NDCG_score is {}'.format(NDCG50))
    result_list.append(NDCG50)
    columns_list.append('NDCG50')

    H50R20 = top_n_recall(eva_data, 50, 10)
    print('H{}R{} is {}'.format(50, 10, H50R20))
    result_list.append(H50R20)
    columns_list.append('H50R10')

    HRK50 = HRK(eva_data, 50)
    print('HRK50 is {}'.format(HRK50))
    result_list.append(HRK50)
    columns_list.append('HRK50')
    return result_list,columns_list







class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, config=None,patience=7, model_path=None, verbose=True, *arg):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.config=config
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = 0
        self.best_model = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        self.best_model = model


    def save_model(self):
        save_path = self.config.model_path + self.config.model + self.config.metric+self.config.dataset + '_' + time.strftime("%m_%d_%H_%M", time.localtime()) + 'loss' + str(
            self.val_loss_min)
        torch.save(self.best_model.state_dict(), save_path)




def pad_sequences(sequences, maxlen=None, dtype='float32', padding='pre',
                  truncating='pre', value=0.):
    """ pad_sequences
    """
    lengths = [len(s) for s in sequences]
    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = (np.ones((nb_samples, maxlen)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)
        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



def show_info(config):
    print('model:{}'.format(config.model))
    print('dataset:{}'.format(config.dataset))
    print('metric:{}'.format(config.metric))
    print('batch_size:{}'.format(config.batch_size))
    print('query_min_len:{}'.format(config.query_min_len))
    print('data_min_len:{}'.format(config.data_min_len))
    print('task:{}'.format(config.task))
    print('num_hidden:{}'.format(config.num_hiddens))
