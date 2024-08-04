import os
import sys
import time

import h5py
import torch.cuda.amp
sys.path.append('..')
from torch.nn.utils.rnn import pad_sequence
from singleQuery import *
from models import RSSE
import pandas as pd
import torch.utils.data as Data
import torch
import numpy as np
import warnings
# from osf import osf
warnings.filterwarnings("ignore")
queryResult=[]
all_pair=[]


class get_Dataset():
    def __init__(self, pair_table):  # 设置初始信息
        '''

        :param score_table:
        :param t_table: trajectory ( id , trajectory list)
        :param q_table: qurey (id , qurey list)
        '''

        self.pair_table = pair_table

    def __len__(self):  # 返回长度
        return len(self.pair_table)

    def __getitem__(self, item):  # 根据item返回数据
        que_list = self.pair_table.iloc[item][0]
        tra_list = self.pair_table.iloc[item][1]

        return tra_list, que_list

class Learning_filter(object):
    def __init__(self,config):
        self.config=config

        self.queryResult = []
        self.all_pair = []
        self.learning_filtering_time=0
    def append_pair(self,x):
        for i in x['cand']:
            self.all_pair.append([x['tra'],i, x['data_index']])



    def to_DataFrame(self,table):
        global all_pair
        all_pair=[]
        table.apply(lambda x: self.append_pair(x),axis=1)
        t=pd.DataFrame(all_pair)
        t.columns=['tra','cand', 'data_index']

        return t







    def collate_fn(self,data):
        score = []
        tra = []
        que = []

        for i in data:
            score.append(torch.tensor(0))
            tra.append(i[0])
            que.append(i[1])

        tra = pad_sequence(tra, batch_first=True)
        que = pad_sequence(que, batch_first=True)

        return tra, que, torch.tensor(score, dtype=torch.float32).view(-1, 1)

    def pre(self,query,dataTrajectorie):
        queryTrajectories=query.copy()
        dataTrajectories=dataTrajectorie.copy()
        f= h5py.File(os.path.join(self.config.data_path, 'tra_Frame_td'+'_{}_{}'.format(self.config.data_min_len,self.config.data_max_len))+'info','r')
        info=f['info']
        x_max,x_min,y_max,y_min = info
        dataTrajectories['tra'] = dataTrajectories['tra'].apply(
            lambda x: [[(i[0] - x_min) / (x_max-x_min), (i[1] - y_min) / (y_max-y_min)] for i in x])

        queryTrajectories['tra']=queryTrajectories['tra'].apply(lambda x: [[(i[0] - x_min) / (x_max-x_min), (i[1] - y_min) / (y_max-y_min)] for i in x])
        f.close()
        pair_list=[]

        def add_pair(x):
            for i in x['cand']:
                pair_list.append([torch.tensor(queryTrajectories.iloc[x['idx']]['tra']),torch.tensor(dataTrajectories.iloc[i]['tra']),x['q_idx'],i])
        queryTrajectories['idx']=queryTrajectories.index
        queryTrajectories.progress_apply(lambda x:add_pair(x),axis=1)


        pair_table=pd.DataFrame(pair_list)


        pair_dataset = get_Dataset(pair_table[[0,1]])
        test_dataloader = Data.DataLoader(
            dataset=pair_dataset,
            batch_size=6000,
            shuffle=False,
            num_workers=1,
            collate_fn=self.collate_fn,
            drop_last=False,
            pin_memory=True

        )
        model = RSSE.Siamese_RSSE(
            self.config.embedding_dim,
            self.config.num_hiddens,
            self.config.num_layers
        )
        state_dict = torch.load(self.config.eval_model_path,map_location='cpu')
        # new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}

        model.load_state_dict(state_dict)

        model = model.to(self.config.device)

        return model ,test_dataloader,pair_table

    def predict(self,model,test_dataloader,pair_table):
        model.eval()
        result = []
        all_time=0
        with torch.no_grad():
            for batch in test_dataloader:
                t, q, y = batch
                t, q= t.type(torch.float), q.type(torch.float)
                t = t.to(self.config.device, non_blocking=True)
                q = q.to(self.config.device, non_blocking=True)
                start=time.time()
                pre_y = model.model(t,q)
                end = time.time()
                all_time=all_time+end-start
                result.append(pre_y.detach().cpu().numpy())
            result = np.concatenate(result, axis=0)
        pair_table['pre']=np.array(result)
        print('learning_time{}'.format(all_time))
        self.learning_filtering_time=all_time
        return pair_table




    def learning_filter(self,query,dataTrajectories,size=50):

        model, test_dataloader,pair_table=self.pre(query,dataTrajectories)
        filter_query=self.predict(model, test_dataloader,pair_table)
        print('predic ok')

        filter_query['pre_rank'] = filter_query.groupby(2)['pre'].rank(ascending=True, method='first')
        filter_query = filter_query[filter_query['pre_rank'] <= size]
        # filter_query=filter_query[['q','d']].groupby('q').apply(lambda x:x['d'].to_list()).reset_index().sort_values(by='q')
        # query['cand']=filter_query[0].values


        return filter_query.drop([0,1],axis=1)




    def read_trajectory(self,data_name):
        table_path = os.path.join(self.config.data_path, data_name)
        table = pd.read_pickle(table_path)
        return table








