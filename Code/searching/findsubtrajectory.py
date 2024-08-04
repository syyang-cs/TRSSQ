import sys
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# dir_path = os.path.dirname(os.path.realpath(__file__))
# parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
# sys.path.insert(0, parent_dir_path)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import Config
# config = Config.Config()
import get_data
# from searching.learning_method import  read_trajectory, pre, to_DataFrame


tqdm.pandas(desc='pandas bar')

from osf import OSF
from learning_method import Learning_filter
from Invert_index_func import Index_filter



class subtrajectory_search(object):
    def __init__(self,config,Learning_filter,Invert_index,OSF,learning_size):
        self.config=config
        self.Learning_filter=Learning_filter
        self.Invert_index=Invert_index
        self.OSF=OSF
        self.learning_size=learning_size
        self.Count_info=[]
        self.List_info=[]
    def show_info(self):
        pass




    def filter_function(self,querys, dataTrajectories):
        if self.config.method == 'osf':
            self.Invert_index.run_get_index(dataTrajectories,0.001)
            querys = self.OSF.osf_filter(querys,self.Invert_index.invert_index,0.1,0.001,dataTrajectories,self.config.metric)
        elif self.config.method == 'grid':
            self.Invert_index.run_get_index(dataTrajectories,0.0005)
            querys = self.grid_filter(querys)
        else:
            self.Invert_index.run_get_index(dataTrajectories, 0.0005)
            querys = self.grid_filter(querys)

        return querys


    def model_function(self,querys, dataTrajectories):
        if self.learning_size > 0:
            querys = self.Learning_filter.learning_filter(querys, dataTrajectories, self.learning_size)
            # c++

            # print(querys)
        else:
            all_row_data=[]
            querys.apply(lambda x: [all_row_data.append([x['q_idx'],i]) for i in x['cand']],axis=1)
            querys=pd.DataFrame(all_row_data)


        return querys


    def calAllScore(self,query_list):
        '''

        :param query_list:
        :param filter: ofs,grid,None
        :param size: int
        :param verify: Exacts Sizes POS PSS
        :param metric: dtw edr erp
        :return:
        '''
        # self.Count_info.append(filter)
        # self.Count_info.append(self.config.metric)
        # print(config.dataset)
        # print(self.config.metric)

        queryTrajectories = self.Learning_filter.read_trajectory('tra_Frame_tq' + '_{}_{}'.format(config.query_min_len, config.query_max_len))
        dataTrajectories = self.Learning_filter.read_trajectory('tra_Frame_td''_{}_{}'.format(config.data_min_len, config.data_max_len))
        querys = queryTrajectories.iloc[query_list].reset_index(drop=True)
        querys['q_idx']=np.array(query_list)


        querys = self.filter_function(querys, dataTrajectories)
        querys['len']=querys['cand'].apply(lambda x:len(x))

        mean_cand_size=querys['len'].mean()
        print('mean_cand_size{}'.format(mean_cand_size))
        self.Count_info.append(mean_cand_size)
        self.List_info.append('mean_cand_size')

        pruning_time=querys['pruning_time'].mean()
        self.Count_info.append(pruning_time)
        self.List_info.append('pruning_time')
        print('pruning_time{}'.format(pruning_time))

        print(querys)

        querys = self.model_function(querys, dataTrajectories)


        learning_time=self.Learning_filter.learning_filtering_time
        self.Count_info.append(learning_time)
        self.List_info.append('learning_time')
        print('learning_time{}'.format(learning_time))
        print(querys)

        querys.to_csv('/root/copy/subtra/add_result/pruning_data/'+config.dataset + '_' + config.metric + '_' + config.method, index=False,header=None)

        result_list = pd.DataFrame(self.Count_info).T
        result_list.columns = self.List_info
        result_list.to_csv('/root/copy/subtra/add_result/pruning_time/' + config.dataset + '_' + config.metric + '_' + config.method, index=False)


        return querys



# @time_cost
    def grid_filter(self,query):
        query['cand'] = query.apply(lambda x: self.Invert_index.get_candi(x, 0.0005),
                                    axis=1)  # grid_based_filtering
        query['pruning_time']=query['cand'].apply(lambda x:x[1])
        query['cand'] = query['cand'].apply(lambda x: x[0])
        return query

    def get_data_framework(self,k=10):
        score_table, t_table, q_table = get_data.get_data(config)
        q_table['idx']=q_table.index
        train_rate=config.train_rate
        val_rate=config.val_rate
        random.seed(2)


        score_table = score_table[score_table['tra_idx'] < score_table['tra_idx'].max() - 1]
        temp = score_table.groupby('que_index')['score'].count().reset_index()
        que_list = temp['que_index'].tolist()
        score_table = score_table[score_table['que_index'].isin(que_list)]

        # score_table=score_table[score_table['que_index'].isin(que_list)].reset_index(drop=True)

        # score_table = score_table.sort_values(by='que_index')
        random.shuffle(que_list)
        dataset_len = len(que_list)

        test_index = que_list[int(dataset_len * (val_rate + train_rate)):]

        q_table=q_table.iloc[test_index]
        q_table.reset_index(inplace=True,)
        q_table['len']=q_table['tra'].apply(lambda x:len(x))
        # random.seed(2)
        # random.shuffle(q_table)
        q_table=q_table.sample(frac=1,random_state=2)

        q_table['bins']=pd.cut(q_table['len'],bins=[0,45,60,75,90],right=False,labels=[0,1,2,3])
        traject_list=[]
        for j in range(4):
            traject_list=traject_list+q_table[q_table['bins']==j].head(k)['idx'].tolist()
        return traject_list






Count_info=[]
if __name__ == '__main__':
    dataset=sys.argv[1]
    # dataset = 'xian'

    config=Config.Config(dataset)
    config.metric = sys.argv[2] #
    # config.metric = 'erp'
    config.eval_model_path= sys.argv[3] #
    # config.eval_model_path = '/root/copy/TRSSQ/TRSSQ/pretrain_model/model_xian/modeledrxian'

    config.method=sys.argv[4]
    # config.method='OSF'
    learning_size=50


    Learning_filter=Learning_filter(config)
    Index_filter=Index_filter(config)
    SSS=subtrajectory_search(config,Learning_filter,Index_filter,OSF,learning_size)

    query_list = SSS.get_data_framework(10)

    query_list=[1,2]
    print(query_list)
    SSS.calAllScore(query_list)

