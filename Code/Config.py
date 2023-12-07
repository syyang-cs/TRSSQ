import os
import torch
class Config(object):
    def __init__(self,dataset):



        self.dataset= dataset # chengdu xian porto
        self.metric='edr' # edr erp dtw
        self.model='TMN'   #T2V,km_test_model,T3S,TMN
        self.task='subtra' # subtra tra

        self.query_min_len = 30 # 10 - 100
        self.query_max_len = 89
        self.data_min_len = 90 # 
        self.data_max_len= 300

        self.train_rate=0.4 # 模型参数
        self.val_rate=0.05

        # path
        self.base_path=r'/root/copy/subtra/'
        self.data_path = self.base_path + 'data' + '/' + self.dataset
        self.result_file = self.base_path+"5_1_result/result_"+self.dataset+'/'
        self.model_path=self.base_path+'5_1_result/'+'model_'+self.dataset+'/'
        self.score_path = self.base_path + '5_1_result/' + 'score_' + self.dataset + '/'





        # train_parameter
        self.epoch = 100


        self.num_workers = 0
        self.patience = 5
        self.pair=True



        # data_parameter


        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 正则化有关参数
        self.x_m=0
        self.x_std=0
        self.y_max = 0
        self.y_std = 0




        # model_parameter
        self.batch_size = 512
        self.embedding_dim = 2
        self.num_hiddens = 64
        self.num_layers = 3
        self.bi=2
