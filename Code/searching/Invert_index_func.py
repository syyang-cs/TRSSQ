import sys
import time
sys.path.append('..')
from Config import Config
import os
import pandas as pd
import pickle
from collections import Counter
# from  findsubtrajectory import config


# scalability
# data_path = config.base_path + "/data_s/"

# data_path = config.base_path + "/data/"

class Index_filter(object):
    def __init__(self,config):
        self.config=config
        self.data_path = config.base_path + "/data/"
        self.invert_index = {}


    def add_index(self,x,grid_size=0.0005):
        for i in x['tra']:
            val=(float(i[0]) - 30.72693) // (grid_size), (float(i[1]) - 104.04761) // (grid_size)
            try:
                self.invert_index[val].add(x['idx'])
            except:
                self.invert_index[val]={x['idx']}

    def run_get_index(self,t_table,grid_size=0.0005):
        print('run_get_index!!')
        if os.path.exists (os.path.join(self.data_path, 'invert_index')+str(grid_size)+self.config.dataset):
            file = open(os.path.join(self.data_path, 'invert_index')+str(grid_size)+self.config.dataset, 'rb')
            exit_invert_index = pickle.load(file)
            file.close()
            print('index_exists    finished!!')
            self.invert_index=exit_invert_index
            return exit_invert_index
        t_table['idx']=t_table.index
        t_table.apply(lambda x:self.add_index(x,grid_size),axis=1)

        file=open(os.path.join(self.data_path, 'invert_index')+str(grid_size)+self.config.dataset,'wb')
        pickle.dump(self.invert_index,file)
        file.close()
        print('run_get_index    finished!!')
    def get_candi(self,x,grid_size=0.0005,num_grid=1):
        candi=[]
        grid_list={((float(i[0]) - 30.72693) // (grid_size), (float(i[1]) - 104.04761) // (grid_size))for i in x['tra']}
        s_time = time.time()
        for i in grid_list:
            try:
                candi = candi + list(self.invert_index[i])
            except:
                pass
        t=Counter(candi)
        candi=[i for i in t if t[i]>num_grid]
        e_time = time.time()


        return [candi,e_time-s_time]

