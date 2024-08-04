import os
import sys
import time

sys.path.append('..')
# from searching.findsubtrajectory import config
# from Invert_index_func import time_cost
from singleQuery import *
from tqdm import tqdm
from tqdm import tqdm
import pandas as pd
tqdm.pandas(desc='pandas bar')
from pandarallel import pandarallel
pandarallel.initialize()
# basePath = config.data_path

# metric = "DTW"
# method = "PSS"



class the_point:
    N=[]
    C=0
    w=0
    v=0
    def __init__(self,n,c):
        self.N = n
        self.C = c

class OSF(object):
    def __init__(self,config):
        self.config=config

    def point2grid(self,point: Tuple[float, float]):
        radius = 0.002
        x = (point[0] - dataDistribution[dataset]["lat"][0]) // radius
        y = (point[1] - dataDistribution[dataset]["lon"][0]) // radius
        return x, y









    def get_candi_osf(self,x, invert_index,rate=0.1,grid_size=0.0005,dataTrajectories=None,metric=''):
        if self.config.dataset=='chengdu':
            eta=0.000242
        elif self.config.dataset=='xian':
            eta=0.00023
        elif self.config.dataset=='porto':
            eta = 0.00079
        if metric=='edr' or metric=='lcss':
            eta = 0.001
            if self.config.dataset=='porto':
                eta=0.003
        # print('=================')

        # c_lis=[]


        candis=set()
        candi_list=[]
        last_val=set()
        temp_time=0
        for i in x['tra']:
            val=(float(i[0]) - 30.72693) // (grid_size), (float(i[1]) - 104.04761) // (grid_size)  # grid_index
            s_time = time.time()
            candi = set()

            if last_val==val:
                continue
            else:
                last_val=val
                for j in [(val[0]-1,val[1]-1),
                          (val[0],val[1]-1),
                          (val[0]+1,val[1]-1),
                          (val[0]-1,val[1]),
                          (val[0],val[1]),
                          (val[0]+1,val[1]),
                          (val[0] - 1, val[1]+1),
                          (val[0], val[1]+1),
                          (val[0] + 1, val[1]+1)
                          ]:
                    try:
                        candi=candi.union(invert_index[j])
                    except:
                        pass
            e_time = time.time()
            temp_time=temp_time+(e_time-s_time)/9
            ##
            f_candi=[] # final candidate
            m_dis = 0.01 # initial  cp
            if candi==None:
                # have no trajecroty in this grid
                print(i)
                # continue
            for k in candi:
                td=np.array(dataTrajectories.iloc[k]['tra'])
                for td_p in td:
                    dis_p=np.linalg.norm(i - td_p)
                    if dis_p<=eta:
                        f_candi.append(k)
                        break
                    else:
                        if dis_p<m_dis:
                            m_dis=dis_p


            candi_list.append([f_candi,m_dis])


        # mincand
        cq=0
        if metric=='edr' or  metric=='lcss' :
            all_cost = sum([1 for i in candi_list]) * rate
            all_point = [the_point(i[0], 1) for i in candi_list]
        else:
            all_cost = sum([i[1] for i in candi_list])* rate
            all_point=[the_point(i[0],i[1]) for i in candi_list]
        s_time=time.time()

        while cq<all_cost:
            for i in range(len(all_point)):
                all_point[i].v=(len(all_point[i].N)-all_point[i].w)/min(all_point[i].C,all_cost-cq)
            all_point.sort(key=lambda x:x.v)
            the_candi=all_point.pop(0)
            for i in range(len(all_point)):
                all_point[i].w=(all_point[i].w+min(all_point[i].C,all_cost-cq)*the_candi.v)

            cq=cq+the_candi.C
            candis = candis.union(the_candi.N)
        e_time=time.time()
        return [candis,e_time-s_time+temp_time]

    def osf_filter(self,query, invert_index,radius=0.1,grid_size=0.0005,dataTrajectories=None,metric=''):
        print('OSF pruning......')
        query['cand'] = query.parallel_apply(lambda x: self.get_candi_osf(x, invert_index,radius,grid_size,dataTrajectories,metric),
                                    axis=1)  # grid_based_filtering
        query['pruning_time']=query['cand'].apply(lambda x:x[1])
        query['cand'] = query['cand'].apply(lambda x: x[0])
        print(query.columns)

        return query






