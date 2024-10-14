import multiprocessing
import sys

import pandas as pd
sys.path.append('models')
import get_data
import torch.utils.data as Data
from utils import train_model,show_info,setup_seed,analizeResult
import Config
from get_data import collate_fn,collate_fn_pair,collate_fn_T3S,collate_fn_T3S_s
import utils
import torch
import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models import T3S


if __name__ == '__main__':

    dataset=sys.argv[1]

    config = Config.Config(dataset)
    config.metric=sys.argv[2]
    config.model=sys.argv[3]
    config.pair_p=float(sys.argv[4])
    config.num_hiddens = int(sys.argv[5])
    # ==data and dateset
    print(config)
    model = T3S.T3S(
        config
    )
    show_info(config)

    # ==data and dateset
    score_table, t_table, q_table,t_grid, q_grid = get_data.get_data_grid(config)

    train_score_table, val_score_table, test_score_table = get_data.split_data_pair(score_table, config)
    train_dataset = get_data.get_Dataset_grid_pair(train_score_table, t_table, q_table, t_grid,
                                              q_grid)
    val_dataset = get_data.get_Dataset_grid_pair(val_score_table, t_table, q_table, t_grid, q_grid)
    test_dataset = get_data.get_Dataset_grid(test_score_table, t_table, q_table, t_grid, q_grid, task=config.task)









    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_T3S,
        drop_last=True

    )
    val_loader = Data.DataLoader(
        dataset=val_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_T3S,
        drop_last=True

    )

    test_dataloader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn_T3S_s,
        drop_last=False,
        pin_memory=True

    )

    model = train_model(model, train_loader, val_loader, config)
    eva_data=utils.evaluate(model, test_score_table,test_dataloader,config)

    result_list,columns_list= analizeResult(eva_data)
    result_list.append(config.dataset),columns_list.append('dataset')
    result_list.append(config.metric),columns_list.append('metric')
    result_list.append(config.model), columns_list.append('model')
    result_list=pd.DataFrame(result_list).T
    result_list.columns=columns_list
    result_list.to_csv('/root/copy/subtra/5_1_result/score/'+config.dataset+'_'+config.metric+'_'+config.model+'_'+str(config.num_hiddens)+'_'+str(config.pair_p),index=False)
