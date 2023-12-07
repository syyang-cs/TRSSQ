import sys

import pandas as pd

sys.path.append('models')
import get_data
import torch.utils.data as Data
from utils import train_model,show_info,setup_seed,analizeResult
import Config
from get_data import collate_fn,collate_fn_pair
import utils
import torch
import warnings
warnings.filterwarnings('ignore')
from models import RSSE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
setup_seed()
if __name__ == '__main__':
    dataset=sys.argv[1]

    config = Config.Config(dataset)
    config.metric=sys.argv[2]
    config.model=sys.argv[3]
    config.pair_p=float(sys.argv[4])
    config.num_hiddens = int(sys.argv[5])
    # ==data and dateset

    show_info(config)


    if config.model=='RSSE':
        model = RSSE.Siamese_RSSE(
            config.embedding_dim,
            config.num_hiddens,
            config.num_layers
        )
        score_table, t_table, q_table = get_data.get_data(config)


        train_score_table, val_score_table, test_score_table = get_data.split_data_pair(score_table,config)



        train_dataset = get_data.get_Dataset_pair(train_score_table, t_table, q_table,task=config.task)
        val_dataset = get_data.get_Dataset_pair(val_score_table, t_table, q_table,task=config.task)
        test_dataset = get_data.get_Dataset(test_score_table, t_table, q_table,task=config.task)
        train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn_pair,
            drop_last=True,
            pin_memory=True

        )
        val_loader = Data.DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn_pair,
            drop_last=True,
            pin_memory=True

        )
        test_dataloader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            drop_last=False,
            pin_memory=True

        )












    show_info(config)

    model = train_model(model, train_loader, val_loader, config)
    eva_data=utils.evaluate(model, test_score_table,test_dataloader,config)

    result_list,columns_list= analizeResult(eva_data)
    result_list.append(config.dataset),columns_list.append('dataset')
    result_list.append(config.metric),columns_list.append('metric')
    result_list.append(config.model), columns_list.append('model')
    result_list=pd.DataFrame(result_list).T
    result_list.columns=columns_list
    result_list.to_csv('/root/copy/subtra/5_1_result/score/'+config.dataset+'_'+config.metric+'_'+config.model+'_'+str(config.num_hiddens)+'_'+str(config.pair_p),index=False)

