from CityDM import CityDM
import argparse
import os
import wandb

def train_main(sweep_config):
    citydm = CityDM(path, sweep_config=sweep_config)
    citydm.train()


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/uniDM_train.yaml")
parser.add_argument("--sample", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--cond", action="store_true", default=False)
parser.add_argument("--eval", action="store_true", default=False)
parser.add_argument("--best", action="store_true", default=False)
parser.add_argument("--sweep", action="store_true", default=False)
parser.add_argument("--sweep_id", type=str, default=None)

if parser.parse_args().sweep:
    sweep_config_train = {
        'method': 'bayes',
        'metric': {
            'name': 'val.FID',
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'min': 0.00005,
                'max': 0.0005
            },
            'objective': {
                'values': ['pred_v', 'pred_noise']
            },
            'beta_schedule':
            {
                'values': ['sigmoid', 'linear', 'cosine']
            },
            'timesteps': {
                'values': [2000, 2500, 3000, 3500]
            },
            'self_condition': {
                'values': [False]
            },
            'opt': {
                'values': ['adam']
            },
            'scheduler': {
                'values': ['cosine', 'step']
            },
        },
        'early_terminate': {
            'type': 'hyperband',
            's': 2,
            'eta': 3,
            'max_iter': 27
        }
    }

else: 
    sweep_config_train = None
    sweep_config_sample = None

path = parser.parse_args().path

if not parser.parse_args().sweep:
    citydm = CityDM(path)

    if parser.parse_args().sample:
        citydm.sample(cond=parser.parse_args().cond, eval=parser.parse_args().eval, best=parser.parse_args().best)    

    elif parser.parse_args().train:
        citydm.train()

else:
    if parser.parse_args().train:
        if parser.parse_args().sweep_id is None:
            sweep_id = wandb.sweep(sweep_config_train, project="CityLayout", entity="913217005")
        else:
            sweep_id = parser.parse_args().sweep_id
        wandb.agent(sweep_id, function=lambda: train_main(sweep_config_train))

    elif parser.parse_args().sample:
        if parser.parse_args().sweep_id is None:
            sweep_id = wandb.sweep(sweep_config_sample, project="CityLayout", entity="913217005")
        else:
            sweep_id = parser.parse_args().sweep_id
        wandb.agent(sweep_id, function=lambda: train_main(sweep_config_sample))


        

    