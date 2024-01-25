from CityDM import CityDM
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default="/home/admin/workspace/yuyuanhong/code/CityLayout/config/new/uniDM_train.yaml")
parser.add_argument("--sample", action="store_true", default=False)
parser.add_argument("--train", action="store_true", default=False)
parser.add_argument("--cond", action="store_true", default=False)
parser.add_argument("--eval", action="store_true", default=False)
parser.add_argument("--best", action="store_true", default=False)
path = parser.parse_args().path


citydm = CityDM(path)

if parser.parse_args().sample:
    citydm.sample(cond=parser.parse_args().cond, eval=parser.parse_args().eval, best=parser.parse_args().best)    

elif parser.parse_args().train:
    citydm.train()

    