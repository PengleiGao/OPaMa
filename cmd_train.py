import os

cmd='python main.py -b configs/train_outdoor_mamba.yaml --name mamba_outdoor'


#cmd='python main.py -b configs/train_indoor_mamba.yaml --name mamba_indoor'

os.system(cmd)
