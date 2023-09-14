#!/usr/bin/env python

import os
import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.decomposition import PCA



if __name__ == '__main__':

    DIRS = [
        'pdgf_pic50_checkpoints_reg',
        'kit_pic50_checkpoints_reg'
    ]
    NAMES  = [
        'pdgf_pic50',
        'kit_pic50',
    ]

    for dir, name in zip(DIRS, NAMES):

        arguments = [
            '--data_path', f'{name}.csv',
            '--dataset_type', 'regression',
            '--save_dir', f'{name}_checkpoints_reg',
            '--epochs', '30',
            '--config_path', f'config/{name}_config.json',
            '--save_smiles_splits'
        ]

        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

        print(mean_score, std_score)
