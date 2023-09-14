#!/usr/bin/env python


import os
import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def plot_parity(y_true, y_pred, title, y_pred_unc=None):

    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)

    r2 = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0]

    plt.plot([axmin, axmax], [axmin, axmax], '--k')

    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, marker='o', markeredgecolor='w', alpha=1, elinewidth=1)

    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))

    ax = plt.gca()
    ax.set_aspect('equal')

    at = AnchoredText(
    f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}\nR2 = {r2:.2f}\nPearson = {pear:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.xlabel('True')
    plt.ylabel('Chemprop Predicted')
    plt.title(title)

    plt.show()

    return



if __name__ == '__main__':

    DIRS = [
        #'pdgf_pic50_checkpoints_reg',
        'kit_pic50_checkpoints_reg'
    ]
    NAMES  = [
        #'pdgf_pic50',
        'kit_pic50',
    ]

    for dir, name in zip(DIRS, NAMES):


        for set_ in ['train', 'test']:
            arguments = [
                '--test_path', f'{name}_checkpoints_reg/fold_0/{set_}_smiles.csv',
                '--preds_path', f'{name}_{set_}_preds_reg.csv',
                '--checkpoint_dir', f'{name}_checkpoints_reg'
            ]

            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args)

            df = pd.read_csv(f'{name}_checkpoints_reg/fold_0/{set_}_full.csv')
            df['preds'] = [x[0] for x in preds]
            plot_parity(df.pic50, df.preds, title=f'{name}\t{set_}')

        # prediction on bcr abl inhibitors from desai et al
        arguments = [
            '--test_path', f'abl_smiles.csv',
            '--preds_path', f'{name}_abl_preds_reg.csv',
            '--checkpoint_dir', f'{name}_checkpoints_reg'
        ]

        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args)
