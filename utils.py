#usr/bin/env python

import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import accuracy_score, roc_auc_score

REG_METRICS = {
       'r2_score': {'callable': r2_score, 'input': 'raw'},
       'mean_absolute_error': {'callable': mean_absolute_error, 'input': 'raw'},
       'mean_squared_error': {'callable': mean_squared_error, 'input': 'raw'},
       'pearsonr': {'callable': pearsonr, 'input': 'raw'},
       'spearmanr': {'callable': spearmanr, 'input': 'raw'},
}

CLA_METRICS = {
    'accuracy': {'callable': accuracy_score, 'input': 'bin'},
    'roc_auc': {'callable': roc_auc_score, 'input': 'raw'},
}


def compute_regression_metrics(
    true,
    pred,
    metrics=['r2_score', 'mean_absolute_error', 'mean_squared_error', 'pearsonr', 'spearmanr'],
):
    '''compute accuracy metrics for regression problem

    Args:
        true (np.ndarray): array of ground truth regression target values
        pred (np.ndarray): array of regressor predictions
        metircs (list): list of regression metrics to evaluate
    '''
    results = {}
    for metric in metrics:
        print('>>> ', metric)
        met = REG_METRICS[metric]
        if met['input'] == 'raw':
            val = met['callable'](true.ravel(), pred.ravel())
            if metric in ['pearsonr', 'spearmanr']:
                results[metric] = val[0]
            else:
                results[metric] = val
        else:
            print('something is wrong ...')
            quit()

    return results

def compute_feasibility_metrics(
    true,
    pred,
    metrics=['accuracy', 'roc_auc'],
):
    '''compute accuracy metrics for classification problem

    Args:
        true (np.ndarray): array of ground truth binary labels
        pred (np.ndarray): array of raw classifier outputs
        metircs (list): list of classification metrics to evaluate
    '''
    results = {}
    for metric in metrics:
        print('>>> ', metric)
        met = CLA_METRICS[metric]
        if met['input'] == 'raw':
            # the numbers here are reversed (reversed between Gryffin source
            # code and the manuscript)
            pred_transform = 1.0-pred
        elif met['input'] == 'bin':
            # the numbers here are reversed (reversed between Gryffin source
            # code and the manuscript)
            pred_transform = np.where(pred >= 0.5, 0, 1)
        val = met['callable'](true, pred_transform)
        results[metric] = val

    return results


def to_binary(val):
    if np.isnan(val):
        return 1 # infeasible measurement
    else:
        return 0 # feasible measuremnet


def build_datasets(observations, df_results, param_names, obj_names, is_feas_name):
    ''' build the datasets of observed and unobserved inputs for the regression and
    classification problems

    Args:
        observations (list): list of current observations in Gryfiin format
        df_results (pd.DataFrame): lookup dataframe
        param_names (list): list of parameter names
        obj_names (list): list of objective names
        is_feas_name (str): lookup dataframe column with feasibility value
    '''
    # add param names columns for more rapid lookup
    df_results['params'] = df_results[param_names[0]].str.cat(
        [df_results[param_names[i+1]] for i in range(len(param_names)-1)], sep='.',
    )

    # get the set of observations that are feasible
    obs_reg_params = []
    obs_cla_params = [] # all the observations
    obs_reg_values = []
    obs_cla_values = [] # all the observations

    for obs_ix, obs in enumerate(observations):
        if np.isnan(obs[obj_names[0]]): # either all objectives are nan or none are (might change in future)
            pass
        else:
            obs_reg_params.append({param_name:obs[param_name] for param_name in param_names})
            obs_reg_values.append([obs[obj_name] for obj_name in obj_names])
        obs_cla_params.append({param_name:obs[param_name] for param_name in param_names})
        obs_cla_values.append([to_binary(obs[obj_name]) for obj_name in obj_names])

    obs_reg_values = np.array(obs_reg_values)
    obs_cla_values = np.array(obs_cla_values)

    # get the unobserved values
    obs_cla_params_cat = [ '.'.join([obs[p] for p in param_names]) for obs in obs_cla_params]
    obs_reg_params_cat = [ '.'.join([obs[p] for p in param_names]) for obs in obs_reg_params]


    unobs_cla_df = df_results[~(df_results.params.isin(obs_cla_params_cat))] # predict on all remaining points
    unobs_reg_df = df_results[~(df_results.params.isin(obs_reg_params_cat))&(df_results[is_feas_name]==0)] # only predict on feasible points

    unobs_cla_params = [row.to_dict() for _, row in unobs_cla_df[param_names].iterrows()]
    unobs_cla_values = unobs_cla_df[is_feas_name].values

    unobs_reg_params = [row.to_dict() for _, row in unobs_reg_df[param_names].iterrows()]
    unobs_reg_values = unobs_reg_df[obj_names].values

    return {
        'obs_cla_params': obs_cla_params, 'obs_cla_values': obs_cla_values,
        'obs_reg_params': obs_reg_params, 'obs_reg_values': obs_reg_values,
        'unobs_cla_params': unobs_cla_params, 'unobs_cla_values': unobs_cla_values,
        'unobs_reg_params': unobs_reg_params, 'unobs_reg_values': unobs_reg_values,
    }


def make_df_results_synthetic(surface_obj, param_type, num_dims=2, num_opts=21, res=50):
    ''' make a lookup table with all the possible
    param naming convention is  "x0", "x1", "x2", ...
    single objective is always called "obj"
    convention for options is "x_0", "x_1", "x_2", ...

    Args:
        surface_obj (obj): Olympus-like surface object with constraints
        param_type (str): "continuous" ot "categorical"
        num_dims (int): number of parameter space dimensions
        num_opts (int): number of options per parameter space dimension
        res (int): per-dimension resolution for the evalauation grid (for continous surfaces only)

    '''
    if param_type == 'continuous':
        pass

    elif param_type == 'categorical':
        # build the cartesian product space (feas and infeas)
        options = [f'x_{i}' for i in range(num_opts)]
        dim_options = [options for _ in range(num_dims)]
        cart_product = list(itertools.product(*dim_options))

        all_obs = []
        for params in cart_product:
            input = {f'x{i}': p for i, p in enumerate(params)}
            input = surface_obj.eval_merit(input)
            if np.isnan(input['obj']):
                input['is_feas'] = 1
            else:
                input['is_feas'] = 0
            all_obs.append(input)

    else:
        raise NotImplementedError

    return pd.DataFrame(all_obs)
