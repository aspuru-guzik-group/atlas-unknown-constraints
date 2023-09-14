#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os
import shutil
from gryffin import Gryffin

# --------
# Settings
# --------

# no budget, we go until one good HOIP is found
repeats = 200

# --------------------
# load tabular results
# --------------------
df_results = pd.read_csv('../reference-and-data/df_results.csv')

# ---------
# Functions
# ---------
def run_experiment(param):
    match = df_results.loc[(df_results['molcat'] == param['molcat']) &
                           (df_results['metal'] == param['metal']) &
                           (df_results['halogen'] == param['halogen'])]
    assert len(match) in [1, 0]
    if len(match) == 0:
        return np.nan, np.nan
    elif len(match) == 1:
        bandgap = np.abs(match.loc[:, 'bandgap'].to_numpy()[0] - 1.25)
        m_star = match.loc[:, 'm_star'].to_numpy()[0]
        return bandgap, m_star
    else:
        raise ValueError()


def eval_merit(param):
    bandgap, m_star = run_experiment(param)
    param['bandgap'] = bandgap
    param['m_star'] = m_star
    return param


def save_pkl_file(data_all_repeats):
    """save pickle file with results so far"""

    if os.path.isfile('results.pkl'):
        shutil.move('results.pkl', 'bkp-results.pkl')  # overrides existing files

    # store run results to disk
    with open("results.pkl", "wb") as content:
        pickle.dump(data_all_repeats, content)



def load_data_from_pkl_and_continue(N):
    """load results from pickle file"""
    data_all_repeats = []
    # if no file, then we start from scratch/beginning
    if not os.path.isfile('results.pkl'):
        return data_all_repeats, N

    # else, we load previous results and continue
    with open("results.pkl", "rb") as content:
        data_all_repeats = pickle.load(content)

    missing_N = N - len(data_all_repeats)

    return data_all_repeats, missing_N

# ------
# Config
# ------
molcats = ['H3S', 'NH4', 'MS', 'MA', 'MP', 'FA', 'EA', 'G', 'AA', 'ED', 'tBA']
metals = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag',
          'Au', 'Zn', 'Cd', 'Hg', 'Ga', 'In', 'Tl', 'Si', 'Ge', 'Sn', 'Pb', 'Bi', 'S', 'Se', 'Te']
halogens = ['F', 'Cl', 'Br', 'I']


def build_all_samples():
    all_combinations = []
    for mol in molcats:
        for met in metals:
            for hal in halogens:
                combination = [mol, met, hal]
                all_combinations.append(combination)
    return all_combinations


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


for num_repeat in range(missing_repeats):
    print(f'   Repeat {len(data_all_repeats)+1}')

    samples = build_all_samples()

    np.random.shuffle(samples)

    observations = []
    for sample in samples:

        param = {}
        param['molcat'] = sample[0]
        param['metal'] = sample[1]
        param['halogen'] = sample[2]

        # evaluate the proposed parameters
        observation = eval_merit(param)

        # append observation
        observations.append(observation)

        # if we find one perovskite with the right targets, we are done
        if observation['bandgap'] < 0.5 and  observation['m_star'] < 4.:
            break

    # store run results into a DataFrame
    data_dict = {}
    for key in observations[0].keys():
        data_dict[key] = [o[key] for o in observations]
    data = pd.DataFrame(data_dict)
    data_all_repeats.append(data)

    # save results to disk
    save_pkl_file(data_all_repeats)
