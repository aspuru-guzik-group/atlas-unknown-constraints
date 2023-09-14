#!/usr/bin/env python

import numpy as np
import pandas as pd
import pickle
import os
import shutil

# --------
# Settings
# --------

budget = 270 # evaluate all possible options
repeats = 50

# --------------------
# load tabular results
# --------------------
df_results = pd.read_csv('../../reference-and-data/lookup_table_multiobj.csv')


# ---------
# Functions
# ---------

def run_experiment(param):
	match = df_results.loc[
		(df_results['template_name']==param['template']) &
		(df_results['alkyne_name']==param['alkyne'])
	]
	assert len(match)==1
	if match.loc[:, 'synthesis_success'].to_numpy()[0] == 1:
		# successful synthesis
		return (
			match.loc[:, 'abl1_pIC50'].to_numpy()[0],
			match.loc[:, 'kit_pIC50'].to_numpy()[0],
			match.loc[:, 'pdgf_pIC50'].to_numpy()[0]
		)
	elif match.loc[:, 'synthesis_success'].to_numpy()[0] == 0:
		# failed synthesis
		return np.nan, np.nan, np.nan
	else:
		raise ValueError()

def eval_merit(param):
	abl1_pIC50, kit_pIC50, pdgf_pIC50 = run_experiment(param)
	param['abl1_pIC50'] = abl1_pIC50
	param['kit_pIC50'] = kit_pIC50
	param['pdgf_pIC50'] = pdgf_pIC50
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
template_names = ['8-1', '8-2', '8-3', '8-4', '8-5', '16-1', '16-2', '16-3', '16-4', '19']
alkyne_names  =  ['22-1', '22-2', '22-3', '22-4', '22-5', '22-6', '22-7', '22-8', '22-9',
                  '22-10', '22-11', '22-12', '22-13', '22-14', '22-15', '22-16', '22-17',
                  '22-18', '22-19', '22-20', '22-21', '22-22', '22-23', '22-24', '22-25',
                  '22-26', '22-27']

def build_all_samples():
    all_combinations = []
    for template in template_names:
        for alkyne in alkyne_names:
            combination = [template, alkyne]
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
        param['template'] = sample[0]
        param['alkyne'] = sample[1]

        # evaluate the proposed parameters
        observation = eval_merit(param)

        # append observation
        observations.append(observation)

    # store run results in DataFrame
    data_dict = {}
    for key in observations[0].keys():
        if key == 'pdgf_pIC50':
            tmp_key = 'pdgf_pIC50_col'
            data_dict[tmp_key] = [o[key] for o in observations]
        else:
            data_dict[key] = [o[key] for o in observations]
    data = pd.DataFrame(data_dict)
    data_all_repeats.append(data)

    # save results to disk
    save_pkl_file(data_all_repeats)
