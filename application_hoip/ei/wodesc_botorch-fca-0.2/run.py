#!/usr/bin/env python

import os, sys
import shutil
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical, ParameterVector

from atlas.optimizers.gp.planner import GPPlanner


#----------
# Settings
#----------

feas_strategy = 'fca'
feas_param = 0.2
with_descriptors = False
descriptors_type = None
type_ = 'categorical'

budget = 1276
repeats = 100
random_seed = None # i.e. use a different random seed each time


#----------------------
# load tabular results
#----------------------
df_results = pd.read_csv('../reference-and-data/df_results.csv')

#------------------
# helper functions
#------------------

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


def get_descriptors(component, desc):
	''' generate a list of descritpors for the particular component
	'''
	desc_vec = []
	for key in desc.keys():
		desc_vec.append(desc[key][component])
	return list(np.array(desc_vec).astype(np.float))



# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


def set_param_space(type_='categorical'):
	param_space = ParameterSpace()

	if type_ == 'continuous':
		raise ValueError()

	elif type_ == 'categorical':
		molcats = ['H3S', 'NH4', 'MS', 'MA', 'MP', 'FA', 'EA', 'G', 'AA', 'ED', 'tBA']
		metals = ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Pd', 'Pt', 'Cu', 'Ag',
				  'Au', 'Zn', 'Cd', 'Hg', 'Ga', 'In', 'Tl', 'Si', 'Ge', 'Sn', 'Pb', 'Bi', 'S', 'Se', 'Te']
		halogens = ['F', 'Cl', 'Br', 'I']

		if with_descriptors:
			# load the descriptors
			with open('../reference-and-data/descriptors/desc_molcats.pkl', 'rb') as content:
				desc_molcats = pickle.load(content)
			with open('../reference-and-data/descriptors/desc_metals.pkl', 'rb') as content:
				desc_metals = pickle.load(content)
			with open('../reference-and-data/descriptors/desc_halogens.pkl', 'rb') as content:
				desc_halogens = pickle.load(content)

			# generate categorty details
			descriptors_molcats = [get_descriptors(x, desc_molcats) for x in molcats]
			descriptors_metals = [get_descriptors(x, desc_metals) for x in metals]
			descriptors_halogens = [get_descriptors(x, desc_halogens) for x in halogens]

		else:
			descriptors_molcats = [None for _ in molcats]
			descriptors_metals = [None for _ in metals]
			descriptors_halogens = [None for _ in halogens]

		x0 = ParameterCategorical(
			name='molcat',
			options=molcats,
			descriptors=descriptors_molcats,
		)
		x1 = ParameterCategorical(
			name='metal',
			options=metals,
			descriptors=descriptors_metals,
		)
		x2 = ParameterCategorical(
			name='halogen',
			options=halogens,
			descriptors=descriptors_halogens,
		)
	param_space.add(x0)
	param_space.add(x1)
	param_space.add(x2)

	value_space = ParameterSpace()
	value_space.add(
		ParameterContinuous(name='bandgap')
	)
	value_space.add(
		ParameterContinuous(name='m_star')
	)
	return param_space, value_space


num_repeat = 0
while num_repeat < missing_repeats:

	# try:

	param_space, value_space = set_param_space(type_=type_)

	planner = GPPlanner(
		goal='minimize', # minimize the Chimera merit
		feas_strategy=feas_strategy,
		feas_param=feas_param,
		use_min_filter=True,
		use_descriptors=with_descriptors,
		acquisition_type='ei',
		acquisition_optimizer_kind='gradient',
		vgp_iters=2000,
		vgp_lr=0.1,
		# moo
		is_moo=True,
		goals=['min', 'min'],
		moo_params={
			'tolerances':[0.5, 4.],
			'absolutes':[True, True],
		},
		value_space=value_space,
		scalarizer_kind='Chimera',
	)

	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	iter = 0
	converged = False
	while len(campaign.observations.get_values()) < budget and not converged:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			print(f'===============================')
			print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {iter+1}')
			print(f'===============================')

			measurement = eval_merit(sample)

			measurement = ParameterVector().from_dict(
				{'bandgap': measurement['bandgap'], 'm_star': measurement['m_star']}
			)

			campaign.add_observation(sample.to_array(), measurement)

			if type_ == 'categorical':
				# if we observe one perovskite with the right targets, we are done
				if measurement['bandgap'] < 0.5 and measurement['m_star'] < 4.:
					print(f'FOUND GOOD MATERIAL AFTER {iter+1} ITERATIONS!')
					converged = True
					break
				
			iter+=1


	# store the results into a DataFrame
	molcat_col = campaign.observations.get_params()[:, 0]
	metal_col = campaign.observations.get_params()[:, 1]
	halogen_col = campaign.observations.get_params()[:, 2]

	bandgap_col = campaign.observations.get_values(as_array=True)[:,0]
	m_star_col  = campaign.observations.get_values(as_array=True)[:, 1]


	data = pd.DataFrame(
		{'molcat': molcat_col, 'metal': metal_col, 'halogen': halogen_col,
		 'bandgap': bandgap_col, 'm_star': m_star_col},
	)
	data_all_repeats.append(data)

	# save results to disk
	save_pkl_file(data_all_repeats)

	num_repeat += 1

	# except Exception as e:
	# 	# if something goes wrong, print the error, skip the iteration and
	# 	# continue on with a different random seed
	# 	print('error : ', e)
