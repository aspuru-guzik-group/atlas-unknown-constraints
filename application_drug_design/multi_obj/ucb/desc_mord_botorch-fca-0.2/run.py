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
from olympus.scalarizers import Scalarizer

from atlas.optimizers.gp.planner import GPPlanner


#----------
# Settings
#----------

feas_strategy = 'fca'
feas_param = 0.2
with_descriptors = True
descriptors_type = 'mord'  # 'mord' or 'pca'
type_ = 'categorical'

budget = 270
repeats = 50
random_seed = None # i.e. use a different random seed each time

#----------------------
# load tabular results
#----------------------
df_results = pd.read_csv('../../reference-and-data/lookup_table_multiobj.csv')

#------------------
# helper functions
#------------------

def run_experiment(param):
	match = df_results.loc[
		(df_results['template_name']==param['template_name']) &
		(df_results['alkyne_name']==param['alkyne_name'])
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


def get_descriptors(frag_name, desc_names, desc):
	desc_vec = []
	for desc_name in desc_names:
		val = desc[desc['name']==frag_name][desc_name].tolist()[0]
		desc_vec.append(val)

	return desc_vec


# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

def set_param_space(type_='categorical'):
	param_space = ParameterSpace()

	if type_ == 'continuous':
		raise ValueError()

	elif type_ == 'categorical':
		template_names = ['8-1', '8-2', '8-3', '8-4', '8-5', '16-1', '16-2', '16-3', '16-4', '19']
		alkyne_names  =  ['22-1', '22-2', '22-3', '22-4', '22-5', '22-6', '22-7', '22-8', '22-9',
						  '22-10', '22-11', '22-12', '22-13', '22-14', '22-15', '22-16', '22-17',
						  '22-18', '22-19', '22-20', '22-21', '22-22', '22-23', '22-24', '22-25',
						  '22-26', '22-27']

		# scramble order of options
		np.random.shuffle(template_names)
		np.random.shuffle(alkyne_names)

		if with_descriptors:
			# load the descriptors
			desc_template_pca = pd.read_csv('../../reference-and-data/descriptors/template_pca_desc.csv')
			desc_alkyne_pca = pd.read_csv('../../reference-and-data/descriptors/alkyne_pca_desc.csv')

			desc_template_mord = pd.read_csv('../../reference-and-data/descriptors/template_mord_desc.csv')
			desc_alkyne_mord = pd.read_csv('../../reference-and-data/descriptors/alkyne_mord_desc.csv')

			# "intuitive" descriptors selected from mordred
			mord_desc_names = [
				'nHeavyAtom', 'nHetero',
				'C1SP2', 'C2SP2', 'C3SP2', 'C1SP3', 'SLogP', 'HybRatio', 'nHBDon',
				'apol', 'bpol', 'nRot', 'RotRatio', 'TopoPSA', 'Diameter', 'Radius', 'MW',
			]
			num_pca_template = 10
			num_pca_alkyne = 20

			if descriptors_type == 'mord':
				# generate categorty details
				descriptors_templates = [get_descriptors(n, mord_desc_names, desc_template_mord) for n in template_names]
				descriptors_alkynes = [get_descriptors(n, mord_desc_names, desc_alkyne_mord) for n in alkyne_names]

			elif descriptors_type == 'pca':
				#raise NotImplementedError
				descriptors_templates = []
				for name in template_names:
					descriptors_templates.append(
						get_descriptors(
							name,
							[f'pc_{i}' for i in range(num_pca_template)],
							desc_template_pca
						)
					)
				descriptors_alkynes = []
				for name in alkyne_names:
					descriptors_alkynes.append(
						get_descriptors(
							name,
							[f'pc_{i}' for i in range(num_pca_alkyne)],
							desc_alkyne_pca
						)
					)

			else:
				raise NotImplementedError

		else:
			# scramble order of options
			np.random.shuffle(template_names)
			np.random.shuffle(alkyne_names)

			descriptors_templates = [None for _ in template_names]
			descriptors_alkynes  = [None for _ in alkyne_names]

		x0 = ParameterCategorical(
			name='template_name',
			options=template_names,
			descriptors=descriptors_templates,
		)
		x1 = ParameterCategorical(
			name='alkyne_name',
			options=alkyne_names,
			descriptors=descriptors_alkynes,
		)
	param_space.add(x0)
	param_space.add(x1)

	return param_space



for num_repeat in range(missing_repeats):

	# define parameter space
	param_space = set_param_space(type_=type_)

	# define the objective space
	value_space = ParameterSpace()
	value_space.add(ParameterContinuous(name='abl1_pIC50'))
	value_space.add(ParameterContinuous(name='kit_pIC50'))
	value_space.add(ParameterContinuous(name='pdgf_pIC50'))


	planner = GPPlanner(
		goal='minimize', # minimize the merit
		feas_strategy=feas_strategy,
		feas_param=feas_param,
		use_min_filter=True,
		use_descriptors=with_descriptors,
		acquisition_type='ucb',
		acquisition_optimizer_kind='gradient',
		num_init_design=10,
		is_moo=True,
		value_space=value_space,
		scalarizer_kind='Hypervolume',
		moo_params={},
		goals=['max', 'max', 'max'],
	)
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)
	campaign.set_value_space(value_space)

	for num_iter in range(budget):
		print(f'===============================')
		print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {num_iter+1}')
		print(f'===============================')

		samples = planner.recommend(campaign.observations)

		sample = samples[0]

		measurement = eval_merit(sample)

		meas_param_vect = ParameterVector().from_dict(
			{'abl1_pIC50': measurement['abl1_pIC50'], 'kit_pIC50': measurement['kit_pIC50'], 'pdgf_pIC50': measurement['pdgf_pIC50']}
		)
		campaign.add_observation(sample, meas_param_vect)

		print(f'\nSAMPLE : {sample}\nMEASUREMENT : {meas_param_vect}\n')


	# store the results into a DataFrame
	template_col = campaign.observations.get_params()[:, 0]
	alkyne_col = campaign.observations.get_params()[:, 1]
	abl1_pIC50_col = campaign.observations.get_values(as_array=True)[:,0]
	kit_pIC50_col = campaign.observations.get_values(as_array=True)[:,1]
	pdgf_pIC50_col = campaign.observations.get_values(as_array=True)[:,2]

	data = pd.DataFrame({
		'template': template_col, 'alkyne': alkyne_col, 'abl1_pIC50': abl1_pIC50_col,
		'kit_pIC50': kit_pIC50_col, 'pdgf_pIC50_col': pdgf_pIC50_col,
	})
	data_all_repeats.append(data)

	# save results to disk
	save_pkl_file(data_all_repeats)
