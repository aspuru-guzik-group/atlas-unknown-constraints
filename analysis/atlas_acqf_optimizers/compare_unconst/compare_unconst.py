#!/usr/bin/env python3

import sys
sys.path.append('../../../benchmarks_unknown/')

import numpy as np
import pandas as pd
import pickle

from atlas.optimizers.gp.planner import GPPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical
from olympus.surfaces import Surface

from benchmark_functions import save_pkl_file, load_data_from_pkl_and_continue


surface = Surface(kind='StyblinskiTang')

budget = 100
repeats = 40
random_seed=None

data_all_repeats_gradient = []
data_all_repeats_genetic = []



#-------------------------
# gradient acqf optimizer
#-------------------------

def run_gradient():

	for num_repeat in range(repeats):
		timings = []
		planner = GPPlanner(
			goal='minimize',
			num_init_design=5,
			init_design_strategy='random',
			acquisition_type='ucb',
			acquisition_optimizer_kind='gradient',
		)

		planner.set_param_space(surface.param_space)

		campaign = Campaign()
		campaign.set_param_space(surface.param_space)

		for num_iter in range(budget):
			print(f'===============================')
			print(f'   Repeat {num_repeat+1} -- Iteration {num_iter+1}')
			print(f'===============================')

			samples = planner.recommend(campaign.observations)

			sample = samples[0]

			measurement = surface.run(sample)[0][0]
			print('measurement : ', measurement  )

			campaign.add_observation(sample, measurement)

			# print(timings)
			if not hasattr(planner, 'timings_dict'):
				# print('no timings dict yet')
				timings.append(0.)

			else:
				# print(planner.timings_dict)
				timings.append(planner.timings_dict['acquisition_opt'])


		# store the results into a DataFrame
		x0_col = campaign.observations.get_params()[:, 0]
		x1_col = campaign.observations.get_params()[:, 1]
		obj_col = campaign.observations.get_values(as_array=True)

		data = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col, 'timings': timings})
		data_all_repeats_gradient.append(data)

		# save results to disk
		with open('results_gradient.pkl', 'wb') as file:
			pickle.dump(data_all_repeats_gradient, file)

#------------------------
# genetic acqf optimizer
#------------------------

def run_genetic():
	for num_repeat in range(repeats):
		timings = []
		planner = GPPlanner(
			goal='minimize',
			num_init_design=5,
			init_design_strategy='random',
			acquisition_type='ucb',
			acquisition_optimizer_kind='genetic',
		)

		planner.set_param_space(surface.param_space)

		campaign = Campaign()
		campaign.set_param_space(surface.param_space)

		for num_iter in range(budget):
			print(f'===============================')
			print(f'   Repeat {num_repeat+1} -- Iteration {num_iter+1}')
			print(f'===============================')

			samples = planner.recommend(campaign.observations)

			sample = samples[0]

			measurement = surface.run(sample)[0][0]

			print(f'\nSAMPLE : {sample}\tMEASUREMENT : {measurement}\n')

			campaign.add_observation(sample, measurement)

			# print(timings)
			if not hasattr(planner, 'timings_dict'):
				# print('no timings dict yet')
				timings.append(0.)
			else:
				# print(planner.timings_dict)
				timings.append(planner.timings_dict['acquisition_opt'])


		# store the results into a DataFrame
		x0_col = campaign.observations.get_params()[:, 0]
		x1_col = campaign.observations.get_params()[:, 1]
		obj_col = campaign.observations.get_values(as_array=True)

		data = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col, 'timings': timings})
		data_all_repeats_genetic.append(data)

		# save results to disk
		with open('results_genetic.pkl', 'wb') as file:
			pickle.dump(data_all_repeats_genetic, file)



if __name__ == '__main__':
	run_gradient()
	run_genetic()
