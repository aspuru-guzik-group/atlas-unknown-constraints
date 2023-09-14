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
from benchmark_functions import HyperEllipsoidConstr as BenchmarkSurface


def set_param_space(type_='continuous'):
	param_space = ParameterSpace()
	if type_ == 'continuous':
		x0 = ParameterContinuous(name='x0', low=0.0, high=1.0)
		x1 = ParameterContinuous(name='x1', low=0.0, high=1.0)
	elif type_ == 'categorical':
		if with_descriptors:
			descriptors = [[float(i), float(i)] for i in range(21)]
		else:
			descriptors = [None for _ in range(21)]
		x0 = ParameterCategorical(
			name='x0',
			options=[f'x_{i}' for i in range(21)],
			descriptors=descriptors,
		)
		x1 = ParameterCategorical(
			name='x1',
			options=[f'x_{i}' for i in range(21)],
			descriptors=descriptors,
		)
	param_space.add(x0)
	param_space.add(x1)

	return param_space


budget = 50
repeats = 40
random_seed=None
num_init_design=10
init_design_strategy='random'

use_min_filter = False

surface = BenchmarkSurface()

feas_strategy='fwa'
feas_param=0.0

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
			feas_strategy=feas_strategy,
			feas_param=feas_param,
			num_init_design=num_init_design,
			init_design_strategy=init_design_strategy,
			acquisition_type='ucb',
			acquisition_optimizer_kind='gradient',
            use_min_filter=use_min_filter,
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

			measurement = surface.eval_merit(sample.to_dict())

			campaign.add_observation(sample.to_array(), measurement['obj'])

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



if __name__ == '__main__':
	run_gradient()
