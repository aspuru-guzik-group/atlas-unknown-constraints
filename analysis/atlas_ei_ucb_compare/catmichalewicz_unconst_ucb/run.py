#!/usr/bin/env python

import sys
sys.path.append('../../../benchmarks_unknown/')

import numpy as np
import pandas as pd
import pickle

from atlas.optimizers.gp.planner import GPPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical

from benchmark_functions import save_pkl_file, load_data_from_pkl_and_continue
# from benchmark_functions import CatMichalewiczConstr as BenchmarkSurface
from olympus.surfaces import Surface

#----------
# Settings
#----------

feas_strategy = 'naive-0'
feas_param = 0.
with_descriptors = True
num_descriptors = 2
type_ = 'categorical'
num_init_design = 10
acquisition_type = 'ucb'

budget = 442
repeats = 40
random_seed = None # i.e. use a different random seed each time
surface = Surface(kind='CatMichalewicz')

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


def set_param_space(type_='continuous'):
	param_space = ParameterSpace()
	if type_ == 'continuous':
		x0 = ParameterContinuous(name='x0', low=0.0, high=1.0)
		x1 = ParameterContinuous(name='x1', low=0.0, high=1.0)
	elif type_ == 'categorical':
		if with_descriptors:
			descriptors = []
			for i in range(21):
				descriptors.append( [ float(i) for _ in  range(num_descriptors) ] )

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


for num_repeat in range(missing_repeats):

	# param_space = set_param_space(type_=type_)
	param_space = surface.param_space

	planner = GPPlanner(
		goal='minimize',
		feas_strategy=feas_strategy,
		feas_param=feas_param,
		num_init_design=num_init_design,
		init_design_strategy='random',
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind='gradient',
		use_descriptors=with_descriptors,
	)

	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	if type_ == 'categorical':
		OPT = surface.minima


	for num_iter in range(budget):
		print(f'===============================')
		print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {num_iter+1}')
		print(f'===============================')

		samples = planner.recommend(campaign.observations)

		sample = samples[0]

		measurement = surface.run(sample)[0][0]

		campaign.add_observation(sample, measurement)

		if type_ == 'categorical':

			#if (sample.to_dict()['x0'], sample.to_dict()['x1']) == OPT:
			if (sample.to_dict()['param_0'], sample.to_dict()['param_1']) == tuple(OPT[0]['params']):
				print(f'FOUND OPTIMUM AFTER {num_iter+1} ITERATIONS!')
				break


	# store the results into a DataFrame
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	obj_col = campaign.observations.get_values(as_array=True)

	data = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col})
	data_all_repeats.append(data)

	# save results to disk
	save_pkl_file(data_all_repeats)
