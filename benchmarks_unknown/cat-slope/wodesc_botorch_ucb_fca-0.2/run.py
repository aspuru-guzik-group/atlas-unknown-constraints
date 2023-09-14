#!/usr/bin/env python

import sys
sys.path.append('../../')

import numpy as np
import pandas as pd
import pickle

from atlas.optimizers.gp.planner import GPPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical

from benchmark_functions import save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import CatSlopeConstr as BenchmarkSurface

#----------
# Settings
#----------

feas_strategy = 'fca'
feas_param = 0.2
with_descriptors = False
type_ = 'categorical'

budget = 442
repeats = 40
random_seed = None # i.e. use a different random seed each time
surface = BenchmarkSurface()

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)


def set_param_space(type_='continuous'):
	param_space = ParameterSpace()
	if type_ == 'continuous':
		x0 = ParameterContinuous(name='x0', low=0.0, high=1.0)
		x1 = ParameterContinuous(name='x1', low=0.0, high=1.0)
	elif type_ == 'categorical':
		options_order = np.arange(21)
		np.random.shuffle(options_order)
		if with_descriptors:
			descriptors = []
			for i in options_order:
				descriptors.append( [ float(i) for _ in  range(num_descriptors) ] )

		else:
			descriptors = [None for _ in options_order]
		x0 = ParameterCategorical(
			name='x0',
			options=[f'x_{i}' for i in options_order],
			descriptors=descriptors,
		)
		x1 = ParameterCategorical(
			name='x1',
			options=[f'x_{i}' for i in options_order],
			descriptors=descriptors,
		)
	param_space.add(x0)
	param_space.add(x1)

	return param_space



for num_repeat in range(missing_repeats):

	param_space = set_param_space(type_=type_)

	planner = GPPlanner(
		goal='minimize',
		feas_strategy=feas_strategy,
		feas_param=feas_param,
		use_min_filter=True,
		use_descriptors=with_descriptors,
		acquisition_type='ucb',
		acquisition_optimizer_kind='gradient',
		num_init_design=10,
		vgp_iters=1000,
		vgp_lr=0.1,
	)
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	if type_ == 'categorical':
		OPT = surface.best
		print('OPTIMUM : ', OPT)

	iter=0
	while len(campaign.observations.get_values()) < budget:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			print(f'===============================')
			print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {iter+1}')
			print(f'===============================')

			measurement = surface.eval_merit(sample.to_dict())

			campaign.add_observation(sample.to_array(), measurement['obj'])

			if type_ == 'categorical':
				if (sample.to_dict()['x0'], sample.to_dict()['x1']) == OPT:
					print(f'FOUND OPTIMUM AFTER {iter+1} ITERATIONS!')
					break
			
			iter+=1


	# store the results into a DataFrame
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	obj_col = campaign.observations.get_values(as_array=True)

	data = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col})
	data_all_repeats.append(data)

	# save results to disk
	save_pkl_file(data_all_repeats)
