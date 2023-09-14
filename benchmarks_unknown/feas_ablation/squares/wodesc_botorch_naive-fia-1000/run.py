#!/usr/bin/env python

import sys
sys.path.append('../../../')

import numpy as np
import pandas as pd
import pickle

from atlas.optimizers.gp.planner import GPPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical

from benchmark_functions import save_pkl_file


#----------
# Settings
#----------

feas_strategy = 'fia'
feas_param = 1000.
with_descriptors = False
type_ = 'categorical'

budget = 100


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


all_results = {}

# load in the surfaces
binned_surfaces = pickle.load(open('../surfaces.pkl', 'rb'))
bins = list(binned_surfaces.keys())

for bin_ix, bin in enumerate(bins):

	surfaces = binned_surfaces[bin]['objs']

	all_results[bin] = []

	for surface_ix, surface in enumerate(surfaces):

		param_space = set_param_space(type_=type_)

		planner = GPPlanner(
			goal='minimize',
			feas_strategy=feas_strategy,
			feas_param=feas_param,
			use_descriptors=with_descriptors,
		)
		planner.set_param_space(param_space)

		campaign = Campaign()
		campaign.set_param_space(param_space)


		for num_iter in range(budget):
			print(f'===============================')
			print(f'   Bin {bin_ix+1} -- Surface {surface_ix+1} -- Iteration {num_iter+1}')
			print(f'===============================')

			samples = planner.recommend(campaign.observations)

			sample = samples[0]

			measurement = surface.eval_merit(sample.to_dict())

			campaign.add_observation(sample.to_array(), measurement['obj'])


		# store the results into a DataFrame
		x0_col = campaign.observations.get_params()[:, 0]
		x1_col = campaign.observations.get_params()[:, 1]
		obj_col = campaign.observations.get_values(as_array=True)

		data = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col})

		all_results[bin].append(data)

		# save results to disk
		save_pkl_file(all_results)
