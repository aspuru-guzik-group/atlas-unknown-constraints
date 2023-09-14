#!/usr/bin/env python

import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import pickle



from atlas.optimizers.gp.planner import GPPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical

from benchmark_functions import save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import CatCamelConstr as BenchmarkSurface
#from benchmark_functions import CatMichalewiczConstr as BenchmarkSurface

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from atlas.optimizers.utils import forward_normalize

#----------
# Settings
#----------


feas_strategy = 'fwa'
feas_param = 0.0
with_descriptors = True
type_ = 'categorical'
plot=True

budget = 442
repeats = 1
random_seed = None # i.e. use a different random seed each time
surface = BenchmarkSurface()



def plot_reg_surrogate_cat(res_df, planner, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_x0 = [f'x_{i}' for i in range(21)]
	domain_x1 = [f'x_{i}' for i in range(21)]

	Z = np.zeros((len(domain_x0), len(domain_x0)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_x0):
		for y_index, y in enumerate(domain_x1):
			z, _ = planner.reg_surrogate([[x, y]], return_np=True)
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z[0][0])
			Z[x_index, y_index] = z[0][0]


	#Z = np.log(Z)

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]


	x_tick_labels = domain_x0
	y_tick_labels = domain_x1

	ax.set_xticks(np.arange(len(domain_x0)))
	ax.set_yticks(np.arange(len(domain_x1)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('x0', fontsize=12)
	ax.set_ylabel('x1', fontsize=12)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	# # plot observations thus far
	# X_str_ = res_df.loc[:, ['template', 'alkyne']]#[:20]#[:-1] # remove last data point
	# print(X_str_)
	# X_  = pd.DataFrame([[domain_x0.index(x[0]), domain_x0.index(x[1])] for x in X_str_.to_numpy() ],
	# 	columns=X_str_.columns
	# 				  )
	#
	# ax.scatter(X_['alkyne'], X_['template'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)


def plot_cla_surrogate_cat(res_df, planner, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_x0 = [f'x_{i}' for i in range(21)]
	domain_x1 = [f'x_{i}' for i in range(21)]


	Z = np.zeros((len(domain_x0), len(domain_x1)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_x0):
		for y_index, y in enumerate(domain_x1):



			z = planner.cla_surrogate([[x, y]], return_np=True, normalize=False)
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z[0][0])
			Z[x_index, y_index] = z[0][0]

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]


	x_tick_labels = domain_x0
	y_tick_labels = domain_x1

	ax.set_xticks(np.arange(len(domain_x0)))
	ax.set_yticks(np.arange(len(domain_x1)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('x0', fontsize=12)
	ax.set_ylabel('x1', fontsize=12)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	# # plot observations thus far
	# X_str_ = res_df.loc[:, ['template', 'alkyne']]#[:20]#[:-1] # remove last data point
	# print(X_str_)
	# X_  = pd.DataFrame([[domain_x0.index(x[0]), domain_x1.index(x[1])] for x in X_str_.to_numpy() ],
	# 	columns=X_str_.columns
	# 				  )
	#
	# ax.scatter(X_['alkyne'], X_['template'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)


def plot_acquisition_cat(res_df, planner, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_x0 = [f'x_{i}' for i in range(21)]
	domain_x1 = [f'x_{i}' for i in range(21)]

	Z = np.zeros((len(domain_x0), len(domain_x1)))

	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_x0):
		for y_index, y in enumerate(domain_x1):

			z = planner.acquisition_function([[x, y]], return_np=True, normalize=False)
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z[0][0])
			Z[x_index, y_index] = z[0][0]

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]

	x_tick_labels = domain_x0
	y_tick_labels = domain_x1

	ax.set_xticks(np.arange(len(domain_x0)))
	ax.set_yticks(np.arange(len(domain_x1)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('x0', fontsize=12)
	ax.set_ylabel('x1', fontsize=12)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	# # plot observations thus far
	# X_str_ = res_df.loc[:, ['template', 'alkyne']]#[:20]#[:-1] # remove last data point
	# print(X_str_)
	# X_  = pd.DataFrame([[domain_x0.index(x[0]), domain_x1.index(x[1])] for x in X_str_.to_numpy() ],
	# 	columns=X_str_.columns
	# 				  )
	#
	# ax.scatter(X_['alkyne'], X_['template'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)

def plot_constr_surface_cat(res_df, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_x0 = [f'x_{i}' for i in range(21)]
	domain_x1 = [f'x_{i}' for i in range(21)]

	Z = np.zeros((len(domain_x1), len(domain_x0)))
	Z_feas = np.empty((len(domain_x1), len(domain_x0)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_x0):
		for y_index, y in enumerate(domain_x1):

			z = surface.eval_merit({'x0':x, 'x1': y})['obj']

			feas_bool = surface.eval_constr({'x0':x, 'x1': y})
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z)
			Z[x_index, y_index] = z
			Z_feas[x_index, y_index] = feas_bool


	#Z = np.log(Z)

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))
	ax.imshow(Z_feas.T, origin='lower', cmap='gray', alpha=0.55, interpolation='none')

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]


	x_tick_labels = domain_x0
	y_tick_labels = domain_x1

	ax.set_xticks(np.arange(len(domain_x0)))
	ax.set_yticks(np.arange(len(domain_x1)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('x0', fontsize=12)
	ax.set_ylabel('x1', fontsize=12)

	# divider = make_axes_locatable(ax)
	# cax = divider.append_axes("right", size="5%", pad=0.05)
	# plt.colorbar(im, cax=cax)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	# plot observations thus far
	X_str_ = res_df.loc[:, ['x0', 'x1']]#[:20]#[:-1] # remove last data point
	X_  = pd.DataFrame([[domain_x0.index(x[0]), domain_x1.index(x[1])] for x in X_str_.to_numpy() ],
		columns=X_str_.columns
					  )

	ax.scatter(X_['x0'], X_['x1'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)






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

#------------------
# begin experiment
#------------------

if plot:
	fig, axes = plt.subplots(2, 2, figsize=(12, 15), sharex=True)
	axes = axes.flatten()
	plt.ion()


param_space = set_param_space(type_=type_)

planner = GPPlanner(
	goal='minimize',
	feas_strategy=feas_strategy,
	feas_param=feas_param,
	num_init_design=10,
	vgp_iters=3000,
	vgp_lr=0.05,
	acquisition_type='ucb',
	use_descriptors=with_descriptors,
)
planner.set_param_space(param_space)

campaign = Campaign()
campaign.set_param_space(param_space)

if type_ == 'categorical':
	OPT = surface.best
	print('OPTIMUM : ', OPT)


for num_iter in range(budget):

	samples = planner.recommend(campaign.observations)

	sample = samples[0]

	measurement = surface.eval_merit(sample.to_dict())

	campaign.add_observation(sample.to_array(), measurement['obj'])

	# store the results into a DataFrame
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	obj_col = campaign.observations.get_values(as_array=True)

	res_df = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col})


	if plot and num_iter>10:

		plot_constr_surface_cat( res_df, n=0, ax=axes[0])
		axes[0].set_title('objective function')

		plot_reg_surrogate_cat( res_df, planner, n=0, ax=axes[1])
		axes[1].set_title('regression surrogate')

		plot_cla_surrogate_cat( res_df, planner, n=0, ax=axes[2])
		axes[2].set_title('classification surrogate')

		plot_acquisition_cat( res_df, planner, n=0, ax=axes[3])
		axes[3].set_title(f'acquisition function : {feas_strategy}')



		plt.tight_layout()
		plt.pause(2.0)

		for ax in axes:
			ax.clear()




	if type_ == 'categorical':
		if (sample.to_dict()['x0'], sample.to_dict()['x1']) == OPT:
			print(f'FOUND OPTIMUM AFTER {num_iter+1} ITERATIONS!')
			break
