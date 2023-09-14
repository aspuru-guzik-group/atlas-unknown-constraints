#!/usr/bin/env python

import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import torch
import pickle

from atlas.planners.gp.planner import GPPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical, ParameterVector

from benchmark_functions import save_pkl_file, load_data_from_pkl_and_continue
#from benchmark_functions import BraninConstr as BenchmarkSurface
#from benchmark_functions import DejongConstr as BenchmarkSurface
from benchmark_functions import StyblinskiTangConstr as BenchmarkSurface
# from benchmark_functions import HyperEllipsoidConstr as BenchmarkSurface

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import seaborn as sns

from atlas.utils.planner_utils import forward_normalize, reverse_normalize


np.random.seed()

#----------
# Settings
#----------

feas_strategy = 'fca'
feas_param = 0.5
with_descriptors = True
type_ = 'continuous'
init_design_strategy='lhs'
acquisition_type='ucb'
acquisition_optimizer_kind='pymoo'

plot = True

budget = 100
repeats = 1
random_seed = None # i.e. use a different random seed each time
surface = BenchmarkSurface()


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


# Golem colormap
_reference_colors = ['#008080', '#70a494', '#b4c8a8', '#f6edbd', '#edbb8a', '#de8a5a','#ca562c']
_cmap = LinearSegmentedColormap.from_list('golem', _reference_colors)
_cmap_r = LinearSegmentedColormap.from_list('golem_r', _reference_colors[::-1])
plt.register_cmap(cmap=_cmap)
plt.register_cmap(cmap=_cmap_r)

def get_golem_colors(n):
	_cmap = plt.get_cmap('golem')
	return [_cmap(x) for x in np.linspace(0, 1, n)]

def plot_contour(ax, X0, X1, y, xlims, ylims, vlims=[None, None], alpha=0.5, contour_lines=True, contour_labels=True,
				 labels_fs=8, labels_fmt='%d', n_contour_lines=8, contour_color='k', contour_alpha=1, cbar=False, cmap='golem'):
	# background surface
	if contour_lines is True:
		contours = ax.contour(X0, X1, y, n_contour_lines, colors=contour_color, alpha=contour_alpha, linestyles='dashed')
		if contour_labels is True:
			_ = ax.clabel(contours, inline=True, fontsize=labels_fs, fmt=labels_fmt)
	mappable = ax.imshow(y, extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
						 origin='lower', cmap=cmap, alpha=alpha, vmin=vlims[0], vmax=vlims[1])

	if cbar is True:
		cbar = plt.colorbar(mappable=mappable, ax=ax, shrink=0.5)

	return mappable

def plot_constr_surface(res_df, surface, ax=None, N=100):
	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

	x0 = np.linspace(0,1,N)
	x1 = np.linspace(0,1,N)
	X0, X1 = np.meshgrid(x0,x1)
	X = np.array([X0.flatten(), X1.flatten()]).T
	y = np.array(surface.run(X)).flatten()
	Y = np.reshape(y, newshape=np.shape(X0))

	_ = plot_contour(ax, X0, X1, Y, xlims=[0,1], ylims=[0,1], alpha=1, contour_lines=True, contour_labels=True,
				 labels_fs=8, labels_fmt='%d', n_contour_lines=8, contour_alpha=0.8, cbar=False, cmap='golem')
	for param in surface.minima:
		x_min = param['params']
		ax.scatter(x_min[0], x_min[1], s=200, marker='*', color='#ffc6ff', zorder=20)

	y_feas = np.array(surface.eval_constr(X))
	Y_feas = np.reshape(y_feas, newshape=np.shape(X0))
	ax.imshow(Y_feas, extent=[0,1,0,1], origin='lower', cmap='gray', alpha=0.5, interpolation='none')

	X =  res_df.loc[:, ['x0', 'x1']]
	mask = surface.eval_constr(X.to_numpy())
	X_feas = X[mask]
	X_infs = X[~mask]

	ax.scatter(X_feas['x0'], X_feas['x1'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)
	ax.scatter(X_infs['x0'], X_infs['x1'], marker='X', s=100, color='white', edgecolor='k', zorder=10)

def plot_reg_surrogate(res_df, surface, planner, ax=None, N=100):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

	x0 = np.linspace(0,1,N)
	x1 = np.linspace(0,1,N)
	X0, X1 = np.meshgrid(x0,x1)
	X = np.array([X0.flatten(), X1.flatten()]).T

	y, _ = planner.reg_surrogate(X, return_np=True)

	Y = np.reshape(y, newshape=np.shape(X0))

	_ = plot_contour(ax, X0, X1, Y, xlims=[0,1], ylims=[0,1], alpha=1, contour_lines=True, contour_labels=True,
				 labels_fs=8, labels_fmt='%d', n_contour_lines=8, contour_alpha=0.8, cbar=False, cmap='golem')
	for param in surface.minima:
		x_min = param['params']
		ax.scatter(x_min[0], x_min[1], s=200, marker='*', color='#ffc6ff', zorder=20)

	y_feas = np.array(surface.eval_constr(X))
	Y_feas = np.reshape(y_feas, newshape=np.shape(X0))
	ax.imshow(Y_feas, extent=[0,1,0,1], origin='lower', cmap='gray', alpha=0.5, interpolation='none')

	X =  res_df.loc[:, ['x0', 'x1']]
	mask = surface.eval_constr(X.to_numpy())
	X_feas = X[mask]
	X_infs = X[~mask]

	ax.scatter(X_feas['x0'], X_feas['x1'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)
	ax.scatter(X_infs['x0'], X_infs['x1'], marker='X', s=100, color='white', edgecolor='k', zorder=10)

def plot_cla_surrogate(res_df, surface, planner, ax=None, N=100):


	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

	x0 = np.linspace(0,1,N)
	x1 = np.linspace(0,1,N)
	X0, X1 = np.meshgrid(x0,x1)
	X = np.array([X0.flatten(), X1.flatten()]).T

	y = planner.cla_surrogate(X, return_np=True, normalize=True)

	Y = np.reshape(y, newshape=np.shape(X0))

	_ = plot_contour(ax, X0, X1, Y, xlims=[0,1], ylims=[0,1], alpha=1, contour_lines=True, contour_labels=True,
				 labels_fs=8, labels_fmt='%d', n_contour_lines=8, contour_alpha=0.8, cbar=False, cmap='golem')
	for param in surface.minima:
		x_min = param['params']
		ax.scatter(x_min[0], x_min[1], s=200, marker='*', color='#ffc6ff', zorder=20)

	# y_feas = np.array(surface.eval_constr(X))
	# Y_feas = np.reshape(y_feas, newshape=np.shape(X0))
	# ax.imshow(Y_feas, extent=[0,1,0,1], origin='lower', cmap='gray', alpha=0.5, interpolation='none')

	# if we have FCA constraints, shade the infeasible region according to the classification surrogate
	if feas_strategy == 'fca':
		# scale the X values before passing them to the fca constraint function
		X_scaled = forward_normalize(X, planner.params_obj._mins_x, planner.params_obj._maxs_x)
		constraint_val = planner.fca_constraint(torch.tensor(X_scaled)).detach().numpy()
		print('X : ', X.shape)
		cla_surr = 1.-planner.cla_surrogate(X, return_np=True, normalize=False)
		cla_surr_min = np.amin(cla_surr)
		cla_surr_max = np.amax(cla_surr)
		print('cla surr shape : ', cla_surr.shape)
		print('cla surr min : ', cla_surr_min)
		print('cla surr max : ', cla_surr_max)

		cutoff_val = (cla_surr_max-cla_surr_min)*feas_param + cla_surr_min

		print('cutoff val : ', cutoff_val)

		constraint_val = np.where(cla_surr<cutoff_val, False, True)
		print(f'num feasible : {np.sum(constraint_val)}/{cla_surr.shape[0]}')

		constraint_val = np.reshape(constraint_val, newshape=np.shape(X0))
		ax.imshow(constraint_val, extent=[0,1,0,1], origin='lower', cmap='gray', alpha=0.5, interpolation='none')

	X =  res_df.loc[:, ['x0', 'x1']]
	mask = surface.eval_constr(X.to_numpy())
	X_feas = X[mask]
	X_infs = X[~mask]

	ax.scatter(X_feas['x0'], X_feas['x1'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)
	ax.scatter(X_infs['x0'], X_infs['x1'], marker='X', s=100, color='white', edgecolor='k', zorder=10)

def plot_acquisition(res_df, surface, planner, ax=None, N=100):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

	x0 = np.linspace(0,1,N)
	x1 = np.linspace(0,1,N)
	X0, X1 = np.meshgrid(x0,x1)
	X = np.array([X0.flatten(), X1.flatten()]).T

	y = planner.acquisition_function(X, return_np=True, normalize=True)

	Y = np.reshape(y, newshape=np.shape(X0))

	_ = plot_contour(ax, X0, X1, Y, xlims=[0,1], ylims=[0,1], alpha=1, contour_lines=True, contour_labels=True,
				 labels_fs=8, labels_fmt='%d', n_contour_lines=8, contour_alpha=0.8, cbar=False, cmap='golem')
	for param in surface.minima:
		x_min = param['params']
		ax.scatter(x_min[0], x_min[1], s=200, marker='*', color='#ffc6ff', zorder=20)

	y_feas = np.array(surface.eval_constr(X))
	Y_feas = np.reshape(y_feas, newshape=np.shape(X0))
	ax.imshow(Y_feas, extent=[0,1,0,1], origin='lower', cmap='gray', alpha=0.5, interpolation='none')

	X_ =  res_df.loc[:, ['x0', 'x1']]
	mask = surface.eval_constr(X_.to_numpy())
	X_feas = X_[mask]
	X_infs = X_[~mask]

	ax.scatter(X_feas['x0'], X_feas['x1'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)
	ax.scatter(X_infs['x0'], X_infs['x1'], marker='X', s=100, color='white', edgecolor='k', zorder=10)

	# plot the maximum of the acquisition function
	max_idx = np.argmax(y)
	max_params = X[max_idx]
	ax.scatter(max_params[0], max_params[1], s=200, marker='D', color='#274690', zorder=20)

	# if FCA, plot the max of the acquisition function s.t. the constriant
	if feas_strategy == 'fca':
		cla_surr = 1.-planner.cla_surrogate(X, return_np=True, normalize=False)
		cla_surr_min = np.amin(cla_surr)
		cla_surr_max = np.amax(cla_surr)
		cutoff_val = (cla_surr_max-cla_surr_min)*feas_param + cla_surr_min

		constraint_val = np.where(cla_surr<cutoff_val, False, True)

		print(X.shape, y.shape, constraint_val.shape)
		X_feas_fca = X[constraint_val[:,0],:]
		y_feas_fca = y[constraint_val[:,0],:]

		max_idx_feas_fca = np.argmax(y_feas_fca)
		max_X_feas_fca = X_feas_fca[max_idx_feas_fca]
		max_y_feas_fca = y_feas_fca[max_idx_feas_fca]
		ax.scatter(max_X_feas_fca[0], max_X_feas_fca[1], s=150, marker='P', color='#a52422', zorder=20)



if plot:
	fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
	axes = axes.flatten()
	plt.ion()

param_space = set_param_space(type_=type_)

planner = GPPlanner(
	goal='minimize',
	feas_strategy=feas_strategy,
	feas_param=feas_param,
	vgp_iters=1000,
	vgp_lr=0.1,
	num_init_design=10,
	init_design_strategy=init_design_strategy,
	acquisition_type=acquisition_type,
	acquisition_optimizer_kind=acquisition_optimizer_kind
)
planner.set_param_space(param_space)

campaign = Campaign()
campaign.set_param_space(param_space)

# add custom initial observations
# custom_init_params = [
# 	{'x0': 0.710183, 'x1': 0.690794},
# 	# {'x0': 0.169314, 'x1': 0.706979},
# 	{'x0': 0.463191, 'x1': 0.655752},
# 	{'x0': 0.317040, 'x1': 0.504390},
# 	{'x0': 0.954457, 'x1': 0.706121},
# 	{'x0': 0.118804, 'x1': 0.026817},
# 	{'x0': 0.466782, 'x1': 0.764056},
# 	# {'x0': 0.958115, 'x1': 0.079982},
# 	{'x0': 0.882438, 'x1': 0.569433},
# 	# {'x0': 0.122223, 'x1': 0.878959},
# 	# duplicates
# 	{'x0': 0.882438, 'x1': 0.569433},
# 	{'x0': 0.882438, 'x1': 0.569433},
# 	{'x0': 0.882438, 'x1': 0.569433},
# ]

# custom_init_values = [
# 	{'obj': 102.960719},
# 	# {'obj': np.nan},
# 	{'obj': 48.018663},
# 	{'obj': 20.683729},
# 	{'obj': 67.797731},
# 	{'obj': 145.796288},
# 	{'obj': 72.070659},
# 	# {'obj': np.nan},
# 	{'obj': 53.837827},
# 	# {'obj': np.nan},
# 	# duplicates
# 	{'obj': 53.837827},
# 	{'obj': 53.837827},
# 	{'obj': 53.837827},
# ]

# for param, value in zip(custom_init_params, custom_init_values):
# 	campaign.add_observation(
# 		ParameterVector().from_dict(param, param_space),
# 		value['obj'],
# 	)

print(campaign.observations.get_params())
print(campaign.observations.get_values())


if type_ == 'categorical':
	OPT = surface.best
	print('OPTIMUM : ', OPT)

for num_iter in range(budget):


	samples = planner.recommend(campaign.observations)

	sample = samples[0]

	measurement = surface.eval_merit(sample.to_dict())

	print(f'ITER : {num_iter}\tSAMPLE : {sample}\tMEASUREMENT : {measurement["obj"]}')

	campaign.add_observation(sample, measurement['obj'])

	# store the results into a DataFrame
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	obj_col = campaign.observations.get_values(as_array=True)

	res_df = pd.DataFrame({'x0': x0_col, 'x1': x1_col, 'obj': obj_col})

	if plot and len(campaign.observations.get_values())>10:

		plot_constr_surface(res_df, surface, ax=axes[0])
		axes[0].set_title('objective function')

		plot_reg_surrogate(res_df, surface,  planner, ax=axes[1])
		axes[1].set_title('regression surrogate')

		plot_cla_surrogate( res_df, surface, planner, ax=axes[2])
		axes[2].set_title('classification surrogate')

		plot_acquisition( res_df, surface, planner, ax=axes[3])
		axes[3].set_title(f'acquisition function : {feas_strategy}')

		plt.tight_layout()
		plt.pause(2.0)

		for ax in axes:
			ax.clear()

	if type_ == 'categorical':
		if (sample.to_dict()['x0'], sample.to_dict()['x1']) == OPT:
			print(f'FOUND OPTIMUM AFTER {num_iter+1} ITERATIONS!')
			break
