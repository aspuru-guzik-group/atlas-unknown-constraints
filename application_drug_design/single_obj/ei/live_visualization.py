#!/usr/bin/emv python

import os, sys
import shutil
from copy import deepcopy
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical

from atlas.optimizers.gp.planner import GPPlanner


#----------
# Settings
#----------

feas_strategy = 'fwa'
feas_param = 0.
with_descriptors = False
descriptors_type = 'mord'  # 'mord' or 'pca'
type_ = 'categorical'
plot = True

budget = 270
repeats = 1
random_seed = None # i.e. use a different random seed each time

best_params = {'template_name': '8-1', 'alkyne_name': '22-5'}

#----------------------
# load tabular results
#----------------------
df_results = pd.read_csv('reference-and-data/lookup_table.csv')

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
		return match.loc[:, 'abl1_pIC50'].to_numpy()[0]
	elif match.loc[:, 'synthesis_success'].to_numpy()[0] == 0:
		# failed synthesis
		return np.nan
	else:
		raise ValueError()

def eval_merit(param):
	abl1_pIC50 = run_experiment(param)
	param['abl1_pIC50'] = abl1_pIC50
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

def plot_reg_surrogate_cat(df, res_df, planner, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_template = df['template_name'].unique().tolist()
	domain_alkyne = df['alkyne_name'].unique().tolist()

	Z = np.zeros((len(domain_template), len(domain_alkyne)))
	# Z_feas = np.empty((len(domain_alkyne), len(domain_template)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_template):
		for y_index, y in enumerate(domain_alkyne):

			row = df[(df['template_name']==x)&(df['alkyne_name']==y)].to_dict('r')[0]

			z, _ = planner.reg_surrogate([[x, y]], return_np=True)
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z[0][0])
			Z[x_index, y_index] = z[0][0]


	#Z = np.log(Z)

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]


	x_tick_labels = domain_template
	y_tick_labels = domain_alkyne

	ax.set_xticks(np.arange(len(domain_template)))
	ax.set_yticks(np.arange(len(domain_alkyne)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('template name', fontsize=12)
	ax.set_ylabel('alkyne name', fontsize=12)


	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	return cax

def plot_cla_surrogate_cat(df, res_df, planner, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_template = df['template_name'].unique().tolist()
	domain_alkyne = df['alkyne_name'].unique().tolist()

	Z = np.zeros((len(domain_template), len(domain_alkyne)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_template):
		for y_index, y in enumerate(domain_alkyne):

			row = df[(df['template_name']==x)&(df['alkyne_name']==y)].to_dict('r')[0]

			z = planner.cla_surrogate([[x, y]], return_np=True, normalize=False)
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z[0][0])
			Z[x_index, y_index] = z[0][0]

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]


	x_tick_labels = domain_template
	y_tick_labels = domain_alkyne

	ax.set_xticks(np.arange(len(domain_template)))
	ax.set_yticks(np.arange(len(domain_alkyne)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('template name', fontsize=12)
	ax.set_ylabel('alkyne name', fontsize=12)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	return cax

def plot_acquisition_cat(df, res_df, planner, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_template = df['template_name'].unique().tolist()
	domain_alkyne = df['alkyne_name'].unique().tolist()

	Z = np.zeros((len(domain_template), len(domain_alkyne)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_template):
		for y_index, y in enumerate(domain_alkyne):

			row = df[(df['template_name']==x)&(df['alkyne_name']==y)].to_dict('r')[0]

			z = planner.acquisition_function([[x, y]], return_np=True, normalize=False)
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z[0][0])
			Z[x_index, y_index] = z[0][0]

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]


	x_tick_labels = domain_template
	y_tick_labels = domain_alkyne

	ax.set_xticks(np.arange(len(domain_template)))
	ax.set_yticks(np.arange(len(domain_alkyne)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)


	ax.set_xlabel('template name', fontsize=12)
	ax.set_ylabel('alkyne name', fontsize=12)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)
	return cax

def plot_constr_surface_cat(df, res_df, n=0, ax=None, mark_best=True):

	if ax is None:
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,5))

	domain_template = df['template_name'].unique().tolist()
	domain_alkyne = df['alkyne_name'].unique().tolist()


	Z = np.zeros((len(domain_template), len(domain_alkyne)))
	Z_feas = np.empty((len(domain_template), len(domain_alkyne)))
	X = []
	X_str = []
	z_flat = []
	for x_index, x in enumerate(domain_template):
		for y_index, y in enumerate(domain_alkyne):

			row = df[(df['template_name']==x)&(df['alkyne_name']==y)].to_dict('r')[0]
			#print(row)
			z = row['abl1_pIC50']
			feas_val = row['synthesis_success'] # 1=success, 0=failure
			if feas_val==1:
				feas_bool=True
			else:
				feas_bool=False
			X.append([x_index, y_index])
			X_str.append([x, y])
			z_flat.append(z)
			Z[x_index, y_index] = z
			Z_feas[x_index, y_index] = feas_bool

	im = ax.imshow(Z.T, origin='lower', cmap = 'RdYlGn')#plt.get_cmap('golem'))
	ax.imshow(Z_feas.T, origin='lower', cmap='gray', alpha=0.55, interpolation='none')

	x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
	y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]

	x_tick_labels = domain_template
	y_tick_labels = domain_alkyne

	ax.set_xticks(np.arange(len(domain_template)))
	ax.set_yticks(np.arange(len(domain_alkyne)))
	ax.set_xticklabels(x_tick_labels, rotation=90)
	ax.set_yticklabels(y_tick_labels)

	ax.set_xlabel('template name', fontsize=12)
	ax.set_ylabel('alkyne name', fontsize=12)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	plt.colorbar(im, cax=cax)

	if mark_best is True:
		best_ix = np.argmax(z_flat)
		X_best = X[best_ix]
		_ = ax.scatter(X_best[0], X_best[1], marker='*', s=200, color='#ffc6ff', linewidth=2, zorder=20)

	# plot observations thus far
	X_str_ = res_df.loc[:, ['template', 'alkyne']]#[:20]#[:-1] # remove last data point
	print(X_str_)
	X_  = pd.DataFrame([[domain_template.index(x[0]), domain_alkyne.index(x[1])] for x in X_str_.to_numpy() ],
		columns=X_str_.columns
					  )

	ax.scatter(X_['template'], X_['alkyne'], marker='X', s=100, color='#adb5bd', edgecolor='k', zorder=10)

	return cax


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

		if with_descriptors:
			# load the descriptors
			desc_template_pca = pd.read_csv('reference-and-data/descriptors/template_pca_desc.csv')
			desc_alkyne_pca = pd.read_csv('reference-and-data/descriptors/alkyne_pca_desc.csv')

			desc_template_mord = pd.read_csv('reference-and-data/descriptors/template_mord_desc.csv')
			desc_alkyne_mord = pd.read_csv('reference-and-data/descriptors/alkyne_mord_desc.csv')


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


#------------------
# begin experiment
#------------------

if plot:
	fig, axes = plt.subplots(2, 2, figsize=(12, 15), sharex=False)
	axes = axes.flatten()
	plt.ion()


param_space = set_param_space(type_=type_)

planner = GPPlanner(
	goal='maximize', # maximize the IC50
	feas_strategy=feas_strategy,
	feas_param=feas_param,
	num_init_design=12,
	vgp_iters=2000,
	vgp_lr=0.2,
)
planner.set_param_space(param_space)

campaign = Campaign()
campaign.set_param_space(param_space)

for num_iter in range(budget):

	print(f'===============================')
	print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {num_iter+1}')
	print(f'===============================')

	samples = planner.recommend(campaign.observations)

	sample = samples[0]

	measurement = eval_merit(sample)

	campaign.add_observation(sample.to_array(), measurement['abl1_pIC50'])

	# store the results into a DataFrame
	template_col = campaign.observations.get_params()[:, 0]
	alkyne_col = campaign.observations.get_params()[:, 1]
	abl1_pIC50_col = campaign.observations.get_values(as_array=True)

	res_df = pd.DataFrame({'template': template_col, 'alkyne': alkyne_col, 'abl1_pIC50': abl1_pIC50_col})


	if plot and num_iter>12:

		cax_surf = plot_constr_surface_cat(df_results, res_df, n=0, ax=axes[0])
		axes[0].set_title('objective function')

		cax_reg = plot_reg_surrogate_cat(df_results, res_df, planner, n=0, ax=axes[1])
		axes[1].set_title('regression surrogate')

		cax_cla = plot_cla_surrogate_cat(df_results, res_df, planner, n=0, ax=axes[2])
		axes[2].set_title('classification surrogate')

		cax_acq = plot_acquisition_cat(df_results, res_df, planner, n=0, ax=axes[3])
		axes[3].set_title(f'acquisition function : {feas_strategy}')




		plt.tight_layout()
		plt.pause(2.0)

		for ax in axes:
			ax.clear()


		cax_surf.clear()
		cax_reg.clear()
		cax_cla.clear()
		cax_acq.clear()




	if type_ == 'categorical':
		if (sample.to_dict()['template_name'], sample.to_dict()['alkyne_name']) == (best_params['template_name'], best_params['alkyne_name']):
			print(f'FOUND OPTIMUM AFTER {num_iter+1} ITERATIONS!')
			break
