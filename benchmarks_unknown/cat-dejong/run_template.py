#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pickle
from gryffin import Gryffin
sys.path.append("../../")

from benchmark_functions import replace_nan_with_worst, write_categories, save_pkl_file, load_data_from_pkl_and_continue
from benchmark_functions import CatDejongConstr as BenchmarkSurface

# --------
# Settings
# --------
feas_approach = _APPROACH_
with_descriptors = _DESCRIPTORS_
dynamic = _DYNAMIC_  # use static/dynamic Gryffin
replace_nan = _NAIVE_  # whether to use feas approach or replace worst
feas_param = _FEAS_

budget = 441  # i.e. run all options
repeats = 100
random_seed = None  # i.e. use different random seed each time
surface = BenchmarkSurface()
sampling_strategies = np.array([1, -1])

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)
# write categories
if with_descriptors is True and dynamic is True:
    num_descs = 2  # dynamic gryffin
else:
    num_descs = 1  # naive (not used), and static
write_categories(num_dims=2, num_opts=21, home_dir='.', num_descs=num_descs, with_descriptors=with_descriptors)

# --------------------------------
# Standardized script from here on
# --------------------------------
config = {
     "general": {
             "auto_desc_gen": dynamic,
             "batches": 1,
             "num_cpus": 4,
             "boosted":  False,
             "caching": True,
             "backend": "tensorflow",
             "save_database": False,
             "sampling_strategies": len(sampling_strategies),
             "random_seed":random_seed,
             "feas_approach":feas_approach,
             "feas_param": feas_param,  # used only if naive is False
             "acquisition_optimizer": "genetic",
             "verbosity": 3  # show stats
                },
    "parameters": [
         {"name": "x0", "type": "categorical", "category_details": "CatDetails/cat_details_x0.pkl"},
         {"name": "x1", "type": "categorical", "category_details": "CatDetails/cat_details_x1.pkl"}
    ],
    "objectives": [
        {"name": "obj", "goal": "min"}
    ]
}

for num_repeat in range(missing_repeats):
    gryffin = Gryffin(config_dict=config)
    observations = []
    for num_iter in range(budget):
        print(f'===============================')
        print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {num_iter+1}')
        print(f'===============================')

        # select alternating sampling strategy
        select_idx = num_iter % len(sampling_strategies)
        sampling_strategy = sampling_strategies[select_idx]
    
        if replace_nan is True:
            # replace nan observations with worst merit seen so far
            observations_without_nan = replace_nan_with_worst(observations)
            # query for new parameters
            params = gryffin.recommend(observations=observations_without_nan, sampling_strategies=[sampling_strategy])
        else:
            # query for new parameters
            params = gryffin.recommend(observations=observations, sampling_strategies=[sampling_strategy])
    
        # select the single set of params we created
        param = params[0]
        
        # evaluate the proposed parameters
        observation = surface.eval_merit(param)
        observations.append(observation)

        # stop optimization if we found best
        if (param['x0'], param['x1']) == surface.best:
            break
    
    # store run results into a DataFrame
    x0_col = [x['x0'] for x in observations]
    x1_col = [x['x1'] for x in observations]
    obj_col = [x['obj'] for x in observations]
    data = pd.DataFrame({'x0':x0_col, 'x1':x1_col, 'obj':obj_col})
    data_all_repeats.append(data)
    
    # save results to disk
    save_pkl_file(data_all_repeats)

