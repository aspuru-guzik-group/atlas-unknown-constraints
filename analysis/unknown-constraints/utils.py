#!/usr/bin/env python

import os, sys
import pickle

STRATEGY_MAP = {
    'random': 'random',
    'naive-0': 'naive',
    'naive-replace': 'naive',
    'naive-fia-1000': 'naive',
    'naive-surrogate': 'naive',
    'fwa-0': 'fwa',
    'fca-0.2': 'fca',
    'fca-0.5': 'fca',
    'fca-0.8': 'fca',
    'fca-1': 'fca',
    'fca-2': 'fca',
    'fia-0.5': 'fia',
    'fia-1': 'fia',
    'fia-2': 'fia',
}


def collect_results_cat(dirs, exp_names, report_missing=True, report_num_runs=True):

    missing_exps = []
    all_data = {}
    num_runs = {}

    for dir_ in dirs:
        surf_name = dir_.split('/')[-1]
        print(f'>>> Unpacking {surf_name}')
        all_data[surf_name] = []

        for exp_name in exp_names:
            results_file = f'{dir_}/{exp_name}/results.pkl'
            split = exp_name.split('_')

            print(split)

            if len(split)==1:
                desc_, planner, acqf, params = 'wodesc', 'random', 'ei', 'random'
            elif len(split)==3:
                desc_, planner, acqf, params = split[0],split[1],'ei',split[2]
            elif len(split)==4:
                desc_, planner, acqf, params = split[0],split[1],split[2],split[3]

            if params == 'fca-1':
                print(desc_, planner, surf_name, acqf, params)
            strategy = STRATEGY_MAP[params]

            if os.path.isfile(results_file):
                with open(results_file, 'rb') as content:
                    res = pickle.load(content)
                    num_runs['-'.join((surf_name,exp_name))] = len(res)
                if len(res)==0:
                    missing_exps.append(results_file)
                else:
                    all_data[surf_name].append(
                        {'method': exp_name, 'planner': planner, 'strategy': strategy, 'params': params, 'acqf':acqf, 'desc':desc_, 'data':res}
                    )

            else:
                missing_exps.append(results_file)


    return all_data, missing_exps, num_runs


def collect_results_cont(dirs, exp_names, report_missing=True, report_num_runs=True):
        missing_exps = []
        all_data = {}
        num_runs = {}

        for dir_ in dirs:
            surf_name = dir_.split('/')[-1]
            print(f'>>> Unpacking {surf_name}')
            all_data[surf_name] = []

            for exp_name in exp_names:
                print('exp_name : ', exp_name)
                results_file = f'{dir_}/{exp_name}/results.pkl'
                split = exp_name.split('_')

                if len(split)==1:
                    #desc_, planner, acqf, params = 'naive', 'gryffin', 'gryffin', split[0]
                    desc_, planner, acqf, params = 'wodesc', 'random', 'random', 'random'
                elif len(split)==2:
                    desc_, planner, acqf, params = 'wodesc', split[0], 'ei', split[1]
                else:
                    desc_, planner, acqf, params = 'wodesc', split[0], split[1], split[2]

                if params == 'fca-1':
                    print(desc_, planner, surf_name, acqf, params)
                strategy = STRATEGY_MAP[params]

                if os.path.isfile(results_file):
                    with open(results_file, 'rb') as content:
                        res = pickle.load(content)
                        num_runs['-'.join((surf_name,exp_name))] = len(res)
                    if len(res)==0:
                        missing_exps.append(results_file)
                    else:
                        all_data[surf_name].append(
                            {'method': exp_name, 'planner': planner, 'strategy':strategy, 'params': params, 'acqf': acqf, 'desc':desc_, 'data':res}
                        )

                else:
                    missing_exps.append(results_file)


        return all_data, missing_exps, num_runs



def collect_results_cont_abl(dir_, exp_names, report_missing=True):
    missing_exps = []
    all_data = {}

    for exp_name in exp_names:
        all_data[exp_name] = {}
        results_file = f'{dir_}/{exp_name}/results.pkl'
        split = exp_name.split('_')

        if len(split)==1:
            desc_, planner, params = 'naive', 'gryffin', split[0]
        else:
            desc_, planner, params = 'wodesc',split[0],split[1]

        if os.path.isfile(results_file):
            with open(results_file, 'rb') as content:
                res = pickle.load(content)
            if len(res)==0:
                missing_exps.append(results_file)
            else:
                bins = list(res.keys())
                for bin in bins:
                    all_data[exp_name][bin] = []
                    data = res[bin]

                    all_data[exp_name][bin].append(
                        {'method': exp_name, 'planner': planner, 'params': params, 'desc':desc_, 'bin': bin, 'data':data}
                    )

        else:
            missing_exps.append(results_file)

    return all_data, missing_exps



def collect_results_cat_abl(dir_, exp_names, report_missing=True):
    missing_exps = []
    all_data = {}

    for exp_name in exp_names:
        all_data[exp_name] = {}
        results_file = f'{dir_}/{exp_name}/results.pkl'
        split = exp_name.split('_')


        desc_, planner, params = split[0], split[1], split[2]

        if os.path.isfile(results_file):
            with open(results_file, 'rb') as content:
                res = pickle.load(content)
            if len(res)==0:
                missing_exps.append(results_file)
            else:
                bins = list(res.keys())
                for bin in bins:
                    all_data[exp_name][bin] = []
                    data = res[bin]

                    all_data[exp_name][bin].append(
                        {'method': exp_name, 'planner': planner, 'params': params, 'desc':desc_, 'bin': bin, 'data':data}
                    )

        else:
            missing_exps.append(results_file)

    return all_data, missing_exps





#{'method': 'naive_gryffin-naive-0', 'planner': 'gryffin', 'params': 'naive-0', 'desc': 'naive', 'data': data_naive_naive_0},

exp_names_cat = [
        # -------
        # GRYFFIN
        # -------
        # naive
        'naive_gryffin_naive-0',
        'naive_gryffin_naive-fia-1000',
        'static_gryffin_naive-0',
        'static_gryffin_naive-fia-1000',
        'dynamic_gryffin_naive-0',
        'dynamic_gryffin_naive-fia-1000',
        # fwa
        'naive_gryffin_fwa-0',
        'static_gryffin_fwa-0',
        'dynamic_gryffin_fwa-0',
        # fca
        'naive_gryffin_fca-0.2',
        'naive_gryffin_fca-0.5',
        'naive_gryffin_fca-0.8',
        'static_gryffin_fca-0.2',
        'static_gryffin_fca-0.5',
        'static_gryffin_fca-0.8',
        'dynamic_gryffin_fca-0.2',
        'dynamic_gryffin_fca-0.5',
        'dynamic_gryffin_fca-0.8',
        # fia
        'naive_gryffin_fca-0.5',
        'naive_gryffin_fca-1',
        'naive_gryffin_fca-2',
        'static_gryffin_fca-0.5',
        'static_gryffin_fca-1',
        'static_gryffin_fca-2',
        'dynamic_gryffin_fca-0.5',
        'dynamic_gryffin_fca-1',
        'dynamic_gryffin_fca-2',
        # ------
        # ATLAS
        # ------
        # naive ei
        'wodesc_botorch_naive-0',
        'wodesc_botorch_naive-fia-1000',
        'desc_botorch_naive-0',
        'desc_botorch_naive-fia-1000',
        # fwa ei
        'wodesc_botorch_fwa-0',
        'desc_botorch_fwa-0',
        # fca ei
        'wodesc_botorch_fca-0.2',
        'wodesc_botorch_fca-0.5',
        'wodesc_botorch_fca-0.8',
        'desc_botorch_fca-0.2',
        'desc_botorch_fca-0.5',
        'desc_botorch_fca-0.8',
        # fia ei
        'wodesc_botorch_fia-0.5',
        'wodesc_botorch_fia-1',
        'wodesc_botorch_fia-2',
        'desc_botorch_fia-0.5',
        'desc_botorch_fia-1',
        'desc_botorch_fia-2',

        # naive ucb
        'wodesc_botorch_ucb_naive-0',
        'wodesc_botorch_ucb_naive-fia-1000',
        'desc_botorch_ucb_naive-0',
        'desc_botorch_ucb_naive-fia-1000',
        # fwa ucb
        'wodesc_botorch_ucb_fwa-0',
        'desc_botorch_ucb_fwa-0',
        # fca ucb
        'wodesc_botorch_ucb_fca-0.2',
        'wodesc_botorch_ucb_fca-0.5',
        'wodesc_botorch_ucb_fca-0.8',
        'desc_botorch_ucb_fca-0.2',
        'desc_botorch_ucb_fca-0.5',
        'desc_botorch_ucb_fca-0.8',
        # fia ucb
        'wodesc_botorch_ucb_fia-0.5',
        'wodesc_botorch_ucb_fia-1',
        'wodesc_botorch_ucb_fia-2',
        'desc_botorch_ucb_fia-0.5',
        'desc_botorch_ucb_fia-1',
        'desc_botorch_ucb_fia-2',
]


dirs_cat = ['../../benchmarks_unknown/cat-camel',
        # '../../benchmarks_unknown/cat-michalewicz',
        # '../../benchmarks_unknown/cat-slope',
        # '../../benchmarks_unknown/cat-dejong',
]


exp_names_cont = [
        # -------
        # GRYFFIN
        # -------
        # naive
        'naive-0',
        'naive-fia-1000',
        # fwa
        'fwa-0',
        # fca
        'fca-0.2',
        'fca-0.5',
        'fca-0.8',
        # fia
        'fca-0.5',
        'fca-1',
        'fca-2',
        # ------
        # ATLAS
        # ------
        # naive ei
        'botorch_naive-0',
        'botorch_naive-fia-1000',
        # fwa ei
        'botorch_fwa-0',
        # fca ei
        'botorch_fca-0.2',
        'botorch_fca-0.5',
        'botorch_fca-0.8',
        # fia ei
        'botorch_fia-0.5',
        'botorch_fia-1',
        'botorch_fia-2',
        # naive ucb
        'botorch_ucb_naive-0',
        'botorch_ucb_naive-fia-1000',
        # fwa ucb
        'botorch_ucb_fwa-0',
        # fca ucb
        'botorch_ucb_fca-0.2',
        'botorch_ucb_fca-0.5',
        'botorch_ucb_fca-0.8',
        # fia ucb
        'botorch_ucb_fia-0.5',
        'botorch_ucb_fia-1',
        'botorch_ucb_fia-2',
]


dirs_cont = [
    '../../benchmarks_unknown/branin',
    # '../../benchmarks_unknown/dejong',
    # '../../benchmarks_unknown/styblinski',
    # '../../benchmarks_unknown/hyperellips',
]


exp_names_cont_abl = [
        # ------
        # ATLAS
        # ------
        # naive
        'botorch_naive-0',
        'botorch_naive-fia-1000',
        # fwa
        'botorch_fwa-0',
        # fca
        'botorch_fca-0.2',
        'botorch_fca-0.5',
        'botorch_fca-0.8',
        # fia
        'botorch_fia-0.5',
        'botorch_fia-1',
        'botorch_fia-2',
]

exp_names_cat_abl = [
        # ------
        # ATLAS
        # ------
        # naive
        'wodesc_botorch_naive-0',
        'wodesc_botorch_naive-fia-1000',
        # fwa
        'wodesc_botorch_fwa-0',
        # fca
        'wodesc_botorch_fca-0.2',
        'wodesc_botorch_fca-0.5',
        'wodesc_botorch_fca-0.8',
        # fia
        'wodesc_botorch_fia-0.5',
        'wodesc_botorch_fia-1',
        'wodesc_botorch_fia-2',
        # naive
        'desc_botorch_naive-0',
        'desc_botorch_naive-fia-1000',
        # fwa
        'desc_botorch_fwa-0',
        # fca
        'desc_botorch_fca-0.2',
        'desc_botorch_fca-0.5',
        'desc_botorch_fca-0.8',
        # fia
        'desc_botorch_fia-0.5',
        'desc_botorch_fia-1',
        'desc_botorch_fia-2',
]



if __name__ == '__main__':

    results, missing_exps = collect_results_cat(dirs_cat, exp_names_cat, report_missing=True)
    #results, missing_exps = collect_results_cont(dirs_cont, exp_names_cont, report_missing=True)


    print('\n\n')

    print(missing_exps)
    print('num missing : ', len(missing_exps))

    print(len(results['cat-camel']))

    # results, missing_exps = collect_results_cat_abl('../../benchmarks_unknown/feas_ablation/squares/', exp_names_cat_abl, report_missing=True)
    #
    # print('results')
    # #print(results)
    #
    # print('num missing : ', len(missing_exps))
    # print('missing exps : ', missing_exps)
