# Anubis

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code to reproduce the experiments reported in "Anubis: Bayesian optimization with unknown feasibility constraints for scientific experimentation".

The constrained Bayesian optimization experiments reported in this work use the `atlas` Python library. You can visit the `atlas` [GitHub repo](), [documentation](https://matter-atlas.readthedocs.io/en/latest/), and [tutorial notebook]() for more information.

![alt text](https://github.com/aspuru-guzik-group/atlas-unknown-constraints/blob/main/static/anubis_banner.png)


## Installation

You can install `atlas` from source by executing the following commands in your terminal

```bash
git clone git@github.com:aspuru-guzik-group/atlas.git
cd atlas
pip install -e .
```


## Usage

The directories in this repo each contain code to reproduce specific experiments from our paper

#### Main text experiments

* `benchmarks_unknown/`: this directory contains the benchmark experiments on analytical surfaces, such as `branin`, `dejong`, etc. Categorical/discrete experiments are in subdirectories beginnig with `cat-`. 

* `application_hoip/`: code to reproduce the inverse design of stable hybrid organic-inorganic halide perovskite materials real-world application. 

* `application_drug_design/`: code to reproduce the (poly)pharmacological design of BCR-Abl fusion protein inhibitors real world application. The single obejctive experiments are in `application_drug_design/single_obj`, and the multi-objective optimization experiments are located in `application_drug_design/multi_obj`. `application_drug_design/reference-and-data` contains the original dataset from Desai _et al._, along with datasets of IC50 values taken from BindingDB for the KIT and PDGF tyrosine kinase inhibitors. This directory also contains the code to emulate the original BCR-Abl inhibitors dataset (using NGBoost classifier and regressor), as well as the scripts used to fit a MPNN to the KIT and PDGF datasets (using chemprop). 

#### Supplementary experiments

* `analysis/atlas_acqf_optimizers/`: comparison of constrained acquisition function optimization strategies SLSQP and DEAP/PyMOO (genetic) on analytical surfaces. 

* `analysis/atlas_ei_ucb_compare/`: comparison of EI and UCB acquisiton functions on constrained analytical benchmark functions.

* `analysis/atlas_filter_compare/`: comparison of mininum-filtered and original FIA and FCA feasibility-aware acquisition functions on analytical surfaces.



## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) license. See `LICENSE` for more information.


## Contact

Academic collaborations and extensions/improvements to the code by the community
are encouraged. Please reach out to [Riley](riley.hickman@mail.utoronto.ca) by email if you have questions.


## Citation

If you utilize the benchmarks experiments in this repository to advance your research project, please cite the following publication.

```
@misc{hickman_unknown_2023,
     author = {Hickman, Riley and Aldeghi, Matteo and Aspuru-Guzik, Al{\'a}n},
	doi = {},
	language = {en},
	month = ,
	publisher = {ChemRxiv},
	title = {Bayesian optimization with unknown feasibility constraints for experiment planning and chemical design},
	urldate = {2023-},
	year = {2023},
}
```

`atlas` is an academic research software. If you use `atlas` in a scientific publication, please cite the following article.

```
@misc{hickman_atlas_2023,
    author = {Hickman, Riley and Sim, Malcolm and Pablo-Garc{\'\i}a, Sergio and Woolhouse, Ivan and Hao, Han and Bao, Zeqing and Bannigan, Pauric and Allen, Christine and Aldeghi, Matteo and Aspuru-Guzik, Al{\'a}n},
	doi = {10.26434/chemrxiv-2023-8nrxx},
	language = {en},
	month = sep,
	publisher = {ChemRxiv},
	shorttitle = {Atlas},
	title = {Atlas: {A} {Brain} for {Self}-driving {Laboratories}},
	urldate = {2023-09-05},
	year = {2023},
}
```
