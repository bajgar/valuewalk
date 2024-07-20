This repository contains the code for the paper "Walking the Values in Bayesian Inverse Reinforcement Learning" by Ondrej Bajgar, Alessandro Abate, Konstantinos Gatsis, and Michael A. Osborne, published at UAI 2024.
The article is available here: https://arxiv.org/abs/2407.10971

## Installation
To install the required packages, run:
```
pip install -r requirements.txt
```
Our experiments were run using Python 3.10, though we expect the code to work also with newer versions of Python.

## Running the experiments

### 1. Gridworld
The complete set of experiments on the GridWorld can be found in the Jupyter notebook `experiments/birl/notebooks/01_gridworld_experiments.ipynb`.

### 2. Classic control environments
The experiment files can be found in the `experiments/birl` folder. To run a single-chain experiment on Cartpole, you can run
```
PYTHONPATH=. python3 experiments/birl/06b_vw_cartpole_loop.py --split 0 --num_trajs 7
```
in the repo directory, where `--split` controls the data split, and `--num_trajs` is the number of expert trajectories (between 1 and 15). The directory contains also corresponding experiment files for the other two environments. The complete set of experiments for the paper can be reproduced by running each experiment for splits 0-4, and num_trajs 1,3,7,10,15, each repeated 4 times (to generate 4 parallel chains, which are then merged during result analysis as is often done in MCMC).

The resulting models will, by default, be saved in `~/results/irl-torch/birl/`. The complete result analysis can be found in `02_results_on_classic_control_envs.ipynb`.

## The algorithm
To understand the algorithm, it's probably best to start with the finite space version, which can be found in `irl_algorithms/value_walk/value_walk_tabular.py`. The continuous-state version is in the same directory. It may help to get basic familiarity with Pyro first.

___

I (Ondrej Bajgar) am continuing work on Bayesian IRL and am interested in talking to others in the area (there aren't many of us), so please do get in touch if you're interested. If you want to build on this repo, I may also be able to give you a quick walkthrough into the structure and how to extend it and possibly share a more recent version of the repo.