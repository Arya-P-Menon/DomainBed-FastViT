# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import getpass
import hashlib
import json
import os
import random
import shutil
import time
import uuid

import numpy as np
import torch

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed import command_launchers

import tqdm
import shlex


class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
                    self.train_args['algorithm'],
                    self.train_args['test_envs'],
                    self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')


def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs. (Original multi-source DG logic)
    """
    assert (n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i + 1, n):
            yield [i, j]


def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, steps,
                   data_dir, task, holdout_fraction, single_test_envs, hparams,
                   single_domain_generalization=False):  # Added new flag
    args_list = []
    for trial_seed in range(n_trials):
        for dataset_name in dataset_names:
            num_envs = datasets.num_environments(dataset_name)

            # Determine the sets of (source_env, test_envs) for this dataset
            if single_domain_generalization:
                # In single DG, for each environment, we treat it as the source
                # and all other environments as the test set for that job.
                configs_for_dataset = []  # This will hold {'source_env_idx', 'test_envs'}
                for source_env_idx in range(num_envs):
                    target_env_indices = [i for i in range(num_envs) if i != source_env_idx]

                    if not target_env_indices:
                        print(
                            f"Warning: Dataset '{dataset_name}' with source environment '{source_env_idx}' has no other environments for testing. Skipping this specific source configuration.")
                        continue

                    # For a single job, 'test_envs' will contain ALL target environments
                    configs_for_dataset.append({
                        'source_env_idx': source_env_idx,  # Just for clarity
                        'test_envs': target_env_indices  # All other environments are test environments for this job
                    })
            else:
                # Original logic for multi-source DG or iterating through single test environments
                configs_for_dataset = []
                if single_test_envs:
                    test_combinations = [[i] for i in range(num_envs)]
                else:
                    test_combinations = all_test_env_combinations(num_envs)

                for test_envs_combo in test_combinations:
                    configs_for_dataset.append({
                        'source_env_idx': None,  # Not applicable for multi-source
                        'test_envs': test_envs_combo
                    })

            for algorithm in algorithms:
                for config in configs_for_dataset:  # Loop through the (source, test_envs) configurations
                    test_envs_for_job = config['test_envs']

                    # --- CORRECTED LOOP FOR HPARAMS_SEED ---
                    for hparams_seed in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset_name
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs_for_job  # Pass the list of all target domains
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['hparams_seed'] = hparams_seed  # <--- NOW hparams_seed IS DEFINED HERE
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['trial_seed'] = trial_seed

                        # The seed hash must uniquely identify this run.
                        # It implicitly includes the source domain because test_envs determines training domains.
                        train_args['seed'] = misc.seed_hash(dataset_name,
                                                            algorithm, test_envs_for_job, hparams_seed, trial_seed)
                        if steps is not None:
                            train_args['steps'] = steps
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
                    # --- END CORRECTED LOOP ---
    return args_list


def ask_for_confirmation():
    # response = input('Are you sure? (y/n) ')
    response = "y"  # Auto-confirm based on your previous code
    if response.lower().strip()[:1] == "y":
        print('Good to go')
    #    exit(0)


DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete_incomplete'])
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=algorithms.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--command_launcher', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--single_test_envs', action='store_true')

    # --- ADD THIS NEW ARGUMENT FOR SINGLE DOMAIN GENERALIZATION ---
    parser.add_argument('--single_domain_generalization', action='store_true',
                        help='If set, each dataset will be treated as a single source domain '
                             'and tested against all other environments in that dataset within the same job.')
    # --- END ADDITION ---

    parser.add_argument('--skip_confirmation', action='store_true')
    args = parser.parse_args()

    args_list = make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        holdout_fraction=args.holdout_fraction,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams,
        single_domain_generalization=args.single_domain_generalization
    )

    jobs = [Job(train_args, args.output_dir) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    # This will print the command for the first job (useful for debugging)
    if jobs:
        print("\nExample command for the first job:")
        print(jobs[0].command_str)
    else:
        print("\nNo jobs generated.")

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state == Job.NOT_LAUNCHED]
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        launcher_fn = command_launchers.REGISTRY[args.command_launcher]
        Job.launch(to_launch, launcher_fn)

    elif args.command == 'delete_incomplete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            ask_for_confirmation()
        Job.delete(to_delete)