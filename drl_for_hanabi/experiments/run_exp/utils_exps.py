import os
import json
import random
import torch
import numpy as np


def set_random_seed(seed_nr):
    torch.manual_seed(seed_nr)
    random.seed(seed_nr)
    np.random.seed(seed_nr)


def update_runtime_json_file(runtime_file, new_runtime, new_session):
    if not os.path.exists(runtime_file):
        if not os.path.exists(os.path.dirname(runtime_file)):
            os.makedirs(os.path.dirname(runtime_file))
        with open(runtime_file, 'w') as the_file:
            the_file.seek(0)
            new_time_dict = {"total_hours": 0}
            json.dump(new_time_dict, the_file)
            the_file.truncate()

    with open(runtime_file, 'r+') as the_file:
        dict_time = json.load(the_file)
        dict_time["total_hours"] += new_runtime
        dict_time[f"session{new_session}"] = new_runtime
        the_file.seek(0)
        json.dump(dict_time, the_file)
        the_file.truncate()


def train_and_test(algo, old_policy, new_policy, data_folder, new_session, test_episodes=50):
    if new_session > 1:
        algo.load_policy(load_path=old_policy)
    algo.train()
    algo.store_training_data(folder=f"{data_folder}s{new_session}/")
    algo.save_policy(save_path=new_policy)
    algo.test(episodes=1000)  # test_episodes)
    algo.store_testing_data(folder=f"{data_folder}s{new_session}/")


def preprocess_reward_settings(exp):
    possible_reward_adjustments = ["success play",
                                   "illegal move",
                                   "out of lives",
                                   "lost one life",
                                   "discard",
                                   "play",
                                   "hint",
                                   "discard playable card",
                                   "discard useless card",
                                   "discard unique card"]
    rewards = {}
    for rew_adj in possible_reward_adjustments:
        if exp.get(rew_adj, 'standard') != 'standard':
            rewards[rew_adj] = (eval(exp[rew_adj]) if isinstance(exp[rew_adj], str) else exp[rew_adj])
    return rewards
