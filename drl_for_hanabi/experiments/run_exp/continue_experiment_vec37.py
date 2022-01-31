import os
import sys
import json
import pandas as pd
import torch.nn as nn
from datetime import datetime
start_time = datetime.now()
Bram_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if Bram_folder not in sys.path:
    sys.path.insert(0, Bram_folder)
from SPG.a12_SPG_cheat_out51 import SPG_Agent
'''
Use this file by going into the drl_for_hanabi/experiments/hpc_scripts folder in the terminal, and then running:
>>> python3 ../run_exps/continue_experiment.py 10
where 10 is just an example number. 
It should be the number of the experiment you want to run.
'''


new_session = 15
new_runtime = 80
start_date_exp = "2020-12-16"
type_exp = "vec37"


# Getting the experiment number from sys.argv, and then the right settings from the csv file
exp_num = int(sys.argv[1])
df_exp_settings = pd.read_csv('../settings/exp_settings_48_vec37.csv')
exp = df_exp_settings.iloc[exp_num]
print("\nEXPERIMENT SETTINGS")
print(exp.to_frame().T)

# Preprocessing the experiments settings to the right format
possible_reward_adjustments = ["out of lives", "illegal move", "lost one life", "discard"]
rewards = {}
for rew_adj in possible_reward_adjustments:
    if exp[rew_adj] != 'standard':
        rewards[rew_adj] = (eval(exp[rew_adj]) if isinstance(exp[rew_adj],str) else exp[rew_adj])
hidden_layers = eval(exp['layers'])
# activation = eval("nn." + exp['activation'])
render_file = f"../renders/{start_date_exp}_{type_exp}/s{new_session}/game_traces-for-Experiment_{exp_num}.txt"

algo = SPG_Agent(epochs=10,
                 runtime=new_runtime,
                 show_epoch_nr=100,
                 print_training=False,
                 batch_size=exp['batch_sizes'],
                 lr=exp['learning_rates'],
                 gamma=exp['gammas'],
                 reward_settings=rewards,
                 hidden_layers=hidden_layers,
                 activation=nn.Tanh,   # activation,
                 obs_vec_type=37,
                 num_actions=11,
                 render_training=False,
                 render_tests=False,
                 render_file=render_file,
                 experiment_num=exp_num,
                 test_episodes=50,
                 save_policy_every_n_epochs=10 ** 3,
                 save_policy_folder=f"../policy_params/{start_date_exp}_{type_exp}/regularly/exp_{exp_num}/")

# Updating the total runtime, just once
if exp_num == 0 or (type_exp == "vec37" and exp_num == 2):
    runtime_file = f"../data/{start_date_exp}_{type_exp}/training_time.json"
    with open(runtime_file, 'r+') as the_file:
        dict_time = json.load(the_file)
        time_so_far = dict_time["total_hours"]
        the_file.seek(0)
        new_time_dict = {"total_hours": time_so_far + new_runtime}
        json.dump(new_time_dict, the_file)
        the_file.truncate()

old_policy = f"../policy_params/{start_date_exp}_{type_exp}/s{new_session-1}/policy_exp_{exp_num}.pth"
new_policy = f"../policy_params/{start_date_exp}_{type_exp}/s{new_session}/policy_exp_{exp_num}.pth"
data_folder = f"../data/{start_date_exp}_{type_exp}/s{new_session}/"

algo.load_policy(load_path=old_policy)
algo.train()
algo.store_training_data(folder=data_folder)
algo.save_policy(save_path=new_policy)
algo.test()
algo.store_testing_data(folder=data_folder)




print("\nRunning time:", datetime.now() - start_time)
