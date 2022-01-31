import os
import sys
import pandas as pd
import utils_exps
Bram_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if Bram_folder not in sys.path:
    sys.path.insert(0, Bram_folder)
from SPG.SPG_cheat import SPG_Agent
'''
Use this file by going into the drl_for_hanabi/experiments/hpc_scripts folder in the terminal, and then running:
>>> python3 ../run_exps/continue_experiment.py 10
where 10 is just an example number. 
It should be the number of the experiment you want to run.
'''


new_session = 3
new_runtime = 80
start_date_exp = "2021-02-18"
type_exp = "spg_entr"


# Getting the experiment number from sys.argv, and then the right settings from the csv file
exp_num = int(sys.argv[1])
df_exp_settings = pd.read_csv(f'../settings/exp_settings_{type_exp}.csv')
exp = df_exp_settings.iloc[exp_num]


# Preprocessing the experiments settings to the right format
rewards = utils_exps.preprocess_reward_settings(exp)
hidden_layers = eval(exp['layers'])
activation = eval("nn." + exp.get('activation', 'Tanh'))
render_file = f"../renders/{start_date_exp}_{type_exp}/s{new_session}/game_traces-for-Experiment_{exp_num}.txt"
save_policy_regularly = f"../policy_params/{start_date_exp}_{type_exp}/regularly/exp_{exp_num}/"
env_config = {"max_life_tokens": exp['life_tokens'],
              "max_information_tokens": exp['hint_tokens'],
              "colors": 5,
              "ranks": 5,
              "players": 2}

algo = SPG_Agent(runtime=new_runtime,
                 show_epoch_nr=50,
                 print_training=False,
                 batch_size=exp['batch_sizes'],
                 lr=exp['learning_rates'],
                 gamma=exp['gammas'],
                 entropy_coeff=exp['entropy_coeff'],
                 reward_settings=rewards,
                 hidden_layers=hidden_layers,
                 activation=activation,
                 obs_vec_type=exp['obs_vec_type'],
                 num_actions=exp['num_actions'],
                 render_training=False,
                 render_tests=False,
                 render_file=render_file,
                 experiment_num=exp_num,
                 save_policy_every_n_epochs=0,  # 10 ** 3,
                 save_policy_folder=save_policy_regularly,
                 env_config=env_config)


old_policy = f"../policy_params/{start_date_exp}_{type_exp}/s{new_session-1}/policy_exp_{exp_num}.pth"
new_policy = f"../policy_params/{start_date_exp}_{type_exp}/s{new_session}/policy_exp_{exp_num}.pth"
data_folder = f"../data/{start_date_exp}_{type_exp}/s{new_session}/"


utils_exps.train_and_test(algo, old_policy, new_policy, data_folder, new_session)


# Updating the total runtime, just once
if exp_num == 0:
    runtime_file = f"../data/{start_date_exp}_{type_exp}/training_time.json"
    utils_exps.update_runtime_json_file(runtime_file, new_runtime, new_session)

