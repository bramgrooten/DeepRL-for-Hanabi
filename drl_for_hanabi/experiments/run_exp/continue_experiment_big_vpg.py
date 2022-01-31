import os
import sys
import pandas as pd
import utils_exps
Bram_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if Bram_folder not in sys.path:
    sys.path.insert(0, Bram_folder)
from VPG.VPG_cheat import VPG_Agent
'''
Use this file by going into the drl_for_hanabi/experiments/hpc_scripts folder in the terminal, and then running:
>>> python3 ../run_exps/continue_experiment.py 10
where 10 is just an example number. 
It should be the number of the experiment you want to run.
'''


new_session = int(sys.argv[3])
new_runtime = int(sys.argv[2])
start_date_exp = "2021-05-05"
type_exp = "big_vpg"


# Getting the experiment number from sys.argv, and then the right settings from the csv file
exp_num = int(sys.argv[1])
df_exp_settings = pd.read_csv(f'../settings/exp_settings_{type_exp}.csv')
exp = df_exp_settings.iloc[exp_num]


# Preprocessing the experiments settings to the right format
rewards = utils_exps.preprocess_reward_settings(exp)
hidden_layers_pi = eval(exp['hidden_layers_pi'])
hidden_layers_vf = eval(exp['hidden_layers_vf'])
activation_pi = eval("nn." + exp.get('activation_pi', 'Tanh'))
activation_vf = eval("nn." + exp.get('activation_vf', 'Tanh'))
render_file = f"../renders/{start_date_exp}_{type_exp}/s{new_session}/game_traces-for-Experiment_{exp_num}.txt"
env_config = {"max_life_tokens": exp['life_tokens'],
              "max_information_tokens": exp['hint_tokens'],
              "colors": 5,
              "ranks": 5,
              "players": 2}


algo = VPG_Agent(runtime=new_runtime,
                 show_epoch_nr=100,
                 print_training=False,
                 lr_pi=exp['lr_pi'],
                 hidden_layers_pi=hidden_layers_pi,
                 activation_pi=activation_pi,
                 train_pi_iters=exp['train_pi_iters'],
                 lr_vf=exp['lr_vf'],
                 hidden_layers_vf=hidden_layers_vf,
                 activation_vf=activation_vf,
                 train_vf_iters=exp['train_vf_iters'],
                 gamma=exp['gamma'],
                 batch_size=exp['batch_size'],
                 entropy_coeff=exp['entropy_coeff'],
                 renormalize_adv=exp['renormalize_adv'],
                 adv_type=exp['adv_type'],
                 lam=exp['lambda'],
                 obs_vec_type=exp['obs_vec_type'],
                 num_actions=exp['num_actions'],
                 reward_settings=rewards,
                 render_training=False,
                 render_tests=False,
                 render_file=render_file,
                 experiment_num=exp_num,
                 env_config=env_config)


old_policy = f"../policy_params/{start_date_exp}_{type_exp}/s{new_session-1}/policy_exp_{exp_num}.pth"
new_policy = f"../policy_params/{start_date_exp}_{type_exp}/s{new_session}/policy_exp_{exp_num}.pth"
data_folder = f"../data/{start_date_exp}_{type_exp}/s{new_session}/"


utils_exps.train_and_test(algo, old_policy, new_policy, data_folder, new_session)


# Updating the total runtime, just once
if exp_num == 0:
    runtime_file = f"../data/{start_date_exp}_{type_exp}/training_time.json"
    utils_exps.update_runtime_json_file(runtime_file, new_runtime, new_session)

