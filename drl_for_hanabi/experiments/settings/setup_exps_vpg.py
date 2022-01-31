import os
import pandas as pd
import json
from itertools import product


### Change settings here:
rew_success_play = [ 10 ]
rew_illegal_move = [ -1 ]
rew_out_of_lives = [ 'standard' ]
rew_lost_one_life = [ -0.1 ]
rew_discard = [ 'standard' ]
rew_play = [ 0.02 ]
rew_hint = [ -0.02 ]
rew_discard_playable_card = [ -0.1 ]
rew_discard_useless_card = [ 0.1 ]
rew_discard_unique_card = [ -0.1 ]

lr_pi = [ 3e-4 ]
hidden_layers_pi = [ [128, 128, 64] ]
activation_pi = [ 'Tanh' ]
train_pi_iters = [ 1 ]

lr_vf = [ 3e-4 ]
hidden_layers_vf = [ [128, 64, 32] ]
activation_vf = [ 'Tanh' ]
train_vf_iters = [ 5 ]

gamma = [ 0.99 ]
batch_size = [ 1000 ]
entropy_coeff = [ 1e-2 ]
renormalize_adv = [ True ]
adv_type = [ 'gae' ]
lam = [ 0.95 ]

# weight_decay = [ 0, 1e-3, 1e-2 ]
# drop_rate = [ 0 , 0.2, 0.5]
obs_vec_type = [ 136 ]
num_actions = [ 11 ]
life_tokens = [ 3 ]
hint_tokens = [ 8 ]

# epsilon_start = [ 0 ]
# epsilon_decay_type = [ "exponential" ]
# epsilon_low_at_epoch = [ 10**5 ]
# epsilon_low = [ 10**(-3) ]

random_seed = [ 3001, 3002, 3003, 3004, 3005 ]



### Name the experiment:
start_date_exp = "2022-01-28"
type_exp = "new_vpg"



experiments = []
count = 0
for exp in product(rew_success_play,                                                               # 0
                   rew_illegal_move, rew_out_of_lives, rew_lost_one_life,                          # 0, 1, 2
                   rew_discard, rew_play, rew_hint,                                                # 3, 4, 5
                   rew_discard_playable_card, rew_discard_useless_card, rew_discard_unique_card,   # 6, 7, 8
                   lr_pi, hidden_layers_pi, activation_pi, train_pi_iters,                         # 9, 10, 11, 12
                   lr_vf, hidden_layers_vf, activation_vf, train_vf_iters,                         # 13, 14, 15, 16
                   gamma, batch_size, entropy_coeff, renormalize_adv,                              # 17, 18, 19, 20
                   adv_type, lam,                                                                  # 21, 22
                   # weight_decay, drop_rate,
                   obs_vec_type, num_actions, life_tokens, hint_tokens,                            # 23, 24, 25, 26
                   random_seed):
                   # epsilon_start, epsilon_decay_type, epsilon_low_at_epoch, epsilon_low):
    # if len(exp[11]) != len(exp[15]):   # hidden_layers_pi != hidden_layers_vf:
    #     continue
    exp_settings = [count] + list(exp)
    experiments.append(exp_settings)
    print("Setup experiment", count)
    count += 1
print(f"There are {count} experiments.")

df_experiments = pd.DataFrame(experiments,
                              columns=['num_exp', 'success play', 'illegal move', 'out of lives', 'lost one life',
                                       'discard', 'play', 'hint',
                                       'discard playable card', 'discard useless card', 'discard unique card',
                                       'lr_pi', 'hidden_layers_pi', 'activation_pi', 'train_pi_iters',
                                       'lr_vf', 'hidden_layers_vf', 'activation_vf', 'train_vf_iters',
                                       'gamma', 'batch_size', 'entropy_coeff', 'renormalize_adv',
                                       'adv_type', 'lambda',
                                       # 'weight_decay', 'drop_rate',
                                       'obs_vec_type', 'num_actions', 'life_tokens', 'hint_tokens',
                                       'random_seed'])
                                       # 'epsilon_start', 'epsilon_decay_type', 'epsilon_low_at_epoch', 'epsilon_low'])


df_experiments.to_csv(f'exp_settings_{type_exp}.csv', index=False)

runtime_file = f"../data/{start_date_exp}_{type_exp}/training_time.json"
if not os.path.exists(os.path.dirname(runtime_file)):
    os.makedirs(os.path.dirname(runtime_file))

with open(runtime_file, 'w') as the_file:
    the_file.seek(0)
    new_time_dict = {"total_hours": 0}
    json.dump(new_time_dict, the_file)
    the_file.truncate()
