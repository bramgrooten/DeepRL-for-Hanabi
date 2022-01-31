import os
import pandas as pd
import json
from itertools import product, cycle, repeat

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

layers = [ [128, 128, 64] ]
activation = [ 'Tanh' ]
batch_size = [ 1000 ]
gamma = [0.99]
learning_rate = [3e-4]
train_iters = [1]
# weight_decay = [ 0, 1e-3, 1e-2 ]
# drop_rate = [ 0 , 0.2, 0.5]
obs_vec_type = [ 136 ]
num_actions = [ 11 ]
entropy_coeff = [ 1e-2 ]
renormalize_gt = [ True ]

# epsilon_start = [ 0 ]
# epsilon_decay_type = [ "exponential" ]
# epsilon_low_at_epoch = [ 10**5 ]
# epsilon_low = [ 10**(-3) ]
# shuffle_hand = [ False ]

life_tokens = [ 3 ]
hint_tokens = [ 8 ]

random_seed = [ 3001, 3002, 3003, 3004, 3005 ]


### Name the experiment:
start_date_exp = "2022-01-28"
type_exp = "new_spg"


# For scaling the rewards, except the success play reward
# divide_by = [1]   # [10, 100, 1000]
# for lst in [rew_illegal_move, rew_out_of_lives, rew_lost_one_life, rew_discard, rew_play, rew_hint,
#             rew_discard_playable_card, rew_discard_useless_card, rew_discard_unique_card]:
#     temp = []
#     idx = 0
#     for _ in range(len(lst)):
#         if lst[idx] != 'standard':
#             setting = lst.pop(idx)
#             for scale in divide_by:
#                 temp.append(setting / scale)
#         else:
#             idx += 1
#     lst += temp
#     # print(lst)
# rews = []
# for rew in zip(rew_illegal_move, cycle(rew_out_of_lives), rew_lost_one_life, cycle(rew_discard), rew_play, rew_hint,
#                rew_discard_playable_card, rew_discard_useless_card, rew_discard_unique_card):
#     rews.append(rew)


experiments = []
count = 0
# for rew in rews:
for exp in product(rew_success_play,
                   rew_illegal_move, rew_out_of_lives, rew_lost_one_life,
                   rew_discard, rew_play, rew_hint,
                   rew_discard_playable_card, rew_discard_useless_card, rew_discard_unique_card,
                   layers, activation, batch_size, gamma, learning_rate, train_iters,
                   # weight_decay, drop_rate,
                   obs_vec_type, num_actions, entropy_coeff, renormalize_gt,
                   life_tokens, hint_tokens,  # shuffle_hand,
                   random_seed):
                   # epsilon_start, epsilon_decay_type, epsilon_low_at_epoch, epsilon_low):
    exp_settings = [count] + list(exp)       # + list(rew) + list(exp)
    experiments.append(exp_settings)
    print("Setup experiment", count)
    count += 1
print(f"There are {count} experiments.")


df_experiments = pd.DataFrame(experiments,
                              columns=['num_exp', 'success play', 'illegal move', 'out of lives', 'lost one life',
                                       'discard', 'play', 'hint',
                                       'discard playable card', 'discard useless card', 'discard unique card',
                                       'layers', 'activation', 'batch_size', 'gamma', 'learning_rate', 'train_iters',
                                       # 'weight_decay', 'drop_rate',
                                       'obs_vec_type', 'num_actions', 'entropy_coeff', 'renormalize_gt',
                                       'life_tokens', 'hint_tokens',  # 'shuffle_hand',
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
