import os
import pandas as pd
import json
from itertools import product


### Change settings here:
batch_sizes = [1]
obs_vec_type = [ 37, 62 ]
num_actions = [ 11 ]
layers = [ [32], [32]*2,
           [64], [64]*2 ]
activation = [ 'Tanh', 'ReLU' ]
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]


### Name the experiment:
start_date_exp = "2022-01-28"
type_exp = "supervised_new"



experiments = []
count = 0
for exp in product(batch_sizes, obs_vec_type, num_actions, layers, activation, learning_rates):
    exp_settings = [count] + list(exp)
    experiments.append(exp_settings)
    print(f"Setup experiment {exp_settings}")
    count += 1
print(f"There are {count} experiments.")

df_experiments = pd.DataFrame(experiments,
                              columns=['num_exp', 'batch_sizes', 'obs_vec_type', 'num_actions',
                                       'layers', 'activation', 'learning_rates'])
df_experiments.to_csv(f'exp_settings_{type_exp}.csv', index=False)



runtime_file = f"../data/{start_date_exp}_{type_exp}/training_time.json"
if not os.path.exists(os.path.dirname(runtime_file)):
    os.makedirs(os.path.dirname(runtime_file))

with open(runtime_file, 'w') as the_file:
    the_file.seek(0)
    new_time_dict = {"total_hours": 0}
    json.dump(new_time_dict, the_file)
    the_file.truncate()
