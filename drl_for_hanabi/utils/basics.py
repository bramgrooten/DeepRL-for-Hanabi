import os
import torch
import torch.nn as nn
import numpy as np


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def neural_network(obs_dim: int, hidden_layers: list, act_dim: int, activation=nn.Tanh):
    # Build a feedforward neural network.
    layers = []
    if hidden_layers:
        layers += [nn.Linear(obs_dim, hidden_layers[0]), activation()]
        for i in range(len(hidden_layers) - 1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i + 1]), activation()]
        layers += [nn.Linear(hidden_layers[-1], act_dim), nn.Identity()]
    else:
        layers += [nn.Linear(obs_dim, act_dim), nn.Identity()]
    return nn.Sequential(*layers)


def save_model_and_tests_after_n_epochs(self):
    self.test(episodes=1000)
    self.store_testing_data(folder=f'{self.save_data_folder}/epoch{self.current_epoch}/')
    self.test_data = []
    param_path = f'{os.path.dirname(os.path.dirname(self.save_params_path))}/' \
                 f'epoch{self.current_epoch}/params_exp{self.experiment_num}.pth'
    self.save_policy(save_path=param_path)


def reward_to_go(rewards, gamma=1.0):
    # Computes the total discounted reward that came after every time step (G_t)
    n = len(rewards)
    rtgs = np.zeros_like(rewards, dtype='float64')
    rtgs[n-1] = rewards[n-1]
    for i in reversed(range(n-1)):
        rtgs[i] = rewards[i] + gamma * rtgs[i + 1]
    return rtgs


def discount_cumsum(x, discount):
    """
     Computing discounted cumulative sums of vector x.
     Same functionality as reward_to_go, but now for torch tensors.
     input: vector x, torch tensor of one dimension
         [x0,
          x1,
          x2]
     output:
         [x0 + discount * x1 + discount^2 * x2,
          x1 + discount * x2,
          x2]
    """
    n = len(x)
    dcs = torch.zeros_like(x)
    dcs[n-1] = x[n-1]
    for i in reversed(range(n-1)):
        dcs[i] = x[i] + discount * dcs[i+1]
    return dcs
