"""Vanilla Policy Gradient Agent."""
import sys
import os
# adding drl folder to sys.path
drl_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.realpath(__file__))))))
if drl_folder not in sys.path:
    sys.path.insert(0, drl_folder)
import random
import torch
import torch.nn as nn
from torch.optim import Adam
from hanabi_learning_environment.rl_env import Agent
from drl_for_hanabi.utils import basics, state_representation_dec, action_representation_dec




class VPG_Dec_Agent(Agent):
    """VPG Agent interface."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent.
        Args:
            config: dict, with settings of the agent. Defaults are shown in the .get() functions.
            Special configs explained:
            - reward_settings: dict, containing the specific reward adjustments
                for example:    reward_settings = { "illegal move": -10,  default: 0
                                                    "out of lives": -50,  default: - current score
                                                    "lost one life": -2,  default: 0
                                                    "discard": -0.05,     default: 0 }
                when a certain reward setting is left out of the dict, the default value is used.
                for example:    reward_settings = { "illegal move": -20 }
            - hidden_layers: list, containing the sizes of the hidden layers
                for example:    hidden_layers = [128, 128, 64]
            - obs_vec_type: int, denoting the different options for observation(state) representation
                options: 7, 37, 62. See file drl_for_hanabi/utils/state_representation.py for details
            - num_actions: int, denoting the different options for action representation
                options: 11, 51. See file drl_for_hanabi/utils action_representation.py for details
            - load_policy: list, denoting the experiment name, session, and sub-experiment number to load from
                for example:    load_policy = ["2021-04-13_combi_vpg", 9, 4]
        """
        self.epochs = config.get('epochs', 50)
        self.runtime = config.get('runtime', 0)
        self.show_epoch_nr = config.get('show_epoch_nr', 10)
        self.print_training = config.get('print_training', True)

        self.lr_pi = config.get('lr_pi', 1e-3)
        self.hidden_layers_pi = config.get('hidden_layers_pi', [64, 64])
        self.activation_pi = eval("nn." + config.get('activation_pi', 'Tanh'))
        self.train_pi_iters = config.get('train_pi_iters', 1)

        self.lr_vf = config.get('lr_vf', 1e-3)
        self.hidden_layers_vf = config.get('hidden_layers_vf', [64, 64])
        self.activation_vf = eval("nn." + config.get('activation_vf', 'Tanh'))
        self.train_vf_iters = config.get('train_vf_iters', 5)

        self.gamma = config.get('gamma', 0.99)  # discount factor
        self.batch_size = config.get('batch_size', 1000)  # number of steps (at least) in one epoch
        self.entropy_coeff = config.get('entropy_coeff', 0.01)
        self.renormalize_adv = config.get('renormalize_adv', True)
        self.adv_type = config.get('adv_type', 'basic')
        self.lam = config.get('lambda', 0.95)

        self.reward_settings = config.get('reward_settings', {})
        self.obs_vec_type = config.get('obs_vec_type', 37)
        self.num_actions = config.get('num_actions', 11)
        self.load_policy = config.get('load_policy', None)

        obs_type_options = {
            7: state_representation_dec.obs_to_vec7_playability,
            37: state_representation_dec.obs_to_vec37_owncards,
            62: state_representation_dec.obs_to_vec62_discards,
            136: state_representation_dec.obs_to_vec136_binary,
            186: state_representation_dec.obs_to_vec186_others_cards
        }
        act_type_options = {
            11: action_representation_dec.choose_action_11position,
            51: action_representation_dec.choose_action_51unique
        }
        self.obs_to_vec_func = obs_type_options[self.obs_vec_type]
        self.action_func = act_type_options[self.num_actions]
        self.policy_network = basics.neural_network(self.obs_vec_type, self.hidden_layers_pi,
                                                    self.num_actions, self.activation_pi)
        self.value_network = basics.neural_network(self.obs_vec_type, self.hidden_layers_vf,
                                                   1, self.activation_vf)
        self.device = basics.get_default_device()
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        self.optimizer_pi = Adam(self.policy_network.parameters(), lr=self.lr_pi)
        self.optimizer_vf = Adam(self.value_network.parameters(), lr=self.lr_vf)
        if self.load_policy is not None:
            load_path = f"{drl_folder}/experiments/policy_params/" \
                        f"{self.load_policy[0]}/s{self.load_policy[1]}/policy_exp_{self.load_policy[2]}.pth"
            self.load_policy_params(load_path)

    def reset(self, config):
        """Reset the agent with a new config"""
        self.__init__(config)

    def act(self, observation, explore=True):
        """Act based on an observation.
        Args:
            observation: dict, containing observation from the view of this agent.
        Returns:
            action: dict, mapping to a legal action taken by this agent.
        """
        # Only act if it's your turn
        if observation['current_player_offset'] != 0:
            return None

        action, act_int, legal = self.choose_action(observation, explore)
        counter = 0
        max_tries = 1e3
        while not legal:
            counter += 1
            action, act_int, legal = self.choose_action(observation, explore)
            if counter > max_tries:
                print(f'Tried to do an illegal action, resampled {max_tries} times, '
                      f'but still not getting a legal action, so doing a random legal move now.')
                return random.choice(observation['legal_moves'])
        if counter > 0:
            print(f'Tried to do an illegal action. Resampled {counter} times before getting a legal action.')
        return action

    def observation_to_vector(self, obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
        """ Options:  (see utils/state_representation.py)
        - obs_to_vec7_playability
        - obs_to_vec37_owncards
        - obs_to_vec62_discards
        """
        return self.obs_to_vec_func(obs)

    def choose_action(self, obs, explore=True):
        """ Options:  (see utils/action_representation.py)
        - choose_action_11position
        - choose_action_51unique
        """
        return self.action_func(self, obs, explore)

    def load_policy_params(self, load_path):
        """Loading the parameters of a previously trained policy."""
        loaded_checkpoint = torch.load(load_path, map_location=self.device)
        # self.current_epoch = loaded_checkpoint["epoch"]
        self.policy_network.load_state_dict(loaded_checkpoint["policy_network"])
        self.optimizer_pi.load_state_dict(loaded_checkpoint["optimizer_pi"])
        self.value_network.load_state_dict(loaded_checkpoint["value_network"])
        self.optimizer_vf.load_state_dict(loaded_checkpoint["optimizer_vf"])
        self.epsilon = loaded_checkpoint.get("epsilon", 0.0)  # default 0, if there was no epsilon saved
        print("\nVPG_Dec_Agent loaded policy parameters from:", load_path)
