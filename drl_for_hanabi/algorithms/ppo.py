import sys
import os
# adding drl folder to sys.path
drl_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if drl_folder not in sys.path:
    sys.path.insert(0, drl_folder)
from datetime import datetime, timedelta
import errno
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torchsummary import summary
from hanabi_learning_environment import rl_env, rl_env_adjusted
from drl_for_hanabi.utils import basics, state_representation, action_representation, reward_shaping, rendering, epsilon_decay

"""
This file implements the Proximal Policy Optimization algorithm
It acts as one centralized agent, in cheat mode.
This agent can choose to "pass" (by giving a random hint)

You have the option to specify:
    a number of epochs
             or
    a maximum runtime in hours
Priority is given to the runtime, except when runtime <= 0
"""

### Training time
RUNTIME = -0.1 / 60
EPOCHS = 10

### Hyperparameters
LR_PI = 1e-3
HIDDEN_LAYERS_PI = [64, 64]
ACTIVATION_PI = nn.Tanh
TRAIN_PI_ITERS = 5

LR_VF = 1e-3
HIDDEN_LAYERS_VF = [64, 64]
ACTIVATION_VF = nn.Tanh
TRAIN_VF_ITERS = 5

CLIP_PARAM = 0.2
GAMMA = 0.99
BATCH_SIZE = 1000
ENTROPY_COEFF = 1e-3
RENORMALIZE_ADVANTAGE = True
ADV_TYPE = 'gae'   # options: gae, basic, no
LAM = 0.95
OBS_VEC_TYPE = 186   # possible observation vector types: 7, 37, 62, 82, 136, 186
NUM_ACTIONS = 11     # possible number of actions: 11, 51

### Exploration
EPSILON_START = 0
EPSILON_DECAY_TYPE = "exponential"  # options: exponential, linear, no
EPSILON_LOW_AT_EPOCH = 10 ** 4
EPSILON_LOW = 10 ** (-3)  # define what low is

### Reward shaping
ADJUST_REW_SUCCESS_PLAY = False
REW_SUCCESS_PLAY = 1
ADJUST_REW_ILLEGAL_MOVE = True
REW_ILLEGAL_MOVE = -1
ADJUST_REW_OUT_OF_LIVES = False
REW_OUT_OF_LIVES = -30
ADJUST_REW_LOST_ONE_LIFE = False
REW_LOST_ONE_LIFE = -0.1
ADJUST_REW_DISCARD = False
REW_DISCARD = -0.01
ADJUST_REW_PLAY = False
REW_PLAY = 0.02
ADJUST_REW_HINT = False
REW_HINT = -0.01
ADJUST_REW_DISCARD_PLAYABLE = False
REW_DISCARD_PLAYABLE = -0.01
ADJUST_REW_DISCARD_USELESS = False
REW_DISCARD_USELESS = 0.01
ADJUST_REW_DISCARD_UNIQUE = False
REW_DISCARD_UNIQUE = -0.01

### Environment
MAX_LIFE_TOKENS = 3    # default: 3
MAX_HINT_TOKENS = 8    # default: 8

### Rendering
RENDER_TRAINING = False
RENDER_TESTS = False
RENDER_FILE = f"renders/{datetime.now().strftime('%Y-%m-%d')}/" \
              f"game_traces-for-{os.path.splitext(os.path.basename(__file__))[0]}.txt"
TEST_EPISODES = 5

### Data
SHOW_EPOCH = 1
PRINT_TRAINING = True
STORE_TRAIN_DATA = False
STORE_TEST_DATA = False
DATA_FOLDER = f"data/ppo/{datetime.now().strftime('%Y-%m-%d')}/"
SAVE_PARAMS_EVERY = 0   # set to e.g. 1000 if you want to save the policy after every 1000 epochs. 0 means: don't save


class PPO_Agent:
    """
    class for the Proximal Policy Optimization algorithm
    """
    def __init__(self, epochs=50, runtime=0.0, show_epoch_nr=10, print_training=True,
                 lr_pi=1e-3, hidden_layers_pi=None, activation_pi=nn.Tanh, train_pi_iters=10,
                 lr_vf=1e-3, hidden_layers_vf=None, activation_vf=nn.Tanh, train_vf_iters=10,
                 clip_param=0.2, gamma=1.0, batch_size=1000, entropy_coeff=1e-3,
                 renormalize_adv=True, adv_type='basic', lam=0.95,
                 obs_vec_type=62, num_actions=11, reward_settings=None,
                 epsilon_start=0.0, epsilon_decay_type="exponential", epsilon_low_at_epoch=10 ** 4,
                 epsilon_low=10 ** (-3),
                 render_training=False, render_tests=False, render_file=None,
                 experiment_num=-1, env_config=None,
                 save_model_every_n_epochs=1e5, save_data_folder="data/",
                 save_params_path=f"params/{datetime.now().strftime('%Y-%m-%d')}/params.pth"):

        self.epochs = epochs
        self.runtime = runtime
        self.show_epoch_nr = show_epoch_nr
        self.print_training = print_training

        self.lr_pi = lr_pi
        self.hidden_layers_pi = [] if (hidden_layers_pi is None) else hidden_layers_pi
        self.activation_pi = activation_pi
        self.train_pi_iters = train_pi_iters

        self.lr_vf = lr_vf
        self.hidden_layers_vf = [] if (hidden_layers_vf is None) else hidden_layers_vf
        self.activation_vf = activation_vf
        self.train_vf_iters = train_vf_iters

        self.clip_param = clip_param
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.renormalize_adv = renormalize_adv
        self.adv_type = adv_type
        self.lam = lam

        switcher_obs = {
            7: state_representation.obs_to_vec7_playability,
            37: state_representation.obs_to_vec37_owncards,
            62: state_representation.obs_to_vec62_discards,
            82: state_representation.obs_to_vec82_one_hot_ranks,
            136: state_representation.obs_to_vec136_binary,
            186: state_representation.obs_to_vec186_others_cards
        }
        self.obs_to_vec_func = switcher_obs.get(obs_vec_type, "Unknown observation_vector_type given.")
        switcher_act = {
            11: action_representation.choose_action_11position,
            51: action_representation.choose_action_51unique
        }
        self.action_func = switcher_act.get(num_actions, "Unknown action_type given.")
        self.num_actions = num_actions
        self.reward_settings = {} if (reward_settings is None) else reward_settings

        self.epsilon = epsilon_start
        self.epsilon_decay_type = epsilon_decay_type
        if epsilon_start > 0:
            self.epsilon_decay_factor = epsilon_decay.compute_decay_factor(
                epsilon_start, epsilon_decay_type, epsilon_low_at_epoch, epsilon_low)

        self.finished_rendering_this_epoch = False
        self.render_training = render_training
        self.render_tests = render_tests
        self.render = (render_training or render_tests)

        self.render_file = None
        if self.render:
            if render_file is None:
                raise ValueError("Provide a render_file or set both render_training and render_tests to False.")
            else:
                render_folder = os.path.dirname(render_file)
                try:
                    os.makedirs(render_folder)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise  # raises the error again, if it's something else than "folder already exists"
                self.render_file = open(render_file, "w")

        self.experiment_num = experiment_num
        self.current_epoch = 0
        self.test_episode = 0
        self.train_data = []
        self.test_data = []
        self.save_model_every_n_epochs = save_model_every_n_epochs
        self.save_data_folder = save_data_folder
        self.save_params_path = save_params_path
        self.illegal_move_max = max(100, self.batch_size)

        # make environment, check spaces, get obs / act dims
        if env_config is None:
            self.env = rl_env.make(environment_name="Hanabi-Cheat", num_players=2)
            self.max_life_tokens = 3
            self.max_hint_tokens = 8
        else:
            self.env = rl_env_adjusted.make(obs_type="CHEAT", config=env_config)
            self.max_life_tokens = env_config.get("max_life_tokens", 3)
            self.max_hint_tokens = env_config.get("max_information_tokens", 8)
        obs0 = self.env.reset()
        self.obs_dim = len(self.observation_to_vector(obs0))
        self.act_dim = num_actions

        self.policy_network = basics.neural_network(self.obs_dim, self.hidden_layers_pi,
                                                    self.act_dim, self.activation_pi)
        self.value_network = basics.neural_network(self.obs_dim, self.hidden_layers_vf,
                                                   1, self.activation_vf)
        # self.print_networks(obs0)  # to print a summary of the networks once
        self.device = basics.get_default_device()
        print("Working on device:", self.device)
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        self.optimizer_pi = Adam(self.policy_network.parameters(), lr=self.lr_pi)
        self.optimizer_vf = Adam(self.value_network.parameters(), lr=self.lr_vf)

    def __del__(self):
        if self.render_file is not None:
            self.render_file.close()

    def get_policy(self, obs_vec_torch):
        """Returns the action distribution for a given observation.
        You may also give a batch of observations as input,
        which will produce a batch of action distributions."""
        logits = self.policy_network(obs_vec_torch.to(self.device))
        return Categorical(logits=logits)

    def get_value_estimate(self, obs_vec_torch):
        """Returns the estimated value corresponding to a given (batch of) observation(s)."""
        return self.value_network(obs_vec_torch.to(self.device)).squeeze()

    def print_networks(self, obs):
        """Visualizes the networks, for a given initial observation."""
        obs_vec = self.observation_to_vector(obs)
        print("\nPolicy network:")
        print("example input (observation):", obs_vec)
        print("input shape: vector of length", len(obs_vec))
        print("output shape: vector of length", self.act_dim)
        summary(self.policy_network, input_size=(1, 1, len(obs_vec)))
        print("\nValue function network:")
        summary(self.value_network, input_size=(1, 1, len(obs_vec)))

    def observation_to_vector(self, obs):
        """Converts obs dict to vector representation. See utils/state_representation.py"""
        return self.obs_to_vec_func(obs, max_life=self.max_life_tokens, max_hint=self.max_hint_tokens)

    def choose_action(self, obs, obs_vec, pick_max_prob=False, epsilon=0.0):
        """Samples an action given the observation. See utils/action_representation.py"""
        return self.action_func(self, obs, obs_vec, pick_max_prob=pick_max_prob, epsilon=epsilon)

    def get_avg_probs(self, policy):
        """Computes the average policy probabilities over this batch of observations.
        Args: policy, Categorical object with a batch of policies
        Returns: avg_probs, numpy list of average policy probabilities
        """
        return torch.mean(policy.probs, dim=0).detach().cpu().numpy()

    def compute_advantage_basic(self, ep_obs, ep_rews):
        """Computes the basic advantage function (A=Q-V),
        with estimates g_t for Q and value network output for V"""
        with torch.no_grad():
            obs = torch.as_tensor(ep_obs, dtype=torch.float32)
            g_t = list(basics.reward_to_go(ep_rews, self.gamma))
            g_t = torch.as_tensor(g_t, dtype=torch.float32).to(self.device)
            adv = g_t - self.get_value_estimate(obs)
            return adv, g_t

    def compute_advantage_gae(self, ep_obs, ep_rews):
        """Computes the Generalized Advantage Estimation (GAE)"""
        with torch.no_grad():
            obs = torch.as_tensor(ep_obs, dtype=torch.float32)
            values = self.get_value_estimate(obs)
            last_val = torch.tensor([0]).to(self.device)
            values_extended = torch.cat((values, last_val))
            rews = torch.as_tensor(ep_rews, dtype=torch.float32).to(self.device)
            deltas = rews + self.gamma * values_extended[1:] - values_extended[:-1]
            adv = basics.discount_cumsum(deltas, self.gamma * self.lam)
            v_target = values + adv
            return adv, v_target

    def get_old_logp(self, obs, act):
        """Gets the log probabilities of the current policy,
        corresponding to a given (observation, action) pair.
        Used by compute_loss_pi to compute the ratio between new and old policies.
        Args:
            obs: torch tensor, a batch of observation vectors
            act: torch tensor, a batch of action integers, corresponding to above observations
        Returns:
            logp_old: torch tensor, a batch of log probability values
            pi_info: dict, extra info about the current policy (for logging)
        """
        with torch.no_grad():
            policy = self.get_policy(obs)
            logp_old = policy.log_prob(act.to(self.device))
            # Also recording some extra info for logging:
            avg_probs = None
            if self.current_epoch % self.show_epoch_nr == 0:
                avg_probs = self.get_avg_probs(policy)
            mean_entropy = torch.mean(policy.entropy())
            pi_info = dict(entropy=mean_entropy, avg_probs=avg_probs)
            return logp_old, pi_info

    def compute_loss_pi(self, obs, act, advantage, logp_old):
        policy = self.get_policy(obs)
        logp = policy.log_prob(act.to(self.device))

        ratio = torch.exp(logp - logp_old)
        clipped_ratio = torch.clamp(ratio, 1-self.clip_param, 1+self.clip_param)
        loss_clip_term = - torch.min(ratio * advantage, clipped_ratio * advantage).mean()

        mean_entropy = torch.mean(policy.entropy())
        loss_entropy_term = - self.entropy_coeff * mean_entropy
        loss_pi = loss_clip_term + loss_entropy_term

        # For logging:
        clipped = ratio.gt(1+self.clip_param) | ratio.lt(1-self.clip_param)
        clip_frac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        return loss_pi, clip_frac

    def compute_loss_vf(self, obs, v_target, update_iter=0):
        """Computes the loss for the value network.
        Args:
            obs: torch tensor, a batch of observation vectors
            v_target: torch tensor, a batch of targets for the value function
            update_iter: int, number of the update iteration within this epoch (in [0,train_vf_iters])
        Returns:
            loss_vf: torch tensor, single element with the loss of the value network
            avg_value: float (or None), the average output of the value network, only not-None for first update iteration
        """
        value_estimate = self.get_value_estimate(obs)
        avg_value = None
        if self.current_epoch % self.show_epoch_nr == 0 and update_iter == 0:
            avg_value = torch.mean(value_estimate).item()
        return ((value_estimate - v_target) ** 2).mean(), avg_value

    def update_step(self, obs, act, advantage, v_target):
        """Updates the parameters of both the policy network and the value network.
        Performs train_pi_iters number of updates to the policy network,
        and train_vf_iters number of updates to the value network.

        Args:
            obs: torch tensor, of observation vectors
            act: torch tensor, of actions (integers)
            advantage: torch tensor, of advantages (floats)
            v_target: torch tensor, of value targets (floats)
        Returns:
            loss_pi: torch tensor, single element giving the loss of the policy network
            loss_vf: torch tensor, single element giving the last loss of the value function network
            pi_info: dict, containing information about the policy (for logging)
            avg_value: float, the average output of the value network this epoch (after first update iteration)
        """
        # compute data needed for policy updates
        avg_value = None
        logp_old, pi_info = self.get_old_logp(obs, act)
        if self.renormalize_adv:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # update the policy network
        for _ in range(self.train_pi_iters):
            self.optimizer_pi.zero_grad()
            loss_pi, clip_frac = self.compute_loss_pi(obs, act, advantage, logp_old)
            loss_pi.backward()
            self.optimizer_pi.step()
            # print("clipped:", clip_frac)
        pi_info['clip_frac'] = clip_frac

        # update the value network
        for iteration in range(self.train_vf_iters):
            self.optimizer_vf.zero_grad()
            loss_vf, value = self.compute_loss_vf(obs, v_target, iteration)
            loss_vf.backward()
            self.optimizer_vf.step()
            if iteration == 0:
                avg_value = value  # just for logging the avg value estimate
        return loss_pi, loss_vf, pi_info, avg_value


    def run_one_episode(self, training=True, pick_max_prob=False):
        # make some empty lists for logging.
        ep_obs, ep_acts = [], []  # for observations, actions
        ep_ret, ep_len, ep_firework = 0, 0, 0  # for episode return, length, and total fireworks regardless of lives
        turn, illegal_move_count = 0, 0
        moves_legal = []
        ep_obs_dicts = []  # list of all obs dictionaries, to do reward shaping at the end
        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout episode

        # do we need to render this episode?
        if training:
            self.render = (self.render_training and not self.finished_rendering_this_epoch)
        else:  # testing mode
            self.render = self.render_tests
            self.epsilon = 0.0

        if self.render:
            rendering.print_start_of_episode(self, training)

        while not done:
            turn += 1

            obs_vec = self.observation_to_vector(obs)
            ep_obs.append(obs_vec)
            ep_obs_dicts.append(obs)
            if self.render:
                rendering.print_current_state(self, turn, obs)

            # act in the environment
            act_int, legal = self.choose_action(obs, obs_vec, pick_max_prob=pick_max_prob, epsilon=self.epsilon)
            moves_legal.append(legal)

            if legal:
                action = (action_representation.convert_hint_to_HLE_int(obs) if act_int == 10 else act_int)
                obs, rew, done, _ = self.env.step(action)
            else:
                illegal_move_count += 1
                rew = self.reward_settings.get("illegal move", 0)
                # state remains the same, but reward may change

            if self.render:
                rendering.print_action(self, act_int, legal)
            # save action, reward
            ep_acts.append(act_int)
            ep_rews.append(rew)

            if done or illegal_move_count > self.illegal_move_max:
                ep_obs_dicts.append(obs)  # save final state of the game
                break

        # episode is over: record info about episode
        ep_rews = reward_shaping.adjust_rewards(self, ep_rews, ep_obs_dicts, ep_acts, moves_legal)
        ep_ret = sum(ep_rews)
        ep_len = len(ep_rews)
        ep_lives = obs['player_observations'][0]['life_tokens']
        for rank in obs['player_observations'][0]['fireworks'].values():
            ep_firework += rank
        ep_score = (ep_firework if ep_lives > 0 else 0)
        if self.adv_type == 'gae' or self.adv_type == 'GAE':
            ep_weights, v_target = self.compute_advantage_gae(ep_obs, ep_rews)
        elif self.adv_type == 'basic':
            ep_weights, v_target = self.compute_advantage_basic(ep_obs, ep_rews)
        else:  # just use reward-to-go from t (G_t)
            ep_weights = torch.as_tensor(list(basics.reward_to_go(ep_rews, gamma=self.gamma)),
                                         dtype=torch.float32).to(self.device)
            v_target = ep_weights

        if self.render:
            rendering.print_end_of_episode(self, obs, ep_score)
        # if illegal_move_count > 0:
        #     print("Tried to do", illegal_move_count, "illegal moves during this episode.")
        return ep_obs, ep_acts, ep_weights, ep_ret, ep_len, \
               ep_firework, ep_score, ep_lives, illegal_move_count, v_target

    # for training the policy
    def train_one_epoch(self):
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for weighting in the objective
        batch_v_targets = []  # for the value network targets
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_fireworks = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_scores = []
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode
        num_eps = 0  # to count the number of episodes in this batch
        total_illegal = 0  # to count the number of illegal moves in this batch
        avg_target = 0  # the target for the value function

        # render first episode of each epoch
        self.finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while len(batch_obs) < self.batch_size:
            ep_obs, ep_acts, ep_weights, ep_ret, ep_len, ep_firework, ep_score, ep_lives, il_moves, ep_v_target = \
                self.run_one_episode(training=True)

            batch_obs += ep_obs
            batch_acts += ep_acts
            batch_weights.append(ep_weights)
            batch_v_targets.append(ep_v_target)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_fireworks.append(ep_firework)
            batch_scores.append(ep_score)
            batch_lives.append(ep_lives)
            num_eps += 1
            total_illegal += il_moves

            # won't render this epoch again
            self.finished_rendering_this_epoch = True

        # convert data to torch tensors
        obs = torch.as_tensor(batch_obs, dtype=torch.float32)
        act = torch.as_tensor(batch_acts, dtype=torch.int32)
        advantage = torch.cat(batch_weights)
        v_target = torch.cat(batch_v_targets)

        # update the parameters of both networks
        loss_pi, loss_vf, pi_info, avg_value = self.update_step(obs, act, advantage, v_target)

        mean_entropy = pi_info['entropy']
        avg_probs = pi_info['avg_probs']
        clip_frac = pi_info['clip_frac']

        if self.current_epoch % self.show_epoch_nr == 0:
            avg_target = torch.mean(v_target).item()
        # if self.epsilon > 0:
        #     self.epsilon = epsilon_decay.compute_next_epsilon(self.epsilon, self.epsilon_decay_type,
        #                                                       self.epsilon_decay_factor)
        return loss_pi, loss_vf, batch_rets, batch_fireworks, batch_scores, batch_lens, batch_acts, batch_lives, \
               num_eps, len(batch_obs), total_illegal, mean_entropy, avg_probs, avg_value, avg_target, clip_frac

    # training loop
    def train(self):
        print(f'\nTraining the algorithm: {os.path.basename(__file__)}')
        if self.runtime <= 0:
            self.train_n_epochs()
        else:
            self.train_n_hours()

    # train based on maximum runtime
    def train_n_hours(self):
        avg_loss_pi, avg_loss_vf, avg_rets, avg_fire, avg_score = [], [], [], [], []
        avg_lens, avg_lives, avg_eps, avg_batch_size, avg_illegal = [], [], [], [], []
        avg_entropy, avg_clip_frac = [], []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.runtime)
        print('Training for:\t\t {} hours'.format(self.runtime))
        print('Start time:\t\t', start_time)
        print('End time:\t\t', end_time)
        while datetime.now() < end_time:
            loss_pi, loss_vf, batch_rets, batch_fire, batch_score, batch_lens, batch_acts, batch_lives, \
            num_eps, batch_sz, num_ill, mean_entropy, avg_probs, avg_value, avg_target, clip_frac = self.train_one_epoch()

            avg_loss_pi.append(loss_pi.item())
            avg_loss_vf.append(loss_vf.item())
            avg_rets.append(np.mean(batch_rets))
            avg_fire.append(np.mean(batch_fire))
            avg_score.append(np.mean(batch_score))
            avg_lens.append(np.mean(batch_lens))
            avg_lives.append(np.mean(batch_lives))
            avg_eps.append(num_eps)
            avg_batch_size.append(batch_sz)
            avg_illegal.append(num_ill)
            avg_entropy.append(mean_entropy.item())
            avg_clip_frac.append(clip_frac)

            if self.current_epoch % self.show_epoch_nr == 0:
                self.train_data.append([self.current_epoch, np.mean(avg_loss_pi), np.mean(avg_loss_vf),
                                        np.mean(avg_rets), np.mean(avg_fire),
                                        np.mean(avg_score), np.mean(avg_lives),
                                        np.mean(avg_lens), np.mean(avg_eps), np.mean(avg_batch_size),
                                        np.mean(avg_illegal), self.epsilon, np.mean(avg_entropy),
                                        avg_probs, avg_value, avg_target, np.mean(avg_clip_frac)])
                if self.print_training:
                    rendering.print_training_epoch_results_VPG(self.current_epoch, avg_loss_pi, avg_loss_vf, avg_rets,
                                                               avg_fire, avg_score, avg_lives, avg_lens, avg_eps,
                                                               avg_batch_size, avg_illegal, avg_entropy)
                avg_loss_pi, avg_loss_vf, avg_rets, avg_fire, avg_score = [], [], [], [], []
                avg_lens, avg_lives, avg_eps, avg_batch_size, avg_illegal = [], [], [], [], []
                avg_entropy, avg_clip_frac = [], []

            if self.save_model_every_n_epochs > 0:
                if self.current_epoch % self.save_model_every_n_epochs == 0 and self.current_epoch != 0:
                    basics.save_model_and_tests_after_n_epochs(self)

            self.current_epoch += 1

        print('Last epoch info:')
        rendering.print_training_epoch_results_VPG(self.current_epoch, loss_pi.item(), loss_vf.item(),
                                                   batch_rets, batch_fire, batch_score, batch_lives, batch_lens,
                                                   num_eps, batch_sz, num_ill, mean_entropy.item())
        print('Actions:\t\t', batch_acts)
        print('Finished running at:\t', datetime.now())

    # train a set number of epochs
    def train_n_epochs(self):
        avg_loss_pi, avg_loss_vf, avg_rets, avg_fire, avg_score = [], [], [], [], []
        avg_lens, avg_lives, avg_eps, avg_batch_size, avg_illegal = [], [], [], [], []
        avg_entropy, avg_clip_frac = [], []
        print(f'Training for:\t\t {self.epochs} epochs')
        for _ in range(self.epochs):
            loss_pi, loss_vf, batch_rets, batch_fire, batch_score, batch_lens, batch_acts, batch_lives, \
            num_eps, batch_sz, num_ill, mean_entropy, avg_probs, avg_value, avg_target, clip_frac = self.train_one_epoch()

            avg_loss_pi.append(loss_pi.item())
            avg_loss_vf.append(loss_vf.item())
            avg_rets.append(np.mean(batch_rets))
            avg_fire.append(np.mean(batch_fire))
            avg_score.append(np.mean(batch_score))
            avg_lens.append(np.mean(batch_lens))
            avg_lives.append(np.mean(batch_lives))
            avg_eps.append(num_eps)
            avg_batch_size.append(batch_sz)
            avg_illegal.append(num_ill)
            avg_entropy.append(mean_entropy.item())
            avg_clip_frac.append(clip_frac)

            if self.current_epoch % self.show_epoch_nr == 0 or self.current_epoch == self.epochs - 1:
                self.train_data.append([self.current_epoch, np.mean(avg_loss_pi), np.mean(avg_loss_vf),
                                        np.mean(avg_rets), np.mean(avg_fire), np.mean(avg_score),
                                        np.mean(avg_lives), np.mean(avg_lens),
                                        np.mean(avg_eps), np.mean(avg_batch_size),
                                        np.mean(avg_illegal), self.epsilon, np.mean(avg_entropy),
                                        avg_probs, avg_value, avg_target, np.mean(avg_clip_frac)])
                if self.print_training:  # only print if requested
                    rendering.print_training_epoch_results_VPG(self.current_epoch, avg_loss_pi, avg_loss_vf,
                                                               avg_rets, avg_fire, avg_score, avg_lives, avg_lens,
                                                               avg_eps, avg_batch_size, avg_illegal, avg_entropy)
                avg_loss_pi, avg_loss_vf, avg_rets, avg_fire, avg_score = [], [], [], [], []
                avg_lens, avg_lives, avg_eps, avg_batch_size, avg_illegal = [], [], [], [], []
                avg_entropy, avg_clip_frac = [], []

            if self.save_model_every_n_epochs > 0:
                if self.current_epoch % self.save_model_every_n_epochs == 0 and self.current_epoch != 0:
                    basics.save_model_and_tests_after_n_epochs(self)

            self.current_epoch += 1

    def test(self, episodes=10, pick_max_prob=False, print_results=True):
        if print_results:
            print('\nTesting', os.path.basename(__file__), 'for', episodes, 'episodes.')
        # make some empty lists for logging.
        batch_rets = []  # for measuring episode returns
        batch_scores = []  # for measuring the Hanabi score
        batch_lens = []  # for measuring episode lengths
        batch_fireworks = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode
        batch_illegals = []  # for seeing the number of illegal moves

        for ep in range(episodes):
            self.test_episode = ep
            _, _, _, ep_ret, ep_len, ep_firework, ep_score, ep_lives, il_move_count, _ = self.run_one_episode(
                training=False, pick_max_prob=pick_max_prob)
            if print_results:
                print(f'episode: {ep:6d}\t return: {ep_ret:10.2f}\t score: {ep_score:4d}\t fireworks: {ep_firework:4d}'
                      f'\t lives left: {ep_lives:4d}\t ep_len: {ep_len:4d}\t ill moves: {il_move_count:4d}')
            self.test_data.append([ep, ep_ret, ep_firework, ep_score, ep_lives, ep_len, il_move_count])
            batch_rets.append(ep_ret)
            batch_scores.append(ep_score)
            batch_lens.append(ep_len)
            batch_fireworks.append(ep_firework)
            batch_lives.append(ep_lives)
            batch_illegals.append(il_move_count)

        if print_results:
            rendering.print_test_results_summary(episodes, batch_rets, batch_scores, batch_lens,
                                                 batch_fireworks, batch_lives, batch_illegals)
        return np.mean(batch_rets), np.mean(batch_scores), np.mean(batch_fireworks), \
               np.mean(batch_lives), np.mean(batch_lens), np.mean(batch_illegals)

    def store_training_data(self, folder='data/'):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again, if it's something else than "folder already exists"
        df = pd.DataFrame(self.train_data, columns=['epoch', 'loss_pi', 'loss_vf',
                                                    'returns', 'fireworks', 'scores', 'lives',
                                                    'epi_length', 'num_epis', 'batch_size', 'illegal_moves',
                                                    'epsilon', 'entropy',
                                                    'avg_probs', 'avg_value', 'avg_target', 'clip_frac'])
        filename = folder + "results_experiment_" + str(self.experiment_num) + ".csv"
        print("\nSaved training data in:", filename)
        df.to_csv(filename, index=False)

    def store_testing_data(self, folder='data/'):
        try:
            os.makedirs(folder + "tests/")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again, if it's something else than "folder already exists"
        df = pd.DataFrame(self.test_data, columns=['episode', 'returns', 'fireworks', 'scores', 'lives',
                                                   'epi_length', 'illegal_moves'])
        filename = folder + "tests/results_experiment_" + str(self.experiment_num) + ".csv"
        print("\nSaved testing data in:", filename)
        df.to_csv(filename, index=False)

    # To save the parameters of the current policy, such that we can continue training later on
    def save_policy(self, save_path=None):
        if save_path is None:
            default_folder = "policy_params/"
            try:
                os.makedirs(default_folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # raises the error again, if it's something else than "folder already exists"
            save_path = default_folder + "policy_" + datetime.now().strftime("%Y-%m-%d_%H%M") + ".pth"
        else:
            folders = os.path.dirname(save_path)
            try:
                os.makedirs(folders)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  # raises the error again, if it's something else than "folder already exists"
        checkpoint = {
            "epoch": self.current_epoch,
            "epsilon": self.epsilon,
            "policy_network": self.policy_network.state_dict(),
            "optimizer_pi": self.optimizer_pi.state_dict(),
            "value_network": self.value_network.state_dict(),
            "optimizer_vf": self.optimizer_vf.state_dict()
        }
        torch.save(checkpoint, save_path)
        print("\nSaved policy parameters in:", save_path)

    # Loading the parameters of a previously trained policy
    def load_policy(self, load_path):
        loaded_checkpoint = torch.load(load_path, map_location=self.device)
        self.current_epoch = loaded_checkpoint["epoch"]
        self.policy_network.load_state_dict(loaded_checkpoint["policy_network"])
        self.optimizer_pi.load_state_dict(loaded_checkpoint["optimizer_pi"])
        self.value_network.load_state_dict(loaded_checkpoint["value_network"])
        self.optimizer_vf.load_state_dict(loaded_checkpoint["optimizer_vf"])
        self.epsilon = loaded_checkpoint.get("epsilon", 0.0)  # default 0, if there was no epsilon saved
        print("\nLoaded policy parameters from:", load_path)


if __name__ == '__main__':
    start_time_main = datetime.now()
    REWARD_SETTINGS = {}
    if ADJUST_REW_SUCCESS_PLAY:
        REWARD_SETTINGS["success play"] = REW_SUCCESS_PLAY
    if ADJUST_REW_ILLEGAL_MOVE:
        REWARD_SETTINGS["illegal move"] = REW_ILLEGAL_MOVE
    if ADJUST_REW_OUT_OF_LIVES:
        REWARD_SETTINGS["out of lives"] = REW_OUT_OF_LIVES
    if ADJUST_REW_LOST_ONE_LIFE:
        REWARD_SETTINGS["lost one life"] = REW_LOST_ONE_LIFE
    if ADJUST_REW_DISCARD:
        REWARD_SETTINGS["discard"] = REW_DISCARD
    if ADJUST_REW_PLAY:
        REWARD_SETTINGS["play"] = REW_PLAY
    if ADJUST_REW_HINT:
        REWARD_SETTINGS["hint"] = REW_HINT
    if ADJUST_REW_DISCARD_PLAYABLE:
        REWARD_SETTINGS["discard playable card"] = REW_DISCARD_PLAYABLE
    if ADJUST_REW_DISCARD_USELESS:
        REWARD_SETTINGS["discard useless card"] = REW_DISCARD_USELESS
    if ADJUST_REW_DISCARD_UNIQUE:
        REWARD_SETTINGS["discard unique card"] = REW_DISCARD_UNIQUE

    ENV_CONFIG = {"max_life_tokens": MAX_LIFE_TOKENS,
                  "max_information_tokens": MAX_HINT_TOKENS,
                  "colors": 5,
                  "ranks": 5,
                  "players": 2}

    algo = PPO_Agent(epochs=EPOCHS,
                     runtime=RUNTIME,
                     show_epoch_nr=SHOW_EPOCH,
                     print_training=PRINT_TRAINING,
                     lr_pi=LR_PI,
                     hidden_layers_pi=HIDDEN_LAYERS_PI,
                     activation_pi=ACTIVATION_PI,
                     train_pi_iters=TRAIN_PI_ITERS,
                     lr_vf=LR_VF,
                     hidden_layers_vf=HIDDEN_LAYERS_VF,
                     activation_vf=ACTIVATION_VF,
                     train_vf_iters=TRAIN_VF_ITERS,
                     clip_param=CLIP_PARAM,
                     gamma=GAMMA,
                     batch_size=BATCH_SIZE,
                     entropy_coeff=ENTROPY_COEFF,
                     renormalize_adv=RENORMALIZE_ADVANTAGE,
                     adv_type=ADV_TYPE,
                     lam=LAM,
                     obs_vec_type=OBS_VEC_TYPE,
                     num_actions=NUM_ACTIONS,
                     reward_settings=REWARD_SETTINGS,
                     epsilon_start=EPSILON_START,
                     epsilon_decay_type=EPSILON_DECAY_TYPE,
                     epsilon_low_at_epoch=EPSILON_LOW_AT_EPOCH,
                     epsilon_low=EPSILON_LOW,
                     render_training=RENDER_TRAINING,
                     render_tests=RENDER_TESTS,
                     render_file=RENDER_FILE,
                     env_config=ENV_CONFIG,
                     save_model_every_n_epochs=SAVE_PARAMS_EVERY,
                     save_data_folder=DATA_FOLDER)

    algo.train()
    if STORE_TRAIN_DATA:
        algo.store_training_data(folder=DATA_FOLDER)

    algo.test(episodes=TEST_EPISODES)
    if STORE_TEST_DATA:
        algo.store_testing_data(folder=DATA_FOLDER)

    print("\nRunning time:", datetime.now() - start_time_main)
