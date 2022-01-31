import sys
import os
# adding drl folder to sys.path
drl_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if drl_folder not in sys.path:
    sys.path.insert(0, drl_folder)
from datetime import datetime, timedelta
import errno
from random import shuffle
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
This file implements the Simple Policy Gradient algorithm
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
LR = 1e-3
GAMMA = 0.99
BATCH_SIZE = 500
HIDDEN_LAYERS = [64, 64]
ACTIVATION = nn.Tanh
TRAIN_ITERS = 1
ENTROPY_COEFF = 1e-2
RENORMALIZE_GT = True
OBS_VEC_TYPE = 186    # possible observation vector types: 7, 37, 62, 82, 136, 186
NUM_ACTIONS = 11      # possible number of actions: 11, 51
SHUFFLE_HAND = False

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
RENDER_FILE = f"renders/{datetime.now().strftime('%Y-%m-%d')}/game_traces-for-{os.path.splitext(os.path.basename(__file__))[0]}.txt"
TEST_EPISODES = 5

### Data
SHOW_EPOCH = 1
PRINT_TRAINING = True
STORE_TRAIN_DATA = False
STORE_TEST_DATA = False
DATA_FOLDER = f"data/spg/{datetime.now().strftime('%Y-%m-%d')}/"
SAVE_PARAMS_EVERY = 0   # set to e.g. 1000 if you want to save the policy after every 1000 epochs. 0 means: don't save


class SPG_Agent:
    """
    class for the Simple Policy Gradient algorithm
    """
    def __init__(self, epochs=50, runtime=0.0, show_epoch_nr=10, print_training=True,
                 batch_size=1000, lr=1e-4, gamma=1.0, train_iters=1,
                 entropy_coeff=1e-3, renormalize_gt=False, reward_settings=None,
                 epsilon_start=0.0, epsilon_decay_type="exponential", epsilon_low_at_epoch=10**4,
                 epsilon_low=10**(-3),
                 hidden_layers=None, activation=nn.Tanh, obs_vec_type=62, num_actions=11,
                 render_training=False, render_tests=False, render_file=None,
                 experiment_num=-1, env_config=None, shuffle_hand=False,
                 save_model_every_n_epochs=1e5, save_data_folder="data/",
                 save_params_path=f"params/{datetime.now().strftime('%Y-%m-%d')}/params.pth"):

        self.epochs = epochs
        self.runtime = runtime
        self.show_epoch_nr = show_epoch_nr
        self.print_training = print_training
        self.batch_size = batch_size  # number of steps (at least) in one epoch
        self.lr = lr
        self.gamma = gamma
        self.train_iters = train_iters
        self.entropy_coeff = entropy_coeff
        self.renormalize_gt = renormalize_gt
        self.reward_settings = {} if (reward_settings is None) else reward_settings
        self.epsilon = epsilon_start
        self.epsilon_decay_type = epsilon_decay_type
        if epsilon_start > 0:
            self.epsilon_decay_factor = epsilon_decay.compute_decay_factor(
                epsilon_start, epsilon_decay_type, epsilon_low_at_epoch, epsilon_low)
        self.hidden_layers = [] if (hidden_layers is None) else hidden_layers
        self.activation = activation
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
        if shuffle_hand:
            self.obs_to_vec_func = state_representation.obs_to_vec62_discards_shuffle
            self.action_func = action_representation.choose_action_11position_shuffle

        self.num_actions = num_actions
        self.shuffle_hand = shuffle_hand
        self.shuffled_indices = [0, 1, 2, 3, 4]

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

        self.policy_network = basics.neural_network(self.obs_dim, self.hidden_layers, self.act_dim, self.activation)
        # self.print_network(obs0)  # to print a summary of the network once
        self.device = basics.get_default_device()
        print("Working on device:", self.device)
        self.policy_network.to(self.device)
        self.optimizer = Adam(self.policy_network.parameters(), lr=self.lr)

    def __del__(self):
        if self.render_file is not None:
            self.render_file.close()

    def get_policy(self, obs_vec_torch):
        """Returns the action distribution for a given observation.
        You may also give a batch of observations as input,
        which will produce a batch of action distributions."""
        logits = self.policy_network(obs_vec_torch.to(self.device))
        return Categorical(logits=logits)

    def print_network(self, obs: dict):
        """Visualizes the network, for a given initial observation."""
        print("\nThe initial policy network")
        obs_vec = self.observation_to_vector(obs)
        print("example input (observation):", obs_vec)
        print("input shape: vector of length", len(obs_vec))
        print("output shape: vector of length", self.act_dim)
        summary(self.policy_network, input_size=(1, 1, len(obs_vec)))

    def observation_to_vector(self, obs: dict):
        """Converts obs dict to vector representation. See utils/state_representation.py"""
        # if self.shuffle_hand:
        #     return self.obs_to_vec_func(self, obs, max_life=self.max_life_tokens, max_hint=self.max_hint_tokens)
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

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(self, obs, act, discounted_return_from_t):
        policy = self.get_policy(obs)

        mean_entropy = torch.mean(policy.entropy())
        avg_probs = None
        if self.current_epoch % self.show_epoch_nr == 0:
            avg_probs = self.get_avg_probs(policy)

        logp = policy.log_prob(act.to(self.device))
        g_t = discounted_return_from_t.to(self.device)
        if self.renormalize_gt:
            g_t = (g_t - g_t.mean()) / (g_t.std() + 1e-8)
        # minus sign in the loss, as we want to maximize logp*g_t, but our optimizer only minimizes
        return -(logp * g_t).mean() - self.entropy_coeff * mean_entropy, mean_entropy, avg_probs

    def update_step(self, batch_obs, batch_acts, batch_weights):
        """Updates the parameters of the policy network.
        Makes train_iters number of update steps (default is 1).

        Args:
            batch_obs: list, of observation vectors (which are lists themselves)
            batch_acts: list, of actions (integers)
            batch_weights: list, of discounted returns (floats)
        Returns:
            loss_pi: torch tensor, single element giving the loss of the policy network
            mean_entropy: torch tensor, single element giving the average entropy of the policy this epoch
            avg_probs: torch tensor, vector of length <num_actions>, giving the mean action distribution
                                                                     of the policy in this epoch
        """
        obs = torch.as_tensor(batch_obs, dtype=torch.float32)
        act = torch.as_tensor(batch_acts, dtype=torch.int32)
        g_t = torch.as_tensor(batch_weights, dtype=torch.float32)
        for iteration in range(self.train_iters):
            self.optimizer.zero_grad()
            loss_pi, entropy, probs = self.compute_loss(obs, act, g_t)
            loss_pi.backward()
            self.optimizer.step()
            if iteration == 0:
                mean_entropy = entropy
                avg_probs = probs
        return loss_pi, mean_entropy, avg_probs

    def run_one_episode(self, training=True, pick_max_prob=False):
        # make some empty lists for logging.
        ep_obs, ep_acts, ep_weights = [], [], []  # for observations, actions, reward-to-go weighting in policy gradient
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
            if self.shuffle_hand:
                shuffle(self.shuffled_indices)

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
        # the weight for each log_prob(a_t|s_t) is reward-to-go from t
        ep_weights += list(basics.reward_to_go(ep_rews, gamma=self.gamma))

        if self.render:
            rendering.print_end_of_episode(self, obs, ep_score)
        # if illegal_move_count > 0:
        #     print("Tried to do", illegal_move_count, "illegal moves during this episode.")
        return ep_obs, ep_acts, ep_weights, ep_ret, ep_len, ep_firework, ep_score, ep_lives, illegal_move_count

    # for training the policy
    def train_one_epoch(self):
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for reward-to-go weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_fireworks = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_scores = []
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode
        num_eps = 0  # to count the number of episodes in this batch
        total_illegal = 0  # to count the number of illegal moves in this batch

        # render first episode of each epoch
        self.finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while len(batch_obs) < self.batch_size:
            ep_obs, ep_acts, ep_weights, ep_ret, ep_len, ep_firework, ep_score, ep_lives, il_moves = \
                self.run_one_episode(training=True)

            batch_obs += ep_obs
            batch_acts += ep_acts
            batch_weights += ep_weights
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_fireworks.append(ep_firework)
            batch_scores.append(ep_score)
            batch_lives.append(ep_lives)
            num_eps += 1
            total_illegal += il_moves

            # won't render again this epoch
            self.finished_rendering_this_epoch = True

        # update the parameters of the policy network
        batch_loss, entropy, avg_probs = self.update_step(batch_obs, batch_acts, batch_weights)

        # if self.epsilon > 0:
        #     self.epsilon = epsilon_decay.compute_next_epsilon(self.epsilon, self.epsilon_decay_type,
        #                                                       self.epsilon_decay_factor)
        return batch_loss, batch_rets, batch_fireworks, batch_scores, batch_lens, batch_acts, batch_lives, \
               num_eps, len(batch_obs), total_illegal, entropy, avg_probs

    # training loop
    def train(self):
        print(f'\nTraining the algorithm: {os.path.basename(__file__)}')
        if self.runtime <= 0:
            self.train_n_epochs()
        else:
            self.train_n_hours()

    # train based on maximum runtime
    def train_n_hours(self):
        avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives = [], [], [], [], [], []
        avg_eps, avg_batch_size, avg_illegal, avg_entropy = [], [], [], []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.runtime)
        print('Training for:\t\t {} hours'.format(self.runtime))
        print('Start time:\t\t', start_time)
        print('End time:\t\t', end_time)
        while datetime.now() < end_time:
            batch_loss, batch_rets, batch_fire, batch_score, batch_lens, batch_acts, batch_lives, \
            num_eps, batch_sz, num_ill, mean_entropy, avg_probs = self.train_one_epoch()

            avg_loss.append(batch_loss.item())
            avg_rets.append(np.mean(batch_rets))
            avg_fire.append(np.mean(batch_fire))
            avg_score.append(np.mean(batch_score))
            avg_lens.append(np.mean(batch_lens))
            avg_lives.append(np.mean(batch_lives))
            avg_eps.append(num_eps)
            avg_batch_size.append(batch_sz)
            avg_illegal.append(num_ill)
            avg_entropy.append(mean_entropy.item())

            if self.current_epoch % self.show_epoch_nr == 0:
                self.train_data.append([self.current_epoch, np.mean(avg_loss), np.mean(avg_rets), np.mean(avg_fire),
                                        np.mean(avg_score), np.mean(avg_lives),
                                        np.mean(avg_lens), np.mean(avg_eps), np.mean(avg_batch_size),
                                        np.mean(avg_illegal), self.epsilon, np.mean(avg_entropy), avg_probs])
                if self.print_training:
                    rendering.print_training_epoch_results(self.current_epoch, avg_loss, avg_rets, avg_fire, avg_score,
                                                           avg_lives, avg_lens, avg_eps, avg_batch_size, avg_illegal,
                                                           avg_entropy)
                avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives = [], [], [], [], [], []
                avg_eps, avg_batch_size, avg_illegal, avg_entropy = [], [], [], []

            if self.save_model_every_n_epochs > 0:
                if self.current_epoch % self.save_model_every_n_epochs == 0 and self.current_epoch != 0:
                    basics.save_model_and_tests_after_n_epochs(self)

            self.current_epoch += 1

        print('Last epoch info:')
        rendering.print_training_epoch_results(self.current_epoch, batch_loss.item(), batch_rets, batch_fire, batch_score,
                                               batch_lives, batch_lens, num_eps, batch_sz, num_ill, mean_entropy.item())
        print('Actions:\t\t', batch_acts)
        print('Finished running at:\t', datetime.now())

    # train a set number of epochs
    def train_n_epochs(self):
        avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives = [], [], [], [], [], []
        avg_eps, avg_batch_size, avg_illegal, avg_entropy = [], [], [], []
        print(f'Training for:\t\t {self.epochs} epochs')
        for _ in range(self.epochs):
            batch_loss, batch_rets, batch_fire, batch_score, batch_lens, batch_acts, batch_lives, \
            num_eps, batch_sz, num_ill, mean_entropy, avg_probs = self.train_one_epoch()

            avg_loss.append(batch_loss.item())
            avg_rets.append(np.mean(batch_rets))
            avg_fire.append(np.mean(batch_fire))
            avg_score.append(np.mean(batch_score))
            avg_lens.append(np.mean(batch_lens))
            avg_lives.append(np.mean(batch_lives))
            avg_eps.append(num_eps)
            avg_batch_size.append(batch_sz)
            avg_illegal.append(num_ill)
            avg_entropy.append(mean_entropy.item())

            if self.current_epoch % self.show_epoch_nr == 0:
                self.train_data.append([self.current_epoch, np.mean(avg_loss), np.mean(avg_rets), np.mean(avg_fire),
                                        np.mean(avg_score), np.mean(avg_lives),
                                        np.mean(avg_lens), np.mean(avg_eps), np.mean(avg_batch_size),
                                        np.mean(avg_illegal), self.epsilon, np.mean(avg_entropy), avg_probs])
                if self.print_training:  # only print if requested
                    rendering.print_training_epoch_results(self.current_epoch, avg_loss, avg_rets, avg_fire, avg_score,
                                                           avg_lives, avg_lens, avg_eps, avg_batch_size, avg_illegal,
                                                           avg_entropy)
                avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives = [], [], [], [], [], []
                avg_eps, avg_batch_size, avg_illegal, avg_entropy = [], [], [], []

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
            _, _, _, ep_ret, ep_len, ep_firework, ep_score, ep_lives, il_move_count = self.run_one_episode(
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
        df = pd.DataFrame(self.train_data, columns=['epoch', 'loss', 'returns', 'fireworks', 'scores', 'lives',
                                                    'epi_length', 'num_epis', 'batch_size', 'illegal_moves',
                                                    'epsilon', 'entropy', 'avg_probs'])
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
            "policy_state": self.policy_network.state_dict(),
            "optim_state": self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)
        print("\nSaved policy parameters in:", save_path)

    # Loading the parameters of a previously trained policy
    def load_policy(self, load_path):
        loaded_checkpoint = torch.load(load_path, map_location=self.device)
        self.current_epoch = loaded_checkpoint["epoch"]
        self.policy_network.load_state_dict(loaded_checkpoint["policy_state"])
        self.optimizer.load_state_dict(loaded_checkpoint["optim_state"])
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

    algo = SPG_Agent(epochs=EPOCHS,
                     runtime=RUNTIME,
                     show_epoch_nr=SHOW_EPOCH,
                     print_training=PRINT_TRAINING,
                     batch_size=BATCH_SIZE,
                     lr=LR,
                     gamma=GAMMA,
                     train_iters=TRAIN_ITERS,
                     entropy_coeff=ENTROPY_COEFF,
                     renormalize_gt=RENORMALIZE_GT,
                     reward_settings=REWARD_SETTINGS,
                     epsilon_start=EPSILON_START,
                     epsilon_decay_type=EPSILON_DECAY_TYPE,
                     epsilon_low_at_epoch=EPSILON_LOW_AT_EPOCH,
                     epsilon_low=EPSILON_LOW,
                     hidden_layers=HIDDEN_LAYERS,
                     activation=ACTIVATION,
                     obs_vec_type=OBS_VEC_TYPE,
                     num_actions=NUM_ACTIONS,
                     render_training=RENDER_TRAINING,
                     render_tests=RENDER_TESTS,
                     render_file=RENDER_FILE,
                     env_config=ENV_CONFIG,
                     shuffle_hand=SHUFFLE_HAND,
                     save_model_every_n_epochs=SAVE_PARAMS_EVERY,
                     save_data_folder=DATA_FOLDER)

    algo.train()
    if STORE_TRAIN_DATA:
        algo.store_training_data(folder=DATA_FOLDER)

    algo.test(episodes=TEST_EPISODES)
    if STORE_TEST_DATA:
        algo.store_testing_data(folder=DATA_FOLDER)

    print("\nRunning time:", datetime.now() - start_time_main)
