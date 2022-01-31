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
from hanabi_learning_environment import rl_env
from drl_for_hanabi.utils import state_representation, action_representation, rendering, epsilon_decay
from drl_for_hanabi.extra.rule_based import rule_based

"""
This file implements a Supervised Learning algorithm
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
EPOCHS = 20

### supervised
TEACHER = rule_based.RB_CheatAgent(),
LOSS_FUNCTION = nn.MSELoss(),

### Hyperparameters
LR = 1e-3
BATCH_SIZE = 500
HIDDEN_LAYERS = [64]
ACTIVATION = nn.Tanh
OBS_VEC_TYPE = 37  # possible observation vector types: 7, 37, 62
NUM_ACTIONS = 11  # possible number of actions: 11, 51

### Rendering
RENDER_TRAINING = False
RENDER_TESTS = False
RENDER_FILE = f"renders/{datetime.now().strftime('%Y-%m-%d')}/game_traces-for-{os.path.splitext(os.path.basename(__file__))[0]}.txt"
TEST_EPISODES = 5

### Data
SHOW_EPOCH = 5
PRINT_TRAINING = True
STORE_TRAIN_DATA = False
STORE_TEST_DATA = False
DATA_FOLDER = f"data/{datetime.now().strftime('%Y-%m-%d')}/"
SAVE_POLICY_EVERY = 0   # set to e.g. 1000 if you want to save the policy after every 1000 epochs. 0 means: don't save
SAVE_POLICY_FOLDER = f"policy_params/{datetime.now().strftime('%Y-%m-%d')}/"


def neural_network(obs_dim: int, hidden_layers: list, act_dim: int, activation=nn.Tanh):
    # Build a feedforward neural network.
    layers = []
    if hidden_layers:
        layers += [nn.Linear(obs_dim, hidden_layers[0]), activation()]
        for i in range(len(hidden_layers)-1):
            layers += [nn.Linear(hidden_layers[i], hidden_layers[i+1]), activation()]
        layers += [nn.Linear(hidden_layers[-1], act_dim), nn.Identity()]
    else:
        layers += [nn.Linear(obs_dim, act_dim), nn.Identity()]
    return nn.Sequential(*layers)


class SupervisedAgent:
    """
    class for the supervised Learning algorithm
    """
    def __init__(self, teacher=rule_based.RB_CheatAgent(), loss_function=nn.MSELoss(),
                 epochs=50, runtime=0.0, show_epoch_nr=10, print_training=True,
                 batch_size=1000, lr=1e-4,
                 hidden_layers=None, activation=nn.Tanh, obs_vec_type=62, num_actions=11,
                 render_training=False, render_tests=False, render_file=None,
                 experiment_num=-1, save_policy_every_n_epochs=0, save_policy_folder="policy_params/"):
        self.teacher = teacher
        self.loss_function = loss_function
        self.epochs = epochs
        self.runtime = runtime
        self.show_epoch_nr = show_epoch_nr
        self.print_training = print_training
        self.batch_size = batch_size  # number of steps (at least) in one epoch
        self.lr = lr
        self.hidden_layers = [] if (hidden_layers is None) else hidden_layers
        self.activation = activation
        switcher_obs = {
            7: state_representation.obs_to_vec7_playability,
            37: state_representation.obs_to_vec37_owncards,
            62: state_representation.obs_to_vec62_discards
        }
        self.obs_to_vec_func = switcher_obs.get(obs_vec_type, "Unknown observation_vector_type given.")
        switcher_act = {
            11: action_representation.choose_action_11position,
            51: action_representation.choose_action_51unique
        }
        self.action_func = switcher_act.get(num_actions, "Unknown action_type given.")
        self.num_actions = num_actions

        self.finished_rendering_this_epoch = False
        self.render_training = render_training
        self.render_tests = render_tests
        self.render = (render_training or render_tests)

        self.render_file = None
        if self.render:
            if render_file is None:
                raise ValueError("Provide a render_file or set both render_training and render_tests to False.")
            else:
                render_folder, _ = os.path.split(render_file)
                try:
                    os.makedirs(render_folder)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise  # raises the error again, if it's something else than "folder already exists"
                self.render_file = open(render_file, "w")

        self.experiment_num = experiment_num
        self.save_policy_every_n_epochs = save_policy_every_n_epochs
        self.save_policy_folder = save_policy_folder

        self.current_epoch = 0
        self.test_episode = 0
        self.train_data = []
        self.test_data = []
        self.illegal_move_max = max(100, self.batch_size)

        # make environment, check spaces, get obs / act dims
        self.env = rl_env.make(environment_name="Hanabi-Cheat", num_players=2)
        obs0 = self.env.reset()
        self.obs_dim = len(self.observation_to_vector(obs0))
        self.act_dim = num_actions
        self.policy_network = neural_network(self.obs_dim, self.hidden_layers, self.act_dim, self.activation)
        # self.print_network(obs0)  # to print a summary of the network once
        self.optimizer = Adam(self.policy_network.parameters(), lr=self.lr)

    def __del__(self):
        if self.render_file is not None:
            self.render_file.close()

    # make function to compute action distribution
    def get_policy(self, obs_vec_torch):
        logits = self.policy_network(obs_vec_torch)
        return Categorical(logits=logits)

    # to visualize the network
    def print_network(self, obs):
        print("\nThe initial policy network")
        obs_vec = self.observation_to_vector(obs)
        print("example input (observation):", obs_vec)
        print("input shape: vector of length", len(obs_vec))
        print("output shape: vector of length", self.act_dim)
        summary(self.policy_network, input_size=(1,1,len(obs_vec)))

    def observation_to_vector(self, obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
        """ Options:  (see utils/state_representation.py)
        - obs_to_vec7_playability
        - obs_to_vec37_owncards
        - obs_to_vec62_discards
        """
        return self.obs_to_vec_func(obs)

    def choose_action(self, obs, pick_max_prob=False, epsilon=0.0):
        """ Options:  (see utils/action_representation.py)
        - choose_action_11position
        - choose_action_51unique
        """
        obs_vec = self.obs_to_vec_func(obs)
        return self.action_func(self, obs, obs_vec, pick_max_prob=pick_max_prob, epsilon=epsilon)

    def compute_loss(self, obs, teacher_action):
        """Computes the loss by comparing the move of our neural network with the move of the rule-based agent.
        Args:
            obs: dict, in HLE form
            teacher_action: dict, action in HLE form
        Returns:
            loss, torch tensor with one item, computed with the initialized loss_function
        """
        obs_vec = self.obs_to_vec_func(obs)
        obs_vec_torch = torch.as_tensor(obs_vec, dtype=torch.float32)
        prob_vec = self.get_policy(obs_vec_torch).probs
        teacher_action_probs = action_representation.convert_move_to_action_probabilities_11(teacher_action)
        target_probs = torch.as_tensor(teacher_action_probs, dtype=torch.float32)
        return self.loss_function(prob_vec, target_probs)

    def run_one_episode(self, training=True, pick_max_prob=False):
        # make some empty lists for logging.
        ep_ret, ep_len, ep_firework = 0, 0, 0  # for episode return, length, and total fireworks regardless of lives
        turn, illegal_move_count = 0, 0
        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout episode
        loss_item = 0
        self.current_epoch += 1

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
            if self.render:
                rendering.print_current_state(self, turn, obs)

            if training:
                action = self.teacher.choose_action(obs)
                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, action)
                loss_item = loss.item()
                loss.backward()
                self.optimizer.step()
                # act in the environment
                obs, rew, done, _ = self.env.step(action)
                if self.render:
                    rendering.print_action_HLE(self, action)
            else:  # testing mode
                act_int, legal = self.choose_action(obs)
                if legal:
                    if self.num_actions == 11:  # comment out for speedup (uncomment if using 51 actions, or testing both)
                        action = (action_representation.convert_hint_to_HLE_int(obs) if act_int == 10 else act_int)
                    else:
                        action = action_representation.convert_51int_to_HLE_int(act_int, obs)
                    obs, rew, done, _ = self.env.step(action)
                else:
                    illegal_move_count += 1
                    rew = -10
                    # obs & done remain the same, but reward may change
                if self.render:
                    rendering.print_action(self, act_int, legal)


            ep_rews.append(rew)
            if done or illegal_move_count > self.illegal_move_max:
                break

        # episode is over: record info about episode
        ep_ret = sum(ep_rews)
        ep_len = len(ep_rews)
        ep_lives = obs['player_observations'][0]['life_tokens']
        for rank in obs['player_observations'][0]['fireworks'].values():
            ep_firework += rank
        ep_score = (ep_firework if ep_lives > 0 else 0)

        if self.render:
            rendering.print_end_of_episode(self, obs, ep_score)
        # if illegal_move_count > 0:
        #     print("Tried to do", illegal_move_count, "illegal moves during this episode.")
        return loss_item, ep_ret, ep_len, ep_firework, ep_score, ep_lives, illegal_move_count

    # training loop
    def train(self):
        print('\nTraining', os.path.basename(__file__), 'with settings:')
        print('Learning rate:\t\t', self.lr)
        print('Number of players:\t', 2)
        print('Neural network:\t\t', len(self.hidden_layers), 'hidden layers with', end=" ")
        for idx, layer in enumerate(self.hidden_layers):
            print(layer, end=", ") if idx < len(self.hidden_layers)-1 else print(layer, end=" ")
        print('nodes.')

        avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives, avg_illegal = [], [], [], [], [], [], []

        if self.runtime <= 0:
            # then we run the given number of epochs
            print('Training for:\t\t {} epochs'.format(self.epochs))
            for _ in range(self.epochs):
                loss_item, ep_ret, ep_len, ep_firework, ep_score, ep_lives, illegal_move_count\
                    = self.run_one_episode(training=True)

                avg_loss.append(loss_item)
                avg_rets.append(ep_ret)
                avg_fire.append(ep_firework)
                avg_score.append(ep_score)
                avg_lens.append(ep_len)
                avg_lives.append(ep_lives)
                avg_illegal.append(illegal_move_count)

                # always print if it's the first or last epoch
                if self.current_epoch == 1 or self.current_epoch == self.epochs:
                    print(f'epoch: {self.current_epoch:10d} \t loss: {np.mean(avg_loss):5.5f} \t return: {np.mean(avg_rets):10.2f}'
                          f' \t fireworks: {np.mean(avg_fire):6.2f} \t lives left: {np.mean(avg_lives):6.2f}'
                          f' \t ep_len: {np.mean(avg_lens):6.2f} \t illegals: {np.mean(avg_illegal):6.2f}')
                if self.current_epoch % self.show_epoch_nr == 0:
                    self.train_data.append([self.current_epoch, np.mean(avg_loss), np.mean(avg_rets), np.mean(avg_fire), np.mean(avg_score), np.mean(avg_lives),
                                  np.mean(avg_lens), np.mean(avg_illegal)])
                    if self.print_training:  # only print if requested
                        print(f'epoch: {self.current_epoch:10d} \t loss: {np.mean(avg_loss):5.5f} \t return: {np.mean(avg_rets):10.2f}'
                              f' \t fireworks: {np.mean(avg_fire):6.2f} \t lives left: {np.mean(avg_lives):6.2f}'
                              f' \t ep_len: {np.mean(avg_lens):6.2f} \t illegals: {np.mean(avg_illegal):6.2f}')
                    avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives, avg_illegal = [], [], [], [], [], [], []


                if self.save_policy_every_n_epochs > 0:
                    if self.current_epoch % self.save_policy_every_n_epochs == 0:
                        self.save_policy(save_path=f"{self.save_policy_folder}policy_at_epoch_{self.current_epoch}.pth")
        else:
            startTime = datetime.now()
            endTime = startTime + timedelta(hours=self.runtime)
            print('Training for:\t\t {} hours'.format(self.runtime))
            print('Start time:\t\t', startTime)
            print('End time:\t\t', endTime)
            while datetime.now() < endTime:
                loss_item, ep_ret, ep_len, ep_firework, ep_score, ep_lives, illegal_move_count\
                    = self.run_one_episode(training=True)

                avg_loss.append(loss_item)
                avg_rets.append(ep_ret)
                avg_fire.append(ep_firework)
                avg_score.append(ep_score)
                avg_lens.append(ep_len)
                avg_lives.append(ep_lives)
                avg_illegal.append(illegal_move_count)

                if self.current_epoch % self.show_epoch_nr == 0:
                    self.train_data.append([self.current_epoch, np.mean(avg_loss), np.mean(avg_rets), np.mean(avg_fire), np.mean(avg_score), np.mean(avg_lives),
                                  np.mean(avg_lens), np.mean(avg_illegal)])
                    if self.print_training:
                        print(f'epoch: {self.current_epoch:10d} \t loss: {np.mean(avg_loss):5.5f} \t return: {np.mean(avg_rets):10.2f}'
                              f' \t fireworks: {np.mean(avg_fire):6.2f} \t lives left: {np.mean(avg_lives):6.2f}'
                              f' \t ep_len: {np.mean(avg_lens):6.2f} \t illegals: {np.mean(avg_illegal):6.2f}')
                    avg_loss, avg_rets, avg_fire, avg_score, avg_lens, avg_lives, avg_illegal = [], [], [], [], [], [], []

                if self.save_policy_every_n_epochs > 0:
                    if self.current_epoch % self.save_policy_every_n_epochs == 0:
                        self.save_policy(save_path=f"{self.save_policy_folder}policy_at_epoch_{self.current_epoch}.pth")

            print('Last epoch info:')
            print(f'epoch: {self.current_epoch:10d} \t loss: {loss_item:5.5f} \t return: {ep_ret:10.2f}'
                  f' \t fireworks: {ep_firework:6.2f} \t lives left: {ep_lives:6.2f}'
                  f' \t ep_len: {ep_len:6.2f} \t illegals: {illegal_move_count:6.2f}')
            print('Finished running at:\t', datetime.now())

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
            _, ep_ret, ep_len, ep_firework, ep_score, ep_lives, il_move_count = self.run_one_episode(training=False, pick_max_prob=pick_max_prob)
            if print_results:
                print(f'episode: {ep:6d}\t return: {ep_ret:10.2f}\t score: {ep_score:4d}\t fireworks: {ep_firework:4d}\t '
                      f'lives left: {ep_lives:4d}\t ep_len: {ep_len:4d}\t ill moves: {il_move_count:4d}')
            self.test_data.append([ep, ep_ret, ep_firework, ep_score, ep_lives, ep_len, il_move_count])
            batch_rets.append(ep_ret)
            batch_scores.append(ep_score)
            batch_lens.append(ep_len)
            batch_fireworks.append(ep_firework)
            batch_lives.append(ep_lives)
            batch_illegals.append(il_move_count)

        if print_results:
            print('\nAfter', episodes, 'test episodes the results are:')
            print(f'Averages:\t return: {np.mean(batch_rets):10.2f}\t score: {np.mean(batch_scores):10.2f}\t '
                  f'fireworks: {np.mean(batch_fireworks):4.2f}\t lives left: {np.mean(batch_lives):4.2f}\t '
                  f'ep_len: {np.mean(batch_lens):4.2f}\t ill moves: {np.mean(batch_illegals):6.2f}')
            print(f'Std.error:\t return: {np.std(batch_rets, ddof=1) / np.sqrt(np.size(batch_rets)):10.2f}\t '
                  f'score: {np.std(batch_scores, ddof=1) / np.sqrt(np.size(batch_rets)):10.2f}\t '
                  f'fireworks: {np.std(batch_fireworks, ddof=1) / np.sqrt(np.size(batch_rets)):4.2f}\t '
                  f'lives left: {np.std(batch_lives, ddof=1) / np.sqrt(np.size(batch_rets)):4.2f}\t '
                  f'ep_len: {np.std(batch_lens, ddof=1) / np.sqrt(np.size(batch_rets)):4.2f}\t '
                  f'ill moves: {np.std(batch_illegals, ddof=1) / np.sqrt(np.size(batch_rets)):6.2f}')

            print(f'\nMinima:\t\t return: {np.min(batch_rets):10.2f}\t score: {np.min(batch_scores):10.2f}\t '
                  f'fireworks: {np.min(batch_fireworks):4.2f}\t lives left: {np.min(batch_lives):4.2f}\t '
                  f'ep_len: {np.min(batch_lens):4.2f}\t ill moves: {np.min(batch_illegals):6.2f}')
            print(f'Maxima:\t\t return: {np.max(batch_rets):10.2f}\t score: {np.max(batch_scores):10.2f}\t '
                  f'fireworks: {np.max(batch_fireworks):4.2f}\t lives left: {np.max(batch_lives):4.2f}\t '
                  f'ep_len: {np.max(batch_lens):4.2f}\t ill moves: {np.max(batch_illegals):6.2f}')

        return np.mean(batch_rets), np.mean(batch_scores), np.mean(batch_fireworks), \
               np.mean(batch_lives), np.mean(batch_lens), np.mean(batch_illegals)

    def store_training_data(self, folder='data/'):
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again, if it's something else than "folder already exists"
        df = pd.DataFrame(self.train_data, columns=['epoch', 'loss', 'returns', 'fireworks', 'scores', 'lives',
                                                    'epi_length', 'illegal_moves'])
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
            # "epsilon": self.epsilon,
            "policy_state": self.policy_network.state_dict(),
            "optim_state": self.optimizer.state_dict()
        }
        torch.save(checkpoint, save_path)
        print("\nSaved policy parameters in:", save_path)

    # Loading the parameters of a previously trained policy
    def load_policy(self, load_path, print_loaded=True, overwrite_print=True):
        loaded_checkpoint = torch.load(load_path)
        self.current_epoch = loaded_checkpoint["epoch"]
        self.policy_network.load_state_dict(loaded_checkpoint["policy_state"])
        self.optimizer.load_state_dict(loaded_checkpoint["optim_state"])
        # self.epsilon = loaded_checkpoint.get("epsilon", 0.0)  # default 0, if there was no epsilon saved
        if print_loaded:
            if overwrite_print:
                print(f"Loaded policy parameters from: {load_path}", end='\r')
            else:
                print("\nLoaded policy parameters from:", load_path)


if __name__ == '__main__':
    start_time_main = datetime.now()

    algo = SupervisedAgent(teacher=rule_based.RB_CheatAgent(),
                           loss_function=nn.MSELoss(),
                           epochs=EPOCHS,
                           runtime=RUNTIME,
                           show_epoch_nr=SHOW_EPOCH,
                           print_training=PRINT_TRAINING,
                           batch_size=BATCH_SIZE,
                           lr=LR,
                           hidden_layers=HIDDEN_LAYERS,
                           activation=ACTIVATION,
                           obs_vec_type=OBS_VEC_TYPE,
                           num_actions=NUM_ACTIONS,
                           render_training=RENDER_TRAINING,
                           render_tests=RENDER_TESTS,
                           render_file=RENDER_FILE,
                           save_policy_every_n_epochs=SAVE_POLICY_EVERY,
                           save_policy_folder=SAVE_POLICY_FOLDER)

    algo.train()
    if STORE_TRAIN_DATA:
        algo.store_training_data(folder=DATA_FOLDER)

    algo.test(episodes=TEST_EPISODES)
    if STORE_TEST_DATA:
        algo.store_testing_data(folder=DATA_FOLDER)

    print("\nRunning time:", datetime.now() - start_time_main)
