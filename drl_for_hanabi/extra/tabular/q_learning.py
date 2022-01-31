import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from hanabi_learning_environment import rl_env

"""
This file implements a Q-learning algorithm
It acts as one centralized agent, in cheat mode.
This agent can choose to "pass" (by giving a random hint)
"""

LR = 0.1
GAMMA = 0.95
START_EPSILON = 0.98
END_EPSILON = 0  # was 0.5
DECAY = 50

TRAIN_EPISODES = 500
TEST_EPISODES = 10
PRINT_EPS = 50
RENDER = False
LOG_FILE = "logs/game_traces-for-" + os.path.splitext(os.path.basename(__file__))[0] + ".txt"


def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile of its color.
    This function might be confusing at first, as you'd think it should say:
        return card['rank'] == fireworks[card['color']] + 1
    However, the ranks of the cards in this program are: 0,1,2,3,4
    while the fireworks are numbered as normal.
    """
    return card['rank'] == fireworks[card['color']]


def obs_to_playable_life_vec(obs, hand_size=5):
    """
        return a vec of length <hand_size>+1 with at each index:
        1 if card is directly playable
        0 if not
        at the last index the number of life tokens is shown
    """
    current_player = obs['current_player']
    obs_vec = []
    fireworks = obs['player_observations'][current_player]['fireworks']
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for idx in range(len(own_cards)):
        if playable_card(own_cards[idx], fireworks):
            obs_vec.append(1)
        else:
            obs_vec.append(0)
    if len(own_cards) < hand_size:  # when deck is emptied, add a zero to keep obs_vec same size
        obs_vec.append(0)

    obs_vec.append(obs['player_observations'][current_player]['life_tokens'])
    return obs_vec


def print_the_state(obs):
    han_list = str(obs['player_observations'][0]['pyhanabi']).splitlines()
    lives = han_list[0]
    hints = han_list[1]
    fireworks = han_list[2]
    deck_size = han_list[-2]
    discards = han_list[-1]

    output = fireworks + "\t" + lives + "\t" + hints + "\n"
    output += "Player 0:  "
    for card in obs['player_observations'][0]['card_knowledge'][0]:
        output += card['color']
        output += str(card['rank'] + 1) + " "
    output += "\t" + deck_size + "\t" + discards + "\n"
    output += "Player 1:  "
    for card in obs['player_observations'][1]['card_knowledge'][0]:
        output += card['color']
        output += str(card['rank'] + 1) + " "
    output += "\tCurrent player: " + str(obs['current_player'])

    return output


class Q_Agent:
    """
    class for the simple Q-learning algorithm
    """

    def __init__(self, obs_dim, act_dim, render=False, training=False, hand_size=5):
        self.obs_dim = act_dim
        self.act_dim = act_dim
        self.render = render
        self.hand_size = hand_size

        self.possible_hands = 2**hand_size
        self.training = training
        self.action_space = [i for i in range(act_dim)]

        self.Q_table = np.zeros([obs_dim, act_dim])

        self.action = None
        self.observation = None

        self.alpha = LR     # learning rate
        self.gamma = GAMMA  # discount factor
        self.start_epsilon = START_EPSILON  # exploration rate, starts high, then goes down
        self.epsilon = self.start_epsilon
        self.min_epsilon = END_EPSILON
        self.decay = DECAY
        self.counter = 1
        self.decay_data = pd.DataFrame(columns=['num_action', 'epsilon'])

        self.log_file = open(LOG_FILE, "w")
        self.episode = 0
        self.env = rl_env.make(environment_name="Hanabi-Cheat", num_players=2)


    def convert_to_state_num(self, obs_vec):
        """
        converts each possible obs_vec to a unique integer in range(obs_dim)
        obs_vec must look like this example: [0,1,1,0,0,3] (cards with index 1,2 are playable & there are 3 life tokens)
        """
        lives = obs_vec[-1]
        num = 0
        for idx, bit in enumerate(obs_vec[0:self.hand_size]):
            num += bit * 2**idx
        num += (lives-1) * self.possible_hands
        return num


    def act(self, obs):
        obs_vec = obs_to_playable_life_vec(obs)
        state = self.convert_to_state_num(obs_vec)

        self.observation = state
        self.epsilon = max(self.start_epsilon ** (self.counter / self.decay), self.min_epsilon)
        self.decay_data.loc[self.counter] = [self.counter, self.epsilon]
        self.counter += 1

        current_player_id = obs['current_player']
        # hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
        legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']
        legals_HLE = obs['player_observations'][current_player_id]['legal_moves']
        legals = []
        for a in legals_int:
            if a < 10:
                legals.append(a)
            else:
                hint_move = a
                legals.append(10)
                break

        if self.training and random.random() < self.epsilon:
            self.action = random.choice(legals)
            if self.action == 10:
                idx = legals_int.index(hint_move)
                act_HLE = legals_HLE[idx]
            else:
                idx = legals_int.index(self.action)
                act_HLE = legals_HLE[idx]
        else:
            # give illegal actions value -Inf
            action_values = np.full(self.act_dim, -float('inf'))
            if legals:
                action_values[legals] = self.Q_table[state][legals]
            else:
                raise ValueError("No action possible.")

            self.action = np.argmax(action_values)

            if self.action == 10:
                idx = legals_int.index(hint_move)
                act_HLE = legals_HLE[idx]
            else:
                idx = legals_int.index(self.action)
                act_HLE = legals_HLE[idx]

        return act_HLE, self.action


    def q_table_update(self, next_obs: dict, reward: float):
        """Gets called at the end of each step
        :param next_state: int, The new state the agents is in, due to the previous action taken
        :param reward: float, The reward the agent got for the previous action taken
        """
        next_obs_vec = obs_to_playable_life_vec(next_obs)
        next_state = self.convert_to_state_num(next_obs_vec)

        # Update your QTable based on the new state and reward.
        old_Q_value = self.Q_table[self.observation, self.action]
        new_Q_value = reward + self.gamma * np.max(self.Q_table[next_state])

        self.Q_table[self.observation, self.action] = (1-self.alpha) * old_Q_value + self.alpha * new_Q_value


    def save(self, file_name="logs/best_q_table.npy"):
        with open(file_name, "wb") as file:
            np.save(file, self.Q_table)
        print("Q table saved")

    def load(self, file_name="logs/best_q_table.npy"):
        with open(file_name, "rb") as file:
            self.Q_table = np.load(file)
        print("Q table loaded")


    # to run a single game of Hanabi
    def run_one_episode(self):
        # make some empty lists for logging.
        ep_ret, ep_len, ep_firework = 0, 0, 0  # for episode return, length, and total fireworks before last turn
        turn = 0

        # reset episode-specific variables
        obs = self.env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # rendering
        if self.render and not self.training:
            self.log_file.write("\n\nEPISODE: " + str(self.episode))

        while not done:
            turn += 1
            # rendering
            if self.render and not self.training:
                self.log_file.write("\nSTATE IN TURN: " + str(turn) +"\n")
                self.log_file.write(print_the_state(obs))

            # act in the environment
            act_HLE, action = self.act(obs)
            obs, rew, done, _ = self.env.step(act_HLE)

            if self.training:
                self.q_table_update(obs, rew)

            # rendering
            if self.render and not self.training:
                self.log_file.write("\nACTION TAKEN:\n" + str(act_HLE) + "\n")

            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                for rank in obs['player_observations'][0]['fireworks'].values():
                    ep_firework += rank
                ep_lives = obs['player_observations'][0]['life_tokens']

        # rendering last observation
        if self.render and not self.training:
            self.log_file.write("\nFINAL STATE OF THE GAME:\n")
            self.log_file.write(print_the_state(obs))
            if ep_lives == 0:
                self.log_file.write("\nGAME DONE. SCORE: 0\n")
            else:
                self.log_file.write("\nGAME DONE. SCORE: " + str(ep_firework) + "\n")

        return ep_ret, ep_len, ep_firework, ep_lives


    def train(self, episodes=1000):
        print('\nTraining', os.path.basename(__file__), 'for', episodes, 'episodes.')
        print('Printing averages over the last', PRINT_EPS, 'episodes.')
        # make some empty lists for logging.
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_fire = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode

        self.training = True

        for ep in range(episodes):
            self.episode = ep
            ep_ret, ep_len, ep_firework, ep_lives = self.run_one_episode()
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_fire.append(ep_firework)
            batch_lives.append(ep_lives)

            if ep % PRINT_EPS == 0 or ep == episodes-1:
                print('episode: {:6d} \t return: {:10.2f} \t fireworks: {:4.2f} \t lives left: {:4.2f} \t ep_len: {:4.2f}'
                  .format(ep, np.mean(batch_rets), np.mean(batch_fire), np.mean(batch_lives), np.mean(batch_lens)))
                batch_rets, batch_lens, batch_fireworks, batch_lives = [], [], [], []



    def test(self, episodes=10):
        print('\nTesting', os.path.basename(__file__), 'for', episodes, 'episodes.')
        # make some empty lists for logging.
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_fire = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode

        self.training = False

        for ep in range(episodes):
            self.episode = ep
            ep_ret, ep_len, ep_firework, ep_lives = self.run_one_episode()
            print('episode: {:6d} \t return: {:10.2f} \t fireworks: {:4d} \t lives left: {:4d} \t ep_len: {:4d}'
                  .format(ep, ep_ret, ep_firework, ep_lives, ep_len))
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_fire.append(ep_firework)
            batch_lives.append(ep_lives)

        print('After', episodes, 'episodes:')
        print('Averages:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:4.2f} \t lives left: {:4.2f} \t ep_len: {:4.2f}'
              .format(np.mean(batch_rets), np.mean(batch_fire), np.mean(batch_lives), np.mean(batch_lens)))
        print('Minima:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:4.2f} \t lives left: {:4.2f} \t ep_len: {:4.2f}'
              .format(np.min(batch_rets), np.min(batch_fire), np.min(batch_lives), np.min(batch_lens)))
        print('Maxima:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:4.2f} \t lives left: {:4.2f} \t ep_len: {:4.2f}'
              .format(np.max(batch_rets), np.max(batch_fire), np.max(batch_lives), np.max(batch_lens)))


    def print_Q_table(self):
        print("\nQ table:")
        print("\t\t Discard actions (5) \t\t\t\t Play actions (5) \t\t\t\t Hint action (1)")
        for row in self.Q_table:
            for ele in row:
                if ele == 0:
                    print(" 0\t", end="\t")
                else:
                    print("{: 3.3f}".format(ele), end="\t")
            print("")

    def plot_epsilon_decay(self):
        self.decay_data.plot(kind='line', x='num_action', y='epsilon', legend=None, title='epsilon decay')
        plt.savefig('plots/plot-epsilon-decay' + str(self.decay)
                    + 'min' + str(self.min_epsilon)
                    + '_' + datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    + '.png')


if __name__ == '__main__':
    algo = Q_Agent(obs_dim=96, act_dim=11, render=RENDER)  # 96 = 3 * 2^5 possible states
    algo.train(episodes=TRAIN_EPISODES)
    algo.print_Q_table()
    algo.test(episodes=TEST_EPISODES)
    # algo.plot_epsilon_decay()
    algo.log_file.close()
