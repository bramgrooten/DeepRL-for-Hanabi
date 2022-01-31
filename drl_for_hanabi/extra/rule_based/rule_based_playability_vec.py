import os
# from datetime import datetime, timedelta
import random
import numpy as np
from hanabi_learning_environment import rl_env

"""
This file implements a Rule-Based algorithm
It acts as one centralized agent, in cheat mode.
This agent can choose to "pass" (by giving a random hint)

It plays a random playable card.
If there is none, it first tries to hint, and otherwise discards at random.
"""

TEST_EPISODES = 100
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


def obs_to_playable_vec(obs, hand_size=5):
    """
        return a vec of length <hand_size> with a each index:
        1 if card is directly playable
        0 if not
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


class RB_Agent:
    """
    class for the random Rule-Based algorithm
    it plays a random playable card
    """

    def __init__(self, render=False, hand_size=5):
        self.render = render
        self.hand_size = hand_size
        self.card_indeces = [i for i in range(hand_size)]
        self.log_file = open(LOG_FILE, "w")
        self.test_episode = 0
        # make environment, check spaces, get obs / act dims
        self.env = rl_env.make(environment_name="Hanabi-Cheat", num_players=2)

    def choose_action(self, obs, hand_size=5):
        obs_vec = obs_to_playable_vec(obs, hand_size)
        playable_card_indeces = [idx for idx, bit in enumerate(obs_vec) if bit == 1]

        if playable_card_indeces:
            play_idx = random.choice(playable_card_indeces)
            return {
                'action_type': 'PLAY',
                'card_index': play_idx
            }
        else:
            current_player_id = obs['current_player']
            hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
            if hint_possible:
                legals_HLE = obs['player_observations'][current_player_id]['legal_moves']
                return legals_HLE[-1]
            else:
                # there are no playable cards, so discard a random one
                discard_idx = random.choice(self.card_indeces)
                return {
                    'action_type': 'DISCARD',
                    'card_index': discard_idx
                }

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
        if self.render:
            self.log_file.write("\n\nEPISODE: " + str(self.test_episode))

        while not done:
            turn += 1
            # rendering
            if self.render:
                self.log_file.write("\nSTATE IN TURN: " + str(turn) +"\n")
                self.log_file.write(print_the_state(obs))

            # act in the environment
            action = self.choose_action(obs)
            obs, rew, done, _ = self.env.step(action)

            # rendering
            if self.render:
                self.log_file.write("\nACTION TAKEN:\n" + str(action) + "\n")

            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret = sum(ep_rews)
                ep_len = len(ep_rews)
                for rank in obs['player_observations'][0]['fireworks'].values():
                    ep_firework += rank
                ep_lives = obs['player_observations'][0]['life_tokens']

        # rendering last observation
        if self.render:
            self.log_file.write("\nFINAL STATE OF THE GAME:\n")
            self.log_file.write(print_the_state(obs))
            if ep_lives == 0:
                self.log_file.write("\nGAME DONE. SCORE: 0\n")
            else:
                self.log_file.write("\nGAME DONE. SCORE: " + str(ep_firework) + "\n")

        return ep_ret, ep_len, ep_firework, ep_lives

    # testing
    def test(self, episodes=10):
        print('\nTesting', os.path.basename(__file__), 'for', episodes, 'episodes.')
        # make some empty lists for logging.
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_fireworks = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode

        for ep in range(episodes):
            self.test_episode = ep
            ep_ret, ep_len, ep_firework, ep_lives = self.run_one_episode()
            print('episode: {:6d} \t return: {:10.2f} \t fireworks: {:4d} \t lives left: {:4d} \t ep_len: {:4d}'
                  .format(ep, ep_ret, ep_firework, ep_lives, ep_len))
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_fireworks.append(ep_firework)
            batch_lives.append(ep_lives)

        print('After', episodes, 'episodes:')
        print('Averages:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:6.2f} \t lives left: {:6.1f} \t ep_len: {:6.2f}'
              .format(np.mean(batch_rets), np.mean(batch_fireworks), np.mean(batch_lives), np.mean(batch_lens)))
        print('Minima:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:6.2f} \t lives left: {:6.1f} \t ep_len: {:6.2f}'
              .format(np.min(batch_rets), np.min(batch_fireworks), np.min(batch_lives), np.min(batch_lens)))
        print('Maxima:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:6.2f} \t lives left: {:6.1f} \t ep_len: {:6.2f}'
              .format(np.max(batch_rets), np.max(batch_fireworks), np.max(batch_lives), np.max(batch_lens)))


if __name__ == '__main__':
    algo = RB_Agent(render=RENDER)
    algo.test(episodes=TEST_EPISODES)
    algo.log_file.close()
