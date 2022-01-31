import os
import errno
from datetime import datetime
import random
import numpy as np
import pandas as pd
from hanabi_learning_environment import rl_env
"""
This file implements a Rule-Based algorithm
It acts as one centralized agent, in cheat mode.
This agent can choose to "pass" (by giving a random hint)
"""

TEST_EPISODES = 100
PRINT_EVERY_N_EPISODES = 5
RENDER = False
RENDER_FILE = f"renders/{datetime.now().strftime('%Y-%m-%d')}/game_traces-for-{os.path.splitext(os.path.basename(__file__))[0]}.txt"


def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile of its color.
    This function might be confusing at first, as you'd think it should say:
        return card['rank'] == fireworks[card['color']] + 1
    However, the ranks of the cards in this program are: 0,1,2,3,4
    while the fireworks are numbered as normal.

    Args:
        card: dict, for example {'color': 'Y', 'rank': 3}
            Possible ranks are: 0,1,2,3,4
        fireworks: dict, for example {'R': 0, 'Y': 5, 'G': 1, 'W': 4, 'B': 1}
            Possible ranks are: 0,1,2,3,4,5
    Returns:
        boolean, True if the card is directly playable, False if it is not.
    """
    return card['rank'] == fireworks[card['color']]


def useless_card(card, fireworks, discard_pile):
    """A card is useless if it cannot be placed on
    the fireworks pile of its color anymore (not even in the future).
    Args:
        card: dict, for example {'color': 'Y', 'rank': 3}
            Possible ranks are: 0,1,2,3,4
        fireworks: dict, for example {'R': 0, 'Y': 5, 'G': 1, 'W': 4, 'B': 1}
            Possible ranks are: 0,1,2,3,4,5
        discard_pile: list, of cards that are discarded
    Returns:
        boolean, True if the card is provably useless, False if it might be useful.
    """
    if card['rank'] < fireworks[card['color']]:
        return True  # definitely useless when firework is bigger
    elif card['rank'] == fireworks[card['color']]:
        return False  # not useless: it's playable!
    else:
        num_discards_per_rank = [0] * 5
        for discarded in discard_pile:
            if discarded['color'] == card['color']:
                num_discards_per_rank[discarded['rank']] += 1
        for lower_rank in range(card['rank']):
            if lower_rank == 0:
                if num_discards_per_rank[lower_rank] >= 3:
                    return True  # useless: all three rank 1 cards have been discarded
            elif lower_rank <= 3:
                if num_discards_per_rank[lower_rank] >= 2:
                    return True  # useless: both rank 2, 3, or 4 cards have been discarded
            # else: this case is unreachable
            # (highest rank_idx is 4, so all lower_ranks have already been dealt with)
        # We could have duplicate cards in our hand. Then 1 of them is not useless.
        # This is dealt with in find_best_discard_option
    return False


def find_best_discard_option(cards, fireworks, discard_pile):
    """This function assumes that none of your cards are playable and
    that none of your cards are proven useless and
    that hinting is illegal at the moment. You must discard, but we try to minimize the damage.
    Notice that you cannot have a rank 1 card in your hand. It would either be playable or useless.
    Args:
        cards: list, your own cards in dict form: for example {'color': 'Y', 'rank': 3}
            Possible ranks are: 0,1,2,3,4
        fireworks: dict, for example {'R': 0, 'Y': 5, 'G': 1, 'W': 4, 'B': 1}
            Possible ranks are: 0,1,2,3,4,5
        discard_pile: list, of cards
    Returns:
        int, index of card to discard
    """
    # First check for duplicates
    possible_discards = []
    for idx, card in enumerate(cards):
        for card2 in cards:
            if card['color'] == card2['color'] and card['rank'] == card2['rank']:
                possible_discards.append(idx)
    if possible_discards:
        return random.choice(possible_discards)

    # No duplicates from here on
    may_discard = [True] * len(cards)  # every card may initially be a good option to discard
    for idx, card in enumerate(cards):
        if card['rank'] >= 4:  # don't discard the rank 5 cards
            may_discard[idx] = False

    if not any(may_discard):  # hand full of 5's, then we discard the 5 of color with the lowest firework so far
        discard_idx = 0
        lowest_fire = 5
        for idx, card in enumerate(cards):
            if fireworks[card['color']] < lowest_fire:
                lowest_fire = fireworks[card['color']]
                discard_idx = idx
        return discard_idx

    # From now on, the hand is not full of 5's. Still tricky though:
    # imagine that you have [R5,G5,Y5,B4,B5]. This does not automatically mean that we should
    # discard the B4 (when the other B4 has been discarded already).
    # This is what we implement here though.

    # Check if there are other critical cards (next to the 5's).
    # A card is critical when its existing duplicate has been discarded already.
    # We don't want to discard critical cards.
    for card in discard_pile:
        for idx_own, card_own in enumerate(cards):
            if card['color'] == card_own['color'] and card['rank'] == card_own['rank']:
                may_discard[idx_own] = False

    if any(may_discard):
        # Pick the highest card we may discard.
        highest_rank = 0
        for idx, can_discard in enumerate(may_discard):
            if can_discard and cards[idx]['rank'] > highest_rank:
                highest_rank = cards[idx]['rank']
        possible_discards = []
        for idx, can_discard in enumerate(may_discard):
            if can_discard and cards[idx]['rank'] == highest_rank:
                possible_discards.append(idx)
        return random.choice(possible_discards)
    else:
        # Discard the highest card which is not a 5
        highest_rank_not_5 = 0
        for idx, card in enumerate(cards):
            if card['rank'] > highest_rank_not_5 and card['rank'] != 4:
                highest_rank_not_5 = card['rank']
        possible_discards = []
        for idx, card in enumerate(cards):
            if card['rank'] == highest_rank_not_5:
                possible_discards.append(idx)
        return random.choice(possible_discards)


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


class RB_CheatAgent:
    """
    class for the Rule-Based algorithm that plays well in cheat mode
    """

    def __init__(self, hand_size=5, render=False, render_file=None):
        self.render = render
        self.hand_size = hand_size
        self.card_indeces = [i for i in range(hand_size)]
        self.render_file = None
        if render:
            if render_file is not None:
                render_folder = os.path.dirname(render_file)
                try:
                    os.makedirs(render_folder)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise  # raises the error again, if it's something else than "folder already exists"
                self.render_file = open(render_file, "w")
            else:
                raise ValueError("Provide render_file or set render to False.")
        self.test_episode = 0
        self.test_data = []
        # make environment
        env_type = "Hanabi-Simplified"
        # print(env_type)
        self.env = rl_env.make(environment_name=env_type, num_players=2)

    def __del__(self):
        if self.render_file is not None:
            self.render_file.close()

    def choose_action(self, obs, max_hint=8):
        """Chooses the best action to play, based on a full cheat observation
        Args:
            obs: dict, the full HLE observation
            max_hint: int, the maximum number of hint tokens
        Returns:
            dict, an HLE action
        """
        current_player_id = obs['current_player']
        my_cards = obs['player_observations'][current_player_id]['card_knowledge'][0]

        # See if we have playable cards
        playable_cards = []
        fireworks = obs['player_observations'][current_player_id]['fireworks']
        for idx, card in enumerate(my_cards):
            if playable_card(card, fireworks):
                playable_cards.append((idx, card))

        if playable_cards:
            lowest_rank = min(playable_cards, key=lambda x: x[1]['rank'])[1]['rank']
            highest_rank = max(playable_cards, key=lambda x: x[1]['rank'])[1]['rank']
            if highest_rank == 4:
                indices_4 = []
                for card_tuple in playable_cards:
                    if card_tuple[1]['rank'] == 4:
                        indices_4.append(card_tuple[0])
                return {'action_type': 'PLAY', 'card_index': random.choice(indices_4)}
            else:
                indices_lowest = []
                for card_tuple in playable_cards:
                    if card_tuple[1]['rank'] == lowest_rank:
                        indices_lowest.append(card_tuple[0])
            return {'action_type': 'PLAY', 'card_index': random.choice(indices_lowest)}

        # No playable cards from here on. Then we first try to discard provably useless cards
        discarding_legal = (obs['player_observations'][current_player_id]['information_tokens'] < max_hint)
        if discarding_legal:
            discard_pile = obs['player_observations'][current_player_id]['discard_pile']
            useless_indices = []
            for idx, card in enumerate(my_cards):
                if useless_card(card, fireworks, discard_pile):
                    useless_indices.append(idx)

            if useless_indices:
                discard_idx = random.choice(useless_indices)
                return {'action_type': 'DISCARD', 'card_index': discard_idx}

            # No useless cards here, but we may discard. We would rather hint ('pass') in this situation though.
            hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
            if hint_possible:
                legals_HLE = obs['player_observations'][current_player_id]['legal_moves']
                return legals_HLE[-1]
            else:
                # We don't have any useless cards, but hinting is also not allowed.
                # So unfortunately, we must discard.
                discard_idx = find_best_discard_option(my_cards, fireworks, discard_pile)
                return {'action_type': 'DISCARD', 'card_index': discard_idx}

        else:  # Hinting must be legal, if discarding is not.
            legals_HLE = obs['player_observations'][current_player_id]['legal_moves']
            return legals_HLE[-1]

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
            self.render_file.write("\n\nEPISODE: " + str(self.test_episode))

        while not done:
            turn += 1
            # rendering
            if self.render:
                self.render_file.write("\nSTATE IN TURN: " + str(turn) +"\n")
                self.render_file.write(print_the_state(obs))

            # act in the environment
            action = self.choose_action(obs)
            obs, rew, done, _ = self.env.step(action)
            ep_rews.append(rew)

            # rendering
            if self.render:
                self.render_file.write("\nACTION TAKEN:\n" + str(action) + "\n")


        # if episode is over, record info about episode
        ep_ret = sum(ep_rews)
        ep_len = len(ep_rews)
        for rank in obs['player_observations'][0]['fireworks'].values():
            ep_firework += rank
        ep_lives = obs['player_observations'][0]['life_tokens']

        # rendering last observation
        if self.render:
            self.render_file.write("\nFINAL STATE OF THE GAME:\n")
            self.render_file.write(print_the_state(obs))
            if ep_lives == 0:
                self.render_file.write("\nGAME DONE. SCORE: 0\n")
            else:
                self.render_file.write("\nGAME DONE. SCORE: " + str(ep_firework) + "\n")

        return ep_ret, ep_len, ep_firework, ep_lives

    # testing
    def test(self, episodes=10, print_every_n_episodes=1):
        print('\nTesting', os.path.basename(__file__), 'for', episodes, 'episodes.')
        # make some empty lists for logging.
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths
        batch_fireworks = []  # for seeing the final fireworks (just before 3rd life token is lost)
        batch_lives = []  # for seeing the (average) number of lives left at the end of an episode

        for ep in range(episodes):
            self.test_episode = ep
            ep_ret, ep_len, ep_firework, ep_lives = self.run_one_episode()
            if ep % print_every_n_episodes == 0:
                print(f'episode: {ep:6d} \t return: {ep_ret:10.2f} \t fireworks: {ep_firework:4d} '
                      f'\t lives left: {ep_lives:4d} \t ep_len: {ep_len:4d}')
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)
            batch_fireworks.append(ep_firework)
            batch_lives.append(ep_lives)
            self.test_data.append([ep, ep_ret, ep_firework, ep_lives, ep_len])

        print('\nAfter', episodes, 'episodes:')
        print('Minima:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:6.2f} \t lives left: {:6.1f} \t ep_len: {:6.2f}'
              .format(np.min(batch_rets), np.min(batch_fireworks), np.min(batch_lives), np.min(batch_lens)))
        print('Averages:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:6.2f} \t lives left: {:6.1f} \t ep_len: {:6.2f}'
              .format(np.mean(batch_rets), np.mean(batch_fireworks), np.mean(batch_lives), np.mean(batch_lens)))
        print('Maxima:')
        print('\t'*5 + ' return: {:10.2f} \t fireworks: {:6.2f} \t lives left: {:6.1f} \t ep_len: {:6.2f}'
              .format(np.max(batch_rets), np.max(batch_fireworks), np.max(batch_lives), np.max(batch_lens)))

    def store_testing_data(self, folder='data/'):
        try:
            os.makedirs(folder + "tests/")
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise  # raises the error again, if it's something else than "folder already exists"
        df = pd.DataFrame(self.test_data, columns=['episode', 'returns', 'fireworks', 'lives', 'epi_length'])
        filename = f"{folder}tests/results_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        print("\nSaved testing data in:", filename)
        df.to_csv(filename, index=False)


if __name__ == '__main__':
    algo = RB_CheatAgent(render=RENDER, render_file=RENDER_FILE)
    algo.test(episodes=TEST_EPISODES, print_every_n_episodes=PRINT_EVERY_N_EPISODES)
    # algo.store_testing_data()
