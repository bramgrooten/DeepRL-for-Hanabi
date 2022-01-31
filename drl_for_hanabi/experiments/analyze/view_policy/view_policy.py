import pandas as pd
import random
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from datetime import datetime
from time import sleep
from drl_for_hanabi.extra.supervised.supervised_algo import SupervisedAgent
from drl_for_hanabi.algorithms.spg import SPG_Agent

"""
Example state and action:

STATE IN TURN: 21
Fireworks: R0 Y0 G0 W1 B1 	Life tokens: 3	Info tokens: 2
Player 0:  Y5 G4 R1 G3 G2 	Deck size: 32	Discards: B4 W4 G2 R2 W1 B3
Player 1:  W4 Y2 R3 W5 R2 	Current player: 0
ACTION TAKEN:
legal: {'action_type': 'PLAY', 'card_index': 2}

which is:
    fireworks = "R0 Y0 G0 W1 B1"
    cards_p0 = "Y5 G4 R1 G3 G2"
    cards_p1 = "W4 Y2 R3 W5 R2"
    life_tokens = 3
    info_tokens = 2
    current_player = 0
    deck_size = 32
    discards = "B4 W4 G2 R2 W1 B3"
"""


def generate_random_state(num_players=2, hand_size=5, max_life=3, max_hint=8):
    """Generates a random possible state of Hanabi
    Args:
        num_players: int, number in [2,5]
        hand_size: int, number of cards that each player has, in [4,5]
        max_life: int, max number of life tokens, >= 1
        max_hint: int, max number of hint tokens, >= 0
    Returns:
        a random Hanabi state in text representation
    """
    # Generate deck
    deck = []
    colors = ['R', 'Y', 'G', 'W', 'B']
    ranks = [1, 2, 3, 4, 5]
    for c in colors:
        for r in ranks:
            card = {'color': c, 'rank': r-1}  # card ranks represented as: 0,1,2,3,4 like in HLE
            if r == 1:
                deck += [card] * 3
            elif r == 5:
                deck += [card]
            else:
                deck += [card] * 2

    # Generate fireworks
    fire_str = ""
    # fire_dict = {}
    for color in colors:
        fire_rank = random.randint(0,5)
        fire_str += f"{color}{fire_rank} "
        # fire_dict[color] = fire_rank
        if fire_rank > 0:
            # Take cards out of deck that must have been played
            for rank in range(fire_rank):
                deck.remove({'color': color, 'rank': rank})
    random.shuffle(deck)

    # Generate players' hands
    hands = []
    for player in range(num_players):
        hand = ""
        for _ in range(hand_size):
            card = deck.pop(0)
            hand += f"{card['color']}{card['rank']+1} "
        hands.append(hand)

    # Randomly discard some cards
    discards = ""
    num_discards = random.randint(0, len(deck))
    for _ in range(num_discards):
        card = deck.pop(0)
        discards += f"{card['color']}{card['rank']+1} "
    deck_size = len(deck)

    # Randomize the rest
    life_tokens = random.randint(1, max_life)
    info_tokens = random.randint(0, max_hint)
    current_player = random.randint(0, num_players-1)

    return fire_str, hands, life_tokens, info_tokens, current_player, deck_size, discards


def print_the_state(fireworks, hands, life_tokens, info_tokens, current_player, deck_size, discards):
    """Prints the state (text representation)."""
    print(f"Fireworks: {fireworks}")
    for player, hand in enumerate(hands):
        print(f"Cards player {player}: {hand}")
    print(f"Life tokens: {life_tokens}")
    print(f"Hint tokens: {info_tokens}")
    print(f"Current player: {current_player}")
    print(f"Deck size: {deck_size}")
    print(f"Discards: {discards}")


def convert_txt_state_to_cheat_obs(fireworks: str, cards_p0: str, cards_p1: str,
                           life_tokens: int, info_tokens: int,
                           current_player: int, deck_size: int, discards: str):
    """Converts a Hanabi state from text representation to dictionary representation."""
    obs = {'player_observations': [{},{}],
           'current_player': current_player}

    fire_lst = fireworks.split()
    cards_p0_lst = cards_p0.split()
    cards_p1_lst = cards_p1.split()
    discard_lst = discards.split()

    for p in range(2):
        obs['player_observations'][p]['life_tokens'] = life_tokens
        obs['player_observations'][p]['information_tokens'] = info_tokens
        obs['player_observations'][p]['num_players'] = 2
        obs['player_observations'][p]['deck_size'] = deck_size
        obs['player_observations'][p]['fireworks'] = {'R': 0, 'Y': 0, 'G': 0, 'W': 0, 'B': 0}
        obs['player_observations'][p]['discard_pile'] = []
        for fire in fire_lst:
            obs['player_observations'][p]['fireworks'][fire[0]] = int(fire[1])
        for discard in discard_lst:
            obs['player_observations'][p]['discard_pile'].append({
                'color': discard[0],
                'rank': int(discard[1]) - 1
            })
        obs['player_observations'][p]['card_knowledge'] = []
        obs['player_observations'][p]['card_knowledge'].append([])  # your own cards
        obs['player_observations'][p]['card_knowledge'].append([])  # the other player's cards

    for card in cards_p0_lst:
        obs['player_observations'][0]['card_knowledge'][0].append({
             'color': card[0],
             'rank': int(card[1]) - 1
         })
        obs['player_observations'][1]['card_knowledge'][1].append({
            'color': card[0],
            'rank': int(card[1]) - 1
        })
    for card in cards_p1_lst:
        obs['player_observations'][0]['card_knowledge'][1].append({
             'color': card[0],
             'rank': int(card[1]) - 1
         })
        obs['player_observations'][1]['card_knowledge'][0].append({
            'color': card[0],
            'rank': int(card[1]) - 1
        })
    return obs


def setup_algorithm(exps, main_exp, exp_num_inside, session=None, load_path=None):
    type_exp = exps[main_exp]['type_exp']
    total_exp_num = exps[main_exp]['num_experiments']
    start_date_exp = exps[main_exp]['date_exp']
    if session is None and load_path is None:
        raise ValueError("Provide a session nr or a path to load the policy from.")
    elif session is not None and load_path is not None:
        raise ValueError("Only provide a session OR a load_path, not both.")
    elif session is not None:
        policy = f"../../policy_params/{start_date_exp}_{type_exp}/s{session}/policy_exp_{exp_num_inside}.pth"
    elif load_path is not None:
        policy = load_path
    else:  # this should be unreachable
        raise ValueError

    if type_exp in ["vec37", "vec62", "act51"]:
        df_exp_settings = pd.read_csv(f'../../settings/exp_settings_{total_exp_num}_{type_exp}.csv')
    elif type_exp in ["2vec37", "2vec62"]:
        df_exp_settings = pd.read_csv(f'../../settings/exp_settings_{total_exp_num}_{type_exp[1:]}.csv')
    elif type_exp == "2redo7":
        df_exp_settings = pd.read_csv(f'../../settings/exp_settings_{type_exp[1:]}.csv')
    else:
        df_exp_settings = pd.read_csv(f'../../settings/exp_settings_{type_exp}.csv')
    exp_setting = df_exp_settings.iloc[exp_num_inside]
    
    # Preprocessing the experiments settings to the right format
    if type_exp != "supervised":
        possible_reward_adjustments = ["out of lives", "illegal move", "lost one life", "discard"]
        rewards = {}
        for rew_adj in possible_reward_adjustments:
            if exp_setting[rew_adj] != 'standard':
                rewards[rew_adj] = (
                    eval(exp_setting[rew_adj]) if isinstance(exp_setting[rew_adj], str) else exp_setting[rew_adj])
    hidden_layers = eval(exp_setting['layers'])
    render_file = f"renders_extra/{start_date_exp}_{type_exp}/s{session}/game_traces-for-Experiment_{exp_num_inside}.txt"
    activation = eval("nn." + exp_setting.get('activation', 'Tanh'))
    if type_exp in ["vec37", "vec62", "2vec37", "2vec62"]:
        obs_vec_type = int(type_exp[-2:])
    else:
        obs_vec_type = exp_setting["obs_vec_type"]
    num_actions = exp_setting.get('num_actions', 11)
    weight_decay = exp_setting.get('weight_decay', 0.0)
    drop_rate = exp_setting.get('drop_rate', 0.0)

    if type_exp == "supervised":
        algo = SupervisedAgent(epochs=10,
                               runtime=1 / 60,
                               show_epoch_nr=100,
                               print_training=False,
                               batch_size=exp_setting['batch_sizes'],
                               lr=exp_setting['learning_rates'],
                               hidden_layers=hidden_layers,
                               activation=activation,
                               obs_vec_type=exp_setting['obs_vec_type'],
                               num_actions=exp_setting['num_actions'],
                               render_training=False,
                               render_tests=False,
                               render_file=render_file,
                               experiment_num=exp_num_inside)
    else:
        algo = SPG_Agent(epochs=10,
                         runtime=1 / 60,
                         show_epoch_nr=100,
                         print_training=False,
                         batch_size=exp_setting['batch_sizes'],
                         lr=exp_setting['learning_rates'],
                         gamma=exp_setting['gammas'],
                         # weight_decay=weight_decay,
                         # drop_rate=drop_rate,
                         reward_settings=rewards,
                         hidden_layers=hidden_layers,
                         activation=activation,
                         obs_vec_type=obs_vec_type,
                         num_actions=num_actions,
                         render_training=False,
                         render_tests=False,
                         render_file=render_file,
                         experiment_num=exp_num_inside)

    algo.load_policy(load_path=policy)

    # data_folder = f"../../data/{start_date_exp}_{type_exp}/s{session}/"
    # algo.test()
    # algo.store_testing_data(folder=data_folder)
    return algo


def get_policy_probabilities(algo, obs):
    obs_vec = algo.observation_to_vector(obs)
    logits = algo.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution
    pol_lst = policy_given_obs.probs.tolist()
    rounded = [round(num, 4) for num in pol_lst]
    print()
    print(pol_lst)
    print(rounded)
    return rounded


def make_policy_histogram(policy, title="Policy probabilities", save_path="figures/policy.png"):
    if len(policy) == 11:
        x = list(range(11))
        fig, ax = plt.subplots(1)
        ax.bar(x=x, height=policy)
        ax.set_xticks(x)
        ax.set_xticklabels(['d1', 'd2', 'd3', 'd4', 'd5', 'p1', 'p2', 'p3', 'p4', 'p5', 'hint'])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        fig.savefig(save_path)
        # plt.show()
        plt.close(fig)
    elif len(policy) == 51:
        x = list(range(51))
        fig, ax = plt.subplots(1, figsize=(14, 6))
        ax.bar(x=x, height=policy)
        ax.set_xticks(x)
        moves = []
        for act in ['p', 'd']:
            for color in ['R', 'Y', 'G', 'W', 'B']:
                for rank in range(5):
                    moves.append(f"{color}{rank+1}")
        moves.append('h')
        ax.set_xticklabels(moves)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        fig.subplots_adjust(bottom=0.15)
        fig.text(0.32, 0.05, "PLAY", fontsize=14)
        fig.text(0.504, 0.05, "|", fontsize=14)
        fig.text(0.65, 0.05, "DISCARD", fontsize=14)
        fig.text(0.85, 0.05, "| HINT", fontsize=14)
        fig.savefig(save_path)
        # plt.show()
        plt.close(fig)
    else:
        print("Unknown number of actions.")


def make_policy_added_exploration_histogram(policy, epsilon=0.3,
                                            title="Policy (with added exploration) probabilities",
                                            save_path="figures/policy_added_exploration.png"):
    """Computes new probs, for the policy with added exploration.
    Assumes: that the number of actions is 11."""
    new_policy = [epsilon * 1/11 + (1-epsilon)*old_prob for old_prob in policy]
    print()
    print("old policy:", policy)
    print("new policy:", [round(p, 3) for p in new_policy])

    x = list(range(11))
    fig, ax = plt.subplots(1)
    ax.bar(x=x, height=new_policy)
    ax.set_xticks(x)
    ax.set_xticklabels(['d1', 'd2', 'd3', 'd4', 'd5', 'p1', 'p2', 'p3', 'p4', 'p5', 'hint'])
    ax.set_ylim(0, 1)
    ax.set_title(title)
    fig.savefig(save_path)
    # plt.show()
    plt.close(fig)



if __name__ == '__main__':
    start_time = datetime.now()


    ### Available experiments to view
    exps = [
        {"date_exp": "2021-02-26",
         "type_exp": "shuf",
         "num_experiments": 5,
         "num_continued_sessions": 25},
    ]


    ### Choose which experiment you want to see here:
    main_exp = 0
    exp_num_inside = 5
    session = 25

    ### Setting the state here:
    fireworks = "R0 Y0 G2 W1 B3"
    cards_p0 =  "R5 B4 R2 W2 G1"

    cards_p1 = "G3 W1 Y1 G3 B3"
    life_tokens = 2
    info_tokens = 5
    current_player = 0
    deck_size = 31
    discards = "Y4 B1 R1"

    ### or generate a random state, and print it, here:
    # fireworks, hands, life_tokens, info_tokens, current_player, deck_size, discards = generate_random_state(num_players=2)
    # cards_p0, cards_p1 = hands
    # print_the_state(fireworks, hands, life_tokens, info_tokens, current_player, deck_size, discards)


    algo = setup_algorithm(exps, main_exp, exp_num_inside, session=session)
    obs = convert_txt_state_to_cheat_obs(fireworks, cards_p0, cards_p1, life_tokens, info_tokens, current_player, deck_size, discards)
    policy = get_policy_probabilities(algo, obs)
    make_policy_histogram(policy, save_path="figures/policy_fig.png")
    make_policy_added_exploration_histogram(policy, epsilon=0.5, save_path="figures/policy_fig_added_explore.png")

    print("\nRunning time:", datetime.now() - start_time)
