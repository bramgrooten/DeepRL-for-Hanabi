"""Functions that provide different options of action representation.
For the centralized version of the agents."""
import torch
from torch.distributions.categorical import Categorical
import random


def choose_action_11position(self, obs, obs_vec, pick_max_prob=False, epsilon=0.0):
    """
    Function to convert the output of the policy network to an action.
    And outputs whether the action was legal or not.

    Sometimes some actions are not allowed, for example:
    - hinting when there are no hint tokens left,
    - discarding when the hint tokens are already at maximum,
    - hinting a color/rank to a player who does not have that color/rank in her cards,
    - playing from an index that doesn't have a card anymore (only an implementation problem, not in real life)

    Actions are:
    0,1,2,3,4 mean: discard card at index 0,1,2,3,4
    5,6,7,8,9 mean: play card at index 0,1,2,3,4
    10 means: give a random hint

    Args:
        obs: dict, the full Hanabi observation (centralized)
        obs_vec: list, observation vector
        pick_max_prob: bool, whether the action picked should be the one with the highest probability, or just a sample
        epsilon: float, between [0,1], probability of choosing a legal move uniformly at random
    Returns:
        int: the move's integer, in [0, 10]
        bool: whether the move is legal
    """
    policy_given_obs = self.get_policy(torch.as_tensor(obs_vec, dtype=torch.float32))

    current_player_id = obs['current_player']
    legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']

    if pick_max_prob:  # for testing mode, if you want a deterministic policy set this to True
        act_int = torch.argmax(policy_given_obs.probs).item()
    else:
        # if random.random() < epsilon:    # uncomment if you want to use epsilon
        #     act_int = pick_random_legal_action(legals_int)
        # else:
        act_int = policy_given_obs.sample().item()

    if act_int == 10:  # hint action chosen
        hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
        if hint_possible:
            return act_int, True
        else:
            return act_int, False

    else:  # discard or play action chosen
        if act_int in legals_int:
            return act_int, True
        else:
            return act_int, False
            # chosen action is illegal (must be a discard with max hint tokens)


def choose_action_11position_shuffle(self, obs, obs_vec, pick_max_prob=False, epsilon=0.0):
    """
    Function to convert the output of the policy network to an action.
    And outputs whether the action was legal or not.

    Sometimes some actions are not allowed, for example:
    - hinting when there are no hint tokens left,
    - discarding when the hint tokens are already at maximum,
    - hinting a color/rank to a player who does not have that color/rank in her cards,
    - playing from an index that doesn't have a card anymore (only an implementation problem, not in real life)

    Actions are:
    0,1,2,3,4 mean: discard card at index 0,1,2,3,4
    5,6,7,8,9 mean: play card at index 0,1,2,3,4
    10 means: give a random hint

    Args:
        obs: dict, the full Hanabi observation (centralized)
        obs_vec: list, observation vector
        pick_max_prob: bool, whether the action picked should be the one with the highest probability, or just a sample
        epsilon: float, between [0,1], probability of choosing a legal move uniformly at random
    Returns:
        int: the move's integer, in [0, 10]
        bool: whether the move is legal
    """
    policy_given_obs = self.get_policy(torch.as_tensor(obs_vec, dtype=torch.float32))

    current_player_id = obs['current_player']
    legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']

    if pick_max_prob:  # for testing mode, if you want a deterministic policy set this to True
        act_int = torch.argmax(policy_given_obs.probs).item()
        shuffle_back = True
    else:
        if random.random() < epsilon:
            act_int = pick_random_legal_action(legals_int)
            shuffle_back = False
        else:
            act_int = policy_given_obs.sample().item()
            shuffle_back = True

    if shuffle_back:
        if act_int < 5:
            act_int = self.shuffled_indices[act_int]
        elif act_int < 10:
            act_int = self.shuffled_indices[act_int - 5] + 5

    if act_int == 10:  # hint action chosen
        hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
        if hint_possible:
            return act_int, True
        else:
            return act_int, False
    else:  # discard or play action chosen
        if act_int in legals_int:
            return act_int, True
        else:
            return act_int, False
            # chosen action is illegal
            # (must be a discard with max hint tokens, or play missing card position at end of game)


def pick_random_legal_action(legals_int):
    """Picks a legal action uniformly at random.
    Args:
        legals_int: list, from the HLE observation, containing integers of legal moves
                    This list has an integer for each hint move. In cheat mode we compress these to one integer: 10
                    Assumption: this list is sorted in ascending order.
    Returns:
        int: the integer of the action picked
    """
    legals_int_11 = []
    for move in legals_int:
        if move <= 9:
            legals_int_11.append(move)
        else:
            legals_int_11.append(10)
            break
    return random.choice(legals_int_11)


def convert_hint_to_HLE_int(obs):
    """This converts the hint action from 11 possibilities to an action integer that the HLE uses.
    Assumption: this function is only called when hinting is legal.
    Args:
        obs: dict, the full HLE observation
    Returns:
        HLE_int: int, from legals_int list
    """
    current_player_id = obs['current_player']
    legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']
    return legals_int[-1]


def convert_move_to_action_probabilities_11(move):
    """Converts an HLE move to an action distribution,
    where the chosen move gets probability 1,
    and all other moves get probability 0.

    Args:
        move: dict, an HLE action, e.g. {'action_type': 'PLAY', 'card_index': 0}
    Returns:
        action_probs: list, of length 11, e.g. [0,0,0,0,0,1,0,0,0,0,0]
    """
    if move['action_type'] == 'PLAY':
        idx = move['card_index'] + 5
    elif move['action_type'] == 'DISCARD':
        idx = move['card_index']
    else:
        idx = 10
    action_probs = [0] * 11
    action_probs[idx] = 1
    return action_probs


def choose_action_51unique(self, obs, obs_vec, pick_max_prob=False, epsilon=0.0):
    """
    Choose action and return: in HLE form, integer form, and whether it is legal.
    51 action integers:
    0 = play Red 1 card
    1 = play Red 2 card
    ...
    4 = play Red 5 card
    5 = play Yellow 1 card
    ... Green ... White ...
    24 = play Blue 5 card
    25-49 = discard
    50 = give random hint

    Args:
        obs: dict, the full Hanabi observation (centralized)
        obs_vec: list, the observation vector
        pick_max_prob: bool, whether the action picked should be the one with the highest probability, or just a sample
        epsilon: float, between [0,1], probability of choosing a legal move uniformly at random
    Returns:
        int: the move's integer, in [0,10]
        bool: whether the move is legal
    """
    logits = self.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution

    if pick_max_prob:
        act_int = torch.argmax(policy_given_obs.probs).item()
    else:
        if random.random() < epsilon:
            # choose random move
            # TODO: choose only legal moves
            act_int = random.randint(0, 50)
        else:
            act_int = policy_given_obs.sample().item()

    current_player_id = obs['current_player']
    color_dict2 = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
    own_cards = obs['player_observations'][current_player_id]['card_knowledge'][0]

    if act_int == 50:  # hint action chosen
        hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
        if hint_possible:
            return act_int, True
        else:
            return act_int, False

    elif act_int < 25:  # play action
        rank = act_int % 5  # modulo, can be: 0,1,2,3,4 (which is also how the ranks are named in the HLE)
        color = act_int // 5  # int division, can be: 0,1,2,3,4
        for idx, card in enumerate(own_cards):
            if rank == card['rank'] and color_dict2[color] == card['color']:
                return act_int, True
        return act_int, False  # you don't have this card, so you can't play it

    else:  # discard action, act_int between 25 and 49
        discard_possible = (obs['player_observations'][current_player_id]['information_tokens'] < 8)
        if discard_possible:
            card_num = act_int - 25
            rank = card_num % 5
            color = card_num // 5
            for idx, card in enumerate(own_cards):
                if rank == card['rank'] and color_dict2[color] == card['color']:
                    return act_int, True
            return act_int, False  # discarding is allowed, but you don't have that card
        else:
            return act_int, False  # discarding is not allowed in this case


def convert_51int_to_HLE_int(act_int, obs):
    """This converts the act_int from 51 possibilities to the action integers that the HLE uses.
    Assumption: this function is only called when the move is legal.
    Args:
        act_int: int, in [0,50]
    Returns:
        HLE_int: int, from legals_int list
    """
    current_player_id = obs['current_player']
    legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']
    color_dict2 = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
    own_cards = obs['player_observations'][current_player_id]['card_knowledge'][0]
    if act_int <= 24:
        rank = act_int % 5  # modulo, can be: 0,1,2,3,4 (which is also how the ranks are named in the HLE)
        color = act_int // 5  # int division, can be: 0,1,2,3,4
        for idx, card in enumerate(own_cards):
            if rank == card['rank'] and color_dict2[color] == card['color']:
                return idx + 5
    elif act_int <= 49:
        card_num = act_int - 25
        rank = card_num % 5
        color = card_num // 5
        for idx, card in enumerate(own_cards):
            if rank == card['rank'] and color_dict2[color] == card['color']:
                return idx
    elif act_int == 50:
        return legals_int[-1]
    else:
        return f"UNKNOWN ACTION INTEGER: {act_int}"


def choose_action_11position_old(self, obs, pick_max_prob=False, epsilon=0):
    """
    Old, meaning: still returns a dict as move. New one only returns int.

    Function to convert the output of the policy network to an action.
    And outputs whether the action was legal or not.

    Sometimes some actions are not allowed, for example:
    - hinting when there are no hint tokens left,
    - discarding when the hint tokens are already at maximum,
    - hinting a color/rank to a player who does not have that color/rank in her cards,
    - playing from an index that doesn't have a card anymore (only an implementation problem, not in real life)

    Actions are:
    0,1,2,3,4 mean: discard card at index 0,1,2,3,4
    5,6,7,8,9 mean: play card at index 0,1,2,3,4
    10 means: give a random hint

    Args:
        obs: dict, the full Hanabi observation (centralized)
        pick_max_prob: bool, whether the action picked should be the one with the highest probability, or just a sample
        epsilon: float, between [0,1], probability of choosing a legal move uniformly at random
    Returns:
        dict: the HLE move,
        int: the move's integer,
        bool: whether the move is legal
    """
    obs_vec = self.observation_to_vector(obs)
    logits = self.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution


    current_player_id = obs['current_player']
    legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']
    legals_HLE = obs['player_observations'][current_player_id]['legal_moves']

    if pick_max_prob:  # for testing mode, if you want a deterministic policy set this to True
        act_int = torch.argmax(policy_given_obs.probs).item()
    else:
        if random.random() < epsilon:
            act_int = pick_random_legal_action(legals_int)
        else:
            act_int = policy_given_obs.sample().item()


    if act_int == 10:  # hint action chosen
        hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
        if hint_possible:
            return legals_HLE[-1], act_int, True
            # last legal HLE action is a hint, when hinting is possible
        else:
            return {}, act_int, False

    else:  # discard or play action chosen
        if act_int in legals_int:
            idx = legals_int.index(act_int)
            return legals_HLE[idx], act_int, True
        else:
            # chosen action is illegal
            # (must be a discard with max hint tokens in this case (or play missing card))
            return {}, act_int, False


def choose_action_51unique_old(self, obs, pick_max_prob=False, epsilon=0):
    """
    Old, meaning: still returns a dict as move. New one only returns int.

    Choose action and return: in HLE form, integer form, and whether it is legal.
    51 action integers:
    0 = play Red 1 card
    1 = play Red 2 card
    ...
    4 = play Red 5 card
    5 = play Yellow 1 card
    ... Green ... White ...
    24 = play Blue 5 card
    25-49 = discard
    50 = give random hint

    Args:
        obs: dict, the full Hanabi observation (centralized)
        pick_max_prob: bool, whether the action picked should be the one with the highest probability, or just a sample
        epsilon: float, between [0,1], probability of choosing a legal move uniformly at random
    Returns:
        dict: the HLE move,
        int: the move's integer,
        bool: whether the move is legal
    """
    obs_vec = self.observation_to_vector(obs)
    logits = self.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution

    current_player_id = obs['current_player']
    legals_int = obs['player_observations'][current_player_id]['legal_moves_as_int']

    if pick_max_prob:
        act_int = torch.argmax(policy_given_obs.probs).item()
    else:
        if random.random() < epsilon:
            # choose random move
            # TODO: choose only legal moves
            act_int = random.randint(0,50)
        else:
            act_int = policy_given_obs.sample().item()


    if act_int == 50:  # hint action chosen
        hint_possible = (obs['player_observations'][current_player_id]['information_tokens'] > 0)
        if hint_possible:
            legals_HLE = obs['player_observations'][current_player_id]['legal_moves']
            # last legal HLE action is a hint, when hinting is possible
            return legals_HLE[-1], act_int, True
        else:
            return {}, act_int, False

    elif act_int < 25:  # play action
        rank = act_int % 5  # modulo, can be: 0,1,2,3,4 (which is also how the ranks are named in the HLE)
        color = act_int // 5  # int division, can be: 0,1,2,3,4
        color_dict2 = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
        own_cards = obs['player_observations'][current_player_id]['card_knowledge'][0]
        for idx, card in enumerate(own_cards):
            if rank == card['rank'] and color_dict2[color] == card['color']:
                return {'action_type': 'PLAY', 'card_index': idx}, act_int, True
        return {}, act_int, False  # you don't have this card, so you can't play it

    else:  # discard action, act_int between 25 and 49
        discard_possible = (obs['player_observations'][current_player_id]['information_tokens'] < 8)
        if discard_possible:
            card_num = act_int - 25
            rank = card_num % 5
            color = card_num // 5
            color_dict2 = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
            own_cards = obs['player_observations'][current_player_id]['card_knowledge'][0]
            for idx, card in enumerate(own_cards):
                if rank == card['rank'] and color_dict2[color] == card['color']:
                    return {'action_type': 'DISCARD', 'card_index': idx}, act_int, True
            return {}, act_int, False  # discarding is allowed, but you don't have that card
        else:
            return {}, act_int, False  # discarding is not allowed in this case
