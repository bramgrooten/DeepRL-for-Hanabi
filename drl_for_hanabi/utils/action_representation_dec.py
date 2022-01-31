"""Functions that provide different options of action representation.
For the DECentralized version of the agents."""
import torch
from torch.distributions.categorical import Categorical


def choose_action_11position(self, obs, explore=True):
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
    """
    obs_vec = self.observation_to_vector(obs)
    logits = self.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution

    if explore:
        act_int = policy_given_obs.sample().item()
    else:
        act_int = torch.argmax(policy_given_obs.probs).item()

    legals_int = obs['legal_moves_as_int']
    legals_HLE = obs['legal_moves']
    hint_possible = (obs['information_tokens'] > 0)

    if act_int == 10:  # hint action chosen
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


def choose_action_51unique(self, obs, explore=True, max_hint=8):
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
    """
    obs_vec = self.observation_to_vector(obs)
    logits = self.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution

    if explore:
        act_int = policy_given_obs.sample().item()
    else:
        act_int = torch.argmax(policy_given_obs.probs).item()

    if act_int == 50:  # hint action chosen
        hint_possible = (obs['information_tokens'] > 0)
        if hint_possible:
            legals_HLE = obs['legal_moves']
            # last legal HLE action is a hint, when hinting is possible
            return legals_HLE[-1], act_int, True
        else:
            return {}, act_int, False

    elif act_int < 25:  # play action
        rank = act_int % 5  # modulo, can be: 0,1,2,3,4 (which is also how the ranks are named in the HLE)
        color = act_int // 5  # int division, can be: 0,1,2,3,4
        color_dict2 = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
        own_cards = obs['card_knowledge'][0]
        for idx, card in enumerate(own_cards):
            if rank == card['rank'] and color_dict2[color] == card['color']:
                return {'action_type': 'PLAY', 'card_index': idx}, act_int, True
        return {}, act_int, False  # you don't have this card, so you can't play it

    else:  # discard action, act_int between 25 and 49
        discard_possible = (obs['information_tokens'] < max_hint)
        if discard_possible:
            card_num = act_int - 25
            rank = card_num % 5
            color = card_num // 5
            color_dict2 = {0: 'R', 1: 'Y', 2: 'G', 3: 'W', 4: 'B'}
            own_cards = obs['card_knowledge'][0]
            for idx, card in enumerate(own_cards):
                if rank == card['rank'] and color_dict2[color] == card['color']:
                    return {'action_type': 'DISCARD', 'card_index': idx}, act_int, True
            return {}, act_int, False  # discarding is allowed, but you don't have that card
        else:
            return {}, act_int, False  # discarding is not allowed in this case
