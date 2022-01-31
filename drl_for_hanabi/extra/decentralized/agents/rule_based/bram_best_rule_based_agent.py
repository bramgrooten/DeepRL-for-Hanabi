"""The best rule-based agent I made.
by Bram Grooten"""
import random
from hanabi_learning_environment.rl_env import Agent


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


class BramsBestRuleBasedAgent(Agent):
    """class for the Rule-Based algorithm that plays well in cheat mode"""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        # Extract max info tokens / life tokens or set defaults to 8 / 3.
        self.max_information_tokens = config.get('information_tokens', 8)
        self.max_life_tokens = config.get('life_tokens', 3)

    def reset(self, config=None):
        pass

    def act(self, obs):
        """Chooses the best action to play, based on a full cheat observation
        Args:
            obs: dict, the HLE observation for current player
        Returns:
            dict, an HLE action
        """
        # Only act if it's your turn
        if obs['current_player_offset'] != 0:
            return None

        my_cards = obs['card_knowledge'][0]

        # See if we have playable cards
        playable_cards = []
        fireworks = obs['fireworks']
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
        discarding_legal = (obs['information_tokens'] < self.max_information_tokens)
        if discarding_legal:
            discard_pile = obs['discard_pile']
            useless_indices = []
            for idx, card in enumerate(my_cards):
                if useless_card(card, fireworks, discard_pile):
                    useless_indices.append(idx)

            if useless_indices:
                discard_idx = random.choice(useless_indices)
                return {'action_type': 'DISCARD', 'card_index': discard_idx}

            # No useless cards here, but we may discard. We would rather hint ('pass') in this situation though.
            hint_possible = (obs['information_tokens'] > 0)
            if hint_possible:
                legals_HLE = obs['legal_moves']
                return legals_HLE[-1]
            else:
                # We don't have any useless cards, but hinting is also not allowed.
                # So unfortunately, we must discard.
                discard_idx = find_best_discard_option(my_cards, fireworks, discard_pile)
                return {'action_type': 'DISCARD', 'card_index': discard_idx}

        else:  # Hinting must be legal, if discarding is not.
            legals_HLE = obs['legal_moves']
            return legals_HLE[-1]
