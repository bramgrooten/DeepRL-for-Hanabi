"""Bram Simple Agent."""
from hanabi_learning_environment.rl_env import Agent


def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile.
    This function might be confusing at first, as you'd think it should say:
        return card['rank'] == fireworks[card['color']] + 1
    However, the ranks of the cards in this program are: 0,1,2,3,4
    while the fireworks are numbered as normal.
    """
    return card['rank'] == fireworks[card['color']]


class BramSimpleAgent(Agent):
    """Agent that applies a simple strategy and plays safely."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        # Extract max info tokens / life tokens or set defaults to 8 / 3.
        self.max_information_tokens = config.get('information_tokens', 8)
        self.max_life_tokens = config.get('life_tokens', 3)

    def reset(self, config=None):
        pass

    def act(self, observation):
        """Act based on an observation."""

        # Only act if it's your turn
        if observation['current_player_offset'] != 0:
            return None

        # print("")
        # print("TURN OF AGENT NR: ", observation['current_player'])
        # print(observation['pyhanabi'])
        my_cards = observation['card_knowledge'][0]
        # print("Knowledge about my cards:", my_cards)

        # Play a card if we're certain that it fits
        fireworks = observation['fireworks']
        for idx, card in enumerate(my_cards):
            if card['color'] is not None and card['rank'] is not None:
                if playable_card(card, fireworks):
                    return {'action_type': 'PLAY', 'card_index': idx}

        # Hint a playable card to your teammates (if it's possible)
        if observation['information_tokens'] > 0:
            # Check if there are any playable cards in the hands of the teammates.
            for player_offset in range(1, observation['num_players']):
                player_hand = observation['observed_hands'][player_offset]
                player_hints = observation['card_knowledge'][player_offset]
                # Check if the card in the hand of the opponent is playable.
                for card, hint in zip(player_hand, player_hints):
                    if playable_card(card, fireworks):
                        if hint['rank'] is None:
                            return {
                                'action_type': 'REVEAL_RANK',
                                'rank': card['rank'],
                                'target_offset': player_offset
                            }
                        if hint['color'] is None:
                            return {
                                'action_type': 'REVEAL_COLOR',
                                'color': card['color'],
                                'target_offset': player_offset
                            }

        # If no card is hintable then discard
        if observation['information_tokens'] < self.max_information_tokens:
            for idx, card in enumerate(my_cards):
                if card['color'] is None and card['rank'] is None:
                    return {'action_type': 'DISCARD', 'card_index': idx}
            return {'action_type': 'DISCARD', 'card_index': 0}
        else:
            # if we can't discard (already at max nr of info tokens), then give random hint
            return {'action_type': 'REVEAL_COLOR',
                    'color': observation['observed_hands'][1][0]['color'],
                    'target_offset': 1}
