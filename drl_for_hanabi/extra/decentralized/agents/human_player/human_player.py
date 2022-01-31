"""Human player, input with keyboard."""
from hanabi_learning_environment.rl_env import Agent


class HumanPlayer(Agent):
    """Agent which is controlled by a human with a keyboard."""

    def __init__(self, max_hint=8):
        """Initialize the agent."""
        self.max_hint = max_hint

    def reset(self):
        """Reset the agent."""
        self.__init__()

    def print_the_state(self, obs):
        han_list = str(obs['pyhanabi']).splitlines()
        lives = han_list[0]
        hints = han_list[1]
        fireworks = han_list[2]
        deck_size = han_list[-2]
        discards = han_list[-1]
        output = fireworks + "\t" + lives + "\t" + hints + "\n"
        output += "Your hand: "
        for card in obs['card_knowledge'][0]:
            output += card['color']
            output += str(card['rank'] + 1) + " "
        output += "\t" + deck_size + "\t" + discards + "\n"
        output += "Other's:   "
        for card in obs['card_knowledge'][1]:
            output += card['color']
            output += str(card['rank'] + 1) + " "
        output += "\tCurrent player: " + str(obs['current_player'])
        print(output)

    def act(self, observation):
        """Act based on an observation."""
        # Only act if it's your turn
        if observation['current_player_offset'] != 0:
            return None

        done_choosing = False
        print()
        self.print_the_state(observation)
        while not done_choosing:
            human_action = input("It's your turn. What action do you take?  "
                                 "(options: d1,d2,d3,d4,d5,p1,p2,p3,p4,p5,h)\n")
            if len(human_action) == 0:
                print("Please choose an action.")
                continue
            if human_action[0] == 'h':
                if observation['information_tokens'] <= 0:
                    print("There are no information tokens left, hinting is not allowed. Try again.")
                else:
                    action = observation['legal_moves'][-1]
                    done_choosing = True
            elif human_action[0] == 'p':
                card_idx = int(human_action[1]) - 1
                if card_idx >= len(observation['observed_hands'][0]):
                    print("You don't have a card in that position. Try again.")
                else:
                    action = {'action_type': 'PLAY', 'card_index': card_idx}
                    done_choosing = True
            elif human_action[0] == 'd':
                if observation['information_tokens'] >= self.max_hint:
                    print("Discarding is not allowed, because you already have all information tokens. Try again.")
                else:
                    card_idx = int(human_action[1]) - 1
                    action = {'action_type': 'DISCARD', 'card_index': card_idx}
                    done_choosing = True
            else:
                print("That's not a valid move, try again.")
        return action
