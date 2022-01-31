"""Bram bias Agent."""
from hanabi_learning_environment.rl_env import Agent


class BramBiasAgent(Agent):
    """Agent that just plays from the same position every time."""

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent."""
        self.play_position = config.get('play_position', 0)

    def reset(self, config=None):
        pass

    def act(self, observation):
        """Act based on an observation."""
        # Only act if it's your turn
        if observation['current_player_offset'] != 0:
            return None
        return {'action_type': 'PLAY', 'card_index': self.play_position}


