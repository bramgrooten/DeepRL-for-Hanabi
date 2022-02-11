"""
A simple episode runner using the RL environment.
by Bram Grooten
"""
import torch
from hanabi_learning_environment import rl_env_adjusted
from agents.human_player.human_player import HumanPlayer
from agents.rule_based.bram_simple_agent import BramSimpleAgent
from agents.rule_based.bram_bias import BramBiasAgent
from agents.rule_based.bram_best_rule_based_agent import BramsBestRuleBasedAgent
from agents.spg_dec.spg_dec_agent import SPG_Dec_Agent
from agents.vpg_dec.vpg_dec_agent import VPG_Dec_Agent
from agents.ppo_dec.ppo_dec_agent import PPO_Dec_Agent

CONFIG_SPG = {
    'hidden_layers': [64, 64],
    'obs_vec_type': 62,
    'num_actions': 11,
    "load_policy": ["2021-02-26_shuf", 25, 5]  # experiment name, session, sub-experiment number
}

CONFIG_VPG = {
    'hidden_layers_pi': [128, 128, 64],
    'hidden_layers_vf': [128, 64, 32],
    'obs_vec_type': 136,
    'num_actions': 11,
    "load_policy": ["2021-04-13_combi_vpg", 9, 4]  # experiment name, session, sub-experiment number
}

CONFIG_PPO = {
    'hidden_layers_pi': [128, 128, 64],
    'hidden_layers_vf': [128, 64, 32],
    'obs_vec_type': 186,
    'num_actions': 11,
    "load_policy": ["2021-04-13_combi_ppo", 8, 1]  # experiment name, session, sub-experiment number
}


class Runner(object):
    """Runner class."""
    def __init__(self, agents_config):
        """Initialize runner."""
        game_config = {"colors": 5,
                       "ranks": 5,
                       "players": len(agents_config),
                       "max_information_tokens": 8,
                       "max_life_tokens": 3}
        self.environment = rl_env_adjusted.make(obs_type="CHEAT", config=game_config)
        self.agents = agents_config
        print()
        for idx, agent in enumerate(self.agents):
            name = agent.__class__.__name__
            print(f'Agent {idx} is {name}')
            rl_agents = ['SPG', 'VPG', 'PPO']
            if any(algo in name for algo in rl_agents):
                agent.device = torch.device('cpu')
                agent.policy_network.to(torch.device('cpu'))

    def run(self, nr_episodes=1):
        """Run episodes."""
        rewards = []
        for episode in range(nr_episodes):
            observations = self.environment.reset()
            agents = self.agents

            # print the starting state, if human player involved
            for idx, agent in enumerate(agents):
                if isinstance(agent, HumanPlayer):
                    print('\nNEW GAME:\n')
                    agent.print_the_state(observations['player_observations'][idx])
                    break

            done = False
            episode_reward = 0
            turn = 0
            while not done:
                turn += 1
                for agent_id, agent in enumerate(agents):
                    observation = observations['player_observations'][agent_id]
                    action = agent.act(observation)
                    if observation['current_player'] == agent_id:
                        assert action is not None
                        current_player_action = action
                    else:
                        assert action is None

                # Make an environment step
                print(f"Turn {turn}, agent {observation['current_player']}, action: {current_player_action}")
                observations, reward, done, unused_info = self.environment.step(current_player_action)
                episode_reward += reward

            # print the end state, if human player involved
            for idx, agent in enumerate(agents):
                if isinstance(agent, HumanPlayer):
                    print('\nGAME FINISHED:\n')
                    agent.print_the_state(observations['player_observations'][idx])
                    print(f'\nSCORE: {episode_reward}')
                    if episode_reward == 25:
                        print('\nPERFECT GAME! CONGRATULATIONS!')
                    break

            rewards.append(episode_reward)
            print(f'\nFinished episode nr: {episode} with reward {episode_reward}')
            print(f'Max reward: {max(rewards):.3f}')
            avg = sum(rewards) / len(rewards)
            print(f'Average reward: {avg:.3f}')
        return rewards


if __name__ == "__main__":

    # Select 2 to 5 agents
    agents = [
        VPG_Dec_Agent(config=CONFIG_VPG),
        HumanPlayer(),
        # VPG_Dec_Agent(config=CONFIG_VPG),
        # VPG_Dec_Agent(config=CONFIG_VPG),
        # VPG_Dec_Agent(config=CONFIG_VPG),
        # VPG_Dec_Agent(config=CONFIG_VPG),
        # BramSimpleAgent(config={}),
        # BramBiasAgent(config={'play_position': 2}),
        # BramsBestRuleBasedAgent(config={}),
        # SPG_Dec_Agent(config=CONFIG_SPG),
        # PPO_Dec_Agent(config=CONFIG_PPO),
    ]

    runner = Runner(agents_config=agents)
    runner.run(nr_episodes=1)
