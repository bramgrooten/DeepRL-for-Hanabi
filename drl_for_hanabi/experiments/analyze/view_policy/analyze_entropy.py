import view_policy
from math import log
from datetime import datetime
import torch
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt


def compute_entropy(probs):
    """Computes the entropy of a categorical distribution given by probs.
    Args:
        probs: list, containing the probabilities
    Returns:
        Entropy: float, value computed as H(X) = -sum( p_i * log(p_i))
    """
    entropy = 0
    for p in probs:
        if p > 0:
            entropy += p * log(p)
        # else: print("p was <= 0, namely:", p)
    return -entropy


def get_exact_policy_probabilities(algo, obs):
    """Computes the policy probs for a given observation.
    Args:
        algo: agent class instance, the agent who's policy we compute
        obs: dict, the observation in HLE form
    Returns:
        policy probs, as a list
    """
    obs_vec = algo.observation_to_vector(obs)
    logits = algo.policy_network(torch.as_tensor(obs_vec, dtype=torch.float32))
    policy_given_obs = Categorical(logits=logits)  # action distribution
    pol_lst = policy_given_obs.probs.tolist()
    return pol_lst


def analyze_n_policies(agent, n=1000, num_actions=11, print_results=True, make_histograms=True):
    """Analyzes the policies of n randomly generated states.
    Args:
        agent: agent class instance, the agent who's policy we're analyzing
        n: int, the number of different observations to generate
    Returns:
        average entropy
    """
    entropy_lst = []
    total_entropy = 0
    total_probs_lst = [0] * num_actions
    for iteration in range(n):
        fireworks, hands, life_tokens, info_tokens, current_player, deck_size, discards = \
            view_policy.generate_random_state(num_players=2)
        cards_p0, cards_p1 = hands
        obs = view_policy.convert_txt_state_to_cheat_obs(fireworks, cards_p0, cards_p1, life_tokens, info_tokens,
                                                         current_player, deck_size, discards)
        policy = get_exact_policy_probabilities(agent, obs)
        entropy = compute_entropy(policy)
        entropy_lst.append(entropy)
        total_entropy += entropy
        total_probs_lst = [a + b for a, b in zip(policy, total_probs_lst)]
        # print(f"entropy in iteration {iteration}: {entropy}")
    avg_entropy = total_entropy / n
    avg_probs_lst = [prob / n for prob in total_probs_lst]
    if print_results:
        print(f"\nAfter analyzing the policies in {n} different states:")
        print("avg entropy:", avg_entropy)
        print("avg prob list:", avg_probs_lst)
    if make_histograms:
        view_policy.make_policy_histogram(avg_probs_lst, title="Average policy probabilities",
                                          save_path="figures/avg_policy.png")
        make_entropy_histogram(entropy_lst)
    return avg_entropy, avg_probs_lst



def make_entropy_histogram(entropy_lst, title="Entropy of n policies", save_path="figures/entropy_distr.png"):
    fig, ax = plt.subplots(1)
    ax.hist(x=entropy_lst, bins=100)
    ax.set_xlabel("Entropy")
    ax.set_ylabel("Nr of occurrences")
    ax.set_title(title)
    fig.savefig(save_path)
    plt.close(fig)


if __name__ == '__main__':
    start_time_main = datetime.now()

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

    agent = view_policy.setup_algorithm(exps, main_exp, exp_num_inside, session=session)
    analyze_n_policies(agent, n=100)

    print(f"\nRunning time: {datetime.now() - start_time_main}")
