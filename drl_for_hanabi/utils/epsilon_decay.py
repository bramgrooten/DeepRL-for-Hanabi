

def compute_decay_factor(epsilon_start=0.5, epsilon_decay_type="exponential", epsilon_low_at_epoch=10**4, epsilon_low=10**(-3)):
    """Computes the decay factor for the exploration hyperparameter: epsilon
    Args:
        epsilon_start: float, in (0,1). Assumption: 0 < epsilon_start < 1
        epsilon_decay_type: str, type of decay. Options: exponential, linear, no (for no decay)
        epsilon_low_at_epoch: int, the epoch at which you want epsilon to be low
        epsilon_low: float, define low yourself. For linear, 0 is used.
    Returs:
        epsilon_decay_factor, float.
        Can be used as follows:
        in linear case:      epsilon = start_eps - factor * epoch
        in exponential case: epsilon = start_eps * factor ^ epoch
    """
    if epsilon_decay_type == "linear":
        return epsilon_start / epsilon_low_at_epoch
    elif epsilon_decay_type == "exponential":
        return (epsilon_low / epsilon_start) ** (1/epsilon_low_at_epoch)
    elif epsilon_decay_type == "no":
        return 1  # doesn't matter what the factor is here, it's not used later anyway
    else:
        print(f"Unknown epsilon decay type given: {epsilon_decay_type}")


def compute_next_epsilon(prev_epsilon, epsilon_decay_type, epsilon_decay_factor):
    """Computes the next value of epsilon, based on the previous value
    Args:
        prev_epsilon: float, previous value of epsilon
        epsilon_decay_type: str, type of decay. Options: exponential, linear, no (for no decay)
        epsilon_decay_factor: float, the factor to decay with.
    Returns:
          The next value of epsilon, after one decay step.
    """
    if epsilon_decay_type == "linear":
        return prev_epsilon - epsilon_decay_factor
        # should be max( <above> , 0) but this is unnecessary as random < epsilon will still always be False
    elif epsilon_decay_type == "exponential":
        return prev_epsilon * epsilon_decay_factor
    elif epsilon_decay_type == "no":
        return prev_epsilon
    else:
        print(f"Unknown epsilon decay type given: {epsilon_decay_type}")


def plot_epsilon_decay(epsilon_start=0.3, epsilon_decay_type="exponential", epsilon_low_at_epoch=10**4, epsilon_low=10**-3,
                       show_until_epoch=10**3, title="Epsilon decay", save_path="figures/epsilon_decay.png"):
    decay_factor = compute_decay_factor(epsilon_start, epsilon_decay_type, epsilon_low_at_epoch, epsilon_low)
    epsilons = [epsilon_start]
    prev_epsilon = epsilon_start
    for epoch in range(1, int(show_until_epoch)+1):
        new_epsilon = compute_next_epsilon(prev_epsilon, epsilon_decay_type, decay_factor)
        epsilons.append(new_epsilon)
        prev_epsilon = new_epsilon

    fig, ax = plt.subplots(1, figsize=(14, 6))
    ax.plot(epsilons)
    ax.set_xlabel("epoch")
    ax.set_xlim([0,show_until_epoch])
    ax.set_ylim([0,epsilon_start*1.05])
    ax.set_title(title)
    fig.subplots_adjust(bottom = 0.15)
    fig.text(0.60, 0.70, f"Decay factor: {round(decay_factor,5)}", fontsize=14)
    fig.savefig(save_path)
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plot_epsilon_decay(epsilon_start=0.3,
                       epsilon_decay_type="exponential",
                       epsilon_low_at_epoch=10**4,
                       epsilon_low=10**-3,
                       show_until_epoch=1.2*10**4)
