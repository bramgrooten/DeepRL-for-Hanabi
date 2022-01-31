from itertools import combinations


def compute_biases_without_scaling(avg_probs):
    """Computes play and discard bias, based on a probability distribution
    Bias play    is defined as: max(abs(p_i - p_j)) where p_i,j come from the set of play actions
    Bias discard is defined as: max(abs(p_i - p_j)) where p_i,j come from the set of discard actions

    Args:
        avg_probs: list, of probabilities, length 11
    Returns:
        bias_play: float
        bias_discard: float
    """
    bias_play = 0
    bias_discard = 0
    discards = avg_probs[0:5]
    plays = avg_probs[5:10]
    for prob1, prob2 in combinations(plays, 2):
        new_bias = abs(prob1 - prob2)
        if new_bias > bias_play:
            bias_play = new_bias
    for prob1, prob2 in combinations(discards, 2):
        new_bias = abs(prob1 - prob2)
        if new_bias > bias_discard:
            bias_discard = new_bias
    return bias_play, bias_discard


def compute_biases(avg_probs):
    """Computes play and discard bias, based on a probability distribution.
    First scales all probs such that sum_{play moves i}(p_i) = 1 and sum_{discard moves i}(p_i) = 1
    Bias play    is defined as: max(abs(p_i - p_j)) where p_i,j come from the set of play actions
    Bias discard is defined as: max(abs(p_i - p_j)) where p_i,j come from the set of discard actions

    Args:
        avg_probs: list, of probabilities, length 11
    Returns:
        bias_play: float
        bias_discard: float
    """
    plays = avg_probs[5:10]
    discards = avg_probs[0:5]
    total_play_prob = sum(plays)
    total_discard_prob = sum(discards)
    if total_play_prob == 0:
        bias_play = 0
    else:
        plays = [p/total_play_prob for p in plays]
        bias_play = compute_single_bias(plays)
    if total_discard_prob == 0:
        bias_discard = 0
    else:
        discards = [d/total_discard_prob for d in discards]
        bias_discard = compute_single_bias(discards)
    return bias_play, bias_discard


def compute_single_bias(probs):
    """Computes bias.
    Args: probs, list of probabilities
    Returns: bias, float, defined as: max(abs(p_i - p_j))
    """
    bias = 0
    for prob1, prob2 in combinations(probs, 2):
        new_bias = abs(prob1 - prob2)
        if new_bias > bias:
            bias = new_bias
    return bias


