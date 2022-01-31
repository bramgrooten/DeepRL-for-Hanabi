"""Functions that provide different options of state/observation representation.
For the centralized version of the agents."""


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


def obs_to_vec7_playability(obs, hand_size=5, max_life=3, max_hint=8):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector (shorter than the standard one).
        This function only works in cheat mode

        obs_vec info:
        [
        life tokens left, value in [0..1], for example: 0.66.. means two tokens left when max=3 (length 1)
        hint tokens left, value in [0..1], for example: 0.25   means two tokens left when max=8 (length 1)

        current player card index 0: value 1 if directly playable, 0 if not
        current player card index 1: same, playable?
        current player card index 2: playable?
        current player card index 3: playable?
        current player card index 4: playable?  (length 5)
        ]
        total vector length = 1 + 1 + 5 = 7
    """
    current_player = obs['current_player']
    obs_vec = []

    # life tokens
    obs_vec.append( obs['player_observations'][current_player]['life_tokens'] / max_life )
    # hint tokens
    obs_vec.append( obs['player_observations'][current_player]['information_tokens'] / max_hint )

    # own cards
    fireworks = obs['player_observations'][current_player]['fireworks']
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for card in own_cards:
        if playable_card(card, fireworks):
            obs_vec.append(1)
        else:
            obs_vec.append(0)
    if len(own_cards) < hand_size:  # when deck is emptied, add a zero to keep obs_vec same size
        obs_vec.append(0)

    return obs_vec


def obs_to_vec37_owncards(obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector (shorter than the standard one).
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs)

        obs_vec info:
        [
        life tokens left, value in [0..1], for example: 0.66.. means two tokens left (length 1)
        hint tokens left, value in [0..1], (length 1)

        firework rank for red, value in [0..1], for example: 0.4 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 1)

        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, value in [0..1], for example: 0.2 means rank 2 (indexed: 0,1,2,3,4)
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 6)
        ]

        example:
        [0.667,0.5,  0.4,0.2,0,0,1,  0,1,0,0,0,0.2, 0,0,1,0,0,0.6, 1,0,0,0,0,0.8, 0,0,0,0,1,0, 0,1,0,0,0,0.6]

        total vector length = 1 + 1 + 5 + 5 * 6 = 37
    """
    current_player = obs['current_player']
    obs_vec = []

    # life tokens
    obs_vec.append( obs['player_observations'][current_player]['life_tokens'] / max_life )

    # hint tokens
    obs_vec.append( obs['player_observations'][current_player]['information_tokens'] / max_hint )

    # fireworks
    for color in ['R', 'Y', 'G', 'W', 'B']:
        obs_vec.append( obs['player_observations'][current_player]['fireworks'][color] / num_ranks )

    # own cards
    color_dict = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for card in own_cards:
        color_vec = [0] * num_colors
        color_vec[color_dict[card['color']]] = 1
        obs_vec += color_vec

        obs_vec.append( card['rank'] / num_ranks )

    if len(own_cards) < hand_size:
        # when hand is not full, add 1 missing card without color to keep obs_vec same size
        obs_vec += [0] * num_colors
        obs_vec.append(0)

    return obs_vec


def obs_to_vec62_discards(obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector (shorter than the standard one).
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs)

        obs_vec info:
        [
        life tokens left, value in [0..1], for example: 0.66.. means two tokens left (length 1)
        hint tokens left, value in [0..1], (length 1)

        firework rank for red, value in [0..1], for example: 0.4 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 1)

        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, value in [0..1], for example: 0.2 means rank 1
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 6)

        discarded cards for red, values in [0..1], for example: 0.66 0 0.5 0 0
                                                   means two 1's and one 3 are discarded
        discarded cards for yellow
        discarded cards for green
        discarded cards for white
        discarded cards for blue     (5 times length 5)
        ]

        total vector length = 1 + 1 + 5 + 30 + 25 = 62
    """
    current_player = obs['current_player']
    obs_vec = []

    # life tokens
    obs_vec.append( obs['player_observations'][current_player]['life_tokens'] / max_life )

    # hint tokens
    obs_vec.append( obs['player_observations'][current_player]['information_tokens'] / max_hint )

    # fireworks
    for color in ['R', 'Y', 'G', 'W', 'B']:
        obs_vec.append( obs['player_observations'][current_player]['fireworks'][color] / num_ranks )

    # own cards
    color_dict = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for card in own_cards:
        color_vec = [0] * num_colors
        color_vec[color_dict[card['color']]] = 1
        obs_vec += color_vec

        obs_vec.append( card['rank'] / num_ranks )

    if len(own_cards) < hand_size:
        # when hand is not full, add 1 missing card without color to keep obs_vec same size
        obs_vec += [0] * num_colors
        obs_vec.append(0)

    # discards
    discard_vec = [0] * num_colors * num_ranks  # for each color and rank
    discards = obs['player_observations'][current_player]['discard_pile']
    for card in discards:
        idx = num_colors * color_dict[card['color']]
        rank = card['rank']  # 0, 1, 2, 3, or 4
        if rank == 4:
            discard_vec[idx + 4] = 1
        elif rank == 0:
            discard_vec[idx] += 1/3
        else:  # other ranks than first and last
            discard_vec[idx + rank] += 1/2

    obs_vec += discard_vec
    return obs_vec


def obs_to_vec62_discards_shuffle(self, obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector (shorter than the standard one).
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs)

        obs_vec info:
        [
        life tokens left, value in [0..1], for example: 0.66.. means two tokens left (length 1)
        hint tokens left, value in [0..1], (length 1)

        firework rank for red, value in [0..1], for example: 0.4 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 1)

        Card indices are shuffled! Order of shuffle is saved, to be able to convert to correct action later.
        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, value in [0..1], for example: 0.2 means rank 1
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 6)

        discarded cards for red, values in [0..1], for example: 0.66 0 0.5 0 0
                                                   means two 1's and one 3 are discarded
        discarded cards for yellow
        discarded cards for green
        discarded cards for white
        discarded cards for blue     (5 times length 5)
        ]

        total vector length = 1 + 1 + 5 + 30 + 25 = 62
    """
    current_player = obs['current_player']
    obs_vec = []

    # life tokens
    obs_vec.append( obs['player_observations'][current_player]['life_tokens'] / max_life )

    # hint tokens
    obs_vec.append( obs['player_observations'][current_player]['information_tokens'] / max_hint )

    # fireworks
    for color in ['R', 'Y', 'G', 'W', 'B']:
        obs_vec.append( obs['player_observations'][current_player]['fireworks'][color] / num_ranks )

    # own cards
    color_dict = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]

    for idx in self.shuffled_indices:
        if idx == hand_size - 1 and len(own_cards) < hand_size:
            # when hand is not full, add 1 missing card without color to keep obs_vec same size
            obs_vec += [0] * num_colors
            obs_vec.append(0)
        else:
            color_vec = [0] * num_colors
            color_vec[color_dict[own_cards[idx]['color']]] = 1
            obs_vec += color_vec
            obs_vec.append( own_cards[idx]['rank'] / num_ranks )

    # discards
    discard_vec = [0] * num_colors * num_ranks  # for each color and rank
    discards = obs['player_observations'][current_player]['discard_pile']
    for card in discards:
        idx = num_colors * color_dict[card['color']]
        rank = card['rank']  # 0, 1, 2, 3, or 4
        if rank == 4:
            discard_vec[idx + 4] = 1
        elif rank == 0:
            discard_vec[idx] += 1/3
        else:  # other ranks than first and last
            discard_vec[idx + rank] += 1/2

    obs_vec += discard_vec
    return obs_vec


def obs_to_vec82_one_hot_ranks(obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector (shorter than the standard one).
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs)

        obs_vec info:
        [
        life tokens left, value in [0..1], for example: 0.66.. means two tokens left (length 1)
        hint tokens left, value in [0..1], (length 1)

        firework rank for red, value in [0..1], for example: 0.4 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 1)

        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, one-hot, for example: 10000 means rank 1
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 10)

        discarded cards for red, values in [0..1], for example: 0.66 0 0.5 0 0
                                                   means two 1's and one 3 are discarded
        discarded cards for yellow
        discarded cards for green
        discarded cards for white
        discarded cards for blue     (5 times length 5)
        ]

        total vector length = 1 + 1 + 5 + 50 + 25 = 82
    """
    current_player = obs['current_player']
    obs_vec = []
    # life tokens
    obs_vec.append( obs['player_observations'][current_player]['life_tokens'] / max_life )
    # hint tokens
    obs_vec.append( obs['player_observations'][current_player]['information_tokens'] / max_hint )

    # fireworks
    for color in ['R', 'Y', 'G', 'W', 'B']:
        obs_vec.append( obs['player_observations'][current_player]['fireworks'][color] / num_ranks )

    # own cards
    color_dict = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for card in own_cards:
        color_vec = [0] * num_colors
        color_vec[color_dict[card['color']]] = 1
        obs_vec += color_vec

        rank_vec = [0] * num_ranks
        rank_vec[card['rank']] = 1
        obs_vec += rank_vec

    if len(own_cards) < hand_size:
        # when hand is not full, add 1 missing card without color&rank to keep obs_vec same size
        obs_vec += [0] * num_colors
        obs_vec += [0] * num_ranks

    # discards
    discard_vec = [0] * num_colors * num_ranks  # for each color and rank
    discards = obs['player_observations'][current_player]['discard_pile']
    for card in discards:
        idx = num_colors * color_dict[card['color']]
        rank = card['rank']  # 0, 1, 2, 3, or 4
        if rank == num_ranks - 1:
            discard_vec[idx + 4] = 1
        elif rank == 0:
            discard_vec[idx] += 1/3
        else:  # other ranks than first and last
            discard_vec[idx + rank] += 1/2
    obs_vec += discard_vec
    return obs_vec


def obs_to_vec136_binary(obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector (shorter than the standard one).
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs)

        obs_vec info:
        [
        life tokens left, thermometer, for example: 110 means two tokens left (length 3)
        hint tokens left, for example: 11111100 means six left (length 8)

        firework rank for red, for example: 11000 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 5)

        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, one-hot, for example: 10000 means rank 1
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 10)

        discarded cards for red, for example 110 00 10 00 0 means two red 1's and one red 3 are discarded
        discarded cards for yellow
        discarded cards for green
        discarded cards for white
        discarded cards for blue     (5 times length 10)
        ]

        total vector length = 3 + 8 + 25 + 50 + 50 = 136
    """
    current_player = obs['current_player']
    obs_vec = []

    # life tokens
    obs_vec += [1] * obs['player_observations'][current_player]['life_tokens']
    obs_vec += [0] * (max_life - obs['player_observations'][current_player]['life_tokens'])

    # hint tokens
    obs_vec += [1] * obs['player_observations'][current_player]['information_tokens']
    obs_vec += [0] * (max_hint - obs['player_observations'][current_player]['information_tokens'])

    # fireworks
    for color in ['R', 'Y', 'G', 'W', 'B']:
        obs_vec += [1] * obs['player_observations'][current_player]['fireworks'][color]
        obs_vec += [0] * (num_ranks - obs['player_observations'][current_player]['fireworks'][color])

    # own cards
    color_dict = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for card in own_cards:
        color_vec = [0] * num_colors
        color_vec[color_dict[card['color']]] = 1
        obs_vec += color_vec

        rank_vec = [0] * num_ranks
        rank_vec[card['rank']] = 1
        obs_vec += rank_vec

    if len(own_cards) < hand_size:
        # when hand is not full, add 1 missing card without color&rank to keep obs_vec same size
        obs_vec += [0] * num_colors
        obs_vec += [0] * num_ranks

    # discards
    discard_vec = [0] * 2 * num_colors * num_ranks
    # 2 duplicate cards per color&rank, except first(3) and last(1)
    # 3 + 2 * (num_colors - 2) + 1 = 2 * num_colors
    discards = obs['player_observations'][current_player]['discard_pile']
    for card in discards:
        idx = 2 * num_colors * color_dict[card['color']]
        rank = card['rank']  # 0, 1, 2, 3, or 4  (standard)
        if rank == num_ranks - 1:
            discard_vec[idx + 2 * num_colors - 1] = 1
        elif rank == 0:
            if discard_vec[idx] == 1 and discard_vec[idx+1] == 1:
                discard_vec[idx+2] = 1
            elif discard_vec[idx] == 1:
                discard_vec[idx+1] = 1
            else:
                discard_vec[idx] = 1
        else:  # other ranks than first and last
            idx += 3
            idx += 2 * (rank - 1)
            if discard_vec[idx] == 1:
                discard_vec[idx+1] = 1
            else:
                discard_vec[idx] = 1
    obs_vec += discard_vec

    return obs_vec


def obs_to_vec186_others_cards(obs, hand_size=5, max_life=3, max_hint=8, num_ranks=5, num_colors=5):
    """
        by drl_for_hanabi
        Turns a HLE observation into a vector.
        This function only works in cheat mode (normally not all agents are allowed to view player 0's obs).
        Assumes: there are 2 players.

        obs_vec info:
        [
        life tokens left, thermometer, for example: 110 means two tokens left (length 3)
        hint tokens left, for example: 11111100 means six left (length 8)

        firework rank for red, for example: 11000 means red firework is at 2
        firework rank for yellow
        firework rank for green
        firework rank for white
        firework rank for blue       (5 times length 5)

        current player color card index 0, one-hot, for example: 01000 is yellow
        current player rank  card index 0, one-hot, for example: 10000 means rank 1
        current player card index 1 (color & rank)
        current player card index 2 (color & rank)
        current player card index 3 (color & rank)
        current player card index 4 (color & rank)   (5 times length 10)

        other player color card index 0, one-hot, for example: 01000 is yellow
        other player rank  card index 0, one-hot, for example: 10000 means rank 1
        other player card index 1 (color & rank)
        other player card index 2 (color & rank)
        other player card index 3 (color & rank)
        other player card index 4 (color & rank)   (5 times length 10)

        discarded cards for red, for example 110 00 10 00 0 means two red 1's and one red 3 are discarded
        discarded cards for yellow
        discarded cards for green
        discarded cards for white
        discarded cards for blue     (5 times length 10)
        ]

        total vector length = 3 + 8 + 25 + 50 + 50 + 50 = 186
    """
    current_player = obs['current_player']
    obs_vec = []

    # life tokens
    obs_vec += [1] * obs['player_observations'][current_player]['life_tokens']
    obs_vec += [0] * (max_life - obs['player_observations'][current_player]['life_tokens'])

    # hint tokens
    obs_vec += [1] * obs['player_observations'][current_player]['information_tokens']
    obs_vec += [0] * (max_hint - obs['player_observations'][current_player]['information_tokens'])

    # fireworks
    for color in ['R', 'Y', 'G', 'W', 'B']:
        obs_vec += [1] * obs['player_observations'][current_player]['fireworks'][color]
        obs_vec += [0] * (num_ranks - obs['player_observations'][current_player]['fireworks'][color])

    # own cards
    color_dict = {'R': 0, 'Y': 1, 'G': 2, 'W': 3, 'B': 4}
    own_cards = obs['player_observations'][current_player]['card_knowledge'][0]
    for card in own_cards:
        color_vec = [0] * num_colors
        color_vec[color_dict[card['color']]] = 1
        obs_vec += color_vec

        rank_vec = [0] * num_ranks
        rank_vec[card['rank']] = 1
        obs_vec += rank_vec

    if len(own_cards) < hand_size:
        # when hand is not full, add 1 missing card without color&rank to keep obs_vec same size
        obs_vec += [0] * num_colors
        obs_vec += [0] * num_ranks

    # other player's cards
    other_cards = obs['player_observations'][current_player]['card_knowledge'][1]
    for card in other_cards:
        color_vec = [0] * num_colors
        color_vec[color_dict[card['color']]] = 1
        obs_vec += color_vec

        rank_vec = [0] * num_ranks
        rank_vec[card['rank']] = 1
        obs_vec += rank_vec

    if len(other_cards) < hand_size:
        # when hand is not full, add 1 missing card without color&rank to keep obs_vec same size
        obs_vec += [0] * num_colors
        obs_vec += [0] * num_ranks

    # discards
    discard_vec = [0] * 2 * num_colors * num_ranks
    # 2 duplicate cards per color&rank, except first(3) and last(1)
    # 3 + 2 * (num_colors - 2) + 1 = 2 * num_colors
    discards = obs['player_observations'][current_player]['discard_pile']
    for card in discards:
        idx = 2 * num_colors * color_dict[card['color']]
        rank = card['rank']  # 0, 1, 2, 3, or 4  (standard)
        if rank == num_ranks - 1:
            discard_vec[idx + 2 * num_colors - 1] = 1
        elif rank == 0:
            if discard_vec[idx] == 1 and discard_vec[idx+1] == 1:
                discard_vec[idx+2] = 1
            elif discard_vec[idx] == 1:
                discard_vec[idx+1] = 1
            else:
                discard_vec[idx] = 1
        else:  # other ranks than first and last
            idx += 3
            idx += 2 * (rank - 1)
            if discard_vec[idx] == 1:
                discard_vec[idx+1] = 1
            else:
                discard_vec[idx] = 1
    obs_vec += discard_vec

    return obs_vec


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


def unique_card(card, discard_pile):
    """
    Says whether this card is now unique (meaning there is
    only one left in the game of its color and rank).

    Args:
        card: dict,
        discard_pile: list, of cards
    Returns:
        boolean: True if card is unique, False otherwise
    """
    if card['rank'] == 4:
        return True  # highest rank is always unique
    elif card['rank'] == 0:
        discarded_duplicates = 0
        for discarded in discard_pile:
            if discarded['color'] == card['color'] and discarded['rank'] == card['rank']:
                discarded_duplicates += 1
        if discarded_duplicates == 2:
            return True  # you have the last (3rd) rank 1 card
    else:
        for discarded in discard_pile:
            if discarded['color'] == card['color'] and discarded['rank'] == card['rank']:
                return True  # the duplicate of your card has already been discarded
    return False
