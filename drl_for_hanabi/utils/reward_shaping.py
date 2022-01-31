from drl_for_hanabi.utils.state_representation import playable_card, useless_card, unique_card


def adjust_rewards(self, ep_rews, ep_obs_dicts, ep_acts, moves_legal):
    """
    Adjusts the rewards according to the settings
    Only the illegal move rewards have already been changed in run_one_episode

    Assumptions:
    1. the lengths of ep_rews, ep_acts, and moves_legal are all equal
        (The length is equal to the number of steps in this episode (n).)
    2. ep_obs_dicts has length of n+1, as it also includes the last obs
        (The state after the game has finished.)
    3. The hand size is 5. (So only for 2 or 3 players now)
    4. The obs dict is centralized. (Having all player obs)

    Args:
        self: class instance, for example a SPG/VPG agent
        ep_rews: list, of rewards gathered this episode
        ep_obs_dicts: list, of observation dicts of the episode
        ep_acts: list, of action integers
        moves_legal: list, of boolean values saying whether each action was legal or not
    Returns:
        ep_rews: list, of adjusted rewards for this episode
    """
    if "success play" in self.reward_settings:
        ep_rews = rew_play_successfully(ep_rews, self.reward_settings["success play"],
                                        ep_acts, ep_obs_dicts)
    if "out of lives" in self.reward_settings:
        ep_rews = rew_out_of_lives(ep_rews, self.reward_settings["out of lives"],
                                   ep_obs_dicts[-1]['player_observations'][0]['life_tokens'])
    if "lost one life" in self.reward_settings:
        ep_rews = rew_lost_one_life(ep_rews, self.reward_settings["lost one life"], ep_acts, ep_obs_dicts)
    if "discard" in self.reward_settings:
        ep_rews = rew_discard(ep_rews, self.reward_settings["discard"], ep_acts, moves_legal)
    if "play" in self.reward_settings:
        ep_rews = rew_play(ep_rews, self.reward_settings["play"], ep_acts)
    if "hint" in self.reward_settings:
        ep_rews = rew_hint(ep_rews, self.reward_settings["hint"], ep_acts)
    if "discard playable card" in self.reward_settings:
        ep_rews = rew_discard_playable_card(ep_rews, self.reward_settings["discard playable card"],
                                            ep_acts, ep_obs_dicts, moves_legal)
    if "discard useless card" in self.reward_settings:
        ep_rews = rew_discard_useless_card(ep_rews, self.reward_settings["discard useless card"],
                                           ep_acts, ep_obs_dicts, moves_legal)
    if "discard unique card" in self.reward_settings:
        ep_rews = rew_discard_unique_card(ep_rews, self.reward_settings["discard unique card"],
                                          ep_acts, ep_obs_dicts, moves_legal)

    return ep_rews


def rew_play_successfully(ep_rews, new_val, ep_acts, ep_obs_dicts):
    """
    Adjusts the rewards corresponding to a successfully played card.
    """
    for idx, action in enumerate(ep_acts):
        if 5 <= action <= 9:  # means it was a play action
            if ep_obs_dicts[idx+1]['player_observations'][0]['life_tokens'] == ep_obs_dicts[idx]['player_observations'][0]['life_tokens']:
                # means no life token was lost
                ep_rews[idx] = new_val
    return ep_rews


def rew_out_of_lives(ep_rews, new_val, life_tokens_left):
    """
    Adjusts the final rewards of an episode, if the episode
    was finished because all life tokens were lost.
    """
    if life_tokens_left <= 0:  # we lost all life tokens
        ep_rews[-1] = new_val
    return ep_rews


def rew_lost_one_life(ep_rews, new_val, ep_acts, ep_obs_dicts):
    """
    Adjusts rewards of actions that led to the loss of one life token.
    """
    for idx, action in enumerate(ep_acts):
        if 5 <= action <= 9:  # means it was a play action
            if ep_obs_dicts[idx+1]['player_observations'][0]['life_tokens'] < \
                    ep_obs_dicts[idx]['player_observations'][0]['life_tokens']:
                ep_rews[idx] += new_val
    return ep_rews


def rew_discard(ep_rews, new_val, ep_acts, moves_legal):
    """
    Adjusts rewards corresponding to timesteps with a "discard" action.
    """
    for idx, action in enumerate(ep_acts):
        if moves_legal[idx]:
            if action < 5:  # means it was a discard action
                ep_rews[idx] += new_val
    return ep_rews


def rew_play(ep_rews, new_val, ep_acts):
    """
    Adjusts rewards corresponding to timesteps with a "play" action.
    """
    for idx, action in enumerate(ep_acts):
        if 5 <= action <= 9:  # means it was a play action
            ep_rews[idx] += new_val
    return ep_rews


def rew_hint(ep_rews, new_val, ep_acts):
    """
    Adjusts rewards corresponding to timesteps with a "hint" action.
    """
    for idx, action in enumerate(ep_acts):
        if action >= 10:  # means it was a hint action
            ep_rews[idx] += new_val
    return ep_rews


def rew_discard_playable_card(ep_rews, new_val, ep_acts, ep_obs_dicts, moves_legal):
    """
    Adjusts rewards of actions where a directly playable card was discarded,
    which is (usually) bad.

    Only works in cheat mode now.
    """
    for idx, action in enumerate(ep_acts):
        if moves_legal[idx] and action < 5:  # means it was a discard action
            current_player = ep_obs_dicts[idx]['current_player']
            card = ep_obs_dicts[idx]['player_observations'][current_player]['card_knowledge'][0][action]
            fireworks = ep_obs_dicts[idx]['player_observations'][current_player]['fireworks']
            if playable_card(card, fireworks):
                ep_rews[idx] += new_val
    return ep_rews


def rew_discard_useless_card(ep_rews, new_val, ep_acts, ep_obs_dicts, moves_legal):
    """
    Adjusts rewards of actions where a proven useless card was discarded,
    which is (usually) good.

    Only works in cheat mode now.
    """
    for idx, action in enumerate(ep_acts):
        if moves_legal[idx] and action < 5:  # means it was a discard action
            current_player = ep_obs_dicts[idx]['current_player']
            card = ep_obs_dicts[idx]['player_observations'][current_player]['card_knowledge'][0][action]
            fireworks = ep_obs_dicts[idx]['player_observations'][current_player]['fireworks']
            discard_pile = ep_obs_dicts[idx]['player_observations'][current_player]['discard_pile']
            if useless_card(card, fireworks, discard_pile):
                ep_rews[idx] += new_val
    return ep_rews


def rew_discard_unique_card(ep_rews, new_val, ep_acts, ep_obs_dicts, moves_legal):
    """
    Adjusts rewards of actions where a unique card was discarded,
    which is bad. Unique meaning: only one card of that color and rank left.

    Only works in cheat mode now.
    """
    for idx, action in enumerate(ep_acts):
        if moves_legal[idx] and action < 5:  # means it was a discard action
            current_player = ep_obs_dicts[idx]['current_player']
            card = ep_obs_dicts[idx]['player_observations'][current_player]['card_knowledge'][0][action]
            discard_pile = ep_obs_dicts[idx]['player_observations'][current_player]['discard_pile']
            if unique_card(card, discard_pile):
                ep_rews[idx] += new_val
    return ep_rews
