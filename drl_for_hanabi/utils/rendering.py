import numpy as np
import torch


def print_the_state(obs):
    han_list = str(obs['player_observations'][0]['pyhanabi']).splitlines()
    lives = han_list[0]
    hints = han_list[1]
    fireworks = han_list[2]
    deck_size = han_list[-2]
    discards = han_list[-1]
    output = fireworks + "\t" + lives + "\t" + hints + "\n"
    output += "Player 0:  "
    for card in obs['player_observations'][0]['card_knowledge'][0]:
        output += card['color']
        output += str(card['rank'] + 1) + " "
    output += "\t" + deck_size + "\t" + discards + "\n"
    output += "Player 1:  "
    for card in obs['player_observations'][1]['card_knowledge'][0]:
        output += card['color']
        output += str(card['rank'] + 1) + " "
    output += "\tCurrent player: " + str(obs['current_player'])
    return output


def print_start_of_episode(self, training=True):
    if training:
        self.render_file.write("\n\nEPOCH NUMBER: " + str(self.current_epoch))
    else:
        self.render_file.write("\n\nTEST EPISODE: " + str(self.test_episode))


def print_current_state(self, turn, obs):
    self.render_file.write("\nSTATE IN TURN: " + str(turn) + "\n")
    self.render_file.write(print_the_state(obs))


def print_action_HLE(self, action, legal=None):
    """Prints the action, and whether it was legal.
    Args:
        action: dict, in HLE form
        legal: bool
    """
    if legal is None:
        act_str = f"\nACTION TAKEN:\n{str(action)}\n"
    else:
        if legal:
            act_str = f"\nACTION TAKEN:\nlegal: {str(action)}\n"
        else:
            act_str = f"\nACTION TAKEN:\nillegal: {str(action)}\n"
    self.render_file.write(act_str)


def print_action(self, act_int, legal):
    """Prints the action, and whether it was legal.
    This one only works for the 11 action representation.
    Args:
        act_int: int, in [0,10]
        legal: bool
    """
    print_str = "\nACTION TAKEN:\n"
    print_str += ("legal: " if legal else "illegal: ")
    if act_int <= 4:
        print_str += f"Discard position {act_int+1}"
    elif act_int <= 9:
        print_str += f"Play position {act_int-4}"
    elif act_int == 10:
        print_str += "Give random hint"
    else:
        print_str += f"UNKNOWN ACTION INTEGER: {act_int}"
    print_str += "\n"
    self.render_file.write(print_str)


def print_end_of_episode(self, obs, ep_score):
    self.render_file.write("\nFINAL STATE OF THE GAME:\n")
    self.render_file.write(print_the_state(obs))
    self.render_file.write("\nGAME DONE. SCORE: " + str(ep_score) + "\n")


def print_test_results_summary(episodes, batch_rets, batch_scores, batch_lens,
                               batch_fireworks, batch_lives, batch_illegals):
    print('\nAfter', episodes, 'test episodes the results are:')
    print(f'Averages:\t return: {np.mean(batch_rets):10.2f}\t score: {np.mean(batch_scores):10.2f}\t '
          f'fireworks: {np.mean(batch_fireworks):4.2f}\t lives left: {np.mean(batch_lives):4.2f}\t '
          f'ep_len: {np.mean(batch_lens):4.2f}\t ill moves: {np.mean(batch_illegals):6.2f}')
    print(f'Std.error:\t return: {np.std(batch_rets, ddof=1) / np.sqrt(np.size(batch_rets)):10.2f}\t '
          f'score: {np.std(batch_scores, ddof=1) / np.sqrt(np.size(batch_rets)):10.2f}\t '
          f'fireworks: {np.std(batch_fireworks, ddof=1) / np.sqrt(np.size(batch_rets)):4.2f}\t '
          f'lives left: {np.std(batch_lives, ddof=1) / np.sqrt(np.size(batch_rets)):4.2f}\t '
          f'ep_len: {np.std(batch_lens, ddof=1) / np.sqrt(np.size(batch_rets)):4.2f}\t '
          f'ill moves: {np.std(batch_illegals, ddof=1) / np.sqrt(np.size(batch_rets)):6.2f}')

    print(f'\nMinima:\t\t return: {np.min(batch_rets):10.2f}\t score: {np.min(batch_scores):10.2f}\t '
          f'fireworks: {np.min(batch_fireworks):4.2f}\t lives left: {np.min(batch_lives):4.2f}\t '
          f'ep_len: {np.min(batch_lens):4.2f}\t ill moves: {np.min(batch_illegals):6.2f}')
    print(f'Maxima:\t\t return: {np.max(batch_rets):10.2f}\t score: {np.max(batch_scores):10.2f}\t '
          f'fireworks: {np.max(batch_fireworks):4.2f}\t lives left: {np.max(batch_lives):4.2f}\t '
          f'ep_len: {np.max(batch_lens):4.2f}\t ill moves: {np.max(batch_illegals):6.2f}')


def print_training_epoch_results(current_epoch, avg_loss, avg_rets, avg_fire, avg_score, avg_lives,
                                 avg_lens, avg_eps, avg_batch_size, avg_illegal, avg_entropy):
    print(f'epoch: {current_epoch:10d}'
          f' \t loss: {np.mean(avg_loss):10.2f}'
          f' \t return: {np.mean(avg_rets):10.2f}'
          f' \t fireworks: {np.mean(avg_fire):6.2f}'
          f' \t score: {np.mean(avg_score):6.2f}'
          f' \t lives left: {np.mean(avg_lives):6.2f}'
          f' \t ep_len: {np.mean(avg_lens):6.2f}'
          f' \t num_eps: {np.mean(avg_eps):6.2f}'
          f' \t batch size: {np.mean(avg_batch_size):6.2f}'
          f' \t illegals: {np.mean(avg_illegal):6.2f}'
          f' \t entropy: {np.mean(avg_entropy):5.3f}')


def print_training_epoch_results_VPG(current_epoch, avg_loss_pi, avg_loss_vf, avg_rets, avg_fire, avg_score, avg_lives,
                                     avg_lens, avg_eps, avg_batch_size, avg_illegal, avg_entropy):
    print(f'epoch: {current_epoch:10d}'
          f' \t loss_pi: {np.mean(avg_loss_pi):10.2f}'
          f' \t loss_vf: {np.mean(avg_loss_vf):10.2f}'
          f' \t return: {np.mean(avg_rets):10.2f}'
          f' \t fireworks: {np.mean(avg_fire):6.2f}'
          f' \t score: {np.mean(avg_score):6.2f}'
          f' \t lives left: {np.mean(avg_lives):6.2f}'
          f' \t ep_len: {np.mean(avg_lens):6.2f}'
          f' \t num_eps: {np.mean(avg_eps):6.2f}'
          f' \t batch size: {np.mean(avg_batch_size):6.2f}'
          f' \t illegals: {np.mean(avg_illegal):6.2f}'
          f' \t entropy: {np.mean(avg_entropy):5.3f}')
