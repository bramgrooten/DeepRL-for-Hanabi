import os, sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
drl_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if drl_folder not in sys.path:
    sys.path.insert(0, drl_folder)
from experiments.analyze.view_policy.analyze_bias import compute_biases


def get_all_exp_settings(type_exp, num_experiments):
    if type_exp in ["vec37", "vec62", "act51"]:
        all_settings_exp = pd.read_csv(f'../settings/exp_settings_{num_experiments}_{type_exp}.csv')
    elif type_exp in ["supRL"]:
        all_settings_exp = pd.read_csv(f'../settings/exp_settings_supervised.csv')
    else:
        all_settings_exp = pd.read_csv(f'../settings/exp_settings_{type_exp}.csv')
    if "num_exp" in list(all_settings_exp.columns):
        del all_settings_exp['num_exp']
    return all_settings_exp


def get_data_from_experiment(date_exp, type_exp, num_exp, num_continued_sessions):
    skip = False
    if type_exp == "vec37" and (num_exp % 3 != 2 or num_exp > 23):
        skip = True  # with the next iteration of the loop (skip the rest of this iteration)
    elif type_exp == "supRL" and num_exp == 0:
        skip = True  # this one exp didn't go through (json error)
    elif type_exp == "shuf" and num_exp not in [5]:
        skip = True
    elif type_exp == "rews" and num_exp not in [32, 33]:
        skip = True
    elif type_exp == "scale" and num_exp not in [1, 5]:
        skip = True
    elif type_exp == "ppo" and num_exp not in [0]:
        skip = True
    # elif type_exp == "vpg_gae" and num_exp not in [0, 1, 4]:
    #     skip = True
    elif type_exp == "vpg_iters" and num_exp not in [0,1,2,3,4, 5]:
        skip = True
    elif type_exp == "iters" and num_exp not in [0, 1, 2,  4, 5, 6]:
        skip = True
    elif type_exp == "vpg_gae" and num_exp not in [0, 1, 2]:
        skip = True
    elif type_exp == "ppo_gae" and num_exp not in [0, 1, 2]:
        skip = True
    elif type_exp == "rew_ppo" and num_exp not in [0, 1, 2, 3]:
        skip = True
    elif type_exp == "combi_vpg" and num_exp not in [2, 3, 4, 5]:
        skip = True

    if skip:
        return None, skip

    data_frames = []
    for session in range(num_continued_sessions):

        data_file = f'../data/{date_exp}_{type_exp}/s{session + 1}/results_experiment_{num_exp}.csv'

        if os.path.exists(data_file):
            data_exp = pd.read_csv(data_file)
            data_frames.append(data_exp)
        else:
            print("File path does not exist:", data_file)
            return
    return pd.concat(data_frames), skip


def text_reward_str(exp_setting, column):
    if exp_setting[column] == 'standard' and column != "out of lives":
        s = f"std (0)"
    elif exp_setting[column] == 'standard' and column == "out of lives":
        s = f"std (-total score)"
    else:
        s = exp_setting[column]
    return s


def add_settings_text_vpg(fig, all_settings_exp, num_exp, rows=1, type_exp=""):
    exp_setting = all_settings_exp.iloc[num_exp]
    reward_str = f"Settings of experiment {num_exp}:\n\n"
    rew2_str = ""
    pi_str = ""
    v_str = ""
    more_str = ""
    split_3 = 13 if "ppo" in type_exp else 14
    split_4 = 17 if "ppo" in type_exp else 18
    for column in all_settings_exp.columns[:4]:
        s = text_reward_str(exp_setting, column)
        reward_str += f"{column}: {s}\n"
    for column in all_settings_exp.columns[4:9]:
        s = text_reward_str(exp_setting, column)
        rew2_str += f"{column}: {s}\n"
    for column in all_settings_exp.columns[9:split_3]:
        pi_str += f"{column}: {exp_setting[column]}\n"
    for column in all_settings_exp.columns[split_3:split_4]:
        v_str += f"{column}: {exp_setting[column]}\n"
    for column in all_settings_exp.columns[split_4:-4]:
        more_str += f"{column}: {exp_setting[column]}\n"
    if rows == 1:
        fig.subplots_adjust(bottom=0.45)
    else:
        fig.subplots_adjust(bottom=0.25)
    start = 0.10
    offset = 0.18
    for idx, txt_str in enumerate([reward_str, rew2_str, pi_str, v_str, more_str]):
        fig.text(start + idx * offset, 0.0, txt_str, fontsize=14)


def add_settings_text_spg(fig, all_settings_exp, num_exp, rows=1, text_cols=4):
    exp_setting = all_settings_exp.iloc[num_exp]
    reward_str = f"Settings of experiment {num_exp}:\n\n"
    setting_str = ""
    more_str = ""
    epsilon_str = ""
    for column in all_settings_exp.columns[:4]:
        s = text_reward_str(exp_setting, column)
        reward_str += f"{column}: {s}\n"
    for column in all_settings_exp.columns[4:9]:
        if text_cols == 4:
            s = text_reward_str(exp_setting, column)
        else:
            s = exp_setting[column]
        setting_str += f"{column}: {s}\n"
    for column in all_settings_exp.columns[9:16]:
        more_str += f"{column}: {exp_setting[column]}\n"
    for column in all_settings_exp.columns[16:-4]:
        epsilon_str += f"{column}: {exp_setting[column]}\n"
    if rows == 1:
        fig.subplots_adjust(bottom=0.45)
    else:
        fig.subplots_adjust(bottom=0.20)
    if text_cols == 3:
        fig.text(0.25, 0.0, reward_str, fontsize=14)
        fig.text(0.45, 0.0, setting_str, fontsize=14)
        fig.text(0.65, 0.0, more_str, fontsize=14)
    elif text_cols == 4:
        offset = 0.10
        fig.text(0.25 - offset, 0.0, reward_str, fontsize=14)
        fig.text(0.45 - offset, 0.0, setting_str, fontsize=14)
        fig.text(0.65 - offset, 0.0, more_str, fontsize=14)
        fig.text(0.75, 0.0, epsilon_str, fontsize=14)


def map_supRL_exp_nums(num_exp):
    mapping = {0: 21, 1: 22, 2: 26, 3: 27, 4: 31, 5: 32, 6: 36, 7: 37,
               8: 61, 9: 62, 10: 66, 11: 67, 12: 71, 13: 72, 14: 76, 15: 77}
    return mapping[num_exp]


def save_figure(fig, fig_path):
    folders = os.path.dirname(fig_path)
    if not os.path.exists(folders):
        os.makedirs(folders)
    fig.savefig(fig_path, transparent=False)
    # possible extensions: png, eps, svg, pdf, pgf
    # use png for quick image, pdf for good quality (able to zoom in)
    plt.close(fig)


def set_epoch_axis_style(ax):
    ax.get_xaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: f"{x:1.1e}".replace("e+0", "e+")))


def make_all_action_probs_plot(ax, data_all_sessions):
    action_names = ['d0', 'd1', 'd2', 'd3', 'd4',
                    'p0', 'p1', 'p2', 'p3', 'p4', 'h']
    for action in action_names:
        data_all_sessions.plot(kind='line', x='epoch', y=action, ax=ax)
        set_epoch_axis_style(ax)
        ax.set_title("average action probabilities")
        ax.legend()  # bbox_to_anchor=(1.05, 1))


def make_avg_probs_histogram(ax, data_all_sessions, color=None):
    x = list(range(11))
    action_names = ['d0', 'd1', 'd2', 'd3', 'd4',
                    'p0', 'p1', 'p2', 'p3', 'p4', 'h']
    probs = []
    p25million = data_all_sessions.loc[data_all_sessions['epoch'] == 2_500_000]
    # print("epoch 2.5 million: ")
    # print(p25million)
    for action in action_names:
        # p = data_all_sessions.iloc[-1][action]
        p = p25million[action].values[0]
        # print("p: ", p)
        probs.append(p)
    if color is None:
        ax.bar(x=x, height=probs)
    else:
        ax.bar(x=x, height=probs, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    # ax.set_ylim(0, 1)
    ax.set_title("Average policy probabilities")  # " at last epoch"


def plot_one_row(the_axs, values_to_plot, fig_indices, data_all_sessions):
    for idx, value in zip(fig_indices, values_to_plot):
        # only when plotting just one picture:
        if max(fig_indices) == 0:
            the_axs = [the_axs]

        data_all_sessions.plot(kind='line', x='epoch', y=value, legend=None, ax=the_axs[idx])
        set_epoch_axis_style(the_axs[idx])

        if idx == 1:
            the_axs[idx].set_title("scores & fireworks")
            the_axs[idx].legend()
        else:
            the_axs[idx].set_title(value)

        if value == 'returns':
            # current_bottom, current_top = the_axs[idx].get_ylim()
            # the_axs[idx].set_ylim(bottom=max(-10, current_bottom - 1), top=min(25, current_top + 1))
            pass
        elif value == 'epsilon':
            the_axs[idx].set_ylim(bottom=0)
        elif value == 'loss':
            current_top = the_axs[idx].get_ylim()[1]
            the_axs[idx].set_ylim(bottom=0, top=max(0.1, current_top))


def plot_two_rows(the_axs, values_to_plot, fig_indices, data_all_sessions):
    plot_one_row(the_axs[0], values_to_plot, fig_indices, data_all_sessions)
    data_all_sessions = put_ratios_and_biases_into_df(data_all_sessions)
    make_entropy_plot(the_axs[1][0], data_all_sessions)
    make_action_ratio_plot(the_axs[1][1], data_all_sessions)
    make_biases_plot(the_axs[1][2], data_all_sessions)
    make_max_probability_plot(the_axs[1][3], data_all_sessions)


def plot_three_rows(the_axs, values_to_plot, fig_indices, data_all_sessions, type_exp):
    plot_two_rows(the_axs, values_to_plot, fig_indices, data_all_sessions)
    make_value_plot(the_axs[2][0], data_all_sessions)
    if 'ppo' in type_exp:
        make_clip_plot(the_axs[2][1], data_all_sessions)
    data_all_sessions = put_all_action_probs_in_df(data_all_sessions)
    make_all_action_probs_plot(the_axs[2][2], data_all_sessions)
    make_avg_probs_histogram(the_axs[2][3], data_all_sessions)


def make_value_plot(ax, data):
    for val in ['avg_value', 'avg_target']:
        data.plot(kind='line', x='epoch', y=val, ax=ax)
    ax.legend(["avg_value", "avg_target"])
    ax.set_title("value network")
    set_epoch_axis_style(ax)


def make_clip_plot(ax, data):
    data.plot(kind='line', x='epoch', y='clip_frac', legend=None, ax=ax)
    ax.set_title("fraction clipped")
    set_epoch_axis_style(ax)


def make_entropy_plot(ax, data_all_sessions):
    data_all_sessions.plot(kind='line', x='epoch', y='entropy', legend=None, ax=ax)
    ax.set_title("entropy")
    set_epoch_axis_style(ax)


def make_action_ratio_plot(ax, data_all_sessions):
    for value in ['play_prob', 'discard_prob', 'hint_prob']:
        data_all_sessions.plot(kind='line', x='epoch', y=value, ax=ax)
    ax.legend(["play", "discard", "hint"])
    ax.set_title("action probabilities")
    set_epoch_axis_style(ax)


def make_biases_plot(ax, data_all_sessions):
    for value in ['bias_play', 'bias_discard']:
        data_all_sessions.plot(kind='line', x='epoch', y=value, ax=ax)
    ax.legend(["play bias", "discard bias"])
    ax.set_title("positional biases")
    set_epoch_axis_style(ax)


def make_max_probability_plot(ax, data_all_sessions):
    data_all_sessions.plot(kind='line', x='epoch', y='max_prob', legend=None, ax=ax)
    ax.set_title("maximum action probability")
    set_epoch_axis_style(ax)


def put_ratios_and_biases_into_df(data_all_sessions):
    play_prob_lst = []
    discard_prob_lst = []
    hint_prob_lst = []
    bias_play_lst = []
    bias_discard_lst = []
    max_prob_lst = []
    for idx, data_row in data_all_sessions.iterrows():
        str_probs = data_row["avg_probs"][1:-1].split(" ")
        probs = get_avg_probs_from_str(str_probs)
        play, discard, hint = compute_ratio_play_discard_hint(probs)
        bias_play, bias_discard = compute_biases(probs)
        play_prob_lst.append(play)
        discard_prob_lst.append(discard)
        hint_prob_lst.append(hint)
        bias_play_lst.append(bias_play)
        bias_discard_lst.append(bias_discard)
        max_prob_lst.append(max(probs))
    data_all_sessions["play_prob"] = play_prob_lst
    data_all_sessions["discard_prob"] = discard_prob_lst
    data_all_sessions["hint_prob"] = hint_prob_lst
    data_all_sessions["bias_play"] = bias_play_lst
    data_all_sessions["bias_discard"] = bias_discard_lst
    data_all_sessions["max_prob"] = max_prob_lst
    return data_all_sessions


def put_all_action_probs_in_df(df):
    all_probs = [ [] for _ in range(11) ]
    for idx, data_row in df.iterrows():
        str_probs = data_row["avg_probs"][1:-1].split(" ")
        probs = get_avg_probs_from_str(str_probs)
        for i in range(11):
            all_probs[i].append(probs[i])
    action_names = ['d0', 'd1', 'd2', 'd3', 'd4',
                    'p0', 'p1', 'p2', 'p3', 'p4', 'h']
    for i in range(11):
        df[action_names[i]] = all_probs[i]
    return df


def get_avg_probs_from_str(str_probs):
    probs = []
    for ele in str_probs:
        if len(ele) == 0:
            continue
        else:
            probs.append(float(ele))
    return probs


def compute_ratio_play_discard_hint(probs):
    """Assumes 11 actions."""
    play = sum(probs[5:10])
    discard = sum(probs[0:5])
    hint = probs[10]
    return play, discard, hint


def make_plots(all_experiments, values_to_plot, fig_indices, plot_entropy_and_bias, folder=""):
    num_figs = max(fig_indices)+1
    for exp_idx, full_exp in enumerate(all_experiments):
        num_experiments = full_exp["num_experiments"]
        num_continued_sessions = full_exp["num_continued_sessions"]
        date_exp = full_exp["date_exp"]
        type_exp = full_exp["type_exp"]
        txt_cols = full_exp.get("txt_cols", 3)
        third_row = full_exp.get("third_row", False)

        if plot_entropy_and_bias and third_row:
            rows = 3
        elif plot_entropy_and_bias:
            rows = 2
        else:
            rows = 1

        all_settings_exp = get_all_exp_settings(type_exp, num_experiments)

        print(f"\nPlotting new set of experiments: {type_exp}")
        for num_exp in range(num_experiments):
            data_all_sessions, skip = get_data_from_experiment(date_exp, type_exp, num_exp, num_continued_sessions)
            if skip:
                continue

            # sns.set(style="darkgrid", font_scale=1.5)
            # color = sns.color_palette()[exp_idx]
            # fig, ax = plt.subplots()
            # df = put_all_action_probs_in_df(data_all_sessions)
            # # make_all_action_probs_plot(ax, data_all_sessions)
            # make_avg_probs_histogram(ax, df, color)

            fig, the_axs = plt.subplots(rows, num_figs, figsize=(6*num_figs, 6*rows))
            if rows == 1:
                plot_one_row(the_axs, values_to_plot, fig_indices, data_all_sessions)
            elif rows == 2:
                plot_two_rows(the_axs, values_to_plot, fig_indices, data_all_sessions)
            elif rows == 3:
                plot_three_rows(the_axs, values_to_plot, fig_indices, data_all_sessions, type_exp)
            fig.tight_layout(h_pad=5.0, w_pad=1.0)

            if type_exp == "supRL":
                num_exp = map_supRL_exp_nums(num_exp)
            if 'vpg' in type_exp or 'ppo' in type_exp:
                add_settings_text_vpg(fig, all_settings_exp, num_exp, rows, type_exp)
            else:
                add_settings_text_spg(fig, all_settings_exp, num_exp, rows, txt_cols)

            fig_path = f'{folder}{date_exp}_{type_exp}/s{num_continued_sessions}/exp{num_exp}_{type_exp}_plots.png'
            save_figure(fig, fig_path)
            print(f"Plotted experiment {num_exp} of {type_exp} session {num_continued_sessions}, saved in {fig_path}")

            # For tweaking plots:
        #     break
        # break


if __name__ == '__main__':
    start_time = datetime.now()

    # uncomment the experiment you want to plot now
    all_experiments = [
        # {"date_exp": "2021-04-13",
        #  "type_exp": "combi_spg",
        #  "num_experiments": 8,  # 10,
        #  "num_continued_sessions": 8,
        #  "txt_cols": 4},
        # {"date_exp": "2021-04-13",
        #  "type_exp": "combi_vpg",
        #  "num_experiments": 8,  # 10,
        #  "num_continued_sessions": 9,
        #  "third_row": True},
        # {"date_exp": "2021-04-13",
        #  "type_exp": "combi_ppo",
        #  "num_experiments": 8,  # 10,
        #  "num_continued_sessions": 8,
        #  "third_row": True},

        {"date_exp": "2021-11-30",
         "type_exp": "repeat_spg",
         "num_experiments": 5,
         "num_continued_sessions": 13,
         "txt_cols": 4},

        {"date_exp": "2021-11-30",
         "type_exp": "repeat_vpg",
         "num_experiments": 5,
         "num_continued_sessions": 13,
         "third_row": True},

        {"date_exp": "2021-11-30",
         "type_exp": "repeat_ppo",
         "num_experiments": 5,
         "num_continued_sessions": 13,
         "third_row": True},

    ]

    folder = ""  # leave empty to put the plots in the current folder (where plot_training.py is)
                 # we recommend to save your plots in another cloud directory, not github
    values_to_plot = ['returns', 'fireworks', 'scores', 'lives', 'illegal_moves']
    fig_indices = [0, 1, 1, 2, 3]   # determines which values go into the same figure
    plot_entropy_and_bias = True    # set to False if you only want to see values_to_plot

    make_plots(all_experiments, values_to_plot, fig_indices, plot_entropy_and_bias, folder)

    print("\nRunning time:", datetime.now() - start_time)
