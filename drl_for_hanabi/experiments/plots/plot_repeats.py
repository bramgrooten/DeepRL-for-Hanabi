import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import Union
from plot_training import put_ratios_and_biases_into_df


def find_max_session_nr(experiment_date, experiment_name):
    max_ses = 0
    for folder in glob.glob(f'../data/{experiment_date}_{experiment_name}/s*'):
        session = int(folder.split('/')[-1][1:])
        if session > max_ses:
            max_ses = session
    return max_ses


def get_data_per_session(exp_date, exp_name, sub_exp, max_sessions):
    data_frames = []
    for session in range(max_sessions):
        data_file = f'../data/{exp_date}_{exp_name}/s{session + 1}/results_experiment_{sub_exp}.csv'
        data_session = pd.read_csv(data_file)
        data_frames.append(data_session)
    return pd.concat(data_frames)


def set_algorithm_label(experiment_name):
    if 'repeat' in experiment_name:
        return experiment_name[-3:].upper()
    else:
        return experiment_name


def get_datasets(experiments, max_epoch: Union[str, float] = 'max'):
    min_last_epoch = float('inf')
    datasets = []

    for exp in experiments:
        date = exp['date_exp']
        name = exp['type_exp']
        sub_exps = exp['sub_exps']
        max_ses = find_max_session_nr(date, name)

        for sub_exp in sub_exps:
            train_data = get_data_per_session(date, name, sub_exp, max_ses)
            last_epoch = train_data['epoch'].iloc[-1]
            print(f"Last epoch of sub experiment {sub_exp} of {name} was: {last_epoch}")
            if last_epoch < min_last_epoch:
                min_last_epoch = last_epoch

            train_data.insert(0, 'sub_exp', sub_exp)
            train_data.insert(0, 'algorithm', set_algorithm_label(name))

            train_data = put_ratios_and_biases_into_df(train_data)

            datasets.append(train_data)

    if max_epoch == 'max':
        datasets = remove_unnecessary_epochs(datasets, min_last_epoch)
    else:
        datasets = remove_unnecessary_epochs(datasets, max_epoch)
    return datasets


def remove_unnecessary_epochs(datasets, until_epoch):
    """Removes the epochs which are too large,
    This makes sure all experiments end at the same epoch in the plot."""
    new_datasets = []
    for data in datasets:
        new_data = data[data['epoch'] <= until_epoch]
        new_datasets.append(new_data)
    return new_datasets


def compute_avg_bias_last_epoch(datasets):
    bp_spg, bp_vpg, bp_ppo, bd_spg, bd_vpg, bd_ppo = [], [], [], [], [], []
    for sub_data in datasets:
        # epo = sub_data['epoch'].iloc[-1]
        bias_play = sub_data['bias_play'].iloc[-1]
        bias_discard = sub_data['bias_discard'].iloc[-1]
        if sub_data['algorithm'].iloc[-1] == 'SPG':
            bp_spg.append(bias_play)
            bd_spg.append(bias_discard)
        elif sub_data['algorithm'].iloc[-1] == 'VPG':
            bp_vpg.append(bias_play)
            bd_vpg.append(bias_discard)
        elif sub_data['algorithm'].iloc[-1] == 'PPO':
            bp_ppo.append(bias_play)
            bd_ppo.append(bias_discard)
    print(f"SPG bias play: {np.mean(bp_spg)}")
    print(f"SPG bias discard: {np.mean(bd_spg)}")
    print(f"VPG bias play: {np.mean(bp_vpg)}")
    print(f"VPG bias discard: {np.mean(bd_vpg)}")
    print(f"PPO bias play: {np.mean(bp_ppo)}")
    print(f"PPO bias discard: {np.mean(bd_ppo)}")


def stats_on_episode_length(datasets):
    lengths = []
    for sub_data in datasets:
        epo = sub_data['epoch'].iloc[-1]
        sub_exp = sub_data['sub_exp'].iloc[-1]
        algo = sub_data['algorithm'].iloc[-1]
        print(epo, sub_exp, algo)
        avg_length = sub_data['epi_length'].iloc[-1]
        print("avg episode length: ", avg_length)
        lengths.append(avg_length)
    print("avg episode length over all algorithms: ", np.mean(lengths))


def make_bias_plots(data, **kwargs):
    sns.tsplot(data=data, time='epoch', value='bias_play', condition='algorithm', unit='sub_exp', ci='sd', **kwargs)
    sns.tsplot(data=data, time='epoch', value='bias_discard', condition='algorithm', unit='sub_exp', ci='sd', linestyle="dotted", **kwargs)
    plt.ylabel("")
    plt.ylim(bottom=0)
    legend1 = plt.legend(['SPG', 'VPG', 'PPO'], loc='upper center', ncol=6, handlelength=1, borderaxespad=-2, prop={'size': 13})
    lines = plt.gca().get_lines()
    legend2 = plt.legend([lines[i] for i in [0, 3]], ["play bias", "discard bias"])  # , loc='center left', bbox_to_anchor=(0, 0.85))
    legend2.legendHandles[0].set_color('black')
    legend2.legendHandles[1].set_color('black')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.tight_layout(pad=1.5)


def make_fireworks_and_scores_plot(data, **kwargs):
    sns.tsplot(data=data, time='epoch', value='fireworks', condition='algorithm', unit='sub_exp', ci='sd', **kwargs)
    sns.tsplot(data=data, time='epoch', value='scores', condition='algorithm', unit='sub_exp', ci='sd', linestyle="dotted", **kwargs)
    plt.ylabel("")
    legend1 = plt.legend(['SPG', 'VPG', 'PPO'], loc='upper center', ncol=6, handlelength=1, borderaxespad=-2,
                         prop={'size': 13})
    lines = plt.gca().get_lines()
    if max_epoch == 'max' or max_epoch == 2.5e6:
        plt.ylim(0, 25)
        legend2 = plt.legend([lines[i] for i in [0, 3]], ["firework", "score"], loc='lower right')  # , bbox_to_anchor=(0, 0.85))
    else:
        plt.ylim(bottom=0)
        legend2 = plt.legend([lines[i] for i in [0, 3]], ["firework", "score"], loc='center left', bbox_to_anchor=(0, 0.85))
    legend2.legendHandles[0].set_color('black')
    legend2.legendHandles[1].set_color('black')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    plt.tight_layout(pad=1.5)


def make_separate_scores_plot(experiments, fig_path='figure.png', smooth=80, max_epoch: Union[str, float] = 'max'):
    data = get_datasets(experiments, max_epoch=max_epoch)
    data = smooth_the_data(data, value='scores', smooth=smooth)
    fig, ax = plt.subplots()
    sns.set(style="darkgrid", font_scale=1.5)
    colors = sns.color_palette()
    xscale = np.max(np.asarray(data[0]['epoch'])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    for idx, df in enumerate(data):
        # sns.tsplot(data=df, time='epoch', value='scores')
        df.plot(kind='line', x='epoch', y='scores', legend=None, ax=ax) # , color=colors[idx])
        print(f"hi, color {idx}")
    plt.ylim(0, 25)
    plt.ylabel('score')
    save_figure(fig_path)


def make_standard_tsplot(data, xaxis, value, condition, unit, **kwargs):
    sns.tsplot(data=data, time=xaxis, value=value, condition=condition, unit=unit, ci='sd', **kwargs)
    plt.legend(loc='upper center', ncol=6, handlelength=1, borderaxespad=-2, prop={'size': 13})
    plt.tight_layout(pad=0.5)


def smooth_the_data(data, value, smooth=1):
    """
    smooth data with moving window average. that is,
        smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
    where the "smooth" param is width of that window (2k+1)
    """
    y = np.ones(smooth)
    for datum in data:
        x = np.asarray(datum[value])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
        datum[value] = smoothed_x
    if value == 'fireworks':
        # then also smooth the scores please
        for datum in data:
            x = np.asarray(datum['scores'])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum['scores'] = smoothed_x
    return data


def plot_data(data, xaxis='epoch', value='returns', condition="algorithm", unit="sub_exp", smooth=1, max_epoch='max', **kwargs):
    if smooth > 1:
        if value in ['3probs', 'bias', 'bias_all']:
            for v in ['play_prob', 'discard_prob', 'hint_prob', 'bias_play', 'bias_discard']:
                data = smooth_the_data(data, v, smooth)
        else:
            data = smooth_the_data(data, value, smooth)
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    plt.figure()
    sns.set(style="darkgrid", font_scale=1.5)

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    if value == 'scores':
        make_standard_tsplot(data, xaxis, value, condition, unit)
        plt.ylim(0, 25)
        plt.ylabel('score')
    elif value == 'scores_separate':
        make_separate_scores_plot(data)
    elif value == 'lives':
        sns.tsplot(data=data, time=xaxis, value=value, condition=condition, unit=unit, ci='sd', **kwargs)
        plt.ylabel("life tokens")
        plt.legend(loc='upper center', ncol=6, handlelength=1, borderaxespad=-2, prop={'size': 13})
        plt.tight_layout(pad=0.5)
    elif value == 'fireworks':
        make_fireworks_and_scores_plot(data)
    elif value == '3probs':
        for v_idx, value in enumerate(['play_prob', 'discard_prob', 'hint_prob']):
            color = sns.color_palette()[3 + v_idx]
            sns.tsplot(data=data, time=xaxis, value=value, condition=condition, unit=unit, ci='sd', color=color, **kwargs)
        plt.legend(["play", "discard", "hint"])
        # plt.ylabel("average policy probabilities")
        plt.ylabel('')
        plt.title("Average policy probabilities")
        plt.ylim(bottom=0)
        plt.tight_layout(pad=0.5)
    elif value == 'bias':
        for v_idx, value in enumerate(['bias_play', 'bias_discard']):
            color = sns.color_palette()[3 + v_idx]
            sns.tsplot(data=data, time=xaxis, value=value, condition=condition, unit=unit, ci='sd', color=color, **kwargs)
        plt.legend(["play bias", "discard bias"])
        plt.ylabel('bias')
        # plt.title("Average policy probabilities")
        plt.ylim(bottom=0)
        plt.tight_layout(pad=0.5)
    elif value == 'bias_all':
        make_bias_plots(data)
    elif value in ['bias_play', 'bias_discard']:
        sns.tsplot(data=data, time=xaxis, value=value, condition=condition, unit=unit, ci='sd', **kwargs)
        plt.ylim(bottom=0)
        plt.ylabel('play bias') if value == 'bias_play' else plt.ylabel('discard bias')
        plt.legend(loc='upper center', ncol=6, handlelength=1, borderaxespad=-2, prop={'size': 13})
        plt.tight_layout(pad=0.5)
    elif value == 'entropy':
        make_standard_tsplot(data, xaxis, value, condition, unit)
        plt.ylim(bottom=0)
    else:
        make_standard_tsplot(data, xaxis, value, condition, unit)

    # plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #            borderaxespad=0., prop={'size': 13})
    # For the version of the legend used in the Spinning Up benchmarking page, swap with:
    # plt.legend(loc='upper center', ncol=6, handlelength=1, borderaxespad=-2, prop={'size': 13})
    # Maybe add also: mode = "expand"


def make_avg_probs_histogram(data):
    """
    Args:
        data: pandas dataframe of one sub-experiment
    Returns: histogram
    """
    sns.set(style="darkgrid", font_scale=1.5)
    fig, ax = plt.subplots()
    x = list(range(11))
    action_names = ['d0', 'd1', 'd2', 'd3', 'd4',
                    'p0', 'p1', 'p2', 'p3', 'p4', 'h']
    probs = []
    for action in action_names:
        p = data.iloc[-1][action]
        probs.append(p)
    ax.bar(x=x, height=probs)
    ax.set_xticks(x)
    ax.set_xticklabels(action_names)
    ax.set_ylim(0, 1)
    ax.set_title("Average policy probabilities last epoch")
    return fig


def make_one_plot(experiments, value='returns', fig_path='figure.png', smooth=1, max_epoch: Union[str, float] = 'max'):
    data = get_datasets(experiments, max_epoch=max_epoch)
    # compute_avg_bias_last_epoch(data)
    # stats_on_episode_length(data)

    plot_data(data, value=value, smooth=smooth, max_epoch=max_epoch)
    save_figure(fig_path)


def save_figure(fig_path):
    folders = os.path.dirname(fig_path)
    if not os.path.exists(folders):
        os.makedirs(folders)
    plt.savefig(fig_path, transparent=False)
    # possible extensions: png, pdf, eps, svg, pgf, ...
    # recommend: png for quick image, pdf for good quality (able to zoom in)
    plt.close()
    print(f"Figure saved in: {fig_path}")


if __name__ == '__main__':
    start_time = datetime.now()

    exps = [
        {"date_exp": "2021-11-30",
         "type_exp": "repeat_spg",
         "sub_exps": [0, 1, 2, 3, 4]},

        {"date_exp": "2021-11-30",
         "type_exp": "repeat_vpg",
         "sub_exps": [0, 1, 2, 3, 4]},

        {"date_exp": "2021-11-30",
         "type_exp": "repeat_ppo",
         "sub_exps": [0, 1, 2, 3, 4]},
    ]

    value = 'scores'  # 'epi_length'  # 'fireworks'  # 'lives'  # 'entropy'  # 'bias_play'  # '3probs'  # 'returns'
    max_epoch = 2.5e6  # 'max'  # 2.5e5  # default is 'max', but a number like 1e5 is also possible
    m = f"{max_epoch:.1e}" if type(max_epoch) == float else str(max_epoch)
    smooth = 80  # higher gives smoother graphs, but more error near the edges

    folder = ""  # leave empty to put the plots in the current folder (where plot_repeats.py is)
                 # we recommend to save your plots in another cloud directory, not github
    fig_name = f"{value}/{value}_smooth{smooth}_epoch{m}.png"
    make_one_plot(exps, value, folder + fig_name, smooth=smooth, max_epoch=max_epoch)



    ### to plot runs separately
    # for exp in exps:
    #     new_exps = [exp]
    #     fig_name = f"scores_separate/scores_separate_smooth{smooth}_epoch{m}_color_{exp['type_exp']}.png"
    #     make_separate_scores_plot(new_exps, folder+fig_name, smooth, max_epoch)

    ### to plot with different smoothing params
    # data = get_datasets(exps)
    # for s in [55,60,65,70,75]:
    #     fig_name = f"smoothing/comparing_{value}_y25_smooth{s}.png"
    #     plot_data(data, value=value, smooth=s)
    #     save_figure(folder+fig_name)

    ### to plot 3probs for each exp separately
    # for exp in exps:
    #     new_exps = [exp]
    #     value = '3probs'
    #     max_epoch = 2.5e6
    #     m = f"{max_epoch:.1e}" if type(max_epoch) == float else str(max_epoch)
    #     smooth = 80  # higher gives smoother graphs
    #
    #     folder = ""
    #     fig_name = f"{value}/{value}_smooth{smooth}_epoch{m}_exp-{exp['type_exp']}.pdf"
    #     make_one_plot(new_exps, value, folder + fig_name, smooth=smooth, max_epoch=max_epoch)

    print("\nRunning time to plot:", datetime.now() - start_time)
