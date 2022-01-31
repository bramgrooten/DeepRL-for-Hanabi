import matplotlib.pyplot as plt
from datetime import datetime
import torch
from drl_for_hanabi.experiments.analyze.view_policy.view_policy import setup_algorithm


def make_histogram_of_parameters(algo, main_exp, exp_num_inside):
    all_params = torch.nn.utils.parameters_to_vector(algo.policy_network.parameters()).tolist()

    fig, ax = plt.subplots()
    ax.hist(all_params, bins=20)
    # ax.set_xlim([-1,1])
    # ax.set_ylim([0,500])
    ax.set_xlabel("Value of parameters")
    ax.set_ylabel("Number of parameters")
    ax.set_title("Weights and biases of the policy network")
    fig.tight_layout()
    fig_path = f"nn_params_exp{main_exp}_{exp_num_inside}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print("\nSaved figure in", fig_path)


if __name__ == '__main__':
    start_time = datetime.now()

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

    algo = setup_algorithm(exps, main_exp, exp_num_inside, session=session)
    make_histogram_of_parameters(algo, main_exp, exp_num_inside)

    print("\nRunning time:", datetime.now() - start_time)
