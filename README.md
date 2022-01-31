# Deep RL for Hanabi

This is the repository for the project: Deep Reinforcement Learning for the cooperative card game Hanabi, 
by Bram Grooten and his master thesis supervisors: Jelle Wemmenhove, Maurice Poot, and Jim Portegies. 
The code builds upon the [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment) provided by DeepMind. 

## Folder structure of this repository
```
drl_for_hanabi/                     # directory that we have added
    algorithms/                     # the main algorithms: SPG, VPG, PPO
    experiments/                    # files to train the agents with different settings
    extra/                          # some extra algorithms, not used in the paper
        decentralized/              # files that make decentralized test runs possible
                                    # (all training is done in a centralized manner)
        rule_based/                 # some rule-based agents (no learning involved)
        supervised/                 # imitation learning approach
        tabular/                    # a Q-learning agent
    utils/                          # often used utilities, like action & state representations

hanabi_learning_environment/        # the actual HLE
    rl_env.py                       # the main reinforcement learning environment
    rl_env_adjusted.py              # some adjustments added by us

.gitignore                          # for git
README.md                           # this file
requirements.txt                    # packages that need to be installed

clean_all.sh                        # files from DeepMind's HLE
CMakeLists.txt
LICENSE
pyproject.toml
setup.py
```

## Install

You need some version of Linux to be able to run this. A Windows Subsystem for Linux (WSL) works as well. 
See an installation guide for WSL [here](https://docs.microsoft.com/en-us/windows/wsl/install-win10). 
To be able to use WSL with PyCharm on Windows, see 
[this useful guide](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html).

1. Open your Linux or WSL terminal.
2. `git clone` this repository to your local computer.
3. Create a new [venv](https://hpcwiki.tue.nl/wiki/Specific_tools#Virtual_environments) to accompany this repo, 
and make sure it's activated. 
We use Python 3.8.10, so make sure you have that in your venv.
(See instructions specific to WSL and PyCharm [here](https://www.dropbox.com/s/qk9hd1m0e51a9wl/Using-the-WSL.pdf?dl=0).)
4. This repo is a copy of the [Hanabi Learning Environment](https://github.com/deepmind/hanabi-learning-environment) (HLE) 
with added files and folders. In your Linux or WSL terminal, run `pip install -e .` from within the top directory 
(with the `setup.py` file in it).
The `-e` makes sure that you can adjust the HLE package a bit if you wish, like we did as well. As said on the HLE page, you might need to run these first:
```
sudo apt-get install g++            # if you don't already have a CXX compiler
sudo apt-get install python-pip     # if you don't already have pip
```
5. Now run `pip install -r requirements.txt` from within the top directory (with the `requirements.txt` file in it), 
to install the packages that we use on top of the HLE.


### Play with one of our pre-trained agents

1. Go to the `drl_for_hanabi/extra/decentralized/` directory.
2. Run the file `hanabi_runner.py`. 

At the bottom of that file you can choose which agents you want to play with. 
Select the `HumanPlayer()` as one of the agents if you want to interact with them.

### Train one of our Deep RL agents

1. Go to the `drl_for_hanabi/algorithms/` directory.
2. Run one of the files `ppo.py`, `vpg.py` or `spg.py`.

At the top of each file you can change how long the algorithm trains (`RUNTIME`, in hours) 
or for how many epochs (`EPOCHS`, set `RUNTIME` &le; 0). You can change the hyperparameters as well, 
such as the learning rate `LR` or the discount factor `GAMMA`. 
The network size can easily be adjusted: if you want 2 hidden layers with 96, 64 nodes respectively, 
then set `HIDDEN_LAYERS = [96, 64]`.
