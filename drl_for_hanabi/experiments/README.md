# Experiments

This is the folder where all experiments have been run from.
We will explain the function of each directory here.

`analyze/` contains a few Python files to analyze the entropy, bias, or actual parameters of 
the policy network that has been trained in our experiments.

`data/` is where all the data of the experiments has been collected. Each experiment has a name, often with spg, vpg, or ppo in it.
If there is no algorithm name in the name of the experiment, then it has only been run with SPG.

`hpc_scripts/` is where the bash scripts for the High Performance Cluster are. These can be used to run the experiments. 
The file `trainAll.sh` was the only script that we started on the HPC, which would then call other scripts.

`plots/` is where we have a few Python files to make nice plots of the experimental data.

`policy_params/` is where we save our policy (and value) network's parameters 
to be able to continue with experiments later on.

`run_exp/` contains the Python files that are called by the HPC scripts to run experiments.

`settings/` is where all the settings for the experiments are generated and saved.



