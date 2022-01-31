#!/usr/bin/bash

# This bash script calls all the experiments you want to run
# You can comment out an experiment to exclude it
# Use shift+alt+arrows (up/down) to change the call order

# structure:
# bash <run_file> <running_time_hours> <new_session_num>


run_time=60    # hours

bash trainAllExps_repeat_ppo.run $run_time 14
bash trainAllExps_repeat_vpg.run $run_time 14
bash trainAllExps_repeat_spg.run $run_time 14

#bash trainAllExps_combi_vpg.run $run_time 9
#bash trainAllExps_combi_ppo.run $run_time 8
#bash trainAllExps_combi_spg.run $run_time 8



