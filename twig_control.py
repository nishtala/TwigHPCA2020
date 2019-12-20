#!/usr/bin/env python3

"""
Twig: Multi-agent Task Management for Colocated Latency-critical Cloud Services
Copyright (C) <2019>  <Rajiv Nishtala>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Rajiv Nishtala"
__copyright__ = "NTNU 2019"
__credits__ = ["Rajiv Nishtala", "Vinicius Petrucci", "Paul Carpenter", "Magnus Sjalander"]
__license__ = "GPLv3"
__version__ = "1.0.0"
__maintainer= "Rajiv Nishtala"
__email__ ="rajiv.nishtala@ntnu.no"
__status__ = "development"

# If not py3 exit.
from sys import version_info, exit, argv
if version_info[0] < 3:
    raise Exception("Must be using Python 3")

import numpy as np
import time
import re
### my functions
from common import *
import scheduling as SG
import procedure_continuous_tasks as deepq
### my functions

current_state  = []
previous_state = []
current_reward = []
computed_power = []

current_step = -1

launch_time = time.strftime("%m-%d-%y-%H-%M", time.localtime())

# Check if hyperthreading is disabled
# Check if scaling_driver is acpi-cpufreq
# Check that userspace governor exists
# Check that per-core DVFS is possible
initial_check()

# setup userspace governor and lowest DVFS state.
# Setup output paths.
initial_setup()

workloads_dict = dict()

for idx, _ in enumerate(LC_WORKLOAD_NAME):
    workloads_dict[_] = SG.Scheduler(lc_workload_name = LC_WORKLOAD_NAME[idx],\
            lc_target = LC_TARGET[idx], lc_PPID = LC_WORKLOAD_PPID[idx], current_DVFS = DVFS_states[-1])

    # Find children
    workloads_dict[_].find_child_pids()
    # Do initial allocation
    workloads_dict[_].initial_allocation()
    # Start performance monitoring
    workloads_dict[_].start_performance_monitoring()
    OUTPUT_FILE[idx].write("{}\n".format("current_step, timestamp, power_recorded, power_computed, load, latency, cores, frequency, reward, epsilon"))

time.sleep(sampling_frequency)
for idx, _ in enumerate(LC_WORKLOAD_NAME):
    previous_state.append(workloads_dict[_].summarize_counters_childs())

# Initiate task manager
my_agent = deepq.learn_continuous_tasks(state_space_length=state_space_length,\
        freq_space=action_space_freq,\
        core_space=action_space_core,\
        NUM_APPS= number_of_workloads)


while True:
    time.sleep(sampling_frequency)

    current_load         = None #TODO
    assert type(current_load) is list, 'current load is not provided as list'
    current_tail_latency = None #TODO
    assert type(current_tail_latency) is list, 'current_tail_latency should be a list'
    recorded_power       = None #TODO

    load_in_percentage   = [cl/ml for cl, ml in zip(current_load, MAX_LOAD)]


    for idx, _ in enumerate(LC_WORKLOAD_NAME):
        current_state.append(workloads_dict[_].summarize_counters_childs())
        current_mapping, number_of_cores, current_DVFS = workloads_dict[_].retrieve_mapping()
        computed_power.append(power_model(load = load_in_percentage, \
                num_cores = number_of_cores, \
                DVFS = float(re.findall("\d+\.\d+", current_DVFS)[0])))
        QOS_TARDINESS = current_tail_latency[idx]/LC_TARGET[idx]
        if current_step >= 0:
            # Compute reward
            if QOS_TARDINESS <= 1:
                reward = QOS_TARDINESS + (power_reward_coef * computed_power[idx]/max_power)
            else:
                reward = - max((QOS_TARDINESS)**negative_qos_reward_coef, worst_case_reward)
            current_reward.append(reward)

            # Add to replay buffer
            my_agent.add_to_replay_buff(prev_state = previous_state, action_idxes = action_idxes[idx],\
                    reward = current_reward[idx], new_state = current_state[idx], workload_num=idx)


    if current_step >= 0:
        my_agent.compute_temporal_diff()
        my_agent.update_target_network_weights()

    action, action_idxes, epsilon, random, lr = my_agent.determine_action(current_state[0], current_state[1])

    action_core_count = [int(action[idx][0]) for idx,w in enumerate(LC_WORKLOAD_NAME)]
    action_frequency  = [str(f'{action[idx][1]:.2f}') + 'GHz' for idx,w in enumerate(LC_WORKLOAD_NAME)]

    # Resource arbitration
    colocation, cores_per_app, common_cores = resource_arbitration(action_core_count = action_core_count)

    for idx, _ in enumerate(LC_WORKLOAD_NAME):
        workloads_dict[_].subsequent_core_allocation(core_list=cores_per_app[idx])
        if not colocation:
            workloads_dict[_].subsequent_DVFS_allocation(core_list=cores_per_app[idx], DVFS=action_frequency[idx])

    if colocation:
        DVFS_core_allocation_app, DVFS_allocation_app, max_common_frequency = resource_arbitration_DVFS(common_cores = common_cores, \
                cores_per_app = cores_per_app, \
                action_frequency = action_frequency)
        set_common_cores(cores = common_cores, DVFS = max_common_frequency)
        for idx, _ in enumerate(LC_WORKLOAD_NAME):
            subsequent_DVFS_allocation(core_list = DVFS_core_allocation_app[idx], DVFS = DVFS_allocation_app[idx])

    for idx, _ in enumerate(LC_WORKLOAD_NAME):
        OUTPUT_FILE[idx].write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(current_step, time.time(), recorded_power, \
                computed_power[idx], current_load[idx], current_tail_latency[idx], action_core_count[idx], \
                action_frequency[idx], current_reward[idx], epsilon))
        OUTPUT_FILE[idx].flush()
    current_step += 1

for idx, _ in enumerate(LC_WORKLOAD_NAME):
    OUTPUT_FILE[idx].close()
