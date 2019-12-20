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


import subprocess
import perfmon
import time
import os
import numpy as np
from re import findall
import sys
from itertools import cycle

CURRENT_PATH             = os.getcwd() + "/"
num_cpu_sockets          = int(subprocess.check_output('cat /proc/cpuinfo | grep "physical id" | sort -u | wc -l', shell=True))
total_number_of_cores    = int(subprocess.check_output('lscpu | grep -m1 "CPU(s):" | cut -d \: -f 2', shell=True))
cores_per_socket         = int(subprocess.check_output('lscpu | grep "Core(s) per socket:" | cut -d \: -f 2', shell=True))
DVFS_states              = [str(float(_.decode())/1e6) + "GHz" for _ in subprocess.check_output('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies', shell=True).split()]
num_DVFS_states          = len(DVFS_states) - 1

# Twig dir to the system path.
current_path = os.path.dirname(os.path.abspath(__file__)) + "/Twig"
sys.path.insert(0, current_path)

EVENTS = ['UNHALTED_CORE_CYCLES', 'INSTRUCTION_RETIRED', 'UNHALTED_REFERENCE_CYCLES', \
        'LLC_MISSES', 'BRANCH_INSTRUCTIONS_RETIRED', 'MISPREDICTED_BRANCH_RETIRED', \
        'PERF_COUNT_HW_CPU_CYCLES', 'PERF_COUNT_HW_BRANCH_MISSES', 'PERF_COUNT_HW_CACHE_L1D', \
        'PERF_COUNT_HW_CACHE_L1I', 'UOPS_RETIRED']

max_power              = None #TODO
assert max_power is not None, 'cant be None'

LC_WORKLOAD_NAME       = [None, None] #TODO
LC_TARGET              = [None, None] #TODO
MAX_LOAD               = [None, None] #TODO
LC_WORKLOAD_PPID       = [None, None] #TODO
number_of_workloads    = len(LC_WORKLOAD_NAME)

sampling_frequency     = None #TODO
assert sampling_frequency is not None, 'Cant be None'

max_counter = {
    #TODO: Run microbenchmark stress_cpu.c
    'UNHALTED_CORE_CYCLES' :     None,
    'INSTRUCTION_RETIRED' :      None,
    'PERF_COUNT_HW_CPU_CYCLES':  None,
    'UNHALTED_REFERENCE_CYCLES': None,
    'UOPS_RETIRED':              None,

    #TODO: Run microbenchmark branch_misses.cpp
    'BRANCH_INSTRUCTIONS_RETIRED': None,
    'MISPREDICTED_BRANCH_RETIRED': None,
    'PERF_COUNT_HW_BRANCH_MISSES': None,

    #TODO: Run microbenchmark stream
    'LLC_MISSES':              None,
    'PERF_COUNT_HW_CACHE_L1D': None,
    'PERF_COUNT_HW_CACHE_L1I': None,
    }


state_space_length = len(events)
num_branches       = 2 # cores, and DVFS
action_space_core  = np.arange(0, total_number_of_cores, 1)
action_space_freq  = DVFS_states

normalisation_window_size = 5

# Coefficients for reward
power_reward_coef = None #TODO
# If QoS is not met, this coef is going to determine the impact of negative reward
negative_qos_reward_coef = None #TODO
worst_case_reward = None #TODO

def initial_setup():
    DVFS_states              = [str(float(_.decode())/1e6) + "GHz" for _ in subprocess.check_output('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_frequencies', shell=True).split()]
    initial_DVFS_state       = str(int(subprocess.check_output('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq', shell=True))/1e6) + "GHz"
    current_scaling_governor = subprocess.check_output('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor', shell=True).split()[0].decode()
    if current_scaling_governor is not 'userspace':
        os.system('sudo cpupower frequency-set --governor userspace >/dev/null')
    if DVFS_states[-1] is not initial_DVFS_state:
       os.system('sudo cpupower frequency-set -f ' + DVFS_states[-1] + ' > /dev/null')
    OUTPUT_PATH            = [CURRENT_PATH + "Output/" + _ + "/" for _ in LC_WORKLOAD_NAME]
    for _ in OUTPUT_PATH: os.mkdir(_)
    OUTPUT_FILE            = [open(CURRENT_PATH + "Output/" + _ + "/output.csv", 'w') for _ in LC_WORKLOAD_NAME]

def initial_check():
    hyperthreading_enabled   = int(subprocess.check_output('lscpu | grep "Thread(s) per core:" | cut -d \: -f 2', shell=True))
    assert hyperthreading_enabled == 1, "Disable hyperthreading"
    scaling_driver           =  subprocess.check_output('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver', shell=True).split()[0].decode()
    assert scaling_driver == 'acpi-cpufreq', "Enable acpi-cpufreq"
    userspace_exists         = int(subprocess.check_output('cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors | grep -c "userspace"', shell=True))
    assert userspace_exists == 1, "Userspace does not exist"
    for _ in range(0, total_number_of_cores):
        related_cpus = 'cat /sys/devices/system/cpu/cpu' + str(_) + '/cpufreq/related_cpus'
        assert _ == int(subprocess.check_output(related_cpus, shell=True)), 'per core DVFS not available'

def resource_arbitration(action_core_count = None):
    common_cores = None
    colocation = sum(action_core_count) <= total_number_of_cores -1
    it = cycle(action_space_core)
    cores_per_app = [[next(it) for _ in range(size)] for size in action_core_count]
    if colocation: common_cores = list(set(cores_per_app[0]).intersection(*cores_per_app))
    return colocation, cores_per_app, action_frequency

def resource_arbitration_DVFS(common_cores = None, cores_per_app = None, action_frequency = None):
    strip_str_DVFS = [float(findall("\d+\.\d+", f)[0]) for f in action_frequency]
    max_common_frequency = str(max(strip_str_DVFS)) + 'GHz'
    DVFS_core_allocation_app = []
    DVFS_allocation_app = []

    for idx, _ in enumerate(LC_WORKLOAD_NAME):
        DVFS_core_allocation_app.append(list(set(cores_per_app[idx])^set(common_cores)))
        DVFS_allocation_app.append(action_frequency[idx])
    return DVFS_core_allocation_app, DVFS_allocation_app, max_common_frequency

def set_common_cores(cores=None, DVFS=None):
    mapping = ",".join(str(_) for _ in core)
    os.system('sudo cpupower -c ' + mapping + ' frequency-set -f ' + DVFS + ' > /dev/null')

def power_model(load=None, num_cores=None, DVFS=None):
    assert load > 0, 'Load has to be greater than zero'
    assert load <= 100, 'Load has to be normalised'
    assert num_cores <= len(action_space_core), 'Number of cores should be less than the available number of cores'
    assert num_cores > 0, 'Num cores has to be greater than zero'
    load_coef = None #TODO
    assert load_coef is not None, 'cant be None'
    num_cores_coef = None #TODO
    assert num_cores_coef is not None, 'cant be None'
    DVFS_coef = None #TODO
    assert DVFS_coef is not None, 'cant be None'
    DVFS = float(findall("\d+\.\d+", DVFS)[0]) # in GHz
    return (load_coef * load) + (num_cores_coef * num_cores) + (DVFS_coef * DVFS_coef * DVFS)
