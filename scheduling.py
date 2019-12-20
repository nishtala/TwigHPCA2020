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

from common import *
import os
import numpy as np
from itertools import cycle, islice, zip_longest
import psutil
import struct
import perfmon
from collections import deque


initial_number_cores_per_workload = total_number_of_cores // number_of_workloads
action_space_initial_allocation   = np.array_split(action_space_core, number_of_workloads)
global idx
idx = 0

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

class Scheduler:
    def __init__(self, lc_workload_name=None, lc_target=None, lc_PPID=None, current_DVFS=None):
        self.lc_workload_name = lc_workload_name
        self.lc_target = lc_target
        self.lc_ppid = PPID
        self.LC_WORKLOAD_CHILD_PIDS = list()
        self.LC_WORKLOAD_CHILD_MAP = list()
        self.current_mapping = list()
        self.number_of_cores = None
        self.current_DVFS = current_DVFS

        self.window_size = normalisation_window_size
        self.dataset = {
                'UNHALTED_CORE_CYCLES' : deque([], maxlen=window_size),
                'INSTRUCTION_RETIRED' : deque([], maxlen=window_size),
                'PERF_COUNT_HW_CPU_CYCLES': deque([], maxlen=window_size),
                'UNHALTED_REFERENCE_CYCLES': deque([], maxlen=window_size),
                'UOPS_RETIRED': deque([], maxlen=window_size),
                'BRANCH_INSTRUCTIONS_RETIRED': deque([], maxlen=window_size),
                'MISPREDICTED_BRANCH_RETIRED': deque([], maxlen=window_size),
                'PERF_COUNT_HW_BRANCH_MISSES': deque([], maxlen=window_size),
                'LLC_MISSES': deque([], maxlen=window_size),
                'PERF_COUNT_HW_CACHE_L1D': deque([], maxlen=window_size),
                'PERF_COUNT_HW_CACHE_L1I': deque([], maxlen=window_size),
                }

    def retrieve_mapping(self):
        self.number_of_cores = len(self.current_mapping)
        return self.current_mapping, self.number_of_cores, self.current_DVFS

    def initial_allocation(self):
        action_space_initial_allocation[idx] = list(islice(cycle(action_space_initial_allocation[idx]), \
                len(self.LC_WORKLOAD_CHILD_PIDS)))
        assert len(action_space_initial_allocation[idx]) == len(self.LC_WORKLOAD_CHILD_PIDS), 'have some dangling threads.'
        for child, core in zip(self.LC_WORKLOAD_CHILD_PIDS, action_space_initial_allocation[idx]):
            self.LC_WORKLOAD_CHILD_MAP[child] = psutil.Process(pid=child)
            self.LC_WORKLOAD_CHILD_MAP[child].cpu_affinity([core])
            self.current_mapping.append(core)
        idx += 1

    def subsequent_core_allocation(self, core_list=None):
        self.current_mapping = []
        core_list = list(islice(cycle(core_list), len(self.LC_WORKLOAD_CHILD_PIDS)))
        assert len(core_list) == len(self.LC_WORKLOAD_CHILD_PIDS), 'have some dangling threads.'
        for child, core in zip(self.LC_WORKLOAD_CHILD_PIDS, core_list):
            self.LC_WORKLOAD_CHILD_MAP[child] = psutil.Process(pid=child)
            self.LC_WORKLOAD_CHILD_MAP[child].cpu_affinity([core])
            self.current_mapping.append(core)

    def subsequent_DVFS_allocation(self, core_list=None, DVFS= None):
        mapping = ",".join(str(_) for _ in core_list)
        os.system('sudo cpupower -c ' + mapping + ' frequency-set -f ' + DVFS + ' > /dev/null')
        self.current_DVFS = DVFS

    def find_child_pids():
        assert type(lc_ppid) == int, "lc_PPID for {}:{} has to be int".format(self.lc_workload_name, self.lc_ppid)
        assert psutil.pid_exists(self.lc_ppid) == True, "PPID of {}:{} not alive".format(self.lc_workload_name, self.lc_ppid)
        threads =  os.popen('ps -o spid -T --no-headers ' + str(self.lc_ppid)).read().split('\n')[:-1]
        for _ in threads:
            self.LC_WORKLOAD_CHILD_PIDS.append(int(_.replace(" ","")))

    def start_performance_monitoring(self):
        """
        starts perfmon
        """
        self.events_dict = dict()
        self.prev_count_events_dict = dict()
        self.sessions = dict()

        for child in self.LC_WORKLOAD_CHILD_PIDS:
            self.events_dict[child] = dict(zip_longest(*[iter(EVENTS)] * 2, fillvalue="0"))
            self.prev_count_events_dict[child] = dict(zip_longest(*[iter(EVENTS)] * 2, fillvalue="0"))
            self.sessions[child] = perfmon.PerThreadSession(int(child), EVENTS)
            self.sessions[child].start()
            for i in range(0, len(EVENTS)):
                count = struct.unpack("L", self.sessions[child].read(i))[0]
                self.prev_count_events_dict[child][EVENTS[i]] = count

    def collect_perfmon(self):
        """
        Refreshes events_dict at each sampling interval
        """
        for child in self.LC_WORKLOAD_CHILD_PIDS:
            for i in range(0, len(EVENTS)):
                count = struct.unpack("L", self.sessions[child].read(i))[0]
                vals = count - self.prev_count_events_dict[child][EVENTS[i]]
                self.prev_count_events_dict[child][EVENTS[i]] = count
                self.events_dict[child][EVENTS[i]] = float(vals)

    def summarize_counters_childs(self):
        """
        Current_cores has to be a list, and it spits out the counters only for those cores
        """
        counters = dict((x,[]) for x in EVENTS)
        self.collect_perfmon()
        for e in EVENTS:
            for c in self.LC_WORKLOAD_CHILD_PIDS:
                counters[e].append(self.events_dict[c][e])
            counters[e] = sum(counters[e])
        return self.norm_data(counters)

    def norm_data(self, cur_counter=None):
        state_space = []
        run_mean = []
        for key, val in max_counter.items():
            out = cur_counter[key]/(max_counter[key])
            state_space.append(out)
            self.dataset[key].append(out)
        if len(self.dataset['UNHALTED_CORE_CYCLES']) < self.window_size:
            return np.array(state_space)
        else:
            for key, val in self.dataset.items():
                run_mean.append(running_mean(val, self.window_size)[0])
            return np.array(run_mean)
