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

import numpy as np
import os
import sys
import tensorflow as tf
import zipfile
import datetime
from common import tf_util as U
import build_graph as bg
import logger
from common.schedules import ConstantSchedule, LinearSchedule, PiecewiseSchedule
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import time
import pickle
import csv
import models
import collections

class learn_continuous_tasks:
    def __init__(self,
          #lr=0.0025,
          lr=1e-4,
          gamma=0.99,
          batch_size=64,
          buffer_size=int(1e6),
          prioritized_replay=True,
          prioritized_replay_alpha=0.6,
          prioritized_replay_beta0=0.4,
          prioritized_replay_beta_iters=20000, # was int(2e6) 13thJune
          prioritized_replay_eps=int(1e8),
          grad_norm_clipping=10,
          learning_starts=1,
          target_network_update_freq=150, # was 500 13thJune
          train_freq=1,
          initial_std=0.2,
          final_std=0.2,
          timesteps_std=1e8,
          print_freq=10,
          epsilon_greedy=False,
          loss_type="L2",
          agg_method = 'reduceLocalMean',
          losses_version = 1,
          target_version='mean_across_branches_per_workload',

          state_space_length=None,
          freq_space=None,
          core_space=None,
          NUM_APPS=None,
          model_load=False,
          model_file = None,
          cyclic_learning_rate = False,
          drop_down = 10000,
          drop_out = True
          ):


        self.num_apps = NUM_APPS
        self.action_space_core = core_space
        self.action_space_freq = freq_space
        self.state_space_length = [state_space_length]*self.num_apps
        self.env_name = 'workloads'
        self.time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.lr=lr
        self.grad_norm_clipping=grad_norm_clipping
        self.max_timesteps=int(1e8)
        self.buffer_size=buffer_size
        self.train_freq=train_freq
        self.batch_size=batch_size
        self.print_freq=print_freq
        self.learning_starts=learning_starts
        self.gamma=gamma
        self.target_network_update_freq=target_network_update_freq
        self.prioritized_replay=prioritized_replay
        self.prioritized_replay_alpha=prioritized_replay_alpha
        self.prioritized_replay_beta0=prioritized_replay_beta0
        self.prioritized_replay_beta_iters=prioritized_replay_beta_iters
        self.prioritized_replay_eps=prioritized_replay_eps
        self.num_cpu=16
        self.losses_version= losses_version
        self.epsilon_greedy=False
        self.timesteps_std=timesteps_std
        self.initial_std=initial_std
        self.final_std=final_std
        self.agg_method = agg_method
        self.target_version = target_version
        self.loss_type=loss_type
        self.drop_down = drop_down
        self.drop_out = drop_out
        self.cyclic = cyclic_learning_rate
        self.save_model_frequency = 50000 # steps
        self.model_load = model_load
        self.model_file = model_file

        try:
           self.path_location = os.getcwd() + "/models/"
           os.mkdir(self.path_location)
        except:
            pass



        self.low = np.array( [ np.min(self.action_space_core), np.min(self.action_space_freq) ] )
        self.high = np.array( [ np.max(self.action_space_core), np.max(self.action_space_freq) ] )
        self.actions_range = np.subtract(self.high, self.low)

        self.num_actions_pad = [ \
                [len(self.action_space_core), len(self.action_space_freq)]\
                ] * self.num_apps

        self.num_action_grains = np.array(self.num_actions_pad) - 1

        self.num_action_streams = [2] * self.num_apps

        duel_str = 'Dueling-' + self.agg_method + '_'
        self.method_name = '{}{}{}{}'.format('Branching_', duel_str, 'TD-target-{}_'.format(self.target_version), 'TD-errors-aggregation-v{}'.format(self.losses_version))

        self.sess = U.make_session(num_cpu=self.num_cpu)
        self.sess.__enter__()

        self.current_step = -1

        self.create_deepq_train()
        self.create_exploration()

        self.create_replay_buffer()

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target_network_weights()

        if self.model_load:
            assert self.model_file != None, "Model file has to provided for loading"
            U.load_state(self.model_file)
            self.current_step = int(os.path.basename(os.path.dirname(self.model_file)))


    def create_deepq_train(self):
        def make_obs_ph(name):
            return U.BatchInput((self.state_space_length[0],), name=name)

        self.q_func = models.mlp_branching(
            hiddens_common=[512, 256],
            hiddens_after=[128], # Default
            hiddens_actions=[128],
            hiddens_value=[128],
            num_action_branches=self.num_action_streams,
            aggregator=self.agg_method,
            drop_out_regularization = self.drop_out
        )

        self.act = bg.build_act(
                make_obs_ph,
                self.q_func,
                self.num_actions_pad,
                self.num_action_streams,
                self.num_apps)

        self.deepq_train = bg.build_train(
            num_apps = self.num_apps,
            make_obs_ph=make_obs_ph,
            q_func=self.q_func,
            num_actions=self.num_actions_pad,
            num_action_streams=self.num_action_streams,
            batch_size=self.batch_size,
            optimizer_name="Adam",
            learning_rate=self.lr,
            grad_norm_clipping=self.grad_norm_clipping,
            gamma=self.gamma,
            scope="deepq",
            reuse=None,
            losses_version=self.losses_version,
            target_version=self.target_version,
            loss_type=self.loss_type,
            cyclic_learning_rate=self.cyclic
        )
        self.train, self.update_target, self.debug = self.deepq_train.get_vars()

    def create_exploration(self):
        if not self.epsilon_greedy:
            #approximate_num_iters = 30000
            approximate_num_iters = 25000
            self.exploration = PiecewiseSchedule([(0, 1.0),
                                            (self.drop_down, 0.1),
                                            (approximate_num_iters, 0.01)
                                            ], outside_value=0.01)
        else:
            self.exploration = ConstantSchedule(value=0.0) # greedy policy
            self.std_schedule = LinearSchedule(schedule_timesteps=self.timesteps_std,
                                         initial_p=self.initial_std,
                                         final_p=self.final_std)

    def create_replay_buffer(self):
        # Create the replay buffer
        if self.prioritized_replay:
            replay_buffer = PrioritizedReplayBuffer(self.buffer_size, alpha=self.prioritized_replay_alpha)
            if self.prioritized_replay_beta_iters is None:
                prioritized_replay_beta_iters = self.max_timesteps
            self.beta_schedule = LinearSchedule(self.prioritized_replay_beta_iters,
                                           initial_p=self.prioritized_replay_beta0,
                                           final_p=1.0)
        else:
            replay_buffer = ReplayBuffer(self.buffer_size)
            self.beta_schedule = None
        self.replay_buffers = [replay_buffer for _ in range(self.num_apps)]


    # Update target network periodically
    def update_target_network_weights(self):
        if self.current_step > self.learning_starts and self.current_step % self.target_network_update_freq == 0:
            self.update_target()

    def add_to_replay_buff(self, prev_state=None, action_idxes=None, reward=None, new_state=None, workload_num=None, done=False):
        self.replay_buffers[workload_num].add(prev_state, action_idxes, reward, new_state, done)

    def compute_temporal_diff(self):
        # Minimize the error in Bellman's equation on a batch sampled from replay buffer
        if self.current_step > self.learning_starts and self.current_step % self.train_freq == 0:
            idxes = self.replay_buffers[0].which_dataset(self.batch_size)
            if self.prioritized_replay:
                experience = self.replay_buffers[0].sample(idxes, beta=self.beta_schedule.value(self.current_step))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                experience = self.replay_buffers[1].sample(idxes, beta=self.beta_schedule.value(self.current_step))
                (obses_t1, actions1, rewards1, obses_tp11, dones1, weights1, batch_idxes1) = experience
            else:
                experience = self.replay_buffers[0].sample(idxes)
                (obses_t, actions, rewards, obses_tp1, dones) = experience
                weights, batch_idxes = np.ones_like(rewards), None
                experience = self.replay_buffers[1].sample(idxes)
                (obses_t1, actions1, rewards1, obses_tp11, dones1) = experience
                weights1, batch_idxes1 = np.ones_like(rewards), None

            td_errors, raw_val, learning_rate = self.train(obses_t, actions, rewards, obses_tp1, dones, weights,\
                    obses_t1, actions1, rewards1, obses_tp11, dones1, weights1, self.current_step)
            if self.cyclic: self.lr = learning_rate

            if self.prioritized_replay:
                for wkld in range(self.num_apps):
                    new_priorities = np.abs(td_errors[wkld]) + self.prioritized_replay_eps
                    self.replay_buffers[wkld].update_priorities(batch_idxes, new_priorities)

    def determine_action(self, current_state_a, current_state_b):
        self.current_step += 1
        # Select action and update exploration probability
        act_away, random = np.array(self.act(observation_0=np.array(current_state_a)[None], \
                        observation_1 = np.array(current_state_b)[None],\
                        update_eps=self.exploration.value(self.current_step)))


        actions=list()
        for wkld in range(self.num_apps):
            action_idxes = np.array(act_away[wkld])
            # Convert sub-actions indexes (discrete sub-actions) to continuous controls # BODMAS
            action = action_idxes / self.num_action_grains[wkld] * self.actions_range + self.low
            actions.append(action)
        return actions, act_away, self.exploration.value(self.current_step), random, self.lr
