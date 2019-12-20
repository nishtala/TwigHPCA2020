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

import clr
import tensorflow as tf
from common import tf_util as U
import numpy as np

def build_act(make_obs_ph, q_func, num_actions, num_action_streams, num_apps, scope="deepq", reuse=None):
    inputs = []
    with tf.variable_scope(scope, reuse=reuse):
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        for wkld in range(num_apps):
            inputs.append(U.ensure_tf_input(make_obs_ph("observation_{}".format(wkld))))
        q_values = q_func([_.get() for _ in inputs], num_actions, scope="q_func")
        random_act = []
        main_output_actions = [[] for _ in range(num_apps)]
        for wkld in range(num_apps):
            output_actions = []
            for dim in range(num_action_streams[wkld]):
                q_values_batch = q_values[wkld][dim][0] # TODO better: does not allow evaluating actions over a whole batch
                deterministic_action = tf.argmax(q_values_batch)
                random_action = tf.random_uniform([], minval=0, maxval=num_actions[wkld][dim], dtype=tf.int64)
                chose_random = tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32) < eps
                stochastic_action = tf.cond(chose_random, lambda: random_action, lambda: deterministic_action)
                output_action = tf.cond(stochastic_ph, lambda: stochastic_action, lambda: deterministic_action)
                output_actions.append(output_action)
            main_output_actions[wkld] = output_actions
            random_act.append(chose_random)

        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        inputs += [stochastic_ph, update_eps_ph]
        act_f = U.function(inputs=inputs,
                         outputs=[main_output_actions,random_act],
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        return act_f

class build_train:
    def __init__(self, num_apps=None, make_obs_ph=None, q_func=None, num_actions=None, num_action_streams=None, batch_size=None, optimizer_name=None, learning_rate=None, grad_norm_clipping=None, gamma=0.99, scope="deepq", reuse=None, losses_version=1, target_version="mean", loss_type="L2", cyclic_learning_rate=False):
        self.num_apps = num_apps
        self.make_obs_ph = make_obs_ph
        self.q_func = q_func
        self.num_actions = num_actions
        self.num_action_streams = num_action_streams
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.const_learning_rate = learning_rate
        self.grad_norm_clipping = grad_norm_clipping
        self.gamma = gamma
        self.scope = scope
        self.reuse = reuse
        self.losses_version = losses_version
        self.target_version = target_version
        self.loss_type = loss_type
        self.loss_function = tf.square
        self.cyclic_learning_rate = cyclic_learning_rate


        self.create_value_function()

    def create_value_function(self):
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.observations = []
            self.actions = []
            self.rewards = []
            self.observations_tp1 = []
            self.dones = []
            self.wght_importance = []

            # Setup variable for learning rate
            global_step  = tf.get_variable("global_step", (), initializer=tf.constant_initializer(0))
            current_step = tf.placeholder(tf.float32, (), name="current_step")
            update_global = global_step.assign(current_step)
            self.learning_rate=clr.cyclic_learning_rate(global_step=global_step, mode='triangular2')
            if self.cyclic_learning_rate:
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            else:
                self.optimizer = tf.train.AdamOptimizer(self.const_learning_rate)


            for wkld in range(self.num_apps):
                # Set up placeholders
                self.observations.append( U.ensure_tf_input( self.make_obs_ph( "obs_t_{}".format(wkld) ) ) )
                self.actions.append( tf.placeholder( tf.int32, [None, self.num_action_streams[wkld]], name="action_{}".format(wkld) ) )
                self.rewards.append( tf.placeholder(tf.float32, [None], name="reward_{}".format(wkld) ) )
                self.observations_tp1.append( U.ensure_tf_input( self.make_obs_ph( "obs_tp1_{}".format(wkld) ) ) )
                self.dones.append( tf.placeholder(tf.float32, [None], name="done_{}".format(wkld) ) )
                self.wght_importance.append( tf.placeholder(tf.float32, [None], name="weight_{}".format(wkld) ) )

            # Q-network evaluation
            q_t = self.q_func([_.get() for _ in self.observations], self.num_actions, scope="q_func", reuse=True) # reuse parameters from act
            self.q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

            # Target Q-network evalution
            self.q_tp1 = self.q_func( [_.get() for _ in self.observations_tp1], self.num_actions, scope="target_q_func")
            target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

            self.selection_q_tp1 = self.q_func([_.get() for _ in self.observations_tp1], self.num_actions, scope="q_func", reuse=True)


            self.q_values = [ [] for _ in range(self.num_apps) ]
            for wkld in range(self.num_apps):
                for dim in range(self.num_action_streams[wkld]):
                    selected_a = tf.squeeze(tf.slice(self.actions[wkld], [0, dim], [self.batch_size, 1])) # TODO better?
                    self.q_values[wkld].append(tf.reduce_sum(tf.one_hot(selected_a, self.num_actions[wkld][dim]) * q_t[wkld][dim], axis=1))

            self.compute_td_error()
            self.compute_loss()

            # Target Q-network parameters are periodically updated with the Q-network's
            update_target_expr = []
            for var, var_target in zip(sorted(self.q_func_vars, key=lambda v: v.name),
                                       sorted(target_q_func_vars, key=lambda v: v.name)):
                update_target_expr.append(var_target.assign(var))
            update_target_expr = tf.group(*update_target_expr)

            xinputs = []
            for wkld in range(self.num_apps):
                xinputs.append(self.observations[wkld])
                xinputs.append(self.actions[wkld])
                xinputs.append(self.rewards[wkld])
                xinputs.append(self.observations_tp1[wkld])
                xinputs.append(self.dones[wkld])
                xinputs.append(self.wght_importance[wkld])

            xinputs.append(current_step)
            self.train = U.function(
                inputs=xinputs,
                outputs=[self.td_errors, self.raw_td_val, self.learning_rate],
                updates=[self.optimize_exprs, update_global, self.learning_rate]
            )

            self.update_target = U.function([], [], updates=[update_target_expr])

            self.q_values = U.function([self.observations[0]], q_t[0])

    def compute_td_error(self):
        assert (self.target_version in ['mean_across_branches_per_workload',
        'min_across_branches_per_workload',
        'max_across_branches_per_workload']), 'appropriate TD method needs to be set'

        #assert (self.target_version in ['mean_across_branches_per_workload',
        #'min_across_branches_per_workload',
        #'max_across_branches_per_workload',
        #'min_across_workload_per_branch',
        #'max_across_workload_per_branch',
        #'mean_across_workload_per_branch']), 'appropriate TD method needs to be set'

        self.target_q_values = []
        if 'per_workload' in self.target_version:
            for wkld in range(self.num_apps):
                for dim in range(self.num_action_streams[wkld]):
                    selected_a = tf.argmax(self.selection_q_tp1[wkld][dim], axis=1)
                    selected_q = tf.reduce_sum(tf.one_hot(selected_a, self.num_actions[wkld][dim]) \
                            * self.q_tp1[wkld][dim], axis=1)
                    masked_selected_q = (1.0 - self.dones[wkld]) * selected_q
                    if dim == 0:
                        mean_next_q_values = masked_selected_q
                    else:
                        if self.target_version == 'mean_across_branches_per_workload':
                            mean_next_q_values += masked_selected_q
                        elif self.target_version == 'min_across_branches_per_workload':
                            mean_next_q_values = tf.minimum(mean_next_q_values, masked_selected_q)
                        elif self.target_version == 'max_across_branches_per_workload':
                            mean_next_q_values = tf.maximum(mean_next_q_values, masked_selected_q)
                if self.target_version == 'mean_across_branches_per_workload':
                    mean_next_q_values /= self.num_action_streams[wkld]
                self.target_q_value = [self.rewards[wkld] + self.gamma * mean_next_q_values] * self.num_action_streams[wkld] # TODO better?
                self.target_q_values.append(self.target_q_value)
        else:
            assert False, 'unsupported target version ' + str(self.target_version)

    def compute_loss(self):
        self.optimize_exprs = []
        self.td_errors = [[0.] for _ in range(self.num_apps)]
        self.raw_td_val = [[0.] for _ in range(self.num_apps)]
        if self.losses_version == 1:
            for wkld in range(self.num_apps):
                stream_losses = []
                q_vars = []
                for q in self.q_func_vars:
                    scnd = q.name.split("/")[2]
                    if scnd.startswith('after'):
                        q_vars.append(q)
                    elif scnd.endswith(str(wkld)):
                        q_vars.append(q)

                for dim in range(self.num_action_streams[wkld]):
                    dim_td_error = self.q_values[wkld][dim] - tf.stop_gradient(self.target_q_values[wkld][dim])
                    self.raw_td_val[wkld] += dim_td_error
                    dim_loss = self.loss_function(dim_td_error)
                    # Scaling of learning based on importance sampling weights is optional, either way works
                    stream_losses.append(tf.reduce_mean(dim_loss * self.wght_importance[wkld]))
                    #stream_losses.append(tf.reduce_mean(dim_loss)) # without scaling
                    #if dim == 0:
                    #    self.td_errors[wkld] = tf.abs(dim_td_error)
                    #else:
                    self.td_errors[wkld] += tf.abs(dim_td_error)

                mean_loss = sum(stream_losses) / self.num_action_streams[wkld]
                optimize_expr = U.minimize_and_clip(self.optimizer,
                                                    mean_loss,
                                                    var_list=q_vars, #self.q_func_vars,
                                                    total_n_streams=(self.num_action_streams[wkld] + 1),
                                                    clip_val=self.grad_norm_clipping)
                self.optimize_exprs.append(optimize_expr)
        elif self.losses_version == 2:
            # self.q_func_vars should only contain those we need to update for that workload
            for dim in range(self.num_action_streams[0]):
                stream_losses = []
                for wkld in range(self.num_apps):
                    dim_td_error = self.q_values[wkld][dim] - tf.stop_gradient(self.target_q_values[wkld][dim])
                    dim_loss = self.loss_function(dim_td_error)
                    # Scaling of learning based on importance sampling weights is optional, either way works
                    stream_losses.append(tf.reduce_mean(dim_loss * self.wght_importance[wkld]))
                    #stream_losses.append(tf.reduce_mean(dim_loss)) # without scaling
                    self.td_errors[wkld] += tf.abs(dim_td_error)

                mean_loss = sum(stream_losses) / self.num_apps
                optimize_expr = U.minimize_and_clip(self.optimizer,
                                                    mean_loss,
                                                    var_list=self.q_func_vars,
                                                    total_n_streams=(self.num_action_streams[wkld] + 1),
                                                    clip_val=self.grad_norm_clipping)
                self.optimize_exprs.append(optimize_expr)
        else:
            assert False, 'unsupported loss version ' + str(self.losses_version)

    def get_vars(self):
        return self.train, self.update_target, {'q_values': self.q_values}
