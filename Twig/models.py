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

import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np

def _mlp_branching(hiddens_common, hiddens_after, hiddens_actions, \
        hiddens_value, num_action_branches, aggregator, drop_out_regularization,\
        inpt, num_actions, scope, reuse=False):
    drop_bias = False
    single_hidden_layer = False
    drop_out_regularization = True
    input_aggregation = 'add'
    if drop_bias:
        biases_initializer = None
    else:
        biases_initializer = tf.zeros_initializer()
        # If 0 bias does not work, then try this.
        #bias_initializer = tf.constant_initializer(0.01)

    with tf.variable_scope(scope, reuse=reuse):
        out = inpt
        num_apps = len(out)

        assert (aggregator in ['reduceLocalMean','reduceGlobalMean','naive','reduceLocalMax','reduceGlobalMax']), 'appropriate aggregator method needs be set when using dueling architecture'
        assert (hiddens_value), 'state-value network layer size cannot be empty when using dueling architecture'

        workloads = [ [] for _ in range(num_apps) ]
        for wkld in range(num_apps):
            with tf.variable_scope('individual_workload_{}'.format(wkld)):
                workloads[wkld] = layers.fully_connected(out[wkld], num_outputs=hiddens_actions[0],\
                        activation_fn=tf.nn.relu, biases_initializer=biases_initializer)
                if drop_out_regularization: workloads[wkld] = layers.dropout(workloads[wkld])

        # Create the shared network module
        with tf.variable_scope('common_net'):
            if input_aggregation == 'naive':
                out = tf.concat(workloads, 1)
            elif input_aggregation == 'add':
                out = tf.add(workloads[0], workloads[1])
            else:
                assert False, 'select appropriate input aggregation method'

            for hidden in hiddens_common:
                out = layers.fully_connected(out, num_outputs=hidden, \
                        activation_fn=tf.nn.relu, biases_initializer=biases_initializer)
                if drop_out_regularization: out = layers.dropout(out)
                if single_hidden_layer: break

        common_out = out
        # After common_layer networks.
        after_common_network_layers = [ [] for _ in range(num_action_branches[0]) ]
        for action_stream in range(num_action_branches[0]):
            with tf.variable_scope('after_common_{}'.format(action_stream)):
                out = common_out
                for hidden in hiddens_after:
                    out = layers.fully_connected(out, num_outputs=hidden, \
                            activation_fn=tf.nn.relu, biases_initializer=biases_initializer)
                    if drop_out_regularization: out = layers.dropout(out)
                    if single_hidden_layer: break
                after_common_network_layers[action_stream] = out

        # Create as many workloads as necessary for each stream
        total_action_scores    = [ [] for _ in range(num_apps)]
        state_scores           = [ [] for _ in range(num_apps)]
        action_scores_adjusted = [ [] for _ in range(num_apps)]

        # Create the action branches
        for action_stream in range(num_action_branches[0]):
            for wkld in range(num_apps):
                action_out = after_common_network_layers[action_stream]
                with tf.variable_scope('action_value_{}_{}'.format(action_stream, wkld)):
                    for hidden in hiddens_actions:
                        action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=tf.nn.relu, biases_initializer=biases_initializer)
                        if drop_out_regularization: action_out = layers.dropout(action_out)
                    action_scores = layers.fully_connected(action_out, num_outputs=num_actions[wkld][action_stream], activation_fn=None, biases_initializer=biases_initializer)
                    if aggregator == 'reduceLocalMean':
                        action_scores_mean = tf.reduce_mean(action_scores, 1)
                        total_action_scores[wkld].append(action_scores - tf.expand_dims(action_scores_mean, 1))
                    elif aggregator == 'reduceLocalMax':
                        action_scores_max = tf.reduce_max(action_scores, 1)
                        total_action_scores[wkld].append(action_scores - tf.expand_dims(action_scores_max, 1))
                    else:
                        total_action_scores[wkld].append(action_scores)

        for wkld in range(num_apps):
            with tf.variable_scope('state_value_{}'.format(wkld)):
                state_out = workloads[wkld]
                for hidden in hiddens_value:
                    state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=tf.nn.relu, biases_initializer=biases_initializer)
                    if drop_out_regularization: state_out = layers.dropout(state_out)
                state_scores[wkld] = layers.fully_connected(state_out, num_outputs=1, activation_fn=None,biases_initializer=biases_initializer)
            if aggregator == 'reduceLocalMean':
                # Local centering wrt branch's mean value has already been done
                action_scores_adjusted[wkld] = total_action_scores[wkld]
            elif aggregator == 'reduceGlobalMean':
                action_scores_mean = sum(total_action_scores[wkld]) / num_action_branches[wkld]
                action_scores_adjusted[wkld] = total_action_scores - tf.expand_dims(action_scores_mean, 1)
            elif aggregator == 'reduceLocalMax':
                # Local max-reduction has already been done
                action_scores_adjusted[wkld] = total_action_scores[wkld]
            elif aggregator == 'reduceGlobalMax':
                assert False, 'not implemented'
                action_scores_max = max(total_action_scores[wkld])
                action_scores_adjusted[wkld] = total_action_scores[wkld] - tf.expand_dims(action_scores_max, 1)
            elif aggregator == 'naive':
                action_scores_adjusted[wkld] = total_action_scores[wkld]
            else:
                assert (aggregator in ['reduceLocalMean','reduceGlobalMean','naive','reduceLocalMax','reduceGlobalMax']), 'aggregator method is not supported'

        q_vals = []
        for wkld in range(num_apps):
            q = [state_scores[wkld] + action_score_adjusted for action_score_adjusted in action_scores_adjusted[wkld]]
            q_vals.append(q)
        return q_vals


def mlp_branching(hiddens_common=[], hiddens_after=[], hiddens_actions=[], hiddens_value=[], num_action_branches=None, aggregator='reduceLocalMean', drop_out_regularization=True): #, distributed_single_stream=False):
    """This model takes as input an observation and returns values of all sub-actions -- either by
    combining the state value and the sub-action advantages (i.e. dueling), or directly the Q-values.

    Parameters
    ----------
    hiddens_common: [int]
        list of sizes of hidden layers in the shared network module --
        if this is an empty list, then the learners across the branches
        are considered 'independent'

    hiddens_actions: [int]
        list of sizes of hidden layers in the action-value/advantage branches --
        currently assumed the same across all such branches

    hiddens_value: [int]
        list of sizes of hidden layers for the state-value branch

    num_action_branches: int
        number of action branches (= num_action_dims in current implementation)

    dueling: bool
        if using dueling, then the network structure becomes similar to that of
        dueling (i.e. Q = f(V,A)), but with N advantage branches as opposed to only one,
        and if not dueling, then there will be N branches of Q-values

    aggregator: str
        aggregator method used for dueling architecture: {naive, reduceLocalMean, reduceLocalMax, reduceGlobalMean, reduceGlobalMax}

    distributed_single_stream: bool
        True if action value (or advantage) function representation is branched (vs. combinatorial), but
        all sub-actions are represented on the same fully-connected stream

    Returns
    -------
    q_func: function
        q_function for DQN algorithm.
    """
    return lambda *args, **kwargs: _mlp_branching(hiddens_common, hiddens_after, hiddens_actions, hiddens_value, num_action_branches, aggregator, drop_out_regularization, *args, **kwargs)
