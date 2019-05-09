# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from planet.tools import attr_dict
from planet.tools import filter_variables


class CustomOptimizer(object):

  def __init__(
      self, optimizer_cls, step, should_summarize, learning_rate,
      include=None, exclude=None, clipping=None, schedule=None):
    if schedule:
      learning_rate *= schedule(step)
    self._step = step
    self._should_summarize = should_summarize
    self._learning_rate = learning_rate
    self._variables = filter_variables.filter_variables(include, exclude)
    self._clipping = clipping
    self._optimizer = optimizer_cls(learning_rate)

  def minimize(self, loss):
    summaries = []
    gradients, variables = zip(*self._optimizer.compute_gradients(       #  tf.train.AdamOptimizer.compute_gradients(loss, var_list, ...)
        loss, self._variables, colocate_gradients_with_ops=True))        #  returns a list of (gradient, variable) pairs where "gradient" is the gradient for "variable".
    gradient_norm = tf.global_norm(gradients)        # Computes the global norm of multiple tensors.
    if self._clipping:
      gradients, _ = tf.clip_by_global_norm(
          gradients, self._clipping, gradient_norm)




    #tf.get_variable_scope().reuse_variables()
    return gradients, variables



  def _define_summaries(self, graph):   # Returns a scalar Tensor of type string
    summaries = []
    summaries.append(tf.summary.scalar('learning_rate', self._learning_rate))
    #summaries.append(tf.summary.scalar('gradient_norm', graph.gradient_norm))
    #if self._clipping:
      #clipped = tf.minimum(graph.gradient_norm, self._clipping)
      #summaries.append(tf.summary.scalar('clipped_gradient_norm', clipped))
    return tf.summary.merge(summaries)




  def apply(self, ave_grad, variable):
    optimize = self._optimizer.apply_gradients(zip(ave_grad, variable))
    graph = attr_dict.AttrDict(locals())
    summary = tf.cond(
      self._should_summarize, lambda: self._define_summaries(graph), str)

    return optimize, summary

