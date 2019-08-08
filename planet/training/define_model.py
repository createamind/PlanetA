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

import functools
import numpy as np
import tensorflow as tf
from numbers import Number
from planet import tools, NUM_GPU
from planet.training import define_summaries
from planet.training import utility
from planet.networks.sac1 import get_vars
#def define_model_sac(data,trainer,config):
from planet import control
from planet.networks import sac1
def define_model(data, trainer, config):
  tf.logging.info('Build TensorFlow compute graph.')
  dependencies = []
  step = trainer.step
  global_step = trainer.global_step  # tf.train.get_or_create_global_step()
  phase = trainer.phase
  should_summarize = trainer.log
  # alpha = 0.2
  # gamma = 0.99
  # lr = 1e-3
  # polyak = 0.995

  import argparse
  parser = argparse.ArgumentParser()



  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--alpha', type=float,default=0.2, help="alpha can be either 'auto' or float(e.g:0.2).")
  parser.add_argument('--lr', type=float, default=0.001)
  parser.add_argument('--polyak', type=float, default=0.995)
  args = parser.parse_args()

  num_gpu = NUM_GPU

  #  for multi-gpu
  if num_gpu > 1:
      var_for_trainop={}
      grads_dict={}

      # data split for multi-gpu
      data_dict={}
      for loss_head, optimizer_cls in config.optimizers.items():
          grads_dict[loss_head] = []
          var_for_trainop[loss_head] =[]

      for gpu_i in range(num_gpu):
          data_dict[gpu_i]={}

      for data_item in list(data.keys()):
        data_split=tf.split(data[data_item], num_gpu)
        for gpu_j in range(num_gpu):
            data_dict[gpu_j][data_item]= data_split[gpu_j]


  for gpu_k in range(num_gpu):
    with tf.device('/gpu:%s' % gpu_k):
      scope_name = r'.+shared_vars'
      with tf.name_scope('%s_%d' % ("GPU", gpu_k)):   # 'GPU'
          with tf.variable_scope(name_or_scope= 'shared_vars', reuse=tf.AUTO_REUSE):

              #  for multi-gpu
              if num_gpu > 1:
                data = data_dict[gpu_k]

              # Preprocess data.
              # with tf.device('/cpu:0'):
              if config.dynamic_action_noise:
                  data['action'] += tf.random_normal(
                      tf.shape(data['action']), 0.0, config.dynamic_action_noise)
              prev_action = tf.concat(
                  [0 * data['action'][:, :1], data['action'][:, :-1]], 1)  # i.e.: (0 * a1, a1, a2, ..., a49)
              obs = data.copy()
              del obs['length']

              # Instantiate network blocks.
              cell = config.cell()
              kwargs = dict()
              encoder = tf.make_template(
                  'encoder', config.encoder, create_scope_now_=True, **kwargs)
              kwargs = dict(hidden_sizes=[500,400,300])
              #add for sac1

              main_actor_critic = tf.make_template(
                  'main_actor_critic', config.actor_critic, create_scope_now_=True, **kwargs)

              #kwargs = dict(hidden_sizes=[128, 128, 128])
              target_actor_critic = tf.make_template(
                  'target_actor_critic', config.actor_critic, create_scope_now_=True, **kwargs)

              # episode_actor_critic = tf.make_template(
              #     'episode_actor_critic', config.actor_critic, create_scope_now_=True, **kwargs)

              heads = {}
              for key, head in config.heads.items():  # heads: network of 'image', 'reward', 'state'
                  name = 'head_{}'.format(key)
                  kwargs = dict(data_shape=obs[key].shape[2:].as_list())
                  heads[key] = tf.make_template(name, head, create_scope_now_=True, **kwargs)

              # Embed observations and unroll model.
              embedded = encoder(obs)  # encode obs['image']
              # Separate overshooting and zero step observations because computing
              # overshooting targets for images would be expensive.
              zero_step_obs = {}
              overshooting_obs = {}
              for key, value in obs.items():
                  if config.zero_step_losses.get(key):
                      zero_step_obs[key] = value
                  if config.overshooting_losses.get(key):
                      overshooting_obs[key] = value
              assert config.overshooting <= config.batch_shape[1]
              target, prior, posterior, mask = tools.overshooting(                    # prior:{'mean':shape(40,50,51,30), ...}; posterior:{'mean':shape(40,50,51,30), ...}
                  cell, overshooting_obs, embedded, prev_action, data['length'],      # target:{'reward':shape(40,50,51), ...}; mask:shape(40,50,51)
                  config.overshooting + 1)
              losses = []

              # Zero step losses.
              _, zs_prior, zs_posterior, zs_mask = tools.nested.map(
                  lambda tensor: tensor[:, :, :1], (target, prior, posterior, mask))
              zs_target = {key: value[:, :, None] for key, value in zero_step_obs.items()}

              features = cell.features_from_state(zs_posterior)  # [s,h]
              #+++++++++++++++++++++++++++++++++++++++add for sac+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

              # data for sac
              #features = tf.stop_gradient(features)  # stop gradient for features


              # s0,s1,.....s49
              hidden_next= features[:, 4:] # 46
              hidden = features[:, 3:-1]  # 46
              reward = obs['reward'][:,3:-1]
              action = obs['action'][:,3:-1]
              done = obs['done'][:,3:-1]

              #s = np.random.permutation(46)
              hidden_next = tf.random_shuffle(hidden_next,seed=2)
              hidden = tf.random_shuffle(hidden,seed=2)
              reward = tf.random_shuffle(reward,seed=2)
              action = tf.random_shuffle(action,seed=2)
              done = tf.random_shuffle(done,seed=2)

              done = tf.cast(done, dtype=tf.float32)

              reward = tf.reshape(reward,(-1,1)) #245,1
              action = tf.reshape(action,(-1,2)) #245,2
              done = tf.reshape(done, (-1, 1)) #245,1
              hidden_next = tf.reshape(hidden_next, (-1, 250)) #245,250
              hidden = tf.reshape(hidden, (-1, 250)) #245,250

              # x = tf.placeholder(dtype=tf.float32, shape=(None, 250))
              # a = tf.placeholder(dtype=tf.float32, shape=(None, 2))
              mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = main_actor_critic(hidden,action)
              _, _, logp_pi_, _, _,q1_pi_, q2_pi_ = target_actor_critic(hidden_next,action)
              # _, pi_ep, _, _, _, _, _ = episode_actor_critic(x, a)


              target_init = tf.group([tf.assign(v_targ, v_main) for v_main, v_targ in
                                      zip(get_vars('main_actor_critic'), get_vars('target_actor_critic'))])

              # if alpha == 'auto':
              #     target_entropy = (-2)
              #
              #     log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
              #     alpha = tf.exp(log_alpha)
              #
              #     alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))
              #
              #     alpha_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, name='alpha_optimizer')
              #     train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

              # if args.alpha == 'auto':
              #     target_entropy = (-np.prod([2,1]))
              #
              #     log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=1.0)
              #     alpha = tf.exp(log_alpha)
              #
              #     alpha_loss = tf.reduce_mean(-log_alpha * tf.stop_gradient(logp_pi + target_entropy))
              #
              #     alpha_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr * 0.01, name='alpha_optimizer')
              #     train_alpha_op = alpha_optimizer.minimize(loss=alpha_loss, var_list=[log_alpha])

              min_q_pi = tf.minimum(q1_pi_, q2_pi_)

              # Targets for Q and V regression
              v_backup = tf.stop_gradient(min_q_pi - args.alpha * logp_pi_)
              q_backup = reward + args.gamma * (1 - done) * v_backup

              # Soft actor-critic losses
              pi_loss = tf.reduce_mean(args.alpha * logp_pi - q1_pi)
              q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
              q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
              value_loss = q1_loss + q2_loss
              #loss = value_loss + pi_loss
              # Policy train op
              # (has to be separate from value train op, because q1_pi appears in pi_loss)
              pi_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
              train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main_actor_critic/pi'))

              # Value train op
              # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
              value_optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
              value_params = get_vars('main_actor_critic/q')
              with tf.control_dependencies([train_pi_op]):
                  train_value_op = value_optimizer.minimize(value_loss, var_list=value_params)

              # Polyak averaging for target variables
              # (control flow because sess.run otherwise evaluates in nondeterministic order)
              with tf.control_dependencies([train_value_op]):
                  target_update = tf.group([tf.assign(v_targ, args.polyak * v_targ + (1 - args.polyak) * v_main)
                                            for v_main, v_targ in zip(get_vars('main_actor_critic'), get_vars('target_actor_critic'))])

              var_counts = tuple(sac1.count_vars(scope) for scope in
                                 ['main_actor_critic/pi', 'main_actor_critic/q', 'main_actor_critic'])
              print(('\nNumber of parameters: \t pi: %d, \t' + 'q: %d,  \t total: %d\n') % var_counts)



              # All ops to call during one training step
              if isinstance(args.alpha, Number):
                print("enter into args.alpha")
                step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, tf.identity(args.alpha),
                              train_pi_op, train_value_op, target_update]
              else:
                step_ops = [pi_loss, q1_loss, q2_loss, q1, q2, logp_pi, args.alpha,
                              train_pi_op, train_value_op, target_update, train_alpha_op]


              # train_step_op = tf.cond(
              #     tf.equal(phase, 'sac'),
              #     lambda: pi_loss,
              #     lambda: 0 * tf.get_variable('dummy_loss', (), tf.float32))



              with tf.control_dependencies(step_ops):
                  train_summary = tf.constant('')





# for sac ===================================if you phase is set as sac , it will not enter phase train and test so
# it will not do planning for  episode data .
  collect_summaries = []
  graph = tools.AttrDict(locals())
  with tf.variable_scope('collection'):
    should_collects = []
    for name, params in config.sim_collects.items():
      after, every = params.steps_after, params.steps_every
      should_collect = tf.logical_and(
          tf.equal(phase, 'sac'),
          tools.schedule.binary(step, config.batch_shape[0], after, every))
      collect_summary, score_train = tf.cond(
          should_collect,
          functools.partial(
              utility.simulate_episodes, config, params, graph, name),
          lambda: (tf.constant(''), tf.constant(0.0)),
          name='should_collect_' + params.task.name)
      should_collects.append(should_collect)
      collect_summaries.append(collect_summary)

  # Compute summaries.
  graph = tools.AttrDict(locals())
  with tf.control_dependencies(collect_summaries):
    # summaries, score = tf.cond(
    #     should_summarize,
    #     lambda: define_summaries.define_summaries(graph, config),
    #     lambda: (tf.constant(''), tf.zeros((0,), tf.float32)),
    #     name='summaries')
    summaries=tf.constant('')
    score=tf.zeros((0,), tf.float32)
  with tf.device('/cpu:0'):
    summaries = tf.summary.merge([summaries, train_summary])
    # summaries = tf.summary.merge([summaries, train_summary] + collect_summaries)

    dependencies.append(utility.print_metrics((
        ('score', score_train),
        ('q1_loss', q1_loss),
        ('q2_loss', q2_loss),
        ('pi_loss', pi_loss),
    ), step, config.mean_metrics_every))
  with tf.control_dependencies(dependencies):
    score = tf.identity(score)
  return score, summaries,target_init


def average_gradients(tower_grads):
    average_grads = []
    for grad_gpus in zip(*tower_grads):
        grads = []
        for g in grad_gpus:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        average_grads.append(grad)
    return average_grads