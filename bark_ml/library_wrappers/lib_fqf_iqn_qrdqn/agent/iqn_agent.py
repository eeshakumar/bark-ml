# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe

import torch
from torch.optim import Adam

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import IQN
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class IQNAgent(BaseAgent):

  def __init__(self, env, test_env, params):
    super(IQNAgent, self).__init__(env, test_env, params)

    self._N = self._params["ML"]["IQNAgent"]["N", "", 64]
    self._N_dash = self._params["ML"]["IQNAgent"]["N_dash", "", 64]
    self._K = self._params["ML"]["IQNAgent"]["N_dash", "", 32]
    self._num_cosines = self._params["ML"]["IQNAgent"]["NumCosines", "", 64]
    self._kappa = self._params["ML"]["IQNAgent"]["Kappa", "", 1.0]

    # Online network.
    self._online_net = IQN(num_channels=env.observation_space.shape[0],
                           num_actions=self._num_actions,
                           num_cosines=self._num_cosines,
                           noisy_net=self._noisy_net,
                           params=self._params).to(self._device)
    # Target network.
    self._target_net = IQN(num_channels=env.observation_space.shape[0],
                           num_actions=self._num_actions,
                           num_cosines=self._num_cosines,
                           noisy_net=self._noisy_net,
                           params=self._params).to(self._device)

    # Copy parameters of the learning network to the target network.
    self.update_target()
    # Disable calculations of gradients of the target network.
    disable_gradients(self._target_net)

    self._optim = Adam(self._online_net.parameters(),
                       lr=self._params["ML"]["IQNAgent"]["LearningRate", "",
                                                         5e-5],
                       eps=1e-2 / self._batch_size)

  def learn(self):
    self._learning_steps += 1
    self._online_net.sample_noise()
    self._target_net.sample_noise()

    if self._use_per:
      (states, actions, rewards, next_states, dones), weights = \
       self._memory.sample(self._batch_size)
    else:
      states, actions, rewards, next_states, dones = \
       self._memory.sample(self._batch_size)
      weights = None

    # Calculate features of states.
    state_embeddings = self._online_net.calculate_state_embeddings(states)

    quantile_loss, mean_q, errors = self._calculate_loss(
        state_embeddings, actions, rewards, next_states, dones, weights)

    update_params(self._optim,
                  quantile_loss,
                  networks=[self._online_net],
                  retain_graph=False,
                  grad_cliping=self._grad_cliping)

    if self._use_per:
      self._memory.update_priority(errors)

    if 4 * self._steps % self._summary_log_interval == 0:
      self._writer.add_scalar('loss/quantile_loss',
                              quantile_loss.detach().item(), 4 * self._steps)
      self._writer.add_scalar('stats/mean_Q', mean_q, 4 * self._steps)

  def calculate_loss(self, state_embeddings, actions, rewards, next_states,
                     dones, weights):
    # Sample fractions.
    taus = torch.rand(self._batch_size,
                      self._N,
                      dtype=state_embeddings.dtype,
                      device=state_embeddings.device)

    # Calculate quantile values of current states and actions at tau_hats.
    current_sa_quantiles = evaluate_quantile_at_action(
        self._online_net.calculate_quantiles(taus,
                                             state_embeddings=state_embeddings),
        actions)
    assert current_sa_quantiles.shape == (self._batch_size, self._N, 1)

    with torch.no_grad():
      # Calculate Q values of next states.
      if self._double_q_learning:  # note: double q learning set to always false.
        # Sample the noise of online network to decorrelate between
        # the action selection and the quantile calculation.
        self._online_net.sample_noise()
        next_q = self._online_net.calculate_q(states=next_states)
      else:
        next_state_embeddings = \
         self._target_net.calculate_state_embeddings(next_states)
        next_q = self._target_net.calculate_q(
            state_embeddings=next_state_embeddings)

      # Calculate greedy actions.
      next_actions = torch.argmax(next_q, dim=1, keepdim=True)
      assert next_actions.shape == (self._batch_size, 1)

      # Calculate features of next states.
      if self._double_q_learning:
        next_state_embeddings = \
         self._target_net.calculate_state_embeddings(next_states)

      # Sample next fractions.
      tau_dashes = torch.rand(self._batch_size,
                              self._N_dash,
                              dtype=state_embeddings.dtype,
                              device=state_embeddings.device)

      # Calculate quantile values of next states and next actions.
      next_sa_quantiles = evaluate_quantile_at_action(
          self._target_net.calculate_quantiles(
              tau_dashes, state_embeddings=next_state_embeddings),
          next_actions).transpose(1, 2)
      assert next_sa_quantiles.shape == (self._batch_size, 1, self._N_dash)

      # Calculate target quantile values.
      target_sa_quantiles = rewards[..., None] + (
          1.0 - dones[..., None]) * self._gamma_n * next_sa_quantiles
      assert target_sa_quantiles.shape == (self._batch_size, 1, self._N_dash)

    td_errors = target_sa_quantiles - current_sa_quantiles
    assert td_errors.shape == (self._batch_size, self._N, self._N_dash)

    quantile_huber_loss = calculate_quantile_huber_loss(td_errors, taus,
                                                        weights, self._kappa)

    return quantile_huber_loss, next_q.detach().mean().item(), \
        td_errors.detach().abs()
