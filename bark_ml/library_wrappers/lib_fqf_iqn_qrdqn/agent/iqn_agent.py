# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License -Copyright (c) 2020 Toshiki Watanabe
import logging
import torch
from torch.optim import Adam

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import IQN
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
 import disable_gradients, update_params, \
 calculate_quantile_huber_loss, calculate_supervised_margin_classification_loss, evaluate_quantile_at_action
from .base_agent import BaseAgent


class IQNAgent(BaseAgent):

  def __init__(self, *args, **kwargs):
    super(IQNAgent, self).__init__(*args, **kwargs)

  def reset_params(self, params):
    super(IQNAgent, self).reset_params(params)
    self.N = params["ML"]["IQNAgent"]["N", "", 64]
    self.N_dash = params["ML"]["IQNAgent"]["N_dash", "", 64]
    self.K = params["ML"]["IQNAgent"]["N_dash", "", 32]
    self.num_cosines = params["ML"]["IQNAgent"]["NumCosines", "", 64]
    self.kappa = params["ML"]["IQNAgent"]["Kappa", "", 1.0]

  def init_always(self):
    super(IQNAgent, self).init_always()
    # Online network.
    self.online_net = IQN(num_channels=self.observer.observation_space.shape[0],
                           num_actions=self.num_actions,
                           num_cosines=self.num_cosines,
                           noisy_net=self.noisy_net,
                           params=self._params).to(self.device)
    # Target network.
    self.target_net = IQN(num_channels=self.observer.observation_space.shape[0],
                           num_actions=self.num_actions,
                           num_cosines=self.num_cosines,
                           noisy_net=self.noisy_net,
                           params=self._params).to(self.device)

    # Copy parameters of the learning network to the target network.
    self.update_target()
    # Disable calculations of gradients of the target network.
    disable_gradients(self.target_net)

    self.optim = Adam(self.online_net.parameters(),
                       lr=self._params["ML"]["IQNAgent"]["LearningRate", "",
                                                         5e-5],
                       eps=1e-2 / self.batch_size)

  def clean_pickables(self, pickables):
    super(IQNAgent, self).clean_pickables(pickables)
    del pickables["optim"]

  def update_target(self):
    self.target_net.dqn_net.load_state_dict(
        self.online_net.dqn_net.state_dict())
    self.target_net.quantile_net.load_state_dict(
        self.online_net.quantile_net.state_dict())
    self.target_net.cosine_net.load_state_dict(
        self.online_net.cosine_net.state_dict())

  def learn(self):
    self.learning_steps += 1
    self.online_net.sample_noise()
    self.target_net.sample_noise()

    if self.use_per:
      if self.is_learn_from_demonstrations:
        (states, actions, rewards, next_states, dones, is_demos), weights = \
        self.memory.sample(self.batch_size)
      else:
        (states, actions, rewards, next_states, dones), weights = \
        self.memory.sample(self.batch_size)
    else:
      states, actions, rewards, next_states, dones = \
       self.memory.sample(self.batch_size)
      weights = None

    # average rewards from batch
    mean_batch_reward = torch.mean(rewards)

    # Calculate features of states.
    state_embeddings = self.online_net.calculate_state_embeddings(states)

    if self.is_learn_from_demonstrations:
      quantile_loss, mean_q, errors = self.calculate_loss(
          state_embeddings, actions, rewards, next_states, dones, weights, states, is_demos)
    else:
      quantile_loss, mean_q, errors = self.calculate_loss(
          state_embeddings, actions, rewards, next_states, dones, weights, states)

    gradient = update_params(self.optim,
                  quantile_loss,
                  networks=[self.online_net],
                  retain_graph=False,
                  grad_cliping=self.grad_cliping)

    if self.use_per:
      if self.is_learn_from_demonstrations:
        self.memory.update_priority(errors, is_demos)
      else:
        self.memory.update_priority(errors)

    if 4 * self.steps % self.summary_log_interval == 0:
      self.writer.add_scalar('loss/quantile_loss',
                              quantile_loss.detach().item(), 4 * self.steps)
      self.writer.add_scalar('stats/mean_Q', mean_q, 4 * self.steps)
      self.writer.add_scalar('stats/mean_batch_reward', mean_batch_reward, 4 * self.steps)
      if gradient is not None:
        print("Registering grad")
        self.writer.add_scalar('loss/grad', gradient.detach().item(), 4 * self.steps)

  def calculate_loss(self, state_embeddings, actions, rewards, next_states,
                     dones, weights, states, is_demos=None):
    # Sample fractions.
    taus = torch.rand(self.batch_size,
                      self.N,
                      dtype=state_embeddings.dtype,
                      device=state_embeddings.device)

    # get current q values from network
    self.online_net.sample_noise()
    current_q = self.online_net.calculate_q(states=states)

    # Calculate quantile values of current states and actions at tau_hats.
    current_sa_quantiles = evaluate_quantile_at_action(
        self.online_net.calculate_quantiles(taus,
                                             state_embeddings=state_embeddings),
        actions)
    assert current_sa_quantiles.shape == (self.batch_size, self.N, 1)

    with torch.no_grad():
      # Calculate Q values of next states.
      if self.double_q_learning:  # note: double q learning set to always false.
        # Sample the noise of online network to decorrelate between
        # the action selection and the quantile calculation.
        self.online_net.sample_noise()
        next_q = self.online_net.calculate_q(states=next_states)
      else:
        next_state_embeddings = \
         self.target_net.calculate_state_embeddings(next_states)
        next_q = self.target_net.calculate_q(
            state_embeddings=next_state_embeddings)

      # Calculate greedy actions.
      next_actions = torch.argmax(next_q, dim=1, keepdim=True)
      assert next_actions.shape == (self.batch_size, 1)

      # Calculate features of next states.
      if self.double_q_learning:
        next_state_embeddings = \
         self.target_net.calculate_state_embeddings(next_states)

      # Sample next fractions.
      tau_dashes = torch.rand(self.batch_size,
                              self.N_dash,
                              dtype=state_embeddings.dtype,
                              device=state_embeddings.device)

      # Calculate quantile values of next states and next actions.
      next_sa_quantiles = evaluate_quantile_at_action(
          self.target_net.calculate_quantiles(
              tau_dashes, state_embeddings=next_state_embeddings),
          next_actions).transpose(1, 2)
      assert next_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

      # Calculate target quantile values.
      target_sa_quantiles = rewards[..., None] + (
          1.0 - dones[..., None]) * self.gamma_n * next_sa_quantiles
      assert target_sa_quantiles.shape == (self.batch_size, 1, self.N_dash)

    td_errors = target_sa_quantiles - current_sa_quantiles
    assert td_errors.shape == (self.batch_size, self.N, self.N_dash)

    quantile_huber_loss = calculate_quantile_huber_loss(td_errors, taus,
                                                        weights, self.kappa)
    total_loss = quantile_huber_loss

    return total_loss, next_q.detach().mean().item(), \
        td_errors.detach().abs()
