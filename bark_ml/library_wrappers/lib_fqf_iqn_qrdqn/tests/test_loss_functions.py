try:
    import debug_settings
except:
    pass


import unittest
import torch
import os
import numpy as np
import gym
from gym import spaces
import matplotlib
import time
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import \
  DiscreteHighwayBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import FQFAgent, IQNAgent
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model import IQN
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.network import initialize_weights_he
from bark_ml.observers.nearest_state_observer import NearestAgentsObserver
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils import calculate_expert_loss,\
    calculate_supervised_classification_quantile_loss, calculate_huber_loss, \
    evaluate_quantile_at_action, get_margin_loss, update_params


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

class TestDQN(nn.Module):

  def __init__(self, num_channels, hidden=4, embedding_dim=1):
    super(TestDQN, self).__init__()

    self.net = nn.Sequential(
        torch.nn.Linear(num_channels, hidden),
        torch.nn.Linear(hidden, embedding_dim),
    ).apply(initialize_weights_he)

    self.embedding_dim = embedding_dim

  def forward(self, states):
    batch_size = states.shape[0]
    # Calculate embeddings of states.
    state_embedding = self.net(states)
    assert state_embedding.shape == (batch_size, self.embedding_dim)

    return state_embedding


class TestCosineEmbeddingNet(nn.Module):

  def __init__(self, num_cosines=4, embedding_dim=1, noisy_net=False):
    super(TestCosineEmbeddingNet, self).__init__()
    linear = nn.Linear

    self.net = nn.Sequential(linear(num_cosines, embedding_dim), nn.ReLU())
    self.num_cosines = num_cosines
    self.embedding_dim = embedding_dim

  def forward(self, taus):
    batch_size = taus.shape[0]
    N = taus.shape[1]

    # Calculate i * \pi (i=1,...,N).
    i_pi = np.pi * torch.arange(start=1,
                                end=self.num_cosines + 1,
                                dtype=taus.dtype,
                                device=taus.device).view(
                                    1, 1, self.num_cosines)

    # Calculate cos(i * \pi * \tau).
    cosines = torch.cos(taus.view(batch_size, N, 1) * i_pi).view(
        batch_size * N, self.num_cosines)

    # Calculate embeddings of taus.
    tau_embeddings = self.net(cosines).view(batch_size, N, self.embedding_dim)

    return tau_embeddings


class TestQuantileNet(nn.Module):

  def __init__(self, num_actions, embedding_dim=1, noisy_net=False):
    super(TestQuantileNet, self).__init__()
    linear = nn.Linear

    self.net = nn.Sequential(
        linear(embedding_dim, 4),
        nn.ReLU(),
        linear(4, num_actions),
    )
    self.num_actions = num_actions
    self.embedding_dim = embedding_dim
    self.noisy_net = noisy_net

  def forward(self, state_embeddings, tau_embeddings):
    assert state_embeddings.shape[0] == tau_embeddings.shape[0]
    assert state_embeddings.shape[1] == tau_embeddings.shape[2]

    # NOTE: Because variable taus correspond to either \tau or \hat \tau
    # in the paper, N isn't neccesarily the same as fqf.N.
    batch_size = state_embeddings.shape[0]
    N = tau_embeddings.shape[1]

    # Reshape into (batch_size, 1, embedding_dim).
    state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)

    # Calculate embeddings of states and taus.
    embeddings = (state_embeddings * tau_embeddings).view(
        batch_size * N, self.embedding_dim)

    # Calculate quantile values.
    quantiles = self.net(embeddings)

    return quantiles.view(batch_size, N, self.num_actions)


class TestIQN(IQN):

    def __init__(self, num_channels, num_actions, params, num_cosines, noisy_net):
        super(TestIQN, self).__init__(num_channels, num_actions, params, num_cosines, noisy_net)
        self.K = 64
        self.N = 64
        self.N_dash = 64
        self.embedding_dim = 1
        # Feature extractor of DQN.
        self.dqn_net = TestDQN(num_channels=num_channels,
                                embedding_dim=self.embedding_dim,
                                hidden=4)
        # Cosine embedding network.
        self.cosine_net = TestCosineEmbeddingNet(num_cosines=num_cosines,
                                                embedding_dim=self.embedding_dim,
                                                noisy_net=noisy_net)
        # Quantile network.
        self.quantile_net = TestQuantileNet(num_actions=num_actions,
                                            embedding_dim=self.embedding_dim,
                                            noisy_net=noisy_net)


class LossTests(unittest.TestCase):

    def test_quantile_huber_loss(self):
        """Test the quantile huber loss formula that switches the
        resulting error between mean square error and abs error
        For more information on Huber Loss:
        """
        td_errors = torch.zeros((1, 2, 1))
        td_errors[:, 0, :] = 0.0
        td_errors[:, 1, :] = 2.0
        taus = torch.rand(1, 2)
        kappa = 1.0
        quantile_huber_loss = calculate_huber_loss(td_errors, kappa=kappa).squeeze()
        assert quantile_huber_loss[0] == 0.0
        assert quantile_huber_loss[1] == kappa * (td_errors[0, 1, 0] - 0.5 * kappa)

    def test_supervised_margin_loss_calculation(self):
        """Sanity test for checking calculation accuracy of the
        Supervised large margin loss
        """
        expert_margin = 0.8
        supervised_loss_weight = 0.5
        num_actions = 2
        batch_size = 1
        state_size = 1
        params = ParameterServer()
        states = torch.rand((batch_size, state_size))
        actions = torch.zeros((batch_size, 1), dtype=torch.int64)
        # random assigment of actions
        actions[states >= 0.5] = 1.0
        is_demos = torch.zeros((batch_size, 1))
        is_demos[(actions.squeeze() == 1.0).nonzero()] = 1.0
        state_shape = spaces.Box(low=np.zeros(state_size), high=np.zeros(state_size))
        test_iqn = TestIQN(num_channels=state_shape.shape[0], num_actions=num_actions, params=params, 
                           num_cosines=4, noisy_net=False)
        taus = torch.rand(batch_size, test_iqn.N)
        state_embeddings = test_iqn.dqn_net(states)

        # get the calculated classification loss
        supervised_classification_loss = calculate_supervised_classification_quantile_loss(actions,
          states, test_iqn, taus, state_embeddings, is_demos, 
          num_actions, 'cpu', supervised_loss_weight, expert_margin)
        
        # individually calculate the margin loss for batch
        resampled_batch_margin_loss = get_margin_loss(actions, num_actions, is_demos, expert_margin, 'cpu')
        # recalculate quantiles to obtain Q values
        recalculated_quantiles = test_iqn.calculate_quantiles(taus, state_embeddings=state_embeddings)
        # calculate the q values to get loss
        recalculated_q = recalculated_quantiles.mean(dim=1)
        # get recalculated loss
        recalculated_loss = calculate_expert_loss(recalculated_q, resampled_batch_margin_loss, is_demos, 
                                                  actions, supervised_loss_weight * is_demos.squeeze())
        assert recalculated_loss.mean () == supervised_classification_loss

    def test_supervised_margin_loss_zero_states(self):
        """Test the supervise margin loss with states are 0
        This is a sanity test to assess basic function"""
        params = ParameterServer()
        states = torch.zeros((1, 4))
        state_shape = spaces.Box(low=np.zeros(4), high=np.zeros(4))
        test_iqn = TestIQN(num_channels=state_shape.shape[0], num_actions=2, params=params, 
                           num_cosines=4, noisy_net=False)
        state_embeddings = test_iqn.dqn_net(states)
        assert(torch.all(state_embeddings == 0.0))
    
    def test_supervised_margin_loss_states(self):
        """Test convergence of ONLY the supervised loss
        This test is for iteratively ensuring convergence of the supervised loss.
        Here, a network is trained with random states and actions iteratively,
        to check for convergence in limited steps"""
        num_actions = 2
        params = ParameterServer()
        batch_size = 512
        state_size = 1
        state_shape = spaces.Box(low=np.zeros(state_size), high=np.zeros(state_size))
        online_iqn = TestIQN(num_channels=state_shape.shape[0], num_actions=num_actions, params=params, 
                           num_cosines=4, noisy_net=False)
        optim = Adam(online_iqn.parameters(),
                       lr=5.5e-3,
                       eps=1e-2 / batch_size)
        online_iqn.train()
        states = torch.rand((batch_size, state_size))
        actions = torch.zeros((batch_size, 1), dtype=torch.int64)
        actions[states >= 0.5] = 1.0
        online_iqn.sample_noise()
        # Create a loss variable in torch for gradient propagation
        loss = Variable(requires_grad=True)

        is_demos = torch.zeros((batch_size, 1))
        is_demos[(actions.squeeze() == 1.0).nonzero()] = 1.0
        # iterate for subsequent online network training
        for i in range(100):
          is_demos[(actions.squeeze() == 1.0).nonzero()] = 1.0
          next_states = torch.rand((batch_size, state_size))
          state_embeddings = online_iqn.dqn_net(states)

          # sample tau random quantiles from online network
          taus = torch.rand(batch_size, 4)
          online_iqn.sample_noise()
          # get next q values to determine next actions
          next_q = online_iqn.calculate_q(states=next_states)
          next_actions = torch.argmax(next_q, dim=1, keepdim=True)
          # use online network, states, next states and actions to get the supervised loss
          supervised_classification_loss = calculate_supervised_classification_quantile_loss(
            actions, states, online_iqn, taus, state_embeddings, is_demos,
            num_actions, 'cpu', 0.5, 0.8)
          # propagate only the supervised margin loss
          loss = supervised_classification_loss
          gradients = update_params(optim, loss, [online_iqn], retain_graph=True, count=i)
          # reassign for the next iteration
          states = next_states
          actions = next_actions
        assert loss == 0.0
        
        
if __name__ == "__main__":
    unittest.main()