# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import unittest
import numpy as np
import os
import gym
import matplotlib
import time

# BARK imports
from bark.runtime.commons.parameters import ParameterServer

# BARK-ML imports
from bark_ml.environments.blueprints import \
  DiscreteHighwayBlueprint, DiscreteMergingBlueprint
from bark_ml.environments.single_agent_runtime import SingleAgentRuntime
import bark_ml.environments.gym
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.model_wrapper \
 import pytorch_script_wrapper
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.agent import IQNAgent, FQFAgent, QRDQNAgent


class PyLibraryWrappersPyTorchAgentTests(unittest.TestCase):
  # make sure the agent works
  def test_agent_wrapping(self):
    params = ParameterServer()
    env = gym.make("highway-v1", params=params)
    env.reset()
    agent = IQNAgent(env=env, test_env=env, params=params)
    agent = FQFAgent(env=env, test_env=env, params=params)
    agent = QRDQNAgent(env=env, test_env=env, params=params)

  # assign as behavior model (to check if trained agent can be used)
  def test_behavior_wrapping(self):
    # create scenario
    params = ParameterServer()
    bp = DiscreteHighwayBlueprint(params, number_of_senarios=10, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)
    #env = gym.make("highway-v1", params=params)
    ml_behaviors = []
    ml_behaviors.append(IQNAgent(env=env, test_env=env, params=params))
    ml_behaviors.append(FQFAgent(env=env, test_env=env, params=params))
    ml_behaviors.append(QRDQNAgent(env=env, test_env=env, params=params))

    for ml_behavior in ml_behaviors:
      # set agent
      env.ml_behavior = ml_behavior
      env.reset()
      done = False
      while done is False:
        action = np.random.randint(low=0, high=env.action_space.n)
        observed_next_state, reward, done, info = env.step(action)
        print(
            f"Observed state: {observed_next_state}, Reward: {reward}, Done: {done}"
        )

      # action is set externally
      ml_behavior._set_action_externally = True
      agent_id = list(env._world.agents.keys())[0]
      observed_world = env._world.Observe([agent_id])[0]

      # do a random action and plan trajectory
      action = np.random.randint(low=1, high=env.action_space.n)
      ml_behavior.ActionToBehavior(action)
      a = ml_behavior.Plan(0.2, observed_world)

      # sample another different random action
      another_action = action
      while another_action == action:
        another_action = np.random.randint(low=1, high=env.action_space.n)

      # plan trajectory for the another action
      ml_behavior.ActionToBehavior(another_action)
      b = ml_behavior.Plan(0.2, observed_world)

      # the trajectory generated by two different actions shoould be different
      self.assertEqual(np.any(np.not_equal(a, b)), True)

      # action will be calculated within the Plan(..) fct.
      ml_behavior._set_action_externally = False
      a = ml_behavior.Plan(0.2, observed_world)
      b = ml_behavior.Plan(0.2, observed_world)

      # same trajectory for same state
      np.testing.assert_array_equal(a, b)

  def test_agents(self):
    params = ParameterServer()
    params["ML"]["BaseAgent"]["NumSteps"] = 2
    params["ML"]["BaseAgent"]["MaxEpisodeSteps"] = 2

    bp = DiscreteHighwayBlueprint(params, number_of_senarios=10, random_seed=0)
    env = SingleAgentRuntime(blueprint=bp, render=False)

    # IQN Agent
    iqn_agent = IQNAgent(env=env, test_env=env, params=params)
    env.ml_behavior = iqn_agent
    self.assertEqual(env.ml_behavior.set_action_externally, False)
    iqn_agent.train()
    self.assertEqual(env.ml_behavior.set_action_externally, True)

    # FQF Agent
    fqf_agent = FQFAgent(env=env, test_env=env, params=params)
    env.ml_behavior = fqf_agent
    self.assertEqual(env.ml_behavior.set_action_externally, False)
    fqf_agent.train()
    self.assertEqual(env.ml_behavior.set_action_externally, True)

    # QRDQN Agent
    qrdqn_agent = QRDQNAgent(env=env, test_env=env, params=params)
    env.ml_behavior = qrdqn_agent
    self.assertEqual(env.ml_behavior.set_action_externally, False)
    qrdqn_agent.train()
    self.assertEqual(env.ml_behavior.set_action_externally, True)

  def test_model_loader(self):
    # env using default params
    env = gym.make("highway-v1")

    networks = ["iqn", "fqf", "qrdqn"]

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.shape[0]

    # a sample random state [0-1] to evaluate actions
    random_state = np.random.rand(state_space_size).tolist()

    # test all networks
    for network in networks:
      # Do inference using C++ wrapped model
      model = pytorch_script_wrapper.ModelLoader(
          os.path.join(
              os.path.dirname(__file__),
              "lib_fqf_iqn_qrdqn_test_data/{}/online_net_script.pt"
              .format(network)), action_space_size, state_space_size)
      model.LoadModel()

      num_iters = 1000  # Number of times to repeat experiment to calcualte runtime

      # Time num_iters iterations for inference using C++ model
      start = time.time()
      for _ in range(num_iters):
        actions_cpp = model.Inference(random_state)
      end = time.time()
      time_cpp = end - start  # todo - how to analyze python vs cpp test time in tests?

      # Load and perform inference using python model
      if network == "iqn":
        agent = IQNAgent(env=env, test_env=env, params=ParameterServer())

      elif network == "fqf":
        agent = FQFAgent(env=env, test_env=env, params=ParameterServer())

      elif network == "qrdqn":
        agent = QRDQNAgent(env=env, test_env=env, params=ParameterServer())

      agent.load_models(
          os.path.join(
              os.path.dirname(__file__),
              "lib_fqf_iqn_qrdqn_test_data",
              network))

      # Time num_iters iterations for inference using python model
      start = time.time()
      for _ in range(num_iters):
        actions_py = agent.calculate_actions(random_state)

      end = time.time()
      time_py = end - start

      # assert that Python and Cpp models are close enough to 6 decimal places
      np.testing.assert_array_almost_equal(
          actions_py.flatten().numpy(),
          np.asarray(actions_cpp),
          decimal=6,
          err_msg="C++ and python models don't match")


if __name__ == '__main__':
  unittest.main()
