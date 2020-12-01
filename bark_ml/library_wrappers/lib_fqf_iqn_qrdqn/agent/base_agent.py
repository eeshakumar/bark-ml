# Copyright (c) 2020 fortiss GmbH
#
# Authors: Patrick Hart, Julian Bernhard, Klemens Esterle,
# Tobias Kessler and Mansoor Nasir
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# The code is adapted from opensource implementation - https://github.com/ku2482/fqf-iqn-qrdqn.pytorch
# MIT License - Copyright (c) 2020 Toshiki Watanabe

import os
import logging
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import pickle
import os
from abc import ABC, abstractmethod

# BARK-ML imports
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils \
  import RunningMeanStats, LinearAnneaer
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.memory \
  import LazyMultiStepMemory, LazyPrioritizedDemMultiStepMemory, LazyPrioritizedMultiStepMemory
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

# TODO: Imports to remove
from bark.runtime.commons.parameters import ParameterServer
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.\
            tests.test_demo_behavior import TestDemoBehavior
from bark_ml.evaluators.evaluator import StateEvaluator


# BARK imports
from bark.core.models.behavior import BehaviorModel

def to_pickle(obj, dir, file):
  path = os.path.join(dir, file)
  with open(path, 'wb') as handle:
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def from_pickle(dir, file):
  path = os.path.join(dir, file)
  with open(path, 'rb') as handle:
    obj = pickle.load(handle)
  return obj

class TrainingBenchmark:
  def __init__(self):
    self.training_env = None
    self.num_episodes = None
    self.max_episode_steps = None
    self.agent = None

  def reset(self, training_env, num_eval_steps, max_episode_steps, agent):
    self.training_env = training_env
    self.num_eval_steps = num_eval_steps
    self.max_episode_steps = max_episode_steps
    self.agent = agent

  def run(self):
    # returns dict with evaluated metrics
    num_episodes = 0
    num_steps = 0
    total_return = 0.0

    while True:
      state = self.training_env.reset()
      episode_steps = 0
      episode_return = 0.0
      done = False
      while (not done) and episode_steps <= self.max_episode_steps:
        if self.agent.is_random(eval=True):
          action = self.agent.explore()
        else:
          action = self.agent.Act(state)

        next_state, reward, done, _ = self.training_env.step(action)
        num_steps += 1
        episode_steps += 1
        episode_return += reward
        state = next_state

      num_episodes += 1
      total_return += episode_return

      if num_steps > self.num_eval_steps:
        break

    mean_return = total_return / num_episodes
    return {"mean_return" : mean_return}, f"Mean return: {mean_return}"

  def is_better(self, eval_result1, than_eval_result2):
    return eval_result1["mean_return"] > than_eval_result2["mean_return"]




class BaseAgent(BehaviorModel):
  def __init__(self, agent_save_dir=None, env=None, params=None, training_benchmark=None, checkpoint_load=None,
               demo_collector = None):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._env = env
    self._training_benchmark = training_benchmark or TrainingBenchmark()
    self._agent_save_dir = agent_save_dir
    if demo_collector is not None:
      self.demonstrator = demo_collector
    else:
      self.demonstrator = None

    if not checkpoint_load and params:
      if not env:
        raise ValueError("Environment must be passed for initialization")
      self.reset_params(self._params)
      self.reset_action_observer(env)
      self.init_always()
      self.reset_training_variables()
    elif checkpoint_load:
      self.load_pickable_members(agent_save_dir)
      self.init_always()
      self.load_models(BaseAgent.check_point_directory(agent_save_dir, checkpoint_load) \
                    if checkpoint_load=="best" else BaseAgent.check_point_directory(agent_save_dir, checkpoint_load) )
    else:
      raise ValueError("Unusual param combination for agent initialization.")


  def init_always(self):
    self.device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")

    self.writer = SummaryWriter(log_dir=BaseAgent.summary_dir(self.agent_save_dir))
    self.train_return = RunningMeanStats(self.summary_log_interval)

    if not os.path.exists(BaseAgent.summary_dir(self.agent_save_dir)):
      os.makedirs(BaseAgent.summary_dir(self.agent_save_dir))

    # NOTE: by default we do not want the action to be set externally
    #       as this enables the agents to be plug and played in BARK.
    self._set_action_externally = False
    self._training_benchmark.reset(self._env, \
        self.num_eval_steps, self.max_episode_steps, self)

  def reset_action_observer(self, env):
    self._observer = self._env._observer
    self._ml_behavior = self._env._ml_behavior

  def clean_pickables(self, pickables):
    del pickables["online_net"]
    del pickables["target_net"]
    del pickables["_env"]
    del pickables["_training_benchmark"]
    del pickables["device"]
    del pickables["writer"]


  def save_pickable_members(self, pickable_dir):
    if not os.path.exists(pickable_dir):
      os.makedirs(pickable_dir)
    pickables = dict(self.__dict__)
    self.clean_pickables(pickables)
    to_pickle(pickables, pickable_dir, "agent_pickables")

  def load_pickable_members(self, agent_save_dir):
    pickables = from_pickle(BaseAgent.pickable_directory(agent_save_dir), "agent_pickables")
    self.__dict__.update(pickables)

  def reset_training_variables(self):
    # Replay memory which is memory-efficient to store stacked frames.
    if self.demonstrator is None:
      if self.use_per:
        beta_steps = (self.num_steps - self.start_steps) / \
              self.update_interval
        self.memory = LazyPrioritizedMultiStepMemory(
            self.memory_size,
            self.observer.observation_space.shape,
            self.device,
            self.gamma,
            self.multi_step,
            beta_steps=beta_steps)
      else:
        self.memory = LazyMultiStepMemory(
            self.memory_size,
            self.observer.observation_space.shape,
            self.device,
            self.gamma,
            self.multi_step)
    else:
      # expect a learning from demonstrations setting, reset use_per to true
      self.use_per = True
      beta_steps = (self.num_steps - self.start_steps) / \
        self.update_interval
      self.memory = LazyPrioritizedDemMultiStepMemory(
            self.memory_size,
            self.observer.observation_space.shape,
            self.device,
            self.gamma,
            self.multi_step,
            beta_steps=beta_steps,
            eps=self.epsilon_train,
            epsilon_demo=self.demonstrator_buffer_params["epsilon_demo", "", 1.0],
            epsilon_alpha=self.demonstrator_buffer_params["epsilon_alpha", "", 0.001],
            alpha=self.demonstrator_buffer_params["alpha", "", 0.4],
            per_beta_steps=self.demonstrator_buffer_params["per_beta_steps", "", 75000],
            per_beta=self.demonstrator_buffer_params["per_beta", "", 0.6],
            demo_ratio=self.demonstrator_buffer_params["demo_capacity_ratio", "", 0.25])

    self.steps = 0
    self.learning_steps = 0
    self.episodes = 0
    self.best_eval_results = None

  def reset_params(self, params):
    self.num_steps = params["ML"]["BaseAgent"]["NumSteps", "", 5000000]
    self.batch_size = params["ML"]["BaseAgent"]["BatchSize", "", 32]

    self.double_q_learning = params["ML"]["BaseAgent"]["Double_q_learning", "", False]
    self.dueling_net = params["ML"]["BaseAgent"]["DuelingNet", "", False]
    self.noisy_net = params["ML"]["BaseAgent"]["NoisyNet", "", False]
    self.use_per = params["ML"]["BaseAgent"]["Use_per", "", False]

    self.reward_log_interval = params["ML"]["BaseAgent"]["RewardLogInterval", "", 5]
    self.summary_log_interval = params["ML"]["BaseAgent"]["SummaryLogInterval", "", 100]
    self.eval_interval = params["ML"]["BaseAgent"]["EvalInterval", "",
                                                         25000]
    self.num_eval_steps = params["ML"]["BaseAgent"]["NumEvalSteps", "",
                                                          12500]
    self.gamma_n = params["ML"]["BaseAgent"]["Gamma", "", 0.99] ** \
        params["ML"]["BaseAgent"]["Multi_step", "", 1]

    self.start_steps = params["ML"]["BaseAgent"]["StartSteps", "", 5000]
    self.epsilon_train = LinearAnneaer(
        1.0, params["ML"]["BaseAgent"]["EpsilonTrain", "", 0.01],
        params["ML"]["BaseAgent"]["EpsilonDecaySteps", "", 25000])
    self.epsilon_eval = params["ML"]["BaseAgent"]["EpsilonEval", "",
                                                        0.001]
    self.update_interval = params["ML"]["BaseAgent"]["Update_interval", "", 4]
    self.target_update_interval = params["ML"]["BaseAgent"]["TargetUpdateInterval", "", 5000]
    self.max_episode_steps = params["ML"]["BaseAgent"]["MaxEpisodeSteps",  "", 10000]
    self.grad_cliping = params["ML"]["BaseAgent"]["GradCliping", "", 5.0]

    self.memory_size = params["ML"]["BaseAgent"]["MemorySize", "", 10**6]
    self.gamma = params["ML"]["BaseAgent"]["Gamma", "", 0.99]
    self.multi_step = params["ML"]["BaseAgent"]["Multi_step", "", 1]

    self.use_cuda = params["ML"]["BaseAgent"]["Cuda", "", False]

    if self.demonstrator:
      self.demonstrator_buffer_params = params.AddChild("ML").AddChild("DemonstratorAgent").AddChild("Buffer")
      self.demonstrator_loss_params = params.AddChild("ML").AddChild("DemonstratorAgent").AddChild("Loss")
      self.demonstrator_agent_params = params.AddChild("ML").AddChild("DemonstratorAgent").AddChild("Agent")

  @property
  def observer(self):
      return self._observer

  @property
  def env(self):
    return self._env

  @property
  def ml_behavior(self):
    return self._ml_behavior

  @property
  def num_actions(self):
    return self.ml_behavior.action_space.n

  @property
  def agent_save_dir(self):
    return self._agent_save_dir

  def run(self):
    assert self.demonstrator is not None, "Run invoked incorrectly, demonstrator not found!"

    # TODO: Remove this evaluator
    class TestEvaluator(StateEvaluator):
      reach_goal = True
      def __init__(self,
                  params=ParameterServer()):
        StateEvaluator.__init__(self, params)
        self.step = 0

      def _evaluate(self, observed_world, eval_results, action):
        """Returns information about the current world state
        """
        done = False
        reward = 0.0
        info = {"goal_r1" : False}
        if self.step > 2:
          done = True
          if self.reach_goal:
            reward = 0.1
            info = {"goal_r1" : True}
        self.step += 1
        return reward, done, info

      def Reset(self, world):
        self._step = 0
        #every second scenario goal is not reached
        TestEvaluator.reach_goal = not TestEvaluator.reach_goal

    demonstrations_save_path = os.path.join(self.agent_save_dir, self.demonstrator_agent_params[
      "save_demo", "", "demonstrations"])
    print("Demonstrations will be saved to", demonstrations_save_path)

    def default_training_evaluators():
      default_config = {"success" : "EvaluatorGoalReached", "collision_other" : "EvaluatorCollisionEgoAgent",
          "out_of_drivable" : "EvaluatorDrivableArea", "max_steps": "EvaluatorStepCount"}
      return default_config

    #TODO: Demonstrator behavior must be updated/passed
    demo_behavior = TestDemoBehavior(self._params)
    #TODO: Evaluator must not be a TestEvaluator
    real_evaluator = self._env._evaluator
    self._env._evaluator = TestEvaluator()
    collection_result = self.demonstrator.CollectDemonstrations(
      self._env, demo_behavior,
      self.demonstrator_agent_params["num_demo_episodes", "", 3000],
      demonstrations_save_path,
      use_mp_runner=False,
      runner_init_params={"deepcopy": False})
    #TODO: evaluator criteria must be updated
    demonstrations = self.demonstrator.ProcessCollectionResult(
      eval_criteria = {"goal_r1": lambda x: x})
    print(type(demonstrations), demonstrations)
    # TODO: Remove this reassignment
    self._env._evaluator = real_evaluator

    # Extract and append demonstrations to memory
    for demo in demonstrations:
      (state, action, reward, next_state, done, is_demo) = demo
      self.memory.append(state, action, reward, next_state, done, is_demo)

    assert self.memory._n == len(demonstrations)
    assert self.memory._dn == len(demonstrations)

    self.online_gradient_update_steps = self.demonstrator_agent_params["online_gradient_update_steps", "", 75000]
    self.train_on_demonstrations()

    self.save()

  def train_on_demonstrations(self):
    while True:
      self.train_step_interval()
      if self.steps > self.online_gradient_update_steps:
        print("Initial gradient updates completed. Totoal episodes", self.episodes)
        break
    self.set_action_externally = True

  def train(self):
    while True:
      self.train_episode()
      if self.steps > self.num_steps:
        break
    self.set_action_externally = True

  def is_update(self):
    return self.steps % self.update_interval == 0 \
        and self.steps >= self.start_steps

  def is_random(self, eval=False):
    # Use e-greedy for evaluation.
    if self.steps < self.start_steps:
      return True
    if eval:
      return np.random.rand() < self.epsilon_eval
    if self.noisy_net:
      return False
    return np.random.rand() < self.epsilon_train.get()

  def update_target(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  def explore(self):
    # Act with randomness.
    action = self.ml_behavior.action_space.sample()
    return action

  @property
  def set_action_externally(self):
    return self.set_action_externally

  @set_action_externally.setter
  def set_action_externally(self, externally):
    self.set_action_externally = externally

  def ActionToBehavior(self, action):
    # NOTE: will either be set externally or internally
    self._action = action

  def Act(self, state):
    # Act without randomness.
    # state = torch.Tensor(state).unsqueeze(0).to(self._device).float()
    actions = self.calculate_actions(state).argmax().item()
    return actions

  def calculate_actions(self, state):
    # Act without randomness.
    state = torch.Tensor(state).unsqueeze(0).to(self.device).float()
    with torch.no_grad():
      actions = self.online_net(states=state)  # pylint: disable=not-callable
    return actions

  def Plan(self, dt, observed_world):
    # NOTE: if training is enabled the action is set externally
    if not self.set_action_externally:
      observed_state = self.observer.Observe(observed_world)
      action = self.Act(observed_state)
      self._action = action

    action = self._action
    # set action to be executed
    self._ml_behavior.ActionToBehavior(action)
    trajectory = self._ml_behavior.Plan(dt, observed_world)
    dynamic_action = self._ml_behavior.GetLastAction()
    # NOTE: BARK requires models to have trajectories of the past
    BehaviorModel.SetLastTrajectory(self, trajectory)
    BehaviorModel.SetLastAction(self, dynamic_action)
    return trajectory

  def learn(self):
    pass

  def Clone(self):
    return self

  @property
  def action_space(self):
    return self._ml_behavior.action_space

  @staticmethod
  def check_point_directory(agent_save_dir, checkpoint_type):
    return os.path.join(agent_save_dir, "checkpoints/", checkpoint_type)

  @staticmethod
  def pickable_directory(agent_save_dir):
    return os.path.join(agent_save_dir, "pickable/")

  @staticmethod
  def summary_dir(agent_save_dir):
    return os.path.join(agent_save_dir, "summaries")

  def save_models(self, checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    torch.save(self.online_net.state_dict(),
               os.path.join(checkpoint_dir, 'online_net.pth'))
    torch.save(self.target_net.state_dict(),
               os.path.join(checkpoint_dir, 'target_net.pth'))
    online_net_script = torch.jit.script(self.online_net)
    online_net_script.save(os.path.join(checkpoint_dir, 'online_net_script.pt'))

  def save(self, checkpoint_type="last"):
    self.save_models(BaseAgent.check_point_directory(self.agent_save_dir, checkpoint_type))
    self.save_pickable_members(BaseAgent.pickable_directory(self.agent_save_dir))

  def load_models(self, checkpoint_dir):
    try:
      self.online_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'online_net.pth')))
    except RuntimeError:
      self.online_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'online_net.pth'), map_location=torch.device('cpu')))
    try:
      self.target_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'target_net.pth')))
    except RuntimeError:
      self.target_net.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, 'target_net.pth'), map_location=torch.device('cpu')))

  def visualize(self, num_episodes=5):
    if not self.env:
      raise ValueError("No environment available for visualization. Was agent reloaded?")
    for _ in range(0, num_episodes):
      state = self._env.reset()
      done = False
      while (not done):
        action = self.Act(state)
        next_state, reward, done, _ = self._env.step(action)
        self._env.render()
        state = next_state

  def train_episode(self):
    self.online_net.train()
    self.target_net.train()

    self.episodes += 1
    episode_return = 0.
    episode_steps = 0

    done = False
    state = self._env.reset()

    while (not done) and episode_steps <= self.max_episode_steps:
      # NOTE: Noises can be sampled only after self._learn(). However, I
      # sample noises before every action, which seems to lead better
      # performances.
      self.online_net.sample_noise()

      if self.is_random(eval=False):
        action = self.explore()
      else:
        action = self.Act(state)

      next_state, reward, done, _ = self._env.step(action)
      if self.episodes % self.reward_log_interval == 0:
        # self._env.render()
        logging.info(f"Reward: {reward:<4}")

      # To calculate efficiently, I just set priority=max_priority here.
      self.memory.append(state, action, reward, next_state, done)

      self.steps += 1
      episode_steps += 1
      episode_return += reward
      state = next_state

      self.train_step_interval()

    # We log running mean of stats.
    self.train_return.append(episode_return)

    # We log evaluation results along with training frames = 4 * steps.
    if self.episodes % self.summary_log_interval == 0:
      self.writer.add_scalar('return/train', self.train_return.get(),
                              4 * self.steps)

    logging.info(f'Episode: {self.episodes:<4}  '
                 f'episode steps: {episode_steps:<4}  '
                 f'return: {episode_return:<5.1f}')

  def train_step_interval(self):
    self.epsilon_train.step()
    if self.demonstrator is not None:
      self.memory.per_beta.step()

    if self.steps % self.target_update_interval == 0:
      self.update_target()

    if self.is_update():
      self.learn()

    if self.steps % self.eval_interval == 0:
      self.evaluate()
      self.save(checkpoint_type='last')
      self.online_net.train()

  def evaluate(self):
    if not self._training_benchmark:
      logging.info("No evaluation performed since no training benchmark available.")
    self.online_net.eval()

    eval_results, formatted_result = self._training_benchmark.run()

    if not self.best_eval_results or \
        self._training_benchmark.is_better(eval_results, self.best_eval_results):
      self.best_eval_results = eval_results
      self.save(checkpoint_type='best')

    # We log evaluation results along with training frames = 4 * steps.
    for eval_result_name, eval_result in eval_results.items():
      self.writer.add_scalar(eval_result_name, eval_result, 4 * self.steps)
    logging.info('-' * 60)
    logging.info('Evaluation result: {}'.format(formatted_result))
    logging.info('-' * 60)

  def __del__(self):
    self.writer.close()
