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
  import LazyMultiStepMemory, LazyPrioritizedMultiStepMemory
from bark_ml.behaviors.discrete_behavior import BehaviorDiscreteMacroActionsML

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

  def reset(self, training_env, num_episodes, max_episode_steps, agent):
    self.training_env = training_env
    self.num_episodes = num_episodes
    self.max_episode_steps = max_episode_steps
    self.agent = agent

  def run(self):
    # returns dict with evaluated metrics
    num_episodes = 0
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
        episode_steps += 1
        episode_return += reward
        state = next_state

      num_episodes += 1
      total_return += episode_return

      if num_episodes > self.num_episodes:
        break

    mean_return = total_return / num_episodes
    return {"mean_return" : mean_return}, f"Mean return: {mean_return}"

  def is_better(self, eval_result1, than_eval_result2):
    return eval_result1["mean_return"] > than_eval_result2["mean_return"]




class BaseAgent(BehaviorModel):
  def __init__(self, env=None, params=None, training_benchmark=None, save_dir=None ):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._env = env
    self._training_benchmark = training_benchmark or TrainingBenchmark()
    
    if not save_dir:
      if not env:
        raise ValueError("Environment must be passed for initialization")
      self.reset_params(self._params["ML"]["BaseAgent"])
      self.reset_action_observer(env)
      self.init_always()
      self.reset_training_variables()
    else:
      self.load_pickable_members(save_dir)
      self.init_always()
      self.load_models(save_dir)


  def init_always(self):
    self.device = torch.device("cuda" if (self.use_cuda and torch.cuda.is_available()) else "cpu")

    self.writer = SummaryWriter(log_dir=self.summary_dir)
    self.train_return = RunningMeanStats(self.summary_log_interval)

    if not os.path.exists(self._model_dir) and self._model_dir:
      os.makedirs(self._model_dir)
    if not os.path.exists(self._summary_dir) and self._summary_dir:
      os.makedirs(self._summary_dir)

    # NOTE: by default we do not want the action to be set externally
    #       as this enables the agents to be plug and played in BARK.
    self._set_action_externally = False
    self._training_benchmark.reset(self._env, \
        self.num_eval_episodes, self.max_episode_steps, self)

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

  def save_pickable_members(self, save_dir):
    pickables = dict(self.__dict__)
    self.clean_pickables(pickables)
    to_pickle(pickables, save_dir, "base_agent_pickables")

  def load_pickable_members(self, save_dir):
    pickables = from_pickle(save_dir, "base_agent_pickables")
    self.__dict__.update(pickables)

  def reset_training_variables(self):
    # Replay memory which is memory-efficient to store stacked frames.
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

    self.steps = 0
    self.learning_steps = 0
    self.episodes = 0
    self.best_eval_results = None

  def reset_params(self, params):
    self.num_steps = params["NumSteps", "", 5000000]
    self.batch_size = params["BatchSize", "", 32]

    self.double_q_learning = params["Double_q_learning", "", False]
    self.dueling_net = params["DuelingNet", "", False]
    self.noisy_net = params["NoisyNet", "", False]
    self.use_per = params["Use_per", "", False]

    self.reward_log_interval = params["RewardLogInterval", "", 5]
    self.summary_log_interval = params["SummaryLogInterval", "", 100]
    self.eval_interval = params["EvalInterval", "",
                                                         25000]
    self.num_eval_episodes = params["NumEvalEpisodes", "",
                                                          12500]
    self.gamma_n = params["Gamma", "", 0.99] ** \
        params["Multi_step", "", 1]

    self.start_steps = params["StartSteps", "", 5000]
    self.epsilon_train = LinearAnneaer(
        1.0, params["EpsilonTrain", "", 0.01],
        params["EpsilonDecaySteps", "", 25000])
    self.epsilon_eval = params["EpsilonEval", "",
                                                        0.001]
    self.update_interval = params["Update_interval", "", 4]
    self.target_update_interval = params["TargetUpdateInterval", "", 5000]
    self.max_episode_steps = params["MaxEpisodeSteps",  "", 10000]
    self.grad_cliping = params["GradCliping", "", 5.0]

    self.summary_dir = params["SummaryPath", "", ""]
    self.model_dir = params["CheckpointPath", "", ""]

    self.memory_size = params["MemorySize", "", 10**6]
    self.gamma = params["Gamma", "", 0.99]
    self.multi_step = params["Multi_step", "", 1]

    self.use_cuda = params["Cuda", "", False] 

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

  def train(self):
    while True:
      self.train_episode()
      if self.steps > self._num_steps:
        break
    self._set_action_externally = True

  def is_update(self):
    return self.steps % self._update_interval == 0 \
        and self.steps >= self._start_steps

  def is_random(self, eval=False):
    # Use e-greedy for evaluation.
    if self.steps < self._start_steps:
      return True
    if eval:
      return np.random.rand() < self._epsilon_eval
    if self._noisy_net:
      return False
    return np.random.rand() < self._epsilon_train.get()

  def update_target(self):
    self._target_net.load_state_dict(self._online_net.state_dict())

  def explore(self):
    # Act with randomness.
    action = self.ml_behavior.action_space.sample()
    return action

  @property
  def set_action_externally(self):
    return self._set_action_externally

  @set_action_externally.setter
  def set_action_externally(self, externally):
    self._set_action_externally = externally

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
    state = torch.Tensor(state).unsqueeze(0).to(self._device).float()
    with torch.no_grad():
      actions = self._online_net(states=state)  # pylint: disable=not-callable
    return actions

  def Plan(self, dt, observed_world):
    # NOTE: if training is enabled the action is set externally
    if not self._set_action_externally:
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

  def save_models(self, save_dir):
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    torch.save(self._online_net.state_dict(),
               os.path.join(save_dir, 'online_net.pth'))
    torch.save(self._target_net.state_dict(),
               os.path.join(save_dir, 'target_net.pth'))
    # online_net_script = torch.jit.script(self._online_net)
    # online_net_script.save(os.path.join(save_dir, 'online_net_script.pt'))

  def save(self, save_dir):
    self.save_models(save_dir)
    self.save_pickable_members(save_dir)

  def load_models(self, save_dir):
    try: 
      self.online_net.load_state_dict(
        torch.load(os.path.join(save_dir, 'online_net.pth')))
    except RuntimeError:
      self.online_net.load_state_dict(
        torch.load(os.path.join(save_dir, 'online_net.pth'), map_location=torch.device('cpu')))
    try: 
      self.target_net.load_state_dict(
        torch.load(os.path.join(save_dir, 'target_net.pth')))
    except RuntimeError:
      self.target_net.load_state_dict(
        torch.load(os.path.join(save_dir, 'target_net.pth'), map_location=torch.device('cpu')))

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
    self._online_net.train()
    self._target_net.train()

    self.episodes += 1
    episode_return = 0.
    episode_steps = 0

    done = False
    state = self._env.reset()

    while (not done) and episode_steps <= self._max_episode_steps:
      # NOTE: Noises can be sampled only after self._learn(). However, I
      # sample noises before every action, which seems to lead better
      # performances.
      self._online_net.sample_noise()

      if self.is_random(eval=False):
        action = self.explore()
      else:
        action = self.Act(state)

      next_state, reward, done, _ = self._env.step(action)
      if self.episodes % self._reward_log_interval == 0:
        # self._env.render()
        logging.info(f"Reward: {reward:<4}")

      # To calculate efficiently, I just set priority=max_priority here.
      self._memory.append(state, action, reward, next_state, done)

      self.steps += 1
      episode_steps += 1
      episode_return += reward
      state = next_state

      self.train_step_interval()

    # We log running mean of stats.
    self._train_return.append(episode_return)

    # We log evaluation results along with training frames = 4 * steps.
    if self.episodes % self._summary_log_interval == 0:
      self._writer.add_scalar('return/train', self._train_return.get(),
                              4 * self.steps)

    logging.info(f'Episode: {self.episodes:<4}  '
                 f'episode steps: {episode_steps:<4}  '
                 f'return: {episode_return:<5.1f}')

  def train_step_interval(self):
    self._epsilon_train.step()

    if self.steps % self._target_update_interval == 0:
      self.update_target()

    if self.is_update():
      self.learn()

    if self.steps % self._eval_interval == 0:
      self.evaluate()
      self.save_models(os.path.join(self._model_dir, 'final'))
      self._online_net.train()

  def evaluate(self):
    if not self._training_benchmark:
      logging.info("No evaluation performed since no training benchmark available.")
    self.online_net.eval()
    
    eval_results, formatted_result = self._training_benchmark.run()

    if not self.best_eval_results or \
        self._training_benchmark.is_better(eval_results, self.best_eval_results):
      self.best_eval_results = eval_results
      self.save_models(os.path.join(self.model_dir, 'best'))

    # We log evaluation results along with training frames = 4 * steps.
    for eval_result_name, eval_result in eval_results.items():
      self.writer.add_scalar(eval_result_name, eval_result, 4 * self.steps)
    logging.info('-' * 60)
    logging.info('Evaluation result: {}'.format(formatted_result))
    logging.info('-' * 60)

  def __del__(self):
    self._writer.close()
