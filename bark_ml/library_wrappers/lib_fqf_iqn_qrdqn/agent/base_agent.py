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
               is_learn_from_demonstrations=False, is_checkpoint_run=False, is_be_obs=False, is_common_taus=False,
               is_online_demo=False):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._env = env
    self._training_benchmark = training_benchmark or TrainingBenchmark()
    self._agent_save_dir = agent_save_dir
    self.is_learn_from_demonstrations = is_learn_from_demonstrations
    self.is_checkpoint_run = is_checkpoint_run
    self.is_common_taus = is_common_taus
    self.is_online_demo = is_online_demo
    if not checkpoint_load and params:
      if not env:
        raise ValueError("Environment must be passed for initialization")
      self.reset_params(self._params)
      self.reset_action_observer(env)
      self.init_always()
      self.reset_training_variables()
    elif checkpoint_load:
      self.reset_params(self._params)
      if is_learn_from_demonstrations:
        self.reset_action_observer(env)
      if is_online_demo:
        self.load_only_memory_pickle(agent_save_dir)
      else:
        self.load_pickable_members(agent_save_dir)
      self.init_always()
      self.load_models(BaseAgent.check_point_directory(agent_save_dir, checkpoint_load) \
                    if checkpoint_load=="best" else BaseAgent.check_point_directory(agent_save_dir, checkpoint_load) )
      self.reset_training_variables(is_online_demo=is_online_demo)
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

  def load_only_memory_pickle(self, agent_save_dir):
    logging.info("Pickling memory from: " + BaseAgent.pickable_directory(agent_save_dir))
    pickables = from_pickle(BaseAgent.pickable_directory(agent_save_dir), "agent_pickables")
    self.__dict__['memory'] = pickables['memory']
    self._agent_save_dir = agent_save_dir

  def load_pickable_members(self, agent_save_dir):
    logging.info("Pickling agent from: " + BaseAgent.pickable_directory(agent_save_dir))
    pickables = from_pickle(BaseAgent.pickable_directory(agent_save_dir), "agent_pickables")
    self.__dict__.update(pickables)
    self._agent_save_dir = agent_save_dir

  def reset_training_variables(self, is_online_demo=False):
    # Replay memory which is memory-efficient to store stacked frames.
    self.is_online_demo = is_online_demo
    if not self.is_learn_from_demonstrations:
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
      # do not reset memory if already loaded
      if not is_online_demo:
        beta_steps = (self.num_steps - self.start_steps) / \
          self.update_interval
        # initially all memory expects only demo samples
        self.memory = LazyPrioritizedDemMultiStepMemory(
              self.memory_size,
              self.observer.observation_space.shape,
              self.device,
              self.gamma,
              self.multi_step,
              beta_steps=beta_steps,
              epsilon_demo=self.demonstrator_buffer_params["epsilon_demo", "", 1.0],
              epsilon_alpha=self.demonstrator_buffer_params["epsilon_alpha", "", 0.001],
              alpha=self.demonstrator_buffer_params["alpha", "", 0.4],
              per_beta_steps=self.demonstrator_buffer_params["per_beta_steps", "", 75000],
              per_beta=self.demonstrator_buffer_params["per_beta", "", 0.6],
              demo_ratio=self.demonstrator_buffer_params["demo_ratio", "", 1.0])

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

    if self.is_learn_from_demonstrations:
      self.demonstrator_buffer_params = params.AddChild("ML").AddChild("DemonstratorAgent").AddChild("Buffer")
      self.demonstrator_loss_params = params.AddChild("ML").AddChild("DemonstratorAgent").AddChild("Loss")
      self.demonstrator_agent_params = params.AddChild("ML").AddChild("DemonstratorAgent").AddChild("Agent")
      self.online_gradient_update_steps = self.demonstrator_agent_params["online_gradient_update_steps", "", 75000]

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

  def learn_from_demonstrations(self, demonstrations, learn_only=False, num_episodes=50000):
    if learn_only:
      self.demonstrations = demonstrations
      self.save(checkpoint_type="configured_with_demonstrations")
      assert self.is_learn_from_demonstrations, "Learn from demonstration params not set!"
      assert self.demonstrations is not None, "Run invoked incorrectly, demonstrations not found!"

      # Extract and append demonstrations to memory
      self.load_demonstrations(demonstrations)

      self.train_on_demonstrations()

      # save trained online agent
      self.save(checkpoint_type="trained_only_demonstrations")
    # if learn_only is False, load from previous training checkpoint, explore and learn
    else:
      # update agent dir to current agent dir
      self._agent_save_dir = os.path.join(self._params["Experiment"]["dir"], "agent")
      self.writer = SummaryWriter(log_dir=BaseAgent.summary_dir(self.agent_save_dir))
      if not os.path.exists(BaseAgent.summary_dir(self.agent_save_dir)):
        os.makedirs(BaseAgent.summary_dir(self.agent_save_dir))
      logging.info(f"Exploration learning DIR {self._agent_save_dir}")
      logging.info(f"Summaries DIR {BaseAgent.summary_dir(self._agent_save_dir)}")
      ckp_dir = BaseAgent.check_point_directory(self._agent_save_dir, "")
      logging.info(f"Checkpoints DIR {ckp_dir}")
      pickle_dir = BaseAgent.pickable_directory(self._agent_save_dir)
      logging.info(f"New Pickables at {pickle_dir}")
      self.memory_size = self.memory.capacity
      self.memory.reset_offline(self.memory_size, self.observer.observation_space.shape, 
                                self.device,
                                self.demonstrator_buffer_params["demo_ratio"],
                                per_beta_steps=self.demonstrator_buffer_params["per_beta_steps"])
      logging.info(f"Demo capacity: {self.memory.demo_capacity}/{self.memory.capacity}")
      logging.info(f"{self.memory._dn + 1} demo samples remaining...")
      self.train_episodes(num_episodes=num_episodes)
      logging.info(f"Total learning_steps/steps {self.learning_steps}/{self.steps}")
      logging.info(f"Self generated data last at {self.memory._an}")
      self.save(checkpoint_type="trained_mixed_experiences")

  def load_demonstrations(self, demonstrations):
      for demo in self.demonstrations:
          (state, action, reward, next_state, done, is_demo) = demo
          self.memory.append(state, action, reward, next_state, done, is_demo)  

  def train_on_demonstrations(self):
    while True:
      self.train_step_interval(demo_only=True)
      logging.info(f"Step {self.learning_steps} complete")
      if self.learning_steps > self.online_gradient_update_steps:
        logging.info(f"Initial gradient updates completed. Learning steps {self.learning_steps}")
        break

  def train_episodes(self, num_episodes=50000):
    while True:
      self.train_episode()
      if self.episodes >= num_episodes:
        break

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
    state = torch.Tensor(state).unsqueeze(0).to(self.device).float()
    with torch.no_grad():
      actions = self.online_net(states=state)  # pylint: disable=not-callable
    return actions

  def Plan(self, dt, observed_world):
    # NOTE: if training is enabled the action is set externally
    if not self.set_action_externally:
      observed_state = self.observer.Observe(observed_world)
      # if self.is_be_obs:
        # self.beliefs_info.append(self.observer.beliefs)
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

  def save_beliefs_info(self, filename):
      import pandas as pd
      df = pd.DataFrame(self.beliefs_info)
      print(f"Storing beliefs to {filename}")
      df.to_pickle(filename)

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
      if self.is_learn_from_demonstrations:
        self.memory.append(state, action, reward, next_state, done, False)
      else:
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

  def train_step_interval(self, demo_only=False):
    if demo_only:
      self.online_net.train()
      self.target_net.train()
      self.steps += 1
    else:
      self.epsilon_train.step()
    if self.is_learn_from_demonstrations:
      self.memory.per_beta.step()

    if self.steps % self.target_update_interval == 0:
      self.update_target()

    if demo_only or self.is_update():
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
      if self.is_learn_from_demonstrations and not self.is_online_demo:
        self.save(checkpoint_type='best_lfd')
      else:
        self.save(checkpoint_type='best')

    # We log evaluation results along with training frames = 4 * steps.
    for eval_result_name, eval_result in eval_results.items():
      if self.is_learn_from_demonstrations and not self.is_online_demo:
        self.writer.add_scalar(eval_result_name + "_offline", eval_result, 4 * self.steps)
      else:
        self.writer.add_scalar(eval_result_name, eval_result, 4 * self.steps)
    logging.info('-' * 60)
    logging.info('Evaluation result: {}'.format(formatted_result))
    logging.info('-' * 60)

  def __del__(self):
    self.writer.close()
