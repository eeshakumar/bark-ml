import numpy as np
from modules.runtime.runtime import Runtime

from configurations.bark_agent import BARKMLBehaviorModel
from bark.models.behavior import *
from bark.models.dynamic import *

class RuntimeRL(Runtime):
  """Runtime wrapper for reinforcement learning.
     Extends the runtime with observers and evaluators.
  
  Arguments:
      Runtime {Runtime} -- BaseClass
  """
  def __init__(self,
               action_wrapper,
               observer,
               evaluator,
               step_time,
               viewer,
               scenario_generator=None,
               render=False):
    Runtime.__init__(self,
                     step_time=step_time,
                     viewer=viewer,
                     scenario_generator=scenario_generator,
                     render=render)
    self._action_wrapper = action_wrapper
    self._observer = observer
    self._evaluator = evaluator
    self._behavior_models = []

  def reset(self, scenario=None):
    """Resets the runtime and its objects
    """
    super().reset(scenario=scenario)
    self._world = self._observer.reset(self._world)
    self._world = self._evaluator.reset(self._world)
    idx_list = []
    for idx, agent in self._world.agents.items():
       idx_list.append(idx)
    self._scenario._eval_agent_ids = np.random.choice(idx_list, 1)
    self._world = self._action_wrapper.reset(self._world,
                                             self._scenario._eval_agent_ids)
    
    # replace constant vel. models
    self._behavior_models = []
    for idx, agent in self._world.agents.items():
      if idx != self._scenario._eval_agent_ids[0]:
        self._behavior_models.append(BARKMLBehaviorModel(
              dynamic_model=SingleTrackModel(self._params),
              observer=self._observer,
              ml_agent=self._agent,
              params=self._params))
        agent.behavior_model = self._behavior_models[-1]

    observed_world = self._world.Observe(
      self._scenario._eval_agent_ids)[0]
    return self._observer.observe(observed_world)

  def step(self, action):
    """Steps the world with a specified time dt
    
    Arguments:
        action {any} -- will be passed to the ActionWrapper
    
    Returns:
        (next_state, reward, done, info) -- RL tuple
    """
    self._world = self._action_wrapper.action_to_behavior(world=self._world,
                                                          action=action)
    self._world.Step(self._step_time)
    observed_world = self._world.Observe([self._scenario._eval_agent_ids[0]])[0]
    # next_observed_world = observed_world.PredictWithOthersIDM(self._step_time, action)
    # print(observed_world.Evaluate())
    snapshot =  self.snapshot(
      observed_world=observed_world,
      action=action)
    if self._render:
      self.render()
    return snapshot

  @property
  def action_space(self):
    """Action space of the agent
    """
    return self._action_wrapper.action_space

  @property
  def observation_space(self):
    """Observation space of the agent
    """
    return self._observer.observation_space

  def snapshot(self, observed_world, action):
    """Evaluates and observes the world from the controlled-agents's
       perspective
    
    Arguments:
        world {bark.world} -- Contains all objects of the simulation
        controlled_agents {list} -- list of AgentIds
    
    Returns:
        (next_state, reward, done, info) -- RL tuple
    """
    # TODO(@hart): could be multiple
    next_state = self._observer.observe(observed_world)
    reward, done, info = self._evaluator.evaluate(
      observed_world=observed_world,
      action=action,
      observed_state=next_state)
    return next_state, reward, done, info


