import numpy as np
from bark.models.behavior import BehaviorModel, DynamicBehaviorModel
from bark.models.dynamic import SingleTrackModel
from modules.runtime.commons.parameters import ParameterServer
from configurations.base_configuration import BaseConfiguration
from bark.world import World, ObservedWorld

# include all configurations
from configurations.highway.configuration import HighwayConfiguration

class BARKMLBehaviorModel(BehaviorModel):
  """This class makes a trained agent available as BehaviorModel
     in BARK.
  """
  def __init__(self,
               dynamic_model=None,
               observer=None,
               ml_agent=None,
               params=None):
    BehaviorModel.__init__(self, params)
    self._params = params
    self._observer = observer
    self._ml_agent = ml_agent
    self._dynamic_behavior_model = DynamicBehaviorModel(
      dynamic_model,
      params)

  def Plan(self, delta_time, observed_world):
    observed_state = self._observer.observe(
      observed_world)
    action = self._ml_agent.act(observed_state)
    self._dynamic_behavior_model.SetLastAction(action)
    trajectory = self._dynamic_behavior_model.Plan(delta_time, observed_world)
    super(BARKMLBehaviorModel, self).SetLastTrajectory(trajectory)
    return trajectory

  def Clone(self):
    return self
