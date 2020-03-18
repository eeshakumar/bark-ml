from absl import app
from absl import flags
import tensorflow as tf
import numpy as np
import copy
import pickle

from bark.models.behavior import *
from bark.models.dynamic import *
from bark.world.opendrive import *
from bark.world.goal_definition import *
from bark.geometry import *

from modules.runtime.scenario.scenario_generation.config_with_ease import \
  LaneCorridorConfig, ConfigWithEase


class LeftLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
                road_ids=[16],
                lane_corridor_id=0,
                params=None):
    super(LeftLaneCorridorConfig, self).__init__(road_ids,
                                                  lane_corridor_id,
                                                  params)

  # def position(self, world, min_s=0., max_s=150.):
  #   return super(LeftLaneCorridorConfig, self).position(world, min_s, max_s)

  def ds(self, s_min=25., s_max=35.):
    return np.random.uniform(s_min, s_max)
  
  def controlled_goal(self, world):
    road_corr = world.map.GetRoadCorridor(self._road_ids, XodrDrivingDirection.forward)
    lane_corr = road_corr.lane_corridors[1]
    return GoalDefinitionStateLimitsFrenet(lane_corr.center_line,
                                            (0.4, 0.4),
                                            (0.1, 0.1),
                                            (10., 15.))

  def controlled_ids(self, agent_list):
    return []


class RightLaneCorridorConfig(LaneCorridorConfig):
  def __init__(self,
               road_ids=[16],
               lane_corridor_id=1,
               params=None,
               ml_agent=None,
               observer=None):
    super(RightLaneCorridorConfig, self).__init__(road_ids,
                                                  lane_corridor_id,
                                                  params)
    self._ml_agent = ml_agent
    self._observer = observer

  def ds(self, s_min=30., s_max=35.):
    return np.random.uniform(s_min, s_max)

  def controlled_goal(self, world):
    road_corr = world.map.GetRoadCorridor(self._road_ids, XodrDrivingDirection.forward)
    lane_corr = road_corr.lane_corridors[0]
    return GoalDefinitionStateLimitsFrenet(lane_corr.center_line,
                                            (0.4, 0.4),
                                            (0.1, 0.1),
                                            (10., 15.))

  def behavior_model(self):
    if self._ml_agent != None and self._observer != None:
      bark_ml_agent = BARKMLBehaviorModel(
        dynamic_model=SingleTrackModel(self._params),
        observer=self._observer,
        ml_agent=self._ml_agent,
        params=self._params)
    else:
      return BehaviorIDMClassic(self._params)