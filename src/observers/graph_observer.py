import os, time, json, pickle
from gym import spaces
import numpy as np
import math
import operator
import networkx as nx
from typing import Dict

from bark.models.dynamic import StateDefinition
from bark.world import ObservedWorld
from bark.geometry import Distance, Point2d
from modules.runtime.commons.parameters import ParameterServer
from src.observers.observer import StateObserver

class ObservationGraph(object):
  """
  An abstraction of the observed world in a graph representation.
  """

  def __init__(self, ego_agent_id: str):
    self._graph = nx.Graph(ego_agent_id=ego_agent_id)

  @property
  def nx_graph(self):
    """
    The underlying `nx.Graph` object.
    """
    return self._graph

  @property
  def attributes(self):
    """
    Global attributes of the graph.
    """
    return self._graph.graph

  @property
  def data(self):
    """
    A dictionary representation of the graph.
    """
    return nx.node_link_data(self._graph)

  def add_global(self, key: str, value):
      self._graph.graph[key] = value

  def add_node(self, id: int, attributes: Dict[str, float]):
    """
    Adds a node with the specified identifier and attributes
    to the graph.
    """
    self._graph.add_node(id, **attributes)

  def add_edge(self, source: int, target: int, attributes: Dict[str, float]=None):
    """ 
    Adds an undirected edge between the specified source 
    and target nodes with the given attributes.
    If an edge between these two nodes already exists, 
    it will replaced.
    
    :param source:int: identifier of the source node.
    :param target:int: identifier of the target node.
  
    :param attributes:Dict[str, float]: A dictionary\
      of attributes with string keys and float values.
    """
    assert None not in [source, target], \
      "Specifying a source and target node id is mandatory."
    assert source != target, \
      "Source node id is equal to target node id, which is not allowed."
    self._graph.add_edge(source, target)


class GraphObserver(StateObserver):
  
  def __init__(self,
               normalize_observations=True,
               use_edge_attributes=True,
               params=ParameterServer()):
    StateObserver.__init__(self, params)

    self._observation_len = 1 # i.e. one graph
    self._normalize_observations = normalize_observations
    self._use_edge_attributes = use_edge_attributes

     # the radius an agent can 'see' in meters
    self._visible_distance = 50

  def observe(self, world):
    """see base class"""
    ego_agent = world.ego_agent
    graph = ObservationGraph(ego_agent_id=ego_agent.id)
    graph.add_global("normalization_ref", self.normalization_data)

    actions = {} # generated for now (steering, acceleration)

    # make ego_agent the first element, sort others by id
    # there should be a more elegant way to do this
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents.sort(key=lambda agent: agent.id)
    agents.insert(0, ego_agent)
    
    # add nodes
    for agent in agents:
      # create node
      features = self._extract_features(agent)
      graph.add_node(id=agent.id, attributes=features)

      # generate actions
      actions[agent.id] = self._generate_actions(features)
      
      # create edges to all other agents
      nearby_agents = self._nearby_agents(agent, agents, self._visible_distance)

      for other_agent in nearby_agents:
        graph.add_edge(source=agent.id, target=other_agent.id)

    return graph, actions

  def _nearby_agents(self, center_agent, agents, radius: float):
    """
    Returns all elements from 'agents' within the specified 
    radius of the 'center_agent's position.
    """
    center_agent_pos = self._position(center_agent)
    other_agents = filter(lambda a: a.id != center_agent.id, agents)
    nearby_agents = []

    for agent in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)

      if distance <= radius:
        nearby_agents.append(agent)
    
    return nearby_agents

  def _extract_features(self, agent) -> Dict[str, float]:
    """Returns dict containing all features of the agent"""
    res = {}
    
    # get basic information
    state = agent.state
    res["x"] = state[int(StateDefinition.X_POSITION)] # maybe not needed
    res["y"] = state[int(StateDefinition.Y_POSITION)] # maybe not needed
    res["theta"] = state[int(StateDefinition.THETA_POSITION)]
    res["vel"] = state[int(StateDefinition.VEL_POSITION)]

    # get information related to goal
    goal_center = agent.goal_definition.goal_shape.center[0:2]
    goal_dx = goal_center[0] - res["x"] # distance to goal in x coord
    goal_dy = goal_center[1] - res["y"] # distance to goal in y coord
    goal_theta = np.arctan2(goal_dy, goal_dx) # theta for straight line to goal
    goal_d = np.sqrt(goal_dx**2 + goal_dy**2) # distance to goal
    res["goal_d"] = goal_d
    res["goal_theta"] = goal_theta

    if self._normalization_enabled:
      n = self.normalization_data

      for k in ["x", "y", "theta", "vel"]:
        res[k] = self._normalize_value(res[k], n[k])
      
      res["goal_d"] = self._normalize_value(res["goal_d"], n["distance"])
      res["goal_theta"] = self._normalize_value(res["goal_theta"], n["theta"])
    
    return res

  def _generate_actions(self, features: Dict[str, float]) -> Dict[str, float]:
    labels = dict()
    steering = features["goal_theta"] - features["theta"]
    labels["steering"] = steering
    
    dt = 2 # time constant in s for planning horizon
    acc = 2. * (features["goal_d"] - features["vel"] * dt) / (dt**2)
    labels["acceleration"] = acc
    return labels

  def _normalize_value(self, value, range):
    return (value - range[0]) / (range[1] - range[0])

  def reset(self, world):
    return world

  def _position(self, agent) -> Point2d:
    return Point2d(
      agent.state[int(StateDefinition.X_POSITION)],
      agent.state[int(StateDefinition.Y_POSITION)]
    )

  @property
  def normalization_data(self) -> Dict[str, list]:
    """
    The reference ranges of how certain attributes are normalized.
    Use this info to scale normalized values back to real values.
    
    E.g. the value for key 'x' returns the possible range of 
    the x-element of positions. A normalized value of 1.0 
    corresponds to the maximum of this range (and vice versa).

    Note: This dictionary does not include a range for each
    possible attribute, but rather for each _kind_ of attribute,
    like distances, velocities, angles, etc.
    E.g. all distances (between agents, between objects, etc.)
    are scaled relative to the 'distance' range. 
    """
    d = {}
    d['x'] = self._world_x_range
    d['y'] = self._world_y_range
    d['theta'] = self._theta_range
    d['vel'] = self._velocity_range
    d['distance'] = [0, \
      np.linalg.norm([self._world_x_range[1], self._world_y_range[1]])]
    return d

  @property
  def observation_space(self):
    return spaces.Box(
      low=np.zeros(self._observation_len),
      high=np.ones(self._observation_len))

  @property
  def _len_state(self):
    return 1