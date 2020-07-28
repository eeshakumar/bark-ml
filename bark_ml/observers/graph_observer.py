import os, time, json, pickle
import logging
from gym import spaces
import numpy as np
import math
import operator
import networkx as nx
import tensorflow as tf
from typing import Dict
from collections import OrderedDict
from itertools import islice

from bark.core.models.dynamic import StateDefinition
from bark.core.world import ObservedWorld
from bark.core.geometry import Distance, Point2d
from bark.runtime.commons.parameters import ParameterServer

from bark_ml.observers.observer import StateObserver

class GraphObserver(StateObserver):
  feature_times = []
  edges_times = []
  
  def __init__(self,
               normalize_observations=True,
               output_supervised_data = False,
               params=ParameterServer()):
    StateObserver.__init__(self, params)

    self._normalize_observations = normalize_observations
    self._output_supervised_data = output_supervised_data

    # the number of features of a node in the graph
    self.feature_len = len(GraphObserver.attribute_keys())

    # the number of features of an edge between two nodes
    self.edge_feature_len = 4

    # the maximum number of agents that can be observed
    self._agent_limit = \
      params["ML"]["GraphObserver"]["AgentLimit", "", 4]

     # the radius an agent can 'see' in meters
    self._visibility_radius = \
      params["ML"]["GraphObserver"]["VisibilityRadius", "", 50]

    self.observe_times = []

  @classmethod
  def attribute_keys(cls):
    return ["x", "y", "theta", "vel", "goal_x", "goal_y", 
    "goal_dx", "goal_dy", "goal_theta", "goal_d", "goal_vel"]

  def Observe(self, world):
    """see base class"""
    agents = self._preprocess_agents(world)
    num_agents = len(agents)
    obs = [self._agent_limit, num_agents, self.feature_len]

    # features
    for _, agent in agents:
      obs.extend(list(self._extract_features(agent).values()))

    # fill empty spots (difference between existing and max agents) with zeros
    obs.extend(np.zeros(max(0, self._agent_limit - num_agents)) * self.feature_len)

    # edges
    # second loop for edges necessary
    # -> otherwise order of graph_nodes is disrupted
    edges = []
    edge_features = np.zeros((self._agent_limit, self._agent_limit, self.edge_feature_len))

    for index, agent in agents:
      # create edges to all visible agents
      nearby_agents = self._nearby_agents(
        center_agent=agent, 
        agents=agents, 
        radius=self._visibility_radius)

      for target_index, nearby_agent in nearby_agents:
        edges.append([index, target_index])
        edge_features[index, target_index, :] = self._extract_edge_features(agent, nearby_agent)
    
    # build adjacency matrix and convert to list
    adjacency_matrix = np.zeros((self._agent_limit, self._agent_limit))
    for source, target in edges:
      adjacency_matrix[source, target] = 1
      adjacency_matrix[target, source] = 1
    
    adjacency_list = adjacency_matrix.reshape(-1)
    edge_features = np.reshape(edge_features, -1)
    obs.extend(adjacency_list)
    obs.extend(edge_features)

    assert len(obs) == self._len_state, f'Observation \
      has invalid length ({len(obs)}, expected: {self._len_state})'
    
    obs = tf.convert_to_tensor(obs, dtype=tf.float32, name='observation')

    if self._output_supervised_data == False:
      return obs
    else:
      features, _ = GraphObserver.graph(obs)
      actions = self._generate_actions(features)
      return obs, actions
      
    return obs

  def _extract_edge_features(self, source_agent, target_agent):
    source_features = self._extract_features(source_agent)
    target_features = self._extract_features(target_agent)

    d_x = source_features["x"] - target_features["x"]
    d_y = source_features["y"] - target_features["y"]
    d_vel = source_features["vel"] - target_features["vel"]
    d_theta = source_features["theta"] - target_features["theta"]
    return np.array([d_x, d_y, d_vel, d_theta])

  @classmethod
  def graph(cls, observations, graph_dims, dense=False):
    """ Maps the given batch of observations into a
      graph representation.

      Args:
      observations: The batch of observations of 
        shape (batch_size, observation_size).
      graph_dims: A tuple containing the dimensions of the 
        graph as (num_nodes, num_features, num_edge_features).
        If `dense` is set to True, num_edge_features is ignored.
      dense: Specifies the returned graph representation. If 
        set to True, the edges are returned as a list of nodes 
        indices (relative to the flattened batch) and an additional
        mapping of each node to a graph is returned. If set to 
        False (default), edges are returned as an adjacency matrix
        and an edge feature matrix is additionally returned.

      Returns:
        X: Nodes features of shape (batch_size, num_nodes, num_features)

        If `dense` is True:
        A: Dense representation of edges as a list of node index pairs,
          shape (num_total_edges, 2).
        node_to_graph_map: A 1d list, where each element is the mapping
        of the node (indexed relative to the batch) to a graph.

        If `dense`is False:
        A: Spare binary adjacency matrix of shape 
          (batch_size num_nodes, num_nodes).
        E: Edge features of shape 
          (batch_size, num_nodes, num_edge_features, num_edge_features).
    """
    n_nodes, n_features = graph_dims[0:2]
    batch_size = observations.shape[0]
    
    # remove first three elements of each sample
    # TODO: remove these values from the observation
    obs = observations[:, 3:]
    
    # extract node features F
    F = tf.reshape(obs[:, :n_nodes * n_nodes], [batch_size, n_nodes, n_features])

    # extract adjacency matrix A
    adj_start_idx = n_nodes * n_features
    adj_end_idx = adj_start_idx + n_nodes ** 2
    A = tf.reshape(obs[:, adj_start_idx:adj_end_idx], [batch_size, n_nodes, n_nodes])

    if dense:
      F = tf.reshape(F, [batch_size * n_nodes, n_features])

      # find non-zero elements in the adjacency matrix (edges)
      # and collect there indices
      A = tf.reshape(tf.where(tf.greater(A, 0))[:,1:], [batch_size, -1])
      A = tf.cast(A, tf.int32)

      if batch_size > 1:
        # we need the indices of the source and target nodes to
        # be represented as their indices in the whole batch,
        # in other words: each node index must be the index
        # of the graph in the batch plus the index of the node
        # in the graph. E.g. if each graph has 5 nodes, the 
        # node indices are: graph 0: 0-4, graph 1: 5-9, etc.
        mask = tf.range(batch_size * n_nodes, delta=n_nodes)
        mask = tf.tile(tf.reshape(mask, [-1, 1]), [1, A.shape[1]])

        # add the graph index to the node indices
        A = tf.add(A, mask)
        
      A = tf.reshape(A, [-1, 2])

      # construct a list where each element represents the
      # assignment of a node to a graph via the graph's index
      node_to_graph_map = tf.reshape(tf.range(batch_size), [-1, 1])
      node_to_graph_map = tf.tile(node_to_graph_map, [1, n_nodes])
      node_to_graph_map = tf.reshape(node_to_graph_map, [-1])

      return F, A, node_to_graph_map

    # extract edge features
    n_edge_features = graph_dims[2]
    E_shape = [batch_size, n_nodes, n_nodes, n_edge_features]
    E = tf.reshape(obs[:, adj_end_idx:], E_shape)

    return F, A, E

  def _preprocess_agents(self, world):
    """
    Returns a list of tuples, consisting
    of an index and an agent object element.

    The first element always represents the ego agent.
    The remaining elements resemble other agents, up
    to the limit defined by `self._agent_limit`,
    sorted in ascending order with respect to the agents'
    distance in the world to the ego agent.
    """
    ego_agent = world.ego_agent
    agents = list(world.agents.values())
    agents.remove(ego_agent)
    agents = self._agents_sorted_by_distance(ego_agent, agents)
    agents.insert(0, ego_agent)
    return list(enumerate(agents))[:self._agent_limit]

  def _agents_sorted_by_distance(self, ego_agent, agents):
    def distance(agent):
      return Distance(
        self._position(ego_agent), 
        self._position(agent))
    
    agents.sort(key=distance)
    return agents

  def _nearby_agents(self, center_agent, agents, radius: float):
    """
    Returns all elements from 'agents' within the specified 
    radius of the 'center_agent's position.
    """
    center_agent_pos = self._position(center_agent)
    other_agents = filter(lambda a: a[1].id != center_agent.id, agents)
    nearby_agents = []

    for index, agent in other_agents:
      agent_pos = self._position(agent)
      distance = Distance(center_agent_pos, agent_pos)

      if distance <= radius:
        nearby_agents.append((index, agent))

    return nearby_agents

  def _extract_features(self, agent) -> Dict[str, float]:
    """Returns dict containing all features of the agent"""
    res = OrderedDict()

    # Init data (to keep ordering always equal for reading and writing!!)
    for label in self.attribute_keys():
      res[label] = "inf"
    
    state = agent.state
    res["x"] = state[int(StateDefinition.X_POSITION)]
    res["y"] = state[int(StateDefinition.Y_POSITION)]
    res["theta"] = state[int(StateDefinition.THETA_POSITION)]
    res["vel"] = state[int(StateDefinition.VEL_POSITION)]

    # get information related to goal
    goal_center = agent.goal_definition.goal_shape.center[0:2]
    res["goal_x"] = goal_center[0] # goal position in x
    res["goal_y"] = goal_center[1] # goal position in y
    goal_dx = goal_center[0] - res["x"] # distance to goal in x coord
    res["goal_dx"] = goal_dx
    goal_dy = goal_center[1] - res["y"] # distance to goal in y coord
    res["goal_dy"] = goal_dy
    goal_theta = np.arctan2(goal_dy, goal_dx) # theta for straight line to goal
    res["goal_theta"] = goal_theta
    goal_d = np.sqrt(goal_dx**2 + goal_dy**2) # distance to goal
    res["goal_d"] = goal_d
    
    # goal_velocity = np.mean(agent.goal_definition.velocity_range)
    # res["goal_vel"] = goal_velocity

    if self._normalize_observations:
      n = self.normalization_data

      for k in ["x", "y", "theta", "vel"]:
        res[k] = self._normalize_value(res[k], n[k])
      res["goal_x"] = self._normalize_value(res["goal_x"], n["x"])
      res["goal_y"] = self._normalize_value(res["goal_y"], n["y"])
      res["goal_dx"] = self._normalize_value(res["goal_dx"], n["dx"])
      res["goal_dy"] = self._normalize_value(res["goal_dy"], n["dy"])
      res["goal_d"] = self._normalize_value(res["goal_d"], n["distance"])
      res["goal_theta"] = self._normalize_value(res["goal_theta"], n["theta"])
      res["goal_vel"] = self._normalize_value(res["goal_vel"], n["vel"])
    
    #####################################################
    #   If you change the number/names of features,     #
    #   please adapt self.attributes_keys accordingly.  #
    #####################################################
    assert list(res.keys()) == self.attribute_keys()

    return res

  def _generate_actions(self, all_features) -> Dict[str, float]:
    all_actions = list()
    for features in all_features:
      actions = OrderedDict()
      att_keys = GraphObserver.attribute_keys()
      steering = features[att_keys.index("goal_theta")] - features[att_keys.index("theta")]
    
      v_0 = features[att_keys.index("vel")]
      dv = features[att_keys.index("goal_vel")] - v_0
      acc = (1./features[att_keys.index("goal_d")])*dv*(dv/2+v_0)
    
      if self._normalize_observations:
        range_steering = [-0.1, 0.1]
        range_acc = [-0.6, 0.6]
        steering = self._normalize_value(steering, range_steering)
        acc = self._normalize_value(acc, range_acc)

      actions["steering"] = steering
      actions["acceleration"] = acc
      all_actions.append(actions)

    return actions

  def _normalize_value(self, value, range):
    """norms to [-1, 1] with
    value <= range[0] -> returns -1
    value >= range[1] -> returns 1"""
    normed = 2 * (value - range[0]) / (range[1] - range[0]) - 1
    normed = max(-1, normed) # values lower -1 clipped
    normed = min(1, normed) # values bigger 1 clipped
    return normed

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
    d = OrderedDict()
    x_range = self._world_x_range[1] - self._world_x_range[0]
    y_range = self._world_y_range[1] - self._world_y_range[0]
    max_dist = np.linalg.norm([x_range, y_range])
    d['x'] = self._world_x_range
    d['y'] = self._world_y_range
    d['theta'] = self._ThetaRange
    d['vel'] = self._VelocityRange
    d['distance'] = [-max_dist / 2, max_dist / 2]
    d['dx'] = [-x_range, x_range]
    d['dy'] = [-y_range, y_range]
    d['road'] = [0, 15] # may need adjustment for if 3 lanes are broader than 15 m
    return d

  def sample(self):
    raise NotImplementedError

  @property
  def observation_space(self):
    #  0 ... 100 for the indices of num_agents and num_features
    # -1 ... 1   for all agent attributes
    #  0 ... 1   for the adjacency list
    # -1 ... 1   for the edge attributes
    return spaces.Box(
      low=np.concatenate((
        np.zeros(3),
        np.full(self._agent_limit * self.feature_len, -1),
        np.zeros(self._agent_limit ** 2),
        np.zeros((self._agent_limit ** 2) * self.edge_feature_len))),
      high=np.concatenate((
        np.array([100, 100, 100]), 
        np.ones(self._len_state - 3)
      )))

  @property
  def _len_state(self):
    len_node_features = self._agent_limit * self.feature_len
    len_adjacency = self._agent_limit ** 2
    len_edge_features = len_adjacency * self.edge_feature_len
    return 3 + len_node_features + len_adjacency + len_edge_features

  @classmethod
  def graph_from_observation(cls, observation):
    graph = nx.OrderedGraph()
    
    node_limit = int(observation[0])
    num_nodes = int(observation[1])
    num_features = int(observation[2])

    obs = observation[3:]

    for node_id in range(num_nodes):
      start_idx = node_id * num_features
      end_idx = start_idx + num_features
      features = obs[start_idx:end_idx]

      attributes = dict(zip(GraphObserver.attribute_keys(), features))
      graph.add_node(node_id, **attributes)
    
    adj_start_idx = node_limit * num_features
    adj_list = obs[adj_start_idx:]
    adj_matrix = np.reshape(adj_list, (node_limit, -1))
    
    for (source_id, source_edges) in enumerate(adj_matrix):
      for target_id in np.flatnonzero(source_edges):
        graph.add_edge(source_id, target_id)

    return graph

  def _observation_from_graph(self, graph):
    """ Encodes the given graph into a bounded array with fixed size.

    The returned array 'a' has the following contents:
    a[0]:                            (int) the maximum number of possibly contained nodes
    a[1]:                            (int) the actual number of contained nodes
    a[2]:                            (int) the number of features per node
    a[3: a[1] * a[2]]:               (floats) the node feature values
    a[3 + a[1] * a[2]: a[0] * a[2]]: (int) all entries have value -1
    a[-a[0] ** 2:]:                  (0 or 1) an adjacency matrix in vector form

    :type graph: A nx.Graph object.
    :param graph:
    
    :rtype: list
    """
    num_nodes = len(graph.nodes)
    obs = [self._agent_limit, num_nodes, self.feature_len]
    
    # append node features
    for (node_id, attributes) in graph.nodes.data():
      obs.extend(list(attributes.values()))

    # fill empty spots (difference between existing and max agents) with -1
    if num_nodes < self._agent_limit:
      obs.extend(np.full((self._agent_limit - num_nodes) * self.feature_len, -1))

    # build adjacency matrix and convert to list
    adjacency_matrix = np.zeros((self._agent_limit, self._agent_limit))
    for source, target in graph.edges:
      adjacency_matrix[source, target] = 1
    
    adjacency_list = adjacency_matrix.reshape(-1)
    obs.extend(adjacency_list)

    # Validity check
    assert len(obs) == self._len_state, f'Observation \
      has invalid length ({len(obs)}, expected: {self._len_state})'
    
    #return obs
    return tf.convert_to_tensor(
        obs, 
        dtype=tf.float32, 
        name='observation'
      )
