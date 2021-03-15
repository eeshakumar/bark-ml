import numpy as np
import torch
from .base import MultiStepBuff, LazyMemory

from collections import deque


class MultiStepDemBuff(MultiStepBuff):

    def __init__(self, maxlen=3):
        super().__init__(maxlen)

    def reset(self):
        super().reset()
        self.is_demo = deque(maxlen=self.maxlen)

    def append(self, state, action, reward, is_demo):
        super().append(state, action, reward)
        self.is_demo.append(is_demo)

    def get(self, gamma=0.99):
        assert len(self.rewards) > 0
        state = self.states.popleft()
        action = self.actions.popleft()
        reward = self._nstep_return(gamma)
        is_demo = self.is_demo.popleft()
        return state, action, reward, is_demo


class LazyDemMemory(LazyMemory):
    state_keys = ['state', 'next_state']
    np_keys = ['action', 'reward', 'done']
    is_demo_keys = ['is_demo']
    keys = state_keys + np_keys + is_demo_keys

    def reset_indexing(self, capacity, state_shape, device,
                       demo_ratio=0.25):
        self.demo_ratio = demo_ratio
        self.demo_capacity = int(demo_ratio*capacity)
        self.agent_capacity = capacity - self.demo_capacity
        if demo_ratio < 1.0:
            # assume memory already contains demo samples
            self._dn = self.demo_capacity - 1
            self._n = self._dn
            if self.demo_capacity == 0:
                self._p = 0
            else:
                self._p = self._dn
        else:
            self._dn = 0
        self._an = self.demo_capacity

    def is_full(self):
        return self._n == self.capacity

    def reset_offline(self, capacity, state_shape, device, demo_ratio):
        self.reset_indexing(capacity, state_shape, device, demo_ratio)
        self['state'][self._an:self.capacity] = np.zeros(
            (self.agent_capacity, *state_shape), dtype=np.float32)
        self['next_state'][self._an:self.capacity] = np.zeros(
            (self.agent_capacity, *state_shape), dtype=np.float32)
        self['action'][self._an:self.capacity] = np.empty(
            (self.agent_capacity, 1), dtype=np.int64)
        self['reward'][self._an:self.capacity] = np.empty(
            (self.agent_capacity, 1), dtype=np.float32)
        self['done'][self._an:self.capacity] = np.empty(
            (self.agent_capacity, 1), dtype=np.float32)
        self['is_demo'][self._an: self.capacity] = np.zeros(
            (self.agent_capacity, 1), dtype=np.int8)

    def __init__(self, capacity, state_shape, device, demo_ratio=0.25):
        super().__init__(capacity, state_shape, device)
        self.reset_indexing(capacity, state_shape, device, demo_ratio)

    def reset(self, is_demo=True):
        super().reset()
        # assume at initialization all memory samples are demo samples
        if is_demo:
            self['is_demo'] = np.ones((self.capacity, 1), dtype=np.int8)
        else:
            self['is_demo'] = np.zeros((self.capacity, 1), dtype=np.int8)
        self['state'] = np.zeros((self.capacity, *self.state_shape))
        self['next_state'] = np.zeros((self.capacity, *self.state_shape))

    def append(self, state, action, reward, next_state, done, is_demo):
        self._append(state, action, reward, next_state, done, is_demo)

    def _append(self, state, action, reward, next_state, done, is_demo):
        if is_demo:
            self['state'][self._dn] = state
            self['next_state'][self._dn] = next_state
            self['action'][self._dn] = action
            self['reward'][self._dn] = reward
            self['done'][self._dn] = done
            self['is_demo'][self._dn] = is_demo
            self._dn = (self._dn + 1) % self.demo_capacity
        else:
            self['state'][self._an] = state
            self['next_state'][self._an] = next_state
            self['action'][self._an] = action
            self['reward'][self._an] = reward
            self['done'][self._an] = done
            self['is_demo'][self._an] = is_demo
            if self._an - self.demo_capacity >= (self.agent_capacity - 1):
                self._an = self.demo_capacity
            else:
                self._an = (self._an + 1)

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity
        self.truncate()

    def _sample(self, indices, batch_size):
        states = np.empty((batch_size, *self.state_shape), dtype=np.float32)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.float32)

        for i, index in enumerate(indices):
            states[i, ...] = self['state'][index]
            next_states[i, ...] = self['next_state'][index]

        states = torch.from_numpy(states).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)
        is_demos =torch.FloatTensor(self['is_demo'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones, is_demos


class LazyDemMultiStepMemory(LazyDemMemory):
    def __init__(self, capacity, state_shape, device, gamma=0.99,
                 multi_step=3, demo_ratio=0.25):
        super().__init__(capacity, state_shape, device, demo_ratio)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepDemBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done, is_demo):
        if self.multi_step != 1:
            self.buff.append(state, action, reward, is_demo)
            if self.buff.is_full():
                state, action, reward, is_demo = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, is_demo)
            if done:
                while not self.buff.is_empty():
                    state, action, reward, is_demo = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done, is_demo)
        else:
            self._append(state, action, reward, next_state, done, is_demo)
