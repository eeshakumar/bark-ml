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

    def __init__(self, capacity, state_shape, device, demo_ratio=0.25):
        super().__init__(capacity, state_shape, device)
        # print("State shape", self.state_shape, *self.state_shape)
        self.demo_ratio = demo_ratio
        self.demo_capacity = int(demo_ratio*capacity)
        self.agent_capacity = capacity - self.demo_capacity
        self._dn = 0
        self._an = 0

    def reset(self, is_demo=True):
        super().reset()
        # assume at initialization all memory samples are demo samples
        if is_demo:
            self['is_demo'] = np.ones((self.capacity, 1), dtype=np.int64)
        else:
            self['is_demo'] = np.zeros((self.capacity, 1), dtype=np.int64)
        self['state'] = np.zeros((self.capacity, *self.state_shape))
        self['next_state'] = np.zeros((self.capacity, *self.state_shape))

    def append(self, state, action, reward, next_state, done, is_demo):
        self._append(state, action, reward, next_state, done, is_demo)

    def _append(self, state, action, reward, next_state, done, is_demo):
        # print("_append")
        if is_demo:
            # print("D ST , MEM ST", state.shape, self['state'].shape, self['state'][self._dn].shape)
            self['state'][self._dn] = state
            self['next_state'][self._dn] = next_state
            self['action'][self._dn] = action
            self['reward'][self._dn] = reward
            self['done'][self._dn] = done
            self['is_demo'][self._dn] = is_demo
            self._dn = (self._dn + 1) % self.demo_capacity
        else:
            self._an = (self.demo_capacity + self._an)
            # print("ND ST , MEM ST", state.shape, self['state'].shape, self['state'][self._an].shape)
            self['state'][self._an] = state
            self['next_state'][self._an] = next_state
            self['action'][self._an] = action
            self['reward'][self._an] = reward
            self['done'][self._an] = done
            self['is_demo'][self._an] = is_demo
            self._an = (self._an + 1) % self.agent_capacity

        # self['state'].append(state)
        # self['next_state'].append(next_state)
        # self['state'][self._p] = state
        # self['next_state'][self._p] = next_state
        # self['action'][self._p] = action
        # self['reward'][self._p] = reward
        # self['done'][self._p] = done
        # self['is_demo'][self._p] = is_demo

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity
        # print("N P DN AN", self._n, self._p, self._dn, self._an)
        self.truncate()

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0
        # print('Bias', bias)
        states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)

        # print("Sample Indices", indices)
        for i, index in enumerate(indices):
            # print("index", index)
            _index = np.mod(index + bias, self.capacity)
            # print("_index", _index)
            states[i, ...] = self['state'][index]
            next_states[i, ...] = self['next_state'][index]

        states = torch.ByteTensor(states).to(self.device).float() / 255.
        next_states = torch.ByteTensor(next_states).to(self.device).float() / 255.
        actions = torch.LongTensor(self['action'][indices]).to(self.device)
        rewards = torch.FloatTensor(self['reward'][indices]).to(self.device)
        dones = torch.FloatTensor(self['done'][indices]).to(self.device)
        is_demos =torch.FloatTensor(self['is_demo'][indices]).to(self.device)

        return states, actions, rewards, next_states, dones, is_demos


class LazyDemMultiStepMemory(LazyDemMemory):
    def __init__(self, capacity, state_shape, device, gamma=0.99, multi_step=3):
        super().__init__(capacity, state_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MultiStepDemBuff(maxlen=self.multi_step)

    def append(self, state, action, reward, next_state, done, is_demo):
        # print("Append to memory reward/is_demo", reward, is_demo)
        if self.multi_step != 1:
            print("Buff length", len(self.buff))
            self.buff.append(state, action, reward, is_demo)
            if self.buff.is_full():
                #TODO: priority change
                # print("Buff is full, get and append to replay")
                state, action, reward, is_demo = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, is_demo)
            if done:
                while not self.buff.is_empty():
                    state, action, reward, is_demo = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done, is_demo)
        else:
            self._append(state, action, reward, next_state, done, is_demo)
