import numpy as np
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

    def __init(self, capacity, state_shape, device):
        super().__init__(capacity, state_shape, device)

    def reset(self, is_demo=True):
        super().reset()
        # assume at initialization all memory samples are demo samples
        if is_demo:
            self['is_demo'] = np.ones((self.capacity, 1), dtype=np.int64)
        else:
            self['is_demo'] = np.zeros((self.capacity, 1), dtype=np.int64)

    def append(self, state, action, reward, next_state, done, is_demo):
        self._append(state, action, reward, next_state, done, is_demo)

    def _append(self, state, action, reward, next_state, done, is_demo):
        self['state'].append(state)
        self['next_state'].append(next_state)
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done
        self['is_demo'][self._p] = is_demo

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

        self.truncate()

    def _sample(self, indices, batch_size):
        bias = -self._p if self._n == self.capacity else 0

        states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)
        next_states = np.empty((batch_size, *self.state_shape), dtype=np.uint8)

        for i, index in enumerate(indices):
            _index = np.mod(index + bias, self.capacity)
            states[i, ...] = self['state'][_index]
            next_states[i, ...] = self['next_state'][_index]

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
        if self.multi_step != 1:
            self.buff.append(state, action, reward, is_demo)
            if self.buff.is_full():
                #TODO: priority change
                state, action, reward, is_demo = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, is_demo)
            if done:
                while not self.buff.is_empty():
                    state, action, reward, is_demo = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done, is_demo)
        else:
            self._append(state, action, reward, next_state, done, is_demo)
