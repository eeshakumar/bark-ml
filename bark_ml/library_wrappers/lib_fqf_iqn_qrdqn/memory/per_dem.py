import numpy as np
import torch
from .base_dem import LazyDemMultiStepMemory
from .segment_tree import SumTree, MinTree
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.utils import LinearAnneaer

class LazyPrioritizedDemMultiStepMemory(LazyDemMultiStepMemory):
    def __init__(self,
                 capacity,
                 state_shape,
                 device,
                 gamma=0.99,
                 multi_step=3,
                 alpha=0.4,
                 beta=0.4,
                 beta_steps=2e5,
                 min_pa=0.0,
                 max_pa=1.0,
                 eps=0.01,
                 epsilon_demo=1.0,
                 epsilon_alpha=0.001,
                 per_beta=0.6,
                 per_beta_steps=75000,
                 demo_ratio=1.0):
        super().__init__(capacity, state_shape, device, gamma, multi_step, demo_ratio)

        self.alpha = alpha
        self.beta = beta
        self.beta_diff = (1.0 - beta) / beta_steps
        self.per_beta = LinearAnneaer(per_beta, 1.0, per_beta_steps)
        self.min_pa = min_pa
        self.max_pa = max_pa
        self.eps = eps
        self._cached = None
        self.per_prio_max = 0.0

        it_capacity = 1
        while it_capacity < capacity:
            it_capacity *= 2
        self.it_sum = SumTree(it_capacity)
        self.it_min = MinTree(it_capacity)

        # dem specific priorities.
        self.epsilon_demo = epsilon_demo
        self.epsilon_alpha = epsilon_alpha

    def _pa(self, p, is_demos):
        # take epsilon_demo when is_demo is 1, else take epsilon_alpha
        eps = np.where(is_demos, self.epsilon_demo, self.epsilon_alpha)# self.eps + self.epsilon_demo * is_demos + self.epsilon_alpha * (1.0 - is_demos)
        total_p = ((p + eps) ** self.alpha) / np.sum((p + eps) ** self.alpha)
        return total_p

    def append(self, state, action, reward, next_state, done, is_demo, p=None):
        # TODO: Change priority calculation by defining _pa(with epsilon?)
        if p is None:
            if is_demo:
                pa = max(self.per_prio_max, self.epsilon_demo)
            else:
                pa = max(self.per_prio_max, self.epsilon_alpha)
        else:
            pa = self._pa(p, is_demo)

        if self.multi_step != 1:
            self.buff.append(state, action, reward, is_demo)

            if self.buff.is_full():
                state, action, reward, is_demo = self.buff.get(self.gamma)
                self._append(state, action, reward, next_state, done, is_demo, pa)

            if done:
                while not self.buff.is_empty():
                    state, action, reward, is_demo = self.buff.get(self.gamma)
                    self._append(state, action, reward, next_state, done, is_demo, pa)
        else:
            self._append(state, action, reward, next_state, done, is_demo, pa)

    def _append(self, state, action, reward, next_state, done, is_demo, pa):
        # Store priority, which is done efficiently by SegmentTree.
        self.it_min[self._p] = pa
        self.it_sum[self._p] = pa
        super()._append(state, action, reward, next_state, done, is_demo)

    def _sample_idxes(self, batch_size):
        total_pa = self.it_sum.sum(0, self._n)
        rands = np.random.rand(batch_size) * total_pa
        indices = [self.it_sum.find_prefixsum_idx(r) for r in rands]
        return indices

    def sample(self, batch_size):
        assert self._cached is None, 'Update priorities before sampling.'

        self._cached = self._sample_idxes(batch_size)
        batch = self._sample(self._cached, batch_size)
        weights = self._calc_weights(self._cached)
        return batch, weights

    def _calc_weights(self, indices):
        weights = [(self.it_sum[i] * self.capacity) ** -self.per_beta.get() for i in indices]
        weights = weights / np.sum(weights)
        return torch.FloatTensor(weights).to(self.device).view(-1, 1)

    def update_priority(self, errors, is_demos):
        #TODO: change priority equation.
        assert self._cached is not None, "No Elements were sampled"
        is_demos_expanded = torch.zeros(errors.shape)
        for i in range(is_demos.shape[0]):
            is_demos_expanded[i,:,:] = is_demos[i]
        ps = errors.detach().cpu().abs().numpy().flatten()
        is_demos = is_demos_expanded.detach().cpu().numpy().flatten()
        assert ps.shape == is_demos.shape
        pas = self._pa(ps, is_demos)
        self.per_prio_max = max(pas.max(), self.per_prio_max)

        for index, pa in zip(self._cached, pas):
            assert 0 <= index < self._n
            assert 0 < pa
            self.it_sum[index] = pa
            self.it_min[index] = pa

        self._cached = None

    def reset_offline(self, capacity, state_shape, device, demo_ratio,
                      per_beta=0.6, per_beta_steps=75000):
        super().reset_offline(capacity, state_shape, device, demo_ratio)
        self.per_beta = LinearAnneaer(per_beta, 1.0, per_beta_steps)
        for index in range(self._an, self._an + self.agent_capacity):
            self.it_min[index] = 0
            self.it_sum[index] = 0
    
    @property
    def sampled(self):
        return self._cached