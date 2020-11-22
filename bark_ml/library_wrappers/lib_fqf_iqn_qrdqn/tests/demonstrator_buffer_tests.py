import unittest
import numpy as np
import torch
from bark_ml.library_wrappers.lib_fqf_iqn_qrdqn.memory import LazyPrioritizedDemMultiStepMemory


class DemonstratorBufferTests(unittest.TestCase):

    def test_sampled_space_dims(self):
        state = np.zeros((2, 2))
        batch_size = 1
        capacity = 10
        memory = LazyPrioritizedDemMultiStepMemory(capacity=capacity, max_pa=2.0,
            state_shape=state.shape, device=0, demo_ratio=1.0)
        for i in range(capacity):
            memory.per_beta.step()
            memory.append(state, action=i, reward=i, next_state=state, done=i%2, is_demo=True)
        (states, actions, rewards, next_states, dones, is_demos), weights = memory.sample(batch_size)
        self.assertEqual(states[0].shape, state.shape)
        self.assertEqual(next_states[0].shape, state.shape)

    def test_memory_capacity(self):
        state = np.zeros((2, 2))
        capacity = 10
        demo_ratio = 0.25
        memory = LazyPrioritizedDemMultiStepMemory(capacity=capacity, max_pa=2.0,
            state_shape=state.shape, device=0, demo_ratio=demo_ratio)
        self.assertEqual(int(capacity * demo_ratio), memory.demo_capacity)
        self.assertEqual(capacity - int(capacity * demo_ratio), memory.agent_capacity)

    def test_sampling_accuracy(self):
        state = np.zeros((2, 2))
        batch_size = 1
        capacity = 1000
        errors = torch.zeros((batch_size, 1, 1))
        errors[0,:,:] = 1.0
        memory = LazyPrioritizedDemMultiStepMemory(capacity=capacity, max_pa=2.0,
            state_shape=state.shape, device=0, demo_ratio=1.0)
        for i in range(capacity):
            memory.per_beta.step()
            memory.append(state, action=i, reward=i, next_state=state, done=i%2, is_demo=True)
        sampled_indices = set()
        for i in range(int(capacity/2) - 1):
            (states, actions, rewards, next_states, dones, is_demos), weights = memory.sample(batch_size)
            sampled_indices.update(memory.sampled)
            memory.update_priority(errors, is_demos)
        
        # assert that repetitive sampling is better than random selection
        self.assertLessEqual(len(sampled_indices), capacity/2)

    def test_append_sample(self):
        state = np.zeros((2, 2))
        batch_size = 1
        capacity = 10
        memory = LazyPrioritizedDemMultiStepMemory(capacity=capacity, max_pa=2.0,
            state_shape=state.shape, device=0, demo_ratio=0.5)
        for i in range(capacity - 2):
            memory.per_beta.step()
            memory.append(state, action=i, reward=i, next_state=state, done=i%2, is_demo=(i%2)==0)
        self.assertEqual(capacity - 2, memory._n)
        self.assertEqual(memory.demo_capacity - 1, memory._dn)
        self.assertEqual(capacity - 1, memory._an)

        memory.per_beta.step()
        memory.append(state, action=0, reward=1.0, next_state=state, done=True, is_demo=False)
        self.assertEqual(memory.demo_capacity, memory._an)

        memory.per_beta.step()
        memory.append(state, action=0, reward=1.0, next_state=state, done=True, is_demo=True)
        self.assertEqual(0, memory._dn)

if __name__ == "__main__":
    unittest.main()