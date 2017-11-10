import numpy as np
import random

class ReplayMemory(object):
    def __init__(self, size):
        self.storage = []
        self.size = size
        self.oldest = 0

    def __len__(self):
        return len(self.storage)

    def add(self, obs):
        if len(self.storage) == self.size:
            self.storage[self.oldest] = obs
            self.oldest = (self.oldest + 1) % self.size
        else:
            self.storage.append(obs)

    def sample(self, batch_size):
        rand_idx = [random.randint(0, len(self.storage)-1) for _ in range(batch_size)]
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in rand_idx:
            data = self.storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
