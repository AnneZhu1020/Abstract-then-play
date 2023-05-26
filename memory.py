from collections import namedtuple
import numpy as np
import random 

# State = namedtuple('State', ('obs', 'description', 'inventory', 'state'))
State = namedtuple('State', ('obs', 'description', 'inventory'))
Transition = namedtuple('Transition', ('state', 'act', 'reward', 'next_state', 'next_acts', 'done', 'acts', 'skill'))
Test_tuple = namedtuple('Test_tuple', ('state', 'skill', 'step'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def clear_memory(self):
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Test_tuple(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PrioritizedReplayMemory(object):
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha= alpha
        self.memory = []
        self.priorities = []
        self.position = 0

    def push(self, transition, priority):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = transition
        self.priorities[self.position] = priority 
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        priorities = np.power(priorities + 1e-5, self.alpha)
        p = priorities / np.sum(priorities)
        idxs = np.random.choice(np.arange(len(p)), size=batch_size, p=p)
        return [self.memory[i] for i in idxs]
    
    def update(self, idxs, priorities):
        for i, priority in zip(idxs, priorities):
            self.priorities[i] = priority

    def __len__(self):
        return len(self.memory)


class ABReplayMemory(object):
    def __init__(self, capacity, priority_fraction):
        self.priority_fraction = priority_fraction
        self.alpha_capacity = int(capacity * priority_fraction)
        self.beta_capacity = capacity - self.alpha_capacity
        self.alpha_memory, self.beta_memory = [], []
        self.alpha_position, self.beta_position = 0, 0

    def clear_alpha(self):
        self.alpha_memory = []
        self.alpha_position = 0

    def push(self, transition, is_prior=False):
        """Saves a transition."""
        if self.priority_fraction == 0.0:
            is_prior = False
        if is_prior:
            if len(self.alpha_memory) < self.alpha_capacity:
                self.alpha_memory.append(None)
            self.alpha_memory[self.alpha_position] = transition
            self.alpha_position = (self.alpha_position + 1) % self.alpha_capacity
        else:
            if len(self.beta_memory) < self.beta_capacity:
                self.beta_memory.append(None)
            self.beta_memory[self.beta_position] = transition
            self.beta_position = (self.beta_position + 1) % self.beta_capacity

    def sample(self, batch_size):
        if self.priority_fraction == 0.0:
            from_beta = min(batch_size, len(self.beta_memory))
            res = random.sample(self.beta_memory, from_beta)
        else:
            from_alpha = min(int(self.priority_fraction * batch_size), len(self.alpha_memory))
            from_beta = min(batch_size - int(self.priority_fraction * batch_size), len(self.beta_memory))
            res = random.sample(self.alpha_memory, from_alpha) + random.sample(self.beta_memory, from_beta)
        random.shuffle(res)
        return res

    def __len__(self):
        return len(self.alpha_memory) + len(self.beta_memory)
