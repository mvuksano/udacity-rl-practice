import torch
import random
import numpy as np
from collections import deque, namedtuple

class ReplayBuffer:
    def __init__(self, device, batch_size, memory_size):
        """
        ReplayBuffer is initialised and its size can be specified via size parameter.
        """
        self.batch_size = batch_size
        
        self.memory = deque(maxlen=memory_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.device = device
    
    def __len__(self):
        return len(self.memory)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add new experience to memory.
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        
        Returns a tuple containing tensors of `states`, `actions`, `rewards`, `next_states` and `dones`. The tensors will be placed on device chosen when initialising ReplayBuffer.
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def is_ready(self):
        if len(self) >= self.batch_size:
            return True
        else:
            return False