from replay_buffer import ReplayBuffer
from model import Actor2, Critic2
from noise import OUNoise

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

MEMORY_BATCH_SIZE=1024
MEMORY_SIZE=int(1e6)
GAMMA=0.99
TAU = 1e-3

class Agent:
    def __init__(self, state_size, action_size, rand_seed, actor_lr, critic_lr):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        
        self.mem = ReplayBuffer(device=device, batch_size=MEMORY_BATCH_SIZE, memory_size=MEMORY_SIZE)
        self.noise = OUNoise(action_size, rand_seed)
        
        self.actor_local = Actor2(state_size, action_size, fc1=400, fc2=300, seed=rand_seed)
        self.actor_target = Actor2(state_size, action_size, fc1=400, fc2=300, seed=rand_seed)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
        
        self.critic_local = Critic2(state_size, action_size, fc1=400, fc2=300, seed=rand_seed)
        self.critic_target = Critic2(state_size, action_size, fc1=400, fc2=300, seed=rand_seed)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr, weight_decay=0)
        
    def load_state(self, state_name):
        """
        Restore state (=weights).
        
        Params:
            state_name (string): Name of the state to restore agent from.
        """
        self.actor_local.load_state_dict(torch.load(f'states/actor_{state_name}.pth'))
        self.critic_local.load_state_dict(torch.load(f'states/critic_{state_name}.pth'))
        
    def save_state(self, state_name):
        """
        Save current state (=weights). This can later be restored using `load_state`. 
        
        Params:
            state_name (string): Name to be used to save the state as.
        """
        torch.save(self.actor_local.state_dict(), f'states/actor_{state_name}.pth')
        torch.save(self.critic_local.state_dict(), f'states/critic_{state_name}.pth')
        
    def act(self, states, add_noise=True):
        """
        Get actions to take from given states. Tensors will be returned on the device on which agent is configured to run on.
        
        params:
            states: 1D tenor of states for which to get actions to return.
        """
        states = torch.from_numpy(states).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)
    
    def learn(self):
        """
        Start learning process
        """
        if len(self.mem) > MEMORY_BATCH_SIZE:
            experiences = self.mem.sample()
            self._do_learn(experiences)
            
    def _do_learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences        
        q_targets = rewards + GAMMA * self.critic_target(next_states, self.actor_target(next_states)) * (1 - dones)
        q_expectations = self.critic_local(states, actions)
        critic_loss = F.mse_loss(q_expectations, q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        actions_loss = -self.critic_local(states, self.actor_local(states)).mean()
        self.actor_optimizer.zero_grad()
        actions_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
        
    
    def step(self, sarss):
        """
        Add each of (state, action, reward, next_state, done) tuples to memory.
        
        params:
            sarss: an array of (state, action, reward, next, done) tuples to add to the memory.
        """
        for state, action, reward, next_state, done in sarss:
            self.mem.add(state, action, reward, next_state, done)
        
    