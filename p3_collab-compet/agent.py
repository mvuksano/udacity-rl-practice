from model import Actor, Critic
from noise import OUNoise

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import random

GAMMA=0.99
TAU = 1e-2

class Maddpg(object):
    def __init__(self, state_size, action_size, n_agents, actor_lr, critic_lr, noise_decay, logger, memory, device): 
        self.device = device
        self.n_agents = n_agents
        self.logger = logger
        self.learn_step = 0
        self.mem = memory
        self.agents = []
        for i in range(n_agents):
            self.agents.append(DdpgAgent(i, state_size, action_size, state_size*n_agents, action_size*n_agents, actor_lr, critic_lr, noise_decay, device ))
            
    def reset_noise(self):
        for agent in self.agents:
            agent.reset()
            
    def save_agents(self, model_dir, episode):
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor_local.state_dict(),  f"{model_dir}/checkpoint_actor_agent_{i}_{episode}.pth")
            torch.save(agent.critic_local.state_dict(), f"{model_dir}/checkpoint_critic_agent_{i}_{episode}.pth")
            
    def load_agents(self, model_dir, episode):
        for i, agent in enumerate(self.agents):
            agent.actor_local.state_dict(torch.load(f'{model_dir}/checkpoint_actor_agent_{i}_{episode}.pth'))
            agent.critic_local.state_dict(torch.load(f'{model_dir}/checkpoint_critic_agent_{i}_{episode}.pth'))
        
        
    def act(self, states, eps):
        """
        Decide which action to take for each of the agents. Returns a 2D numpy array of actions for each of the agents.
        
        Params:
            states (2D numpy array): State of the environment for each of the agents. First dim selects an agent.
        """
        all_actions = []
        for i in range(self.n_agents):
            actions = self.agents[i].act(states[i], eps)
            all_actions.append(actions)
            
        return np.array(all_actions)
    
    def step(self, joint_states, joint_actions, joint_rewards, joint_next_states, joint_dones):
        """
        Add 
        """
        # `joint_states` and `joing_next_states` are a (2x24) matrix and due to how ReplayBuffer is implemented
        # we need to reshape those into (1x48)
        self.mem.add(joint_states.reshape(1, -1),
                     joint_actions.reshape(1, -1),
                     np.array(joint_rewards).reshape(1, -1),
                     joint_next_states.reshape(1, -1),
                     np.array(joint_dones).reshape(1, -1))

        
    def learn(self):
        """
        Start learning processs in case there is enough information stored in memory.
        """
        if self.mem.is_ready():
            experiences = [self.mem.sample() for _ in range(self.n_agents)]
            self._do_learn(experiences)
            
    
    def _do_learn(self, experiences):
        """
        """
        joint_predicted_actions_from_state=[]
        joint_predicted_actions_from_next_state=[]

        self.learn_step += 1
        
        for i in range(self.n_agents):
            agent_id = torch.tensor([i]).to(self.device)
            states, _, _, next_states, _ = experiences[i]
            # Get a 3D tensor of states for an agent
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            predicted_action_from_state = self.agents[i].actor_local(state)
            joint_predicted_actions_from_state.append(predicted_action_from_state)
            
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            predicted_action_from_next_state = self.agents[i].actor_target(next_state)
            joint_predicted_actions_from_next_state.append(predicted_action_from_next_state)
            
        for i in range(self.n_agents):
            e = self.fixup_experiences_for_agent(i, experiences[i])
            joint_predicted_actions_from_state = [a if i == j else a.detach() for j, a in enumerate(joint_predicted_actions_from_state)]
            c_loss, a_loss = self.agents[i].learn(e,
                                   torch.cat(joint_predicted_actions_from_state, dim=1),
                                   torch.cat(joint_predicted_actions_from_next_state, dim=1))
            self.logger.add_scalar(f'critic_loss_{i}', c_loss, self.learn_step)
            self.logger.add_scalar(f'actor_loss_{i}', a_loss, self.learn_step)
        
    def fixup_experiences_for_agent(self, agent_idx, experiences):
        """
        Removes information about rewards and `dones` for all other agents besides the one identified with `agent_idx`.
        """
        states, actions, rewards, next_states, dones = experiences
        agent_idx = torch.tensor([agent_idx]).to(self.device)
        
        return (states, actions, rewards.index_select(1, agent_idx), next_states, dones.index_select(1, agent_idx))

class DdpgAgent(object):
    def __init__(self, name, state_size, action_size, joint_state_size, joint_action_size, actor_lr, critic_lr, noise_decay, device):
        self.name = name
        self.device = device
        self.noise = OUNoise(action_size)
        
        self.actor_local = Actor(state_size, action_size, fc1=256, fc2=256).to(device)
        self.actor_target = Actor(state_size, action_size, fc1=256, fc2=256).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
        
        self.critic_local = Critic(joint_state_size, joint_action_size, fc1=256, fc2=256).to(device)
        self.critic_target = Critic(joint_state_size, joint_action_size, fc1=256, fc2=256).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)
        
        self.noise_decay = noise_decay
        self.noise_weight = 0.7
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

    def reset(self):
        self.noise.reset()

    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
        
    def act(self, states, eps, add_noise=True):
        """
        Get actions to take from given states. Tensors will be returned on the device on which agent is configured to run on.
        
        params:
            states: 1D tenor of states for which to get actions to return.
        """
        s = np.random.random()
        if s < eps:
            action = np.random.uniform(-1,1,2)
            return action
        
        states = torch.from_numpy(states).float()
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states.unsqueeze(0)).squeeze(0).data.numpy()
        self.actor_local.train()
        if add_noise:
                action += self.noise.sample() 
        return np.clip(action, -1, 1)
    
    def learn(self, experiences, joint_predicted_actions_from_state, joint_predicted_actions_from_next_state):
        """
        Start learning process
        """
        joint_states, joint_actions, rewards, joint_next_states, dones = experiences
        
        q_targets_next = self.critic_target(joint_next_states, joint_predicted_actions_from_next_state)
        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))
        q_expected = self.critic_local(joint_states, joint_actions)
        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        critic_loss_value = critic_loss.item()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        # print("Joint next states: {}".format(q_targets - q_expected))
        
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic_local(joint_states, joint_predicted_actions_from_state).mean()
        actor_loss_value = actor_loss.item()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)     

        return critic_loss_value, actor_loss_value

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
        
    
        
    