import torch
import torch.optim as optim
from collections import deque

class Trainer:
    def __init__(self, env, brain_name, policy):
        self.env = env
        self.policy = policy
        self.brain_name = brain_name
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        
    def get_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def reinforce(self, n_episodes=1000, n_timesteps=1000, gamma=0.9, print_every=100):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        scores_deque = deque(maxlen=100)
        scores = []
        device = self.get_device()
        self.policy.to(device)
        for episode in range(n_episodes):
            saved_log_probs = []
            rewards = []
            state = env_info.vector_observations[0]
            for timestep in range(n_timesteps):
                action, log_prob = self.policy.act(state)
                saved_log_probs.append(log_prob)
                print("{}".format(action))
                env_info = self.env.step([action])[brain_name]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                state = next_states[0]
                rewards.append(reward)
                if done:
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))
            
            discounts = [gamma ** i for i in range(len(rewards))]
            R = sum([a*b for a, b in zip(discounts, rewards)])
            
            policy_losses = []
            for log_prob in saved_log_probs:
                policy_losses.append(-log_prob*R)
            policy_loss = torch.cat(policy_losses).sum()
            
            self.optim.zero_grad()
            policy_loss.backward()
            self.optim.step()
            
            if episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            if np.mean(scores_deque)>=30.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_deque)))
                break
            
            return scores
    
