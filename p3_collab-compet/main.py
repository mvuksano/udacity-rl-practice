import argparse
import numpy as np
import torch
import os

from agent import Maddpg
from collections import deque
from unityagents import UnityEnvironment
from replay_buffer import ReplayBuffer


RAND_SEED=0
STATE_SIZE=24
ACTION_SIZE=2
NUM_AGENTS=2
AGENT_LR=0.001
CRITIC_LR=0.001
NOISE_DECAY=0.999
LEARN_EVERY=1
GOAL_SCORE=0.5
SAVE_INTERVAL=100

NUM_EPISODES=20000
NUM_AGENTS=2
MAX_TIMESTEPS=5000

MEMORY_BATCH_SIZE=512
MEMORY_SIZE=int(1e6)

def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--episode", type=str, help="File name from which a model should be restored")
    parser.add_argument("--play", action="store_true", help="run play mode.")

    args = parser.parse_args()
    

    seeding(RAND_SEED)

    log_path = os.getcwd()+"/log"
    model_dir= os.getcwd()+"/model_dir"
    os.makedirs(model_dir, exist_ok=True)


    env, brain_name = start_unity_env("Tennis.app")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    
    memory = ReplayBuffer(device=device, batch_size=MEMORY_BATCH_SIZE, memory_size=MEMORY_SIZE)
    maddpg = Maddpg(STATE_SIZE, ACTION_SIZE, NUM_AGENTS, AGENT_LR, CRITIC_LR, NOISE_DECAY, memory, device)
    if args.play:
        print(f"Restoring from: {args.episode}")
        restore_agents(maddpg, model_dir, args.episode)
        play(maddpg, env, brain_name)
    else:
        train(NUM_EPISODES, NUM_AGENTS, MAX_TIMESTEPS, maddpg, env, brain_name, LEARN_EVERY, GOAL_SCORE, SAVE_INTERVAL, model_dir)

def start_unity_env(file_name):
    env = UnityEnvironment(file_name)
    brain_name = env.brain_names[0]
    return env, brain_name

def train(num_episodes, num_agents, max_timesteps, agents, env, brain_name, learn_every, goal_score, save_interval, model_dir):
    scores_window = deque(maxlen=100)
    for e in range(num_episodes):
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agents.reset_noise()
        for t in range(max_timesteps):            
            actions = agents.act(states, 0)
            env_info = env.step(actions.reshape(1, -1).squeeze().tolist())[brain_name]
            next_states = env_info.vector_observations
            dones = env_info.local_done
            agents.step(states, actions, env_info.rewards, next_states, dones)
            
            if t % learn_every == 0:
                agents.learn()

                
            states = next_states
            scores += env_info.rewards
            if np.any(dones):
                break

        episode_reward = np.max(scores)
        scores_window.append(episode_reward)
        current_avg_score_over_window = np.mean(scores_window)
        
        if e % save_interval == 0:
            persist_models(agents, model_dir, e)
            
        if current_avg_score_over_window > goal_score:
            print("Problem solved in {} episodes".format(e))
            persist_models(agents, model_dir, e)
            break

def play(agents, env, brain_name):
    while True:
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations
        done = False
        while not done:
            env_info = env.step(agents.act(state, 0))[brain_name]
            state = env_info.vector_observations
            done = any(env_info.local_done)

def persist_models(agents, model_dir, current_episode):
    agents.save_agents(model_dir, current_episode)

def restore_agents(agents, model_dir, episode):
    agents.load_agents(model_dir, episode)

if __name__=='__main__':
    main()
