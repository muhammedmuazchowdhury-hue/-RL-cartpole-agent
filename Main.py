import argparse
import gymnasium as gym
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# DQN Network
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dim=128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
MEMORY_CAPACITY = 10000
NUM_EPISODES = 500
HIDDEN_DIM = 128
SEED = 42
SOLVED_REWARD = 475
SOLVED_WINDOW = 100
SAVE_PATH = "cartpole_dqn.pth"
LOG_PATH = "cartpole_log.csv"

def select_action(state, policy_net, steps_done, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long), steps_done

def optimize_model(policy_net, target_net, memory, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        # Double DQN: action selection by policy_net, evaluation by target_net
        next_actions = policy_net(non_final_next_states).max(1).indices.view(-1, 1)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def soft_update(target_net, policy_net, tau):
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
    target_net.load_state_dict(target_net_state_dict)

def train(env, policy_net, target_net, memory, optimizer, num_episodes, render=False):
    steps_done = 0
    episode_durations = []
    episode_rewards = []
    best_avg_reward = -float('inf')
    log_data = []

    for i_episode in range(1, num_episodes + 1):
        state, info = env.reset(seed=SEED + i_episode)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0

        for t in count():
            action, steps_done = select_action(state, policy_net, steps_done, env)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(policy_net, target_net, memory, optimizer)
            soft_update(target_net, policy_net, TAU)

            if render:
                env.render()

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(total_reward)
                avg_reward = np.mean(episode_rewards[-SOLVED_WINDOW:]) if len(episode_rewards) >= SOLVED_WINDOW else np.mean(episode_rewards)
                eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                print(f"Episode {i_episode} | Steps: {t+1} | Reward: {total_reward:.1f} | Avg100: {avg_reward:.1f} | Epsilon: {eps:.3f}")
                log_data.append({'episode': i_episode, 'steps': t+1, 'reward': total_reward, 'avg100': avg_reward})

                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    torch.save(policy_net.state_dict(), SAVE_PATH)
                    print(f"  -> Best model saved to {SAVE_PATH}")

                if len(episode_rewards) >= SOLVED_WINDOW and np.mean(episode_rewards[-SOLVED_WINDOW:]) >= SOLVED_REWARD:
                    print(f"Solved in {i_episode} episodes! Average reward over last {SOLVED_WINDOW} episodes: {np.mean(episode_rewards[-SOLVED_WINDOW:]):.1f}")
                    pd.DataFrame(log_data).to_csv(LOG_PATH, index=False)
                    return episode_durations, episode_rewards
                break

    pd.DataFrame(log_data).to_csv(LOG_PATH, index=False)
    return episode_durations, episode_rewards

def evaluate(env, policy_net, num_episodes=10, render=True):
    policy_net.eval()
    rewards = []
    for i in range(num_episodes):
        state, info = env.reset(seed=SEED + 1000 + i)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            done = terminated or truncated
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            if render:
                env.render()
        rewards.append(total_reward)
        print(f"Eval Episode {i+1}: Reward = {total_reward}")
    policy_net.train()
    return rewards

def plot_results(episode_durations, episode_rewards):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(episode_durations, label='Duration', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(episode_rewards, label='Reward', color='tab:orange', alpha=0.7)
    ax2.set_ylabel('Reward', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Training Performance')
    fig.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='DQN for CartPole-v1')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--eval', action='store_true', help='Evaluate the agent')
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, help='Number of training episodes')
    parser.add_argument('--load', type=str, default=None, help='Path to load model')
    parser.add_argument('--save', type=str, default=SAVE_PATH, help='Path to save best model')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')
    args = parser.parse_args()

    global SAVE_PATH, SEED
    SAVE_PATH = args.save
    SEED = args.seed

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    n_actions = env.action_space.n
    state, info = env.reset(seed=SEED)
    n_observations = len(state)

    policy_net = DQN(n_observations, n_actions, HIDDEN_DIM).to(device)
    target_net = DQN(n_observations, n_actions, HIDDEN_DIM).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_CAPACITY)

    if args.load:
        policy_net.load_state_dict(torch.load(args.load, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Loaded model from {args.load}")

    if args.train:
        print("Training...")
        durations, rewards = train(env, policy_net, target_net, memory, optimizer, args.episodes, render=args.render)
        plot_results(durations, rewards)
    elif args.eval:
        print("Evaluating...")
        evaluate(env, policy_net, num_episodes=10, render=args.render)
    else:
        print("Please specify --train or --eval. Use --help for options.")

    env.close()

if __name__ == "__main__":
    main()