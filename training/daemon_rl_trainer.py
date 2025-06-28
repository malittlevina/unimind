

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class DaemonPolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DaemonPolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

def train_daemon(env_name='CartPole-v1', episodes=1000, gamma=0.99, learning_rate=1e-2):
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = DaemonPolicyNet(obs_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for episode in range(episodes):
        obs = env.reset()
        rewards, log_probs = [], []

        done = False
        while not done:
            obs_tensor = torch.from_numpy(obs).float()
            probs = model(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            obs, reward, done, _ = env.step(action.item())
            rewards.append(reward)

        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)

        loss = 0
        for log_prob, G in zip(log_probs, discounted_rewards):
            loss -= log_prob * G

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    env.close()
    return model