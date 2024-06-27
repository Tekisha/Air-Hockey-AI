import random
from collections import namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, input):

        x = F.leaky_relu(self.fc1(input))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def select_action(state, model, eps, n_actions):
    if random.random() > eps:
        with torch.no_grad():
            return model(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long).to(
            state.device
        )


def train(memory, batch_size, policy_net, target_net, optimizer, gamma):
    if len(memory.memory) < batch_size:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    next_state_batch = torch.cat(batch.next_state)
    done_batch = torch.cat(batch.done)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = target_net(next_state_batch).max(1)[0].detach()

    expected_state_action_values = (
        next_state_values * gamma * (1 - done_batch)
    ) + reward_batch

    expected_state_action_values = expected_state_action_values.unsqueeze(1)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
