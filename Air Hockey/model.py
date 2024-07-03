import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Init hidden layer weights between two limits
def hidden_init(layer):
    in_size = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(in_size)
    return (-lim, lim)


class ActorModel(nn.Module):

    def __init__(self, input_size, output_size, seed, fc1_units=256, fc2_units=256):
        super(ActorModel, self).__init__()
        self.seed = torch.manual_seed(seed)  # Set seed for pytorch random
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, output_size)
        self.bn = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.fc1(state))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))  # Scale output between -1 and 1
        return x


class CriticModel(nn.Module):

    def __init__(self, input_size, seed, fc1_units=256, fc2_units=256):

        super(CriticModel, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        xs = torch.cat((states, actions), dim=1)
        x = F.relu(self.fc1(xs))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor_Critic_Models:
    def __init__(self, n_agents, state_size, action_size, seed=0):
        self.actor_local = ActorModel(state_size, action_size, seed).to(device)
        self.actor_target = ActorModel(state_size, action_size, seed).to(device)

        critic_input_size = (state_size + action_size) * n_agents

        self.critic_local = CriticModel(critic_input_size, seed).to(device)
        self.critic_target = CriticModel(critic_input_size, seed).to(device)
