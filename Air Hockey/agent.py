import os
import random

import model
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from noise import OUNoise
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent_DDPG:
    def __init__(
        self,
        agent_id,
        model,
        action_size,
        seed=0,
        tau=1e-3,
        lr_actor=1e-4,  # actor has smaller learning rate
        lr_critic=1e-3,
        weight_decay=0.0,
    ):
        random.seed(seed)
        self.id = agent_id
        self.action_size = action_size
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.actor_local = model.actor_local
        self.actor_target = model.actor_target
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=lr_actor)

        self.critic_local = model.critic_local
        self.critic_target = model.critic_target
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=lr_actor, weight_decay=weight_decay
        )
        self.noise = OUNoise(action_size, seed)

        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

    def hard_copy_weights(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # get action based on state
    def act(self, state, noise_weight=1.0, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            self.noise_val = (
                self.noise.sample() * noise_weight
            )  # add progressivly smaller noise
            action += self.noise_val
        return np.clip(action, -1, 1)  # clip output betwen -1 and 1

    def reset(self):
        self.noise.reset()

    # learn based on replay memory samples
    def learn(self, agent_id, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences

        # update critic

        self.critic_optimizer.zero_grad()
        agent_id = torch.tensor([agent_id]).to(device)
        actions_next = torch.cat(all_next_actions, dim=1).to(device)

        with torch.no_grad():
            q_targets_next = self.critic_target(next_states, actions_next)

        q_expected = self.critic_local(states, actions)

        q_targets = rewards.index_select(1, agent_id) + (
            gamma * q_targets_next * (1 - dones.index_select(1, agent_id))
        )

        critic_loss = F.mse_loss(q_expected, q_targets.detach())
        critic_loss.backward()
        self.critic_optimizer.step()

        # update actor

        self.actor_optimizer.zero_grad()
        actions_pred = [
            actions if i == self.id else actions.detach()
            for i, actions in enumerate(all_actions)
        ]
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        actor_loss.backward()
        self.actor_optimizer.step()

        # update target networks based on: tau * local + tau * target
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                tau * local_param.data + (1.0 * tau) * target_param.data
            )

    # for playing against human
    def load_model(self):
        path = f"checkpoint_agent_1_play.pth"
        if not os.path.isfile(f"./{path}"):
            return
        i = 1
        checkpoint = torch.load(path, map_location=device)

        self.actor_local.load_state_dict(checkpoint[f"model{i}_actor_local"])
        self.actor_target.load_state_dict(checkpoint[f"model{i}_actor_target"])
        self.actor_optimizer.load_state_dict(checkpoint[f"model{i}_actor_optim"])

        self.actor_local.eval()
        self.actor_target.eval()

        print(f"Loaded checkpoints for agent {i} on device {device}")


# class with two ddpg agents
class Agent_MADDPG:

    def __init__(
        self,
        action_size=2,
        state_size=2,
        seed=0,
        n_agents=2,
        buffer_size=10000,
        batch_size=256,
        gamma=0.99,
        update_every=2,
        noise_start=1.0,
        noise_decay=1.0,
        t_stop_noise=30000,
    ):

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.n_agents = n_agents
        self.noise_weight = noise_start
        self.noise_decay = noise_decay
        self.t_step = 0
        self.noise_on = True
        self.t_stop_noise = t_stop_noise  # noise stops after t steps
        self.epoch = 0

        models = [
            model.Actor_Critic_Models(n_agents, state_size, action_size)
            for _ in range(n_agents)
        ]
        self.agents = [Agent_DDPG(i, models[i], action_size) for i in range(n_agents)]

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    # make step based on joined states and rewards from all agents
    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)
        all_next_states = all_next_states.reshape(1, -1)
        self.memory.add(
            all_states, all_actions, all_rewards, all_next_states, all_dones
        )

        if self.t_step > self.t_stop_noise:
            self.noise_on = False

        self.t_step = self.t_step + 1
        # Learn every update_every time steps.
        if self.t_step % self.update_every == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # sample from the replay buffer for each agent
                experiences = [self.memory.sample() for _ in range(self.n_agents)]
                self.learn(experiences, self.gamma)

    def act(self, all_states, add_noise=True):
        # pass each agent's state from the environment and calculate its action
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(
                state, noise_weight=self.noise_weight, add_noise=self.noise_on
            )
            self.noise_weight *= self.noise_decay
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        all_next_actions = []
        all_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(device)
            state = states.reshape(-1, 2, 11).index_select(1, agent_id).squeeze(1)
            action = agent.actor_local(state)
            all_actions.append(action)
            next_state = (
                next_states.reshape(-1, 2, 11).index_select(1, agent_id).squeeze(1)
            )
            next_action = agent.actor_target(next_state)
            all_next_actions.append(next_action)

        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)

    def save_agents(self):
        # Save all local and target networks and their optimizers
        for i, agent in enumerate(self.agents):
            torch.save(
                {
                    "epoch": self.epoch,
                    f"model{i}_actor_local": agent.actor_local.state_dict(),
                    f"model{i}_actor_target": agent.actor_target.state_dict(),
                    f"model{i}_actor_optim": agent.actor_optimizer.state_dict(),
                    f"model{i}_critic_local": agent.critic_local.state_dict(),
                    f"model{i}_critic_target": agent.critic_target.state_dict(),
                    f"model{i}_critic_optim": agent.critic_optimizer.state_dict(),
                },
                f"checkpoint_agent_{i}.pth",
            )

            print(f"Saved checkpoints for agent {i}")

    def load_agents(self):
        # load all data to resume training
        for i, agent in enumerate(self.agents):
            path = f"checkpoint_agent_{i}.pth"
            if not os.path.isfile(f"./{path}"):
                break
            checkpoint = torch.load(path, map_location=device)

            agent.actor_local.load_state_dict(checkpoint[f"model{i}_actor_local"])
            agent.actor_target.load_state_dict(checkpoint[f"model{i}_actor_target"])
            agent.actor_optimizer.load_state_dict(checkpoint[f"model{i}_actor_optim"])

            self.epoch = checkpoint["epoch"]

            agent.critic_local.load_state_dict(checkpoint[f"model{i}_critic_local"])
            agent.critic_target.load_state_dict(checkpoint[f"model{i}_critic_target"])
            agent.critic_optimizer.load_state_dict(checkpoint[f"model{i}_critic_optim"])

            agent.actor_local.train()
            agent.actor_target.train()

            agent.critic_local.train()
            agent.critic_target.train()

            print(f"Loaded checkpoints for agent {i} on device {device}")
