# agents/td3.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.actor import Actor
from models.twin_critic import TwinCritic
from utils.noise import OrnsteinUhlenbeckActionNoise
from utils.replay_buffer import ReplayBuffer
from config.config import *


class TD3_Agent:
    def __init__(self, state_dim, action_dim, max_action):

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TwinCritic(state_dim, action_dim)
        self.critic_target = TwinCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.memory = ReplayBuffer(REPLAY_BUFFER_SIZE)

        self.max_action = max_action
        self.train_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)

        self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(action_dim))

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().detach().numpy()
        action += np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def store(self, s, a, r, s2, d):
        self.memory.add((s, a, r, s2, d))

    def train(self):
        if self.memory.size() < MIN_MEMORY:
            return

        batch = self.memory.sample(BATCH_SIZE)
        s, a, r, s2, d = zip(*batch)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s2 = torch.FloatTensor(s2).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = torch.clamp(
                torch.normal(0, TARGET_POLICY_CLIP, size=a.shape).to(self.device),
                -0.5, 0.5
            )
            a2 = torch.clamp(self.actor_target(s2) + noise, -self.max_action, self.max_action)

            q1_t, q2_t = self.critic_target(s2, a2)
            target_q = r + GAMMA * (1 - d) * torch.min(q1_t, q2_t)

        q1, q2 = self.critic(s, a)
        critic_loss = F.smooth_l1_loss(q1, target_q) + F.smooth_l1_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.train_step % POLICY_DELAY == 0:
            actor_loss = -self.critic(s, self.actor(s))[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(TAU * p.data + (1 - TAU) * tp.data)

        self.train_step += 1