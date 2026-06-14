# models/twin_critic.py

import torch
import torch.nn as nn


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1
        self.fc1_1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2_1 = nn.Linear(512, 512)
        self.fc3_1 = nn.Linear(512, 256)
        self.q1 = nn.Linear(256, 1)

        # Q2
        self.fc1_2 = nn.Linear(state_dim + action_dim, 512)
        self.fc2_2 = nn.Linear(512, 512)
        self.fc3_2 = nn.Linear(512, 256)
        self.q2 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)

        x1 = torch.relu(self.fc1_1(sa))
        x1 = torch.relu(self.fc2_1(x1))
        x1 = torch.relu(self.fc3_1(x1))
        q1 = self.q1(x1)

        x2 = torch.relu(self.fc1_2(sa))
        x2 = torch.relu(self.fc2_2(x2))
        x2 = torch.relu(self.fc3_2(x2))
        q2 = self.q2(x2)

        return q1, q2