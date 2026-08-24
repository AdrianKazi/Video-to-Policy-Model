# utils/noise.py

import numpy as np


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma=0.2, theta=0.15, dt=1e-2):
        self.mean = mean
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.reset()

    def reset(self):
        self.x_prev = np.zeros_like(self.mean)

    def __call__(self):
        noise = (
            self.theta * (self.mean - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(*self.mean.shape)
        )
        self.x_prev += noise
        return self.x_prev