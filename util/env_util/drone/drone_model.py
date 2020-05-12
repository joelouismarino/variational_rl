import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RHO = 1.225
GRAVITY = 9.81

def read_weight(filename):
    model_weight = torch.load(filename)
    model = Network()
    model.load_state_dict(model_weight)
    return model


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(12, 25)
        self.fc2 = nn.Linear(25, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class DroneModel(object):
    """
    A neural network-based model for drone landing.

    Args:
        batch_size (int, optional): number of parallel environments
        stochastic (bool, optional): whether to add acceleration and control noise
    """
    def __init__(self, batch_size=1, stochastic=False):
        self.batch_size = batch_size

        # Drone parameters
        self.mass = 1.47                  # mass
        self.D = 0.23                     # diameter
        self.CT = 0.08937873              # thrust coeff

        # Desired trajectory parameters
        self.h_d = 0.                     # desired landing height

        # NN model for unknown dynamics
        self.Fa_model = read_weight('util/env_util/drone/Fa_net_12_3_full_Lip16.pth')

        # Real states
        self.z = 0                        # height
        self.v = 0                        # velocity
        self.a = 0                        # acceleration
        self.u = 0                        # control signal
        # self.prev_u = 6508                # previous control signal
        self.Fa = 0                       # ground truth Fa
        self.F = 0                        # force

        # Noise
        self.stochastic = stochastic
        self.a_noise_sigma = 0.1
        self.u_noise_sigma = 0.01
        self.a_noise = 0
        self.u_noise = 0

        # Step
        self.step_size = 1e-2
        self.total_step = 0
        self.sim_duration = 5

        self.device = None

        self.reset()

    def step(self, u):
        """
        Steps the model forward one time step.

        Args:
            u (torch.Tensor): the control input [batch_size, 1]
                              in the range [-1, +1]
        """
        if type(u) == np.ndarray:
            u = torch.from_numpy(u).to(self.device)

        # sample acceleration and control noise
        if self.stochastic:
            # Noise freq is 10
            if not self.total_step % int(1 / self.step_size * 0.1):
                self.a_noise = torch.normal(torch.zeros_like(self.a), self.a_noise_sigma)
                self.a_noise = self.a_noise.clamp(-3 * self.a_noise_sigma, 3 * self.a_noise_sigma)
                self.u_noise = torch.normal(torch.zeros_like(u), self.u_noise_sigma)
                self.u_noise = self.u_noise.clamp(-3 * self.u_noise_sigma, 3 * self.u_noise_sigma)

        # Consider control delay
        # u = 0.8 * self.prev_u.detach() + 0.2 * u
        u = u + self.u_noise
        self.u = u
        # self.prev_u = u

        # ODE solver: (4,5) Runge-Kutta
        prev_z = self.z
        prev_v = self.v

        self.dynamics()
        s1_dz = self.v
        s1_dv = self.a

        self.z = prev_z + 0.5 * self.step_size * s1_dz
        self.v = prev_v + 0.5 * self.step_size * s1_dv
        self.dynamics()
        s2_dz = self.v
        s2_dv = self.a

        self.z = prev_z + 0.5 * self.step_size * s2_dz
        self.v = prev_v + 0.5 * self.step_size * s2_dv
        self.dynamics()
        s3_dz = self.v
        s3_dv = self.a

        self.z = prev_z + self.step_size * s3_dz
        self.v = prev_v + self.step_size * s3_dv
        self.dynamics()
        s4_dz = self.v
        s4_dv = self.a

        self.z = prev_z + 1.0 / 6 * self.step_size * \
                      (s1_dz + 2 * s2_dz + 2 * s3_dz + s4_dz)
        self.v = prev_v + 1.0 / 6 * self.step_size * \
                      (s1_dv + 2 * s2_dv + 2 * s3_dv + s4_dv)

        self.a = (self.v - prev_v) / self.step_size

        self.Fa = self.mass * (self.a + GRAVITY) - self.F

        self.total_step += 1

        done = True if self.step_size*self.total_step >= self.sim_duration else False
        cost = self.compute_cost()

        return self.state, -cost, done

    def compute_cost(self):
        """
        Computes the cost of the current state-action pair.
        """
        height_cost = 0.05 * (self.z - self.h_d) ** 2
        control_cost = (self.u + 1.) ** 2
        return height_cost + control_cost

    def dynamics(self):
        """
        Steps the neural network model.
        """
        Fa = self.Fa_model(self.model_state)[:, 2]
        u = (2000. * self.u) + 6000.
        self.F = 4 * self.CT * RHO * u ** 2 * self.D ** 4 / 3600.
        self.a = self.F/self.mass - GRAVITY + self.a_noise # + Fa/self.mass

    @property
    def model_state(self):
        """
        Collects the system state to feed to the model.
        """
        state = torch.zeros([self.batch_size, 12]).to(self.device)
        state[:, 0] = self.z
        state[:, 3] = self.v
        state[:, 7] = torch.ones([self.batch_size, 1]).to(self.device)
        state[:, 8:12] = 0.75 + self.u / 4.
        return state

    @property
    def state(self):
        """
        Collects the system state.
        """
        state = torch.zeros([self.batch_size, 3]).to(self.device)
        state[:, 0] = self.z
        state[:, 1] = self.v
        state[:, 2] = 0.75 + self.u / 4.
        return state

    def set_state(self, state, prev_u=None):
        """
        Set the state of the environment.
        """
        self.z = state[:, 0:1].to(self.device)
        self.v = state[:, 1:2].to(self.device)
        self.u = state[:, 2:3].to(self.device)
        # self.prev_u = prev_u.to(self.device) if prev_u is not None else self.u

    def to(self, device_id):
        """
        Place the environment on the specified device.

        Args:
            device_id (int): GPU index
        """
        self.device = device_id
        self.Fa_model = self.Fa_model.to(device_id)
        self.z = self.z.to(device_id)
        self.v = self.v.to(device_id)
        self.u = self.u.to(device_id)
        # self.prev_u = self.prev_u.to(device_id)
        self.Fa = self.Fa.to(device_id)
        self.F = self.F.to(device_id)
        return self

    def reset(self):
        """
        Reinitialize the model initial state.
        """
        self.z = torch.zeros(self.batch_size, 1).normal_(1.5, 0.25).to(self.device)
        self.v = torch.zeros(self.batch_size, 1).normal_(0., 0.05).to(self.device)
        self.a = torch.zeros(self.batch_size, 1).to(self.device)
        self.u = torch.ones(self.batch_size, 1).uniform_(-1, 1).to(self.device)
        # self.prev_u = torch.ones(self.batch_size, 1).uniform_(-1, 1).to(self.device)
        self.Fa = torch.zeros(self.batch_size, 1).to(self.device)
        self.F = torch.zeros(self.batch_size, 1).to(self.device)
        self.total_step = 0
        return self.state
