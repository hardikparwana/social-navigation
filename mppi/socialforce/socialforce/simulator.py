# coding=utf-8

"""Synthetic pedestrian behavior according to the Social Force model.

See Helbing and Molnár 1998.
"""

import numpy as np

from .potentials import PedPedPotential
from .fieldofview import FieldOfView
from . import stateutils

MAX_SPEED_MULTIPLIER = 1.3  # with respect to initial speed

class Simulator(object):
    """Simulate social force model.

    Main interface is the state. Every pedestrian is an entry in the state and
    represented by a vector (x, y, v_x, v_y, d_x, d_y, [tau]).
    tau is optional in this vector.

    ped_space is an instance of PedSpacePotential.

    delta_t in seconds.
    tau in seconds: either float or numpy array of shape[n_ped].
    """
    def __init__(self, initial_state, ped_space=None, delta_t=0.4, tau=0.5, initial_speed=1.0, v0=2.1, sigma=0.3):
        self.state = initial_state
        self.initial_speeds = np.ones((initial_state.shape[0])) * initial_speed
        self.max_speeds = 0.5 #MAX_SPEED_MULTIPLIER * self.initial_speeds

        self.delta_t = delta_t

        if self.state.shape[1] < 7:
            if not hasattr(tau, 'shape'):
                tau = tau * np.ones(self.state.shape[0])
            self.state = np.concatenate((self.state, np.expand_dims(tau, -1)), axis=-1)

        # potentials
        self.V = PedPedPotential(self.delta_t, v0=v0, sigma=sigma)
        self.U = ped_space

        # field of view
        self.w = FieldOfView()

    def f_ab(self):
        """Compute f_ab."""
        return -1.0 * self.V.grad_r_ab(self.state)

    def f_aB(self):
        """Compute f_aB."""
        if self.U is None:
            return np.zeros((self.state.shape[0], 0, 2))
        return -1.0 * self.U.grad_r_aB(self.state)

    def capped_velocity(self, desired_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, self.max_speeds / desired_speeds)
        return desired_velocity * np.expand_dims(factor, -1)

    def step(self):
        """Do one step in the simulation and update the state in place."""
        # accelerate to desired velocity
        e = stateutils.desired_directions(self.state)
        vel = self.state[:, 2:4]
        tau = self.state[:, 6:7]
        F0 = 1.0 / tau * (np.expand_dims(self.initial_speeds, -1) * e - vel)

        # repulsive terms between pedestrians
        f_ab = self.f_ab()
        w = np.expand_dims(self.w(e, -f_ab), -1)
        F_ab = w * f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = self.f_aB()

        # social force
        F = F0 + np.sum(F_ab, axis=1) + np.sum(F_aB, axis=1)
        # desired velocity
        w = self.state[:, 2:4] + self.delta_t * F
        # velocity
        v = self.capped_velocity(w)

        # update state
        self.state[:, 0:2] += v * self.delta_t
        self.state[:, 2:4] = v

        return self
