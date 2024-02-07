import jax
import jax.numpy as jnp
from jax.random import multivariate_normal

class MPPI():

    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    def __init__(self, horizon=10, samples = 10, input_size = 2, dt=0.05):
        self.key = jax.random.PRNGKey(111)
        self.horizon = horizon
        self.samples = samples
        self.input_size = input_size
        self.dt = dt
        self.control_mu = jnp.zeros(input_size)
        self.control_cov = 1.0 * jnp.eye(input_size)
        self.control_bound_lb = -jnp.array([[1], [1]])
        self.control_bound_ub = -self.control_bound_lb
        # self.U = -0.1 * jnp.ones((self.horizon, input_size))  # T x nu
        self.U = jnp.append(  -0.5 * jnp.ones((self.horizon, 1)), jnp.zeros((self.horizon,1)), axis=1  )

    # Linear dynamics for now
    def dynamics_step(self, state, input):
        return state + input * self.dt

    def rollout_states(self, init_state, perturbed_control, goal, human_states, human_controls ):
        states = jnp.tile( init_state, (self.samples,1) )
        humanX = jnp.copy(human_states)
        cost_total = jnp.zeros(self.samples)
        for i in range(self.horizon):
            states = jnp.append( states, self.dynamics_step( states[:,[i]], perturbed_control[:, i, :].reshape(-1,1) ), axis=1 )
            for j in range(self.samples):
                state = states[2*j:2*j+2,[i]]
                cost_total = cost_total.at[j].set( (3.0 * (state-goal).T @ (state-goal) + 1/max(jnp.min(jnp.linalg.norm(state - humanX, axis=0)), 0.01))[0,0] ) # assume human: horizon x 2 x num_humans
            humanX = humanX + human_controls * self.dt
        return states, cost_total
    
    def rollout_control(self, init_state, actions):
        states = jnp.copy(init_state)
        for i in range(self.horizon):
            states = jnp.append( states, self.dynamics_step(states[:,[-1]], actions[i,:].reshape(-1,1)), axis=1 )
        return states

    def compute_rollout_costs( self, init_state, goal, human_states, human_controls ):

        self.key, subkey = jax.random.split(self.key)
        perturbation = multivariate_normal( subkey, self.control_mu, self.control_cov, shape=( self.samples, self.horizon ) ) # K x T x nu
        perturbation = jnp.clip( perturbation, -0.3, 0.3 )
        perturbed_control = self.U + perturbation
        # perturbed_control = jnp.clip( perturbed_control.T, self.control_bound_lb, self.control_bound_ub ).T
        sampled_states, costs = self.rollout_states(init_state, perturbed_control, goal, human_states, human_controls)

        lambd = 1.0
        weights = jnp.exp( - 1.0/lambd * costs )
        normalization_factor = jnp.sum(weights)
        for i in range(self.samples):
            self.U = self.U + perturbation[i] * weights[i] / normalization_factor
        states_final = self.rollout_control(init_state, self.U)              
        action = self.U[0,:].reshape(-1,1)

        return sampled_states, states_final, action

