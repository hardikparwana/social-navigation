import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax

class MPPI():

    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    samples = []
    horizon = []
    dt = 0.05

    def __init__(self, horizon=10, samples = 10, input_size = 2, dt=0.05):
        self.key = jax.random.PRNGKey(111)
        MPPI.horizon = horizon
        MPPI.samples = samples
        self.input_size = input_size
        MPPI.dt = dt
        self.control_mu = jnp.zeros(input_size)
        self.control_cov = 2.0 * jnp.eye(input_size)
        self.control_bound_lb = -jnp.array([[1], [1]])
        self.control_bound_ub = -self.control_bound_lb
        self.U = jnp.append(  -0.5 * jnp.ones((MPPI.horizon, 1)), jnp.zeros((MPPI.horizon,1)), axis=1  ) # T x nu

    # Linear dynamics for now
    @staticmethod
    def dynamics_step(state, input):
        return state + input * MPPI.dt
    
    @staticmethod
    @jit
    def body_sample(cost_total, states_t, humanX, goal):            
        def body(j, samples):
            cost_total = samples
            state = states_t[ :,[j] ]   #states_t[2*j:2*j+2,0]
            cost_total = cost_total.at[j].set( cost_total[j] + ( 1.0 * (state-goal).T @ (state-goal) + 3/jnp.max( jnp.array([jnp.min(jnp.linalg.norm(state - humanX, axis=0)), 0.01]) ) )[0,0] ) # assume human: horizon x 2 x num_humans
            return cost_total
        return lax.fori_loop( 0, MPPI.samples, body, (cost_total) )
    
    @staticmethod
    @jit
    def rollout_states(init_state, perturbed_control, goal, human_states, human_controls ):
        states = jnp.zeros( (2*MPPI.samples, MPPI.horizon) )
        states = states.at[:,0].set( jnp.tile( init_state, (MPPI.samples,1) )[:,0] )
        humanX = jnp.copy(human_states)
        cost_total = jnp.zeros(MPPI.samples)

        # Loop over horizon
        @jit
        def body_horizon(i, inputs):
            states, humanX, cost_total = inputs     

            # Get cost
            cost_total = MPPI.body_sample( cost_total, states[:,[i]].reshape((2,MPPI.samples), order='F'), humanX, goal  )

            # Update state
            states = states.at[:,i+1].set( MPPI.dynamics_step( states[:,[i]], perturbed_control[:, i, :].reshape(-1,1) )[:,0] )
            humanX = humanX + human_controls * MPPI.dt

            return states, humanX, cost_total   

        states, humanX, cost_total = lax.fori_loop( 0, MPPI.horizon, body_horizon, (states, humanX, cost_total) )
        return states, cost_total
    
    def rollout_control(self, init_state, actions):
        states = jnp.copy(init_state)
        for i in range(MPPI.horizon):
            states = jnp.append( states, self.dynamics_step(states[:,[-1]], actions[i,:].reshape(-1,1)), axis=1 )
        return states
    

    
    @staticmethod
    def weighted_sum(U, perturbation, weights):
        normalization_factor = jnp.sum(weights)
        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U
        return lax.fori_loop( 0, MPPI.samples, body, (U) )
    
        # for i in range(MPPI.samples):
        #     U = U + perturbation[i] * weights[i] / normalization_factor
        # return U


    def compute_rollout_costs( self, init_state, goal, human_states, human_controls ):

        self.key, subkey = jax.random.split(self.key)
        perturbation = multivariate_normal( subkey, self.control_mu, self.control_cov, shape=( MPPI.samples, MPPI.horizon ) ) # K x T x nu
        
        perturbation = jnp.clip( perturbation, -0.8, 0.8 ) #0.3
        perturbed_control = self.U + perturbation

        # perturbed_control = jnp.clip( perturbed_control, -2.0, 2.0 )
        # perturbation = perturbed_control - self.U

        sampled_states, costs = MPPI.rollout_states(init_state, perturbed_control, goal, human_states, human_controls)

        lambd = 10 #1000 #  1.0
        weights = jnp.exp( - 1.0/lambd * costs )        
        print(f" max costs:{jnp.max(costs)}, weights: {jnp.max(weights)} ")
        self.U = MPPI.weighted_sum( self.U, perturbation, weights )

        states_final = self.rollout_control(init_state, self.U)              
        action = self.U[0,:].reshape(-1,1)
        self.U = jnp.append(self.U[1:,:], self.U[[-1],:], axis=0)

        return sampled_states, states_final, action


        
    # def rollout_states(self, init_state, perturbed_control, goal, human_states, human_controls ):
    #     states = jnp.tile( init_state, (MPPI.samples,1) )
    #     humanX = jnp.copy(human_states)
    #     cost_total = jnp.zeros(MPPI.samples)
    #     for i in range(MPPI.horizon):
    #         states = jnp.append( states, self.dynamics_step( states[:,[i]], perturbed_control[:, i, :].reshape(-1,1) ), axis=1 )
    #         for j in range(MPPI.samples):
    #             state = states[2*j:2*j+2,[i]]
    #             cost_total = cost_total.at[j].set( (3.0 * (state-goal).T @ (state-goal) + 3/max(jnp.min(jnp.linalg.norm(state - humanX, axis=0)), 0.01))[0,0] ) # assume human: horizon x 2 x num_humans
    #             # cost_total = cost_total.at[j].set( (3.0 * (state-goal).T @ (state-goal) + 1/max(jnp.min(jnp.linalg.norm(state - humanX, axis=0)), 0.01))[0,0] ) # assume human: horizon x 2 x num_humans
    #         humanX = humanX + human_controls * self.dt
    #     return states, cost_total