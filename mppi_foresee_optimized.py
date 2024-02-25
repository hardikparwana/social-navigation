import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax
from ut_utils_jax import *
import time

class MPPI_FORESEE():

    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    """

    samples = []
    horizon = []
    dt = 0.05
    human_n = 2
    robot_n = 2
    m = 2
    sensing_radius = 2
    human_noise_cov = 0.01
    std_factor = 2.0

    def __init__(self, horizon=10, samples = 10, input_size = 2, dt=0.05, sensing_radius=2, human_noise_cov=0.01, std_factor=1.96, control_bound=7, control_init_ratio=1, u_guess=None):
        self.key = jax.random.PRNGKey(111)
        MPPI_FORESEE.human_n = 2
        MPPI_FORESEE.robot_n = 2
        MPPI_FORESEE.m = 2
        MPPI_FORESEE.horizon = horizon
        MPPI_FORESEE.samples = samples
        MPPI_FORESEE.sensing_radius = sensing_radius
        MPPI_FORESEE.human_noise_cov = human_noise_cov
        MPPI_FORESEE.dt = dt
        MPPI_FORESEE.std_factor = std_factor

        self.input_size = input_size        
        self.control_bound = control_bound
        self.control_mu = jnp.zeros(input_size)
        self.control_cov = 30.0 * jnp.eye(input_size)  #2.0 * jnp.eye(input_size)
        self.control_bound_lb = -jnp.array([[1], [1]])
        self.control_bound_ub = -self.control_bound_lb  
        if u_guess != None:
            self.U = u_guess
        else:
            # self.U = jnp.append(  1.0 * jnp.ones((MPPI_FORESEE.horizon, 1)), control_init_ratio * 1.0 * jnp.ones((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu
            self.U = jnp.append(  1.0 * jnp.ones((MPPI_FORESEE.horizon, 1)), 1.0 * jnp.ones((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu
            # self.U = jnp.append(  -0.5 * jnp.ones((MPPI_FORESEE.horizon, 1)), jnp.zeros((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu

    # Linear dynamics for now
    @staticmethod
    def robot_dynamics_step(state, input):
        return state + input * MPPI_FORESEE.dt
    
    @staticmethod
    @jit
    def single_sample_state_cost(robot_state, human_sigma_points, human_sigma_weights, goal):       
        human_dist_sigma_points = jnp.linalg.norm(robot_state - human_sigma_points, axis=0).reshape(1,-1)
        mu_human_dist, cov_human_dist = get_mean_cov( human_dist_sigma_points, human_sigma_weights )
        # cost_total = cost_total + 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 3.0 / jnp.max(  jnp.array([mu_human_dist[0,0] - MPPI_FORESEE.std_factor * jnp.sqrt(cov_human_dist[0,0]), 0.01 ]) )
        cost = 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 10.0 / jnp.max(  jnp.array([mu_human_dist[0,0] - MPPI_FORESEE.std_factor * jnp.sqrt(cov_human_dist[0,0]), 0.01 ]) )
        return cost
    
    def rollout_control(self, init_state, actions):
        states = jnp.copy(init_state)
        for i in range(MPPI_FORESEE.horizon):
            states = jnp.append( states, self.robot_dynamics_step(states[:,[-1]], actions[i,:].reshape(-1,1)), axis=1 )
        return states
        
    @staticmethod
    @jit
    def weighted_sum(U, perturbation, weights):
        normalization_factor = jnp.sum(weights)
        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U
        return lax.fori_loop( 0, MPPI_FORESEE.samples, body, (U) )
    
    @staticmethod
    @jit
    def true_func(u_human, mu_x, robot_x):
        u_human = u_human - (robot_x - mu_x) / jnp.clip(jnp.linalg.norm( mu_x-robot_x ), 0.01, None) * ( 2.0 * jnp.tanh( 1.0 / jnp.linalg.norm( mu_x-robot_x ) ) )
        return u_human

    @staticmethod
    @jit
    def false_func(u_human, mu_x, robot_x):
        return u_human
  
    def human_dynamics( human_x, robot_x ):
        u = jnp.array([-3.0,0]).reshape(-1,1)
        u_human = lax.cond( jnp.linalg.norm( human_x-robot_x )<MPPI_FORESEE.sensing_radius, MPPI_FORESEE.true_func, MPPI_FORESEE.false_func, u, human_x, robot_x)
        # mu, cov = human_x + u_human * MPPI_FORESEE.dt, 0.0 * jnp.eye(MPPI_FORESEE.human_n)
        # mu, cov = human_x + u_human * MPPI_FORESEE.dt, MPPI_FORESEE.human_noise_cov * jnp.eye(MPPI_FORESEE.human_n) * MPPI_FORESEE.dt**2
        mu, cov = human_x + u_human * MPPI_FORESEE.dt, ( MPPI_FORESEE.human_noise_cov + jnp.max( jnp.array([1.0 / jnp.clip(jnp.linalg.norm(u_human), 0, 2), 0.5]) ) ) * jnp.eye(MPPI_FORESEE.human_n) * MPPI_FORESEE.dt**2
        return mu, cov

    @staticmethod
    @jit
    def human_sigma_point_expand(sigma_points, weights, robot_state):

        new_points = jnp.zeros((MPPI_FORESEE.human_n*(2*MPPI_FORESEE.human_n+1), 2*MPPI_FORESEE.human_n+1 ))
        new_weights = jnp.zeros((2*MPPI_FORESEE.human_n+1, 2*MPPI_FORESEE.human_n+1))

        # loop over sigma points
        def body(i, inputs):
            new_points, new_weights = inputs
            mu, cov = MPPI_FORESEE.human_dynamics( sigma_points[:,[i]], robot_state )
            root_term = get_ut_cov_root_diagonal(cov)           
            temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, jnp.zeros((MPPI_FORESEE.human_n, 1)), 1.0 )
            new_points = new_points.at[:,i].set( temp_points.reshape(-1,1, order='F')[:,0] )
            new_weights = new_weights.at[:,i].set( temp_weights.reshape(-1,1, order='F')[:,0] * weights[:,i] )   
            return new_points, new_weights
        new_points, new_weights = lax.fori_loop( 0, 2*MPPI_FORESEE.human_n+1, body, (new_points, new_weights) )
        return new_points.reshape( (MPPI_FORESEE.human_n, (2*MPPI_FORESEE.human_n+1)**2), order='F' ), new_weights.reshape(( 1,(2*MPPI_FORESEE.human_n+1)**2 ), order='F')
    
    @staticmethod
    @jit
    def single_sample_rollout(goal, robot_states_init, perturbed_control, human_sigma_points_init, human_sigma_weights_init, human_mus_init, human_covs_init):
        
        # Initialize variables
        robot_states = jnp.zeros( ( MPPI_FORESEE.robot_n, MPPI_FORESEE.horizon) )
        human_sigma_points = jnp.zeros( ((2*MPPI_FORESEE.human_n+1)*MPPI_FORESEE.human_n, MPPI_FORESEE.horizon) )
        human_sigma_weights = jnp.zeros( (2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.horizon) )
        human_mus = jnp.zeros( (MPPI_FORESEE.human_n, MPPI_FORESEE.horizon) )
        human_covs = jnp.zeros( (MPPI_FORESEE.human_n, MPPI_FORESEE.horizon) )

        # Set inital value
        robot_states = robot_states.at[:,0].set(robot_states_init)
        human_sigma_points = human_sigma_points.at[:,0].set(human_sigma_points_init)
        human_sigma_weights  = human_sigma_weights.at[:,0].set(human_sigma_weights_init)
        human_mus = human_mus.at[:,0].set(human_mus_init)
        human_covs = human_covs.at[:,0].set(human_covs_init)

        # loop over horizon
        cost_sample = 0
        def body(i, inputs):
            cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs = inputs
            human_sigma_state = human_sigma_points[:,i].reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1), order='F')
            human_sigma_weight = human_sigma_weights[:,[i]].T
            robot_state = robot_states[:,[i]]

            # Get cost
            cost_sample = cost_sample + MPPI_FORESEE.single_sample_state_cost( robot_state, human_sigma_state, human_sigma_weight,  goal)

            # Update robot states
            robot_states = robot_states.at[:,i+1].set( MPPI_FORESEE.robot_dynamics_step( robot_states[:,[i]], perturbed_control[:, [i]] )[:,0] )
            
            # Update human states
            
            expanded_states, expanded_weights = MPPI_FORESEE.human_sigma_point_expand( human_sigma_state, human_sigma_weight, robot_state )
            mu_temp, cov_temp, compressed_states, compressed_weights = sigma_point_compress( expanded_states, expanded_weights )

            # Store states
            human_sigma_points = human_sigma_points.at[:, i+1].set( compressed_states.T.reshape(-1,1)[:,0] )
            human_sigma_weights = human_sigma_weights.at[:, i+1].set( compressed_weights.T[:,0] )
            human_mus = human_mus.at[:, i+1].set( mu_temp[:,0] )   
            human_covs = human_covs.at[:, i+1].set( jnp.diag(cov_temp)  )

            return cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs
        
        cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs = lax.fori_loop( 0, MPPI_FORESEE.horizon-1, body, (cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs) )
        human_sigma_state = human_sigma_points[:,MPPI_FORESEE.horizon-1].reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1), order='F')
        human_sigma_weight = human_sigma_weights[:,[MPPI_FORESEE.horizon-1]].T
        robot_state = robot_states[:,[MPPI_FORESEE.horizon-1]]
        cost_sample = cost_sample + MPPI_FORESEE.single_sample_state_cost( robot_state, human_sigma_state, human_sigma_weight,  goal)
        
        return cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs

    @staticmethod
    @jit
    def rollout_states_foresee(robot_init_state, perturbed_control, goal, human_init_state_mu, human_init_state_cov):

        # Expansion - Compression - Projection. # show something in theory.. for arbitarry dynamics! assume swicthing happens only at discrete time intervals
        
        # Human
        human_sigma_points = jnp.zeros( (MPPI_FORESEE.samples,(2*MPPI_FORESEE.human_n+1)*MPPI_FORESEE.human_n, MPPI_FORESEE.horizon) )
        human_sigma_weights = jnp.zeros( (MPPI_FORESEE.samples,2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.horizon) )
        human_mus = jnp.zeros( (MPPI_FORESEE.samples,MPPI_FORESEE.human_n, MPPI_FORESEE.horizon) )
        human_covs = jnp.zeros( (MPPI_FORESEE.samples,MPPI_FORESEE.human_n, MPPI_FORESEE.horizon) )
        
        ##### Initialize
        points_init, weights_init = generate_sigma_points_gaussian( human_init_state_mu, jnp.diag(jnp.sqrt(human_init_state_cov[:,0])), jnp.zeros((MPPI_FORESEE.human_n,1)), 1.0 )
        human_sigma_points = human_sigma_points.at[:,:,0].set( jnp.tile(points_init.T.reshape(1,-1), (MPPI_FORESEE.samples,1)) )
        human_sigma_weights = human_sigma_weights.at[:,:,0].set( jnp.tile( weights_init, (MPPI_FORESEE.samples,1) ) )
        human_mus = human_mus.at[:,:,0].set( jnp.tile(human_init_state_mu.T, (MPPI_FORESEE.samples,1)) )
        human_covs = human_covs.at[:,:,0].set( jnp.tile(human_init_state_cov.T, (MPPI_FORESEE.samples,1)) )
        
        # Robot
        robot_states = jnp.zeros( (MPPI_FORESEE.samples, MPPI_FORESEE.robot_n, MPPI_FORESEE.horizon) )
        robot_states = robot_states.at[:,:,0].set( jnp.tile( robot_init_state.T, (MPPI_FORESEE.samples,1) ) )

        # Cost
        cost_total = jnp.zeros(MPPI_FORESEE.samples)

        # Loop over samples 
        @jit
        def body_samples(i, inputs):
            robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs = inputs     

            # Get cost
            cost_sample, robot_states_sample, human_sigma_points_sample, human_sigma_weights_sample, human_mus_sample, human_covs_sample = MPPI_FORESEE.single_sample_rollout(goal, robot_states[i,:,0], perturbed_control[i,:,:].T, human_sigma_points[i,:,0], human_sigma_weights[i,:,0], human_mus[i,:,0], human_covs[i,:,0] )
            cost_total = cost_total.at[i].set( cost_sample )
            robot_states = robot_states.at[i,:,:].set( robot_states_sample )
            human_sigma_points= human_sigma_points.at[i,:,:].set( human_sigma_points_sample )
            human_sigma_weights= human_sigma_weights.at[i,:,:].set( human_sigma_weights_sample )
            human_mus = human_mus.at[i,:,:].set( human_mus_sample )
            human_covs = human_covs.at[i,:,:].set( human_covs_sample )
            return robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs  
        
        robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs = lax.fori_loop( 0, MPPI_FORESEE.samples, body_samples, (robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs) )
        return robot_states, cost_total, human_sigma_points, human_sigma_weights, human_mus, human_covs
        
    
    def compute_rollout_costs( self, init_state, goal, human_states_mu, human_states_cov ):

        self.key, subkey = jax.random.split(self.key)
        perturbation = multivariate_normal( subkey, self.control_mu, self.control_cov, shape=( MPPI_FORESEE.samples, MPPI_FORESEE.horizon ) ) # K x T x nu
        
        perturbation = jnp.clip( perturbation, -0.8, 0.8 ) #0.3
        perturbed_control = self.U + perturbation

        perturbed_control = jnp.clip( perturbed_control, -self.control_bound, self.control_bound )
        perturbation = perturbed_control - self.U

        t0 = time.time()
        sampled_robot_states, costs, human_sigma_points, human_sigma_weights, human_mus, human_covs = MPPI_FORESEE.rollout_states_foresee(init_state, perturbed_control, goal, human_states_mu, human_states_cov)
        tf = time.time()

        lambd = 100 #1000 #  1.0
        weights = jnp.exp( - 1.0/lambd * costs )        
        print(f" max costs:{jnp.max(costs)}, weights: {jnp.max(weights)} vel: {jnp.max(perturbed_control)}, time: {tf-t0} ")
        self.U = MPPI_FORESEE.weighted_sum( self.U, perturbation, weights )

        states_final = self.rollout_control(init_state, self.U)              
        action = self.U[0,:].reshape(-1,1)
        self.U = jnp.append(self.U[1:,:], self.U[[-1],:], axis=0)

        sampled_robot_states = sampled_robot_states.reshape(( MPPI_FORESEE.robot_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon ))

        human_mus = human_mus.reshape( (MPPI_FORESEE.human_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon) )
        human_covs = human_covs.reshape( (MPPI_FORESEE.human_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon) )
   
        return sampled_robot_states, states_final, action, human_mus, human_covs