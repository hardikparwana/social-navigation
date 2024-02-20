import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax
from ut_utils_jax import *

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

    def __init__(self, horizon=10, samples = 10, input_size = 2, dt=0.05, sensing_radius=2, human_noise_cov=0.01, std_factor=1.96):
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
        self.control_mu = jnp.zeros(input_size)
        self.control_cov = 2.0 * jnp.eye(input_size)
        self.control_bound_lb = -jnp.array([[1], [1]])
        self.control_bound_ub = -self.control_bound_lb
        self.U = jnp.append(  0.5 * jnp.ones((MPPI_FORESEE.horizon, 1)), 0.5 * jnp.ones((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu
        # self.U = jnp.append(  -0.5 * jnp.ones((MPPI_FORESEE.horizon, 1)), jnp.zeros((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu

    # Linear dynamics for now
    @staticmethod
    def robot_dynamics_step(state, input):
        return state + input * MPPI_FORESEE.dt
    
    @staticmethod
    @jit
    def body_sample(cost_total, robot_states_t, human_sigma_states_t, human_sigma_weights_t, goal):    
        # Loop over samples. this should come first        
        def body(j, samples):
            cost_total = samples
            human_sigma_state = human_sigma_states_t[ :,[j] ]   #states_t[2*j:2*j+2,0]
            human_sigma_weight = human_sigma_weights_t[:, [j]]
            robot_state = robot_states_t[:, [j]]
            cost_total = cost_total.at[j].set(MPPI_FORESEE.body_sample_chance( cost_total[j], robot_state, human_sigma_state.reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1), order='F'), human_sigma_weight.T, goal ))
            return cost_total
        return lax.fori_loop( 0, MPPI_FORESEE.samples, body, (cost_total) )
        
        for j in range(MPPI_FORESEE.samples):
            human_sigma_state = human_sigma_states_t[ :,[j] ]   #states_t[2*j:2*j+2,0]
            human_sigma_weight = human_sigma_weights_t[:, [j]]
            robot_state = robot_states_t[:, [j]]
            cost_total = cost_total.at[j].set(MPPI_FORESEE.body_sample_chance( cost_total[j], robot_state, human_sigma_state.reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1), order='F'), human_sigma_weight.T, goal ))
        return cost_total
        
    @staticmethod
    @jit
    def body_sample_chance(cost_total, robot_state, human_sigma_points, human_sigma_weights, goal):
        # loop over sigma points

        # or do over covariance and expectation
        
        human_dist_sigma_points = jnp.linalg.norm(robot_state - human_sigma_points, axis=0).reshape(1,-1)
        mu_human_dist, cov_human_dist = get_mean_cov( human_dist_sigma_points, human_sigma_weights )
        cost_total = cost_total + 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 3.0 / jnp.max(  jnp.array([mu_human_dist[0,0] - MPPI_FORESEE.std_factor * jnp.sqrt(cov_human_dist[0,0]), 0.01 ]) )
        return cost_total

        def body(j, inputs):
            cost_total = inputs
            human_state = human_sigma_points[:,[j]]
            weight = human_sigma_weights[0,j]
            cost_total = cost_total + weight * ( 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 3.0/jnp.max( jnp.array([jnp.min(jnp.linalg.norm(robot_state - human_state, axis=0)), 0.01]) ) )
            # cost_total = cost_total.at[j].set( cost_total[j] + 1.0 * ((state-goal).T @ (state-goal))[0,0] - ( jnp.linalg.norm(state - humanX_mu, axis=0) - 1.96**2 * humanX_cov[0,0] )[0] )  # assume human: horizon x 2 x num_humans
            return cost_total
        return lax.fori_loop( 0, 2*MPPI_FORESEE.human_n+1, body, (cost_total) )
    
        for j in range(2*MPPI_FORESEE.human_n+1):
            human_state = human_sigma_points[:,[j]]
            weight = human_sigma_weights[0,j]
            cost_total = cost_total + weight * ( 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 3/jnp.max( jnp.array([jnp.min(jnp.linalg.norm(robot_state - human_state, axis=0)), 0.01]) ) )
        return cost_total
    
    def rollout_control(self, init_state, actions):
        states = jnp.copy(init_state)
        for i in range(MPPI_FORESEE.horizon):
            states = jnp.append( states, self.robot_dynamics_step(states[:,[-1]], actions[i,:].reshape(-1,1)), axis=1 )
        return states
        
    @staticmethod
    def weighted_sum(U, perturbation, weights):
        normalization_factor = jnp.sum(weights)
        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U
        return lax.fori_loop( 0, MPPI_FORESEE.samples, body, (U) )
    
    @staticmethod
    def true_func(u_human, mu_x, robot_x):
        u_human = u_human - (robot_x - mu_x) / jnp.clip(jnp.linalg.norm( mu_x-robot_x ), 0.01, None) * ( 2.0 * jnp.tanh( 1.0 / jnp.linalg.norm( mu_x-robot_x ) ) )
        return u_human

    @staticmethod
    def false_func(u_human, mu_x, robot_x):
        return u_human

    @staticmethod
    @jit
    def human_step_noisy(mu_x,cov_x,u,dt):
        return mu_x.reshape(-1,1) + u*dt, cov_x.reshape(-1,1) + MPPI_FORESEE.human_noise_cov*jnp.ones((2,1))*dt*dt
    
    def human_dynamics( human_x, robot_x ):
        u = jnp.array([-3.0,0]).reshape(-1,1)
        u_human = lax.cond( jnp.linalg.norm( human_x-robot_x )<MPPI_FORESEE.sensing_radius, MPPI_FORESEE.true_func, MPPI_FORESEE.false_func, u, human_x, robot_x)
        # mu, cov = human_x + u_human * MPPI_FORESEE.dt, 0.0 * jnp.eye(MPPI_FORESEE.human_n)
        mu, cov = human_x + u_human * MPPI_FORESEE.dt, MPPI_FORESEE.human_noise_cov * jnp.eye(MPPI_FORESEE.human_n) * MPPI_FORESEE.dt**2
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
    
        for i in range(2*MPPI_FORESEE.human_n+1):
            mu, cov = MPPI_FORESEE.human_dynamics( sigma_points[:,[i]], robot_state )
            root_term = get_ut_cov_root_diagonal(cov)           
            temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, jnp.zeros((MPPI_FORESEE.human_n, 1)), 1.0 )
            new_points = new_points.at[:,i].set( temp_points.reshape(-1,1, order='F')[:,0] )
            new_weights = new_weights.at[:,i].set( temp_weights.reshape(-1,1, order='F')[:,0] * weights[:,i] )   

        return new_points.reshape( (MPPI_FORESEE.human_n, (2*MPPI_FORESEE.human_n+1)**2), order='F' ), new_weights.reshape(( 1,(2*MPPI_FORESEE.human_n+1)**2 ), order='F')
    
    @staticmethod
    @jit
    def human_dynamics_step( robot_states, human_sigma_points, human_sigma_weights  ):

        next_sigma_points, next_weights = jnp.zeros(( (2*MPPI_FORESEE.human_n+1)*MPPI_FORESEE.human_n,MPPI_FORESEE.samples) ), jnp.zeros( (2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.samples) )
        mus, covs = jnp.zeros((MPPI_FORESEE.human_n, MPPI_FORESEE.samples)), jnp.zeros((MPPI_FORESEE.human_n, MPPI_FORESEE.samples))
        # loop over samples
        def body(i, inputs):

            next_sigma_points, next_weights, mus, covs = inputs

            human_sigma_state = human_sigma_points[:,i].reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1), order='F')
            human_sigma_weight = human_sigma_weights[:,[i]].T
            robot_state = robot_states[:,[i]]

            #expand states
            expanded_states, expanded_weights = MPPI_FORESEE.human_sigma_point_expand( human_sigma_state, human_sigma_weight, robot_state )
            mu_temp, cov_temp, compressed_states, compressed_weights = sigma_point_compress( expanded_states, expanded_weights )

            next_sigma_points = next_sigma_points.at[:,i].set( compressed_states.T.reshape(-1,1)[:,0] )
            next_weights = next_weights.at[:,i].set( compressed_weights.T[:,0] )
            mus = mus.at[:,i].set( mu_temp[:,0] )
            covs = covs.at[:,i].set( jnp.diag(cov_temp) )

            return next_sigma_points, next_weights, mus, covs
        
        return lax.fori_loop( 0, MPPI_FORESEE.samples, body, (next_sigma_points, next_weights, mus, covs) )
    
        for i in range(MPPI_FORESEE.samples):

            human_sigma_state = human_sigma_points[:,i].reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1), order='F')
            human_sigma_weight = human_sigma_weights[:,[i]].T
            robot_state = robot_states[:,[i]]

            #expand states
            expanded_states, expanded_weights = MPPI_FORESEE.human_sigma_point_expand( human_sigma_state, human_sigma_weight, robot_state )
            mu_temp, cov_temp, compressed_states, compressed_weights = sigma_point_compress( expanded_states, expanded_weights )

            next_sigma_points = next_sigma_points.at[:,i].set( compressed_states.T.reshape(-1,1)[:,0] )
            next_weights = next_weights.at[:,i].set( compressed_weights.T[:,0] )
            mus = mus.at[:,i].set( mu_temp[:,0] )
            covs = covs.at[:,i].set( jnp.diag(cov_temp) )

        return next_sigma_points, next_weights, mus, covs

    @staticmethod
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

        # Loop over horizon
        @jit
        def body_horizon(i, inputs):
            robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs = inputs     

            # Get cost
            cost_total = MPPI_FORESEE.body_sample( cost_total, robot_states[:,:,i].T, human_sigma_points[:,:,i].T, human_sigma_weights[:,:,i].T, goal )

            # Update state
            robot_states = robot_states.at[:,:,i+1].set( MPPI_FORESEE.robot_dynamics_step( robot_states[:,:,i].T, perturbed_control[:, i, :].T ).T )
            next_sigma_points, next_sigma_weights, next_mus, next_covs = MPPI_FORESEE.human_dynamics_step( robot_states[:,:,i].T, human_sigma_points[:, :, i].T, human_sigma_weights[:, :, i].T  )
            human_sigma_points = human_sigma_points.at[:, :, i+1].set( next_sigma_points.T )
            human_sigma_weights = human_sigma_weights.at[:, :, i+1].set( next_sigma_weights.T )
            human_mus = human_mus.at[:, :, i+1].set( next_mus.T )   
            human_covs = human_covs.at[:, :, i+1].set( next_covs.T )
            return robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs  
        robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs = lax.fori_loop( 0, MPPI_FORESEE.horizon, body_horizon, (robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs) )
        return robot_states, cost_total, human_sigma_points, human_sigma_weights, human_mus, human_covs
        
        for i in range(MPPI_FORESEE.horizon):
            # Get cost
            cost_total = MPPI_FORESEE.body_sample( cost_total, robot_states[:,:,i].T, human_sigma_points[:,:,i].T, human_sigma_weights[:,:,i].T, goal )

            # Update state
            robot_states = robot_states.at[:,:,i+1].set( MPPI_FORESEE.robot_dynamics_step( robot_states[:,:,i].T, perturbed_control[:, i, :].T ).T )
            next_sigma_points, next_sigma_weights, next_mus, next_covs = MPPI_FORESEE.human_dynamics_step( robot_states[:,:,i].T, human_sigma_points[:, :, i].T, human_sigma_weights[:, :, i].T  )
            human_sigma_points = human_sigma_points.at[:, :, i+1].set( next_sigma_points.T )
            human_sigma_weights = human_sigma_weights.at[:, :, i+1].set( next_sigma_weights.T )
            human_mus = human_mus.at[:, :, i+1].set( next_mus.T )   
            human_covs = human_covs.at[:, :, i+1].set( next_covs.T )

        return robot_states, cost_total, human_sigma_points, human_sigma_weights, human_mus, human_covs
    
    def compute_rollout_costs( self, init_state, goal, human_states_mu, human_states_cov ):

        self.key, subkey = jax.random.split(self.key)
        perturbation = multivariate_normal( subkey, self.control_mu, self.control_cov, shape=( MPPI_FORESEE.samples, MPPI_FORESEE.horizon ) ) # K x T x nu
        
        perturbation = jnp.clip( perturbation, -0.8, 0.8 ) #0.3
        perturbed_control = self.U + perturbation

        # perturbed_control = jnp.clip( perturbed_control, -2.0, 2.0 )
        # perturbation = perturbed_control - self.U

        sampled_robot_states, costs, human_sigma_points, human_sigma_weights, human_mus, human_covs = MPPI_FORESEE.rollout_states_foresee(init_state, perturbed_control, goal, human_states_mu, human_states_cov)

        lambd = 10 #1000 #  1.0
        weights = jnp.exp( - 1.0/lambd * costs )        
        print(f" max costs:{jnp.max(costs)}, weights: {jnp.max(weights)} ")
        self.U = MPPI_FORESEE.weighted_sum( self.U, perturbation, weights )

        states_final = self.rollout_control(init_state, self.U)              
        action = self.U[0,:].reshape(-1,1)
        self.U = jnp.append(self.U[1:,:], self.U[[-1],:], axis=0)


        # robot_states = jnp.zeros( (MPPI_FORESEE.samples, MPPI_FORESEE.robot_n, MPPI_FORESEE.horizon) )
        sampled_robot_states = sampled_robot_states.reshape(( MPPI_FORESEE.robot_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon ))

        # print(f"states_final:\n x: {states_final[0,:]} \n y: {states_final[1,:]} \n human mus x:{human_mus[0,:,0]}, \n human mus y: {human_mus[1,:,0]}")
        

        human_mus = human_mus.reshape( (MPPI_FORESEE.human_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon) )
        human_covs = human_covs.reshape( (MPPI_FORESEE.human_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon) )
        print(f"human cov max x:{jnp.max(human_covs[0,:])}, y: {jnp.max(human_covs[1,:])}")
        



        return sampled_robot_states, states_final, action, human_mus, human_covs