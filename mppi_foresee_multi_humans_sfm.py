# import os
# os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=100'
import jax
print(f"{jax.devices()}")

import jax
import jax.numpy as jnp
from jax.random import multivariate_normal
from jax import jit, lax
from ut_utils_jax import *
import time
import multiprocessing
import socialforce
from socialforce.potentials_jax import PedPedPotential
from socialforce.fieldofview_jax import FieldOfView
from socialforce import stateutils_jax

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
    use_gpu = True
    key = 0
    control_mu = 0
    conrtol_cov = 0
    control_bound = 0
    control_bound_lb = 0
    control_bound_ub = 0

    human_repulsion_gain = 2.0
    costs_lambda = 300
    cost_goal_coeff = 1.0
    cost_safety_coeff = 10.0
    num_humans = 5
    human_nominal_speed = jnp.tile(jnp.array([-3.0,0]).reshape(-1,1), (1,num_humans))
    num_obstacles = 2
    indices = 0
    hindex_list = []
    aware = True
    humans_interact = True
    obstacles_interact = True

    obstacle_radius = 1.0

    socialforce = 0
    MAX_SPEED_MULTIPLIER = 1.3
    initial_speeds = 1.0
    max_speeds = 1.0

    V = 0
    U = 0
    W = 0

    def __init__(self, horizon=10, samples = 10, input_size = 2, dt=0.05, sensing_radius=2, human_noise_cov=0.01, std_factor=1.96, control_bound=7, control_init_ratio=1, u_guess=None, use_GPU=True, human_nominal_speed = jnp.array([-3.0,0]).reshape(-1,1), human_repulsion_gain = 2.0, costs_lambda = 300, cost_goal_coeff = 1.0, cost_safety_coeff = 10.0, num_humans = 5, num_obstacles = 2, aware=True, humans_interact=True, obstacles_interact=True, ped_space=None, v0=2.1, sigma=0.3):
        MPPI_FORESEE.key = jax.random.PRNGKey(111)
        MPPI_FORESEE.human_n = 4
        MPPI_FORESEE.robot_n = 2 #3 #2
        MPPI_FORESEE.m = 2
        MPPI_FORESEE.horizon = horizon
        MPPI_FORESEE.samples = samples
        MPPI_FORESEE.sensing_radius = sensing_radius
        MPPI_FORESEE.human_noise_cov = human_noise_cov
        MPPI_FORESEE.dt = dt
        MPPI_FORESEE.std_factor = std_factor
        MPPI_FORESEE.use_gpu = use_GPU
        MPPI_FORESEE.human_nominal_speed = human_nominal_speed
        MPPI_FORESEE.human_repulsion_gain = human_repulsion_gain
        MPPI_FORESEE.costs_lambda = costs_lambda
        MPPI_FORESEE.cost_goal_coeff = cost_goal_coeff
        MPPI_FORESEE.cost_safety_coeff = cost_safety_coeff
        MPPI_FORESEE.num_humans = num_humans
        MPPI_FORESEE.num_obstacles = num_obstacles
        MPPI_FORESEE.indices = jnp.arange(0, MPPI_FORESEE.num_humans)
        MPPI_FORESEE.hindex_list = jnp.arange(1,num_humans).reshape(1,-1)
        for i in range(1,num_humans):
            MPPI_FORESEE.hindex_list = jnp.append( MPPI_FORESEE.hindex_list, jnp.delete( jnp.arange(0,num_humans), i ).reshape(1,-1), axis=0 )    
        MPPI_FORESEE.aware = aware
        MPPI_FORESEE.humans_interact = humans_interact
        MPPI_FORESEE.obstacles_interact = obstacles_interact

        self.input_size = input_size        
        MPPI_FORESEE.control_bound = control_bound
        MPPI_FORESEE.control_mu = jnp.zeros(input_size)
        MPPI_FORESEE.control_cov = 3.0 * jnp.eye(input_size)  #2.0 * jnp.eye(input_size)
        MPPI_FORESEE.control_bound_lb = -jnp.array([[1], [1]])
        MPPI_FORESEE.control_bound_ub = -self.control_bound_lb  
        if u_guess != None:
            self.U = u_guess
        else:
            # self.U = jnp.append(  1.0 * jnp.ones((MPPI_FORESEE.horizon, 1)), control_init_ratio * 1.0 * jnp.ones((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu
            self.U = jnp.append(  1.0 * jnp.ones((MPPI_FORESEE.horizon, 1)), 1.0 * jnp.ones((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu
            # self.U = jnp.append(  -0.5 * jnp.ones((MPPI_FORESEE.horizon, 1)), jnp.zeros((MPPI_FORESEE.horizon,1)), axis=1  ) # T x nu


        # SFM stuff
        MPPI_FORESEE.socialforce = socialforce.Simulator( delta_t = MPPI_FORESEE.dt )
        MPPI_FORESEE.MAX_SPEED_MULTIPLIER = 1.3
        initial_speed = 2.0 #1.0
        MPPI_FORESEE.initial_speeds = jnp.ones((MPPI_FORESEE.num_humans+1)) * initial_speed
        MPPI_FORESEE.max_speeds = MPPI_FORESEE.MAX_SPEED_MULTIPLIER * MPPI_FORESEE.initial_speeds
        # tau = 0.5
        # initial_human_pos = jnp.append( jnp.linspace(0,MPPI_FORESEE.num_humans, MPPI_FORESEE.num_humans).reshape(-1,1), jnp.linspace(0,MPPI_FORESEE.num_humans, MPPI_FORESEE.num_humans).reshape(-1,1), axis=1 )
        # initial_state = jnp.append(initial_human_pos
        # tau = tau * jnp.ones(initial_state.shape[0])
        # social_state = jnp.concatenate((initial_state, jnp.expand_dims(tau, -1)), axis=-1)
        # s = socialforce.Simulator( social_state, delta_t = dt )
        # state = social_state

        MPPI_FORESEE.V = PedPedPotential(v0=v0, sigma=sigma)
        MPPI_FORESEE.U = ped_space
        MPPI_FORESEE.w = FieldOfView(twophi=360.0)

    # SFM functions
    @staticmethod
    def f_ab(state, delta_t):
        """Compute f_ab."""
        return -1.0 * MPPI_FORESEE.V.grad_r_ab(state, delta_t)

    @staticmethod
    def f_aB(state, delta_t):
        return jnp.zeros((state.shape[0], 0, 2))
        """Compute f_aB."""
        # if Simulator.U is None:
        #     return jnp.zeros((Simulator.state.shape[0], 0, 2))
        return -1.0 * Simulator.U.grad_r_aB(state)

    @staticmethod
    def capped_velocity(desired_velocity, max_speeds):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = jnp.linalg.norm(desired_velocity, axis=-1)
        factor = jnp.minimum(1.0, max_speeds / desired_speeds)
        return desired_velocity * jnp.expand_dims(factor, -1)
    
    @staticmethod
    @jit
    def sfm_step_(state, initial_speeds, max_speeds, delta_t):
        """Do one step in the simulation and update the state in place."""
        # accelerate to desired velocity
        e = stateutils_jax.desired_directions(state)
        vel = state[:, 2:4]
        tau = state[:, 6:7]
        F0 = 1.0 / tau * (jnp.expand_dims(initial_speeds, -1) * e - vel)

        # repulsive terms between pedestrians
        f_ab = MPPI_FORESEE.f_ab(state, delta_t)
        w = jnp.expand_dims(MPPI_FORESEE.w.__call__(e, -f_ab), -1)
        F_ab = w * f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = MPPI_FORESEE.f_aB(state, delta_t)

        # social force
        F = F0 + jnp.sum(F_ab, axis=1) + jnp.sum(F_aB, axis=1)
        # desired velocity
        w = state[:, 2:4] + delta_t * F
        # velocity
        v = MPPI_FORESEE.capped_velocity(w, max_speeds)

        new_pos = state[ 0, 0:2 ] + v[0,0:2] * delta_t
        new_state_mu = jnp.append( new_pos, v[0,0:2] ).reshape(-1,1)
        new_state_cov = MPPI_FORESEE.human_noise_cov * jnp.eye(MPPI_FORESEE.human_n) * MPPI_FORESEE.dt**2

        return new_state_mu, new_state_cov
    
    @staticmethod
    @jit
    def sfm_step_actual_(state, initial_speeds, max_speeds, delta_t):
        """Do one step in the simulation and update the state in place."""
        # accelerate to desired velocity
        e = stateutils_jax.desired_directions(state)
        vel = state[:, 2:4]
        tau = state[:, 6:7]
        F0 = 1.0 / tau * (jnp.expand_dims(initial_speeds, -1) * e - vel)

        # repulsive terms between pedestrians
        f_ab = MPPI_FORESEE.f_ab(state, delta_t)
        w = jnp.expand_dims(MPPI_FORESEE.w.__call__(e, -f_ab), -1)
        F_ab = w * f_ab

        # repulsive terms between pedestrians and boundaries
        F_aB = MPPI_FORESEE.f_aB(state, delta_t)

        # social force
        F = F0 + jnp.sum(F_ab, axis=1) + jnp.sum(F_aB, axis=1)
        # desired velocity
        w = state[:, 2:4] + delta_t * F
        # velocity
        v = MPPI_FORESEE.capped_velocity(w, max_speeds)

        new_pos = state[ :, 0:2 ] + v[:,0:2] * delta_t
        new_state_mu = jnp.append( new_pos, v, axis=1 )[0:MPPI_FORESEE.num_humans,:]
        new_state_cov = MPPI_FORESEE.human_noise_cov * jnp.eye(MPPI_FORESEE.human_n) * MPPI_FORESEE.dt**2

        return new_state_mu, new_state_cov

    @staticmethod
    def robot_xdot(state, input):

        # single integrator
        return input

    # Linear dynamics for now
    @staticmethod
    def robot_dynamics_step(state, input):
        return state + input * MPPI_FORESEE.dt
    
        # Unicycle
        theta = state[2,0]
        xdot = jnp.array([ [ input[0,0]*jnp.cos(theta) ],
                            [ input[0,0]*jnp.sin(theta) ],
                            [ input[1,0] ]
                            ])
        return state + xdot * MPPI_FORESEE.dt
    
    @staticmethod
    @jit
    def single_sample_state_cost(robot_state, human_sigma_points, human_sigma_weights, goal, obstaclesX):       
        human_dist_sigma_points = jnp.linalg.norm(robot_state[0:2] - human_sigma_points[0:2,:], axis=0).reshape(1,-1)
        mu_human_dist, cov_human_dist = get_mean_cov( human_dist_sigma_points, human_sigma_weights )
        robot_obstacle_dists = jnp.linalg.norm(robot_state[0:2] - obstaclesX, axis=0)
        # cost_total = cost_total + 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 3.0 / jnp.max(  jnp.array([mu_human_dist[0,0] - MPPI_FORESEE.std_factor * jnp.sqrt(cov_human_dist[0,0]), 0.01 ]) )
        cost = MPPI_FORESEE.cost_safety_coeff / jnp.max(  jnp.array([mu_human_dist[0,0] - MPPI_FORESEE.std_factor * jnp.sqrt(cov_human_dist[0,0]), 0.01 ]) )
        cost = cost + MPPI_FORESEE.cost_safety_coeff / jnp.max(jnp.array([jnp.min(robot_obstacle_dists), 0.01])  )
        return cost
    
    @staticmethod
    @jit
    def state_cost(robot_state, human_state, goal, obstaclesX):       
        human_dist = jnp.linalg.norm(robot_state[0:2] - human_state)
        robot_obstacle_dists = jnp.linalg.norm(robot_state[0:2] - obstaclesX, axis=0).reshape(-1,1)
        # cost_total = cost_total + 1.0 * ((robot_state-goal).T @ (robot_state-goal))[0,0] + 3.0 / jnp.max(  jnp.array([mu_human_dist[0,0] - MPPI_FORESEE.std_factor * jnp.sqrt(cov_human_dist[0,0]), 0.01 ]) )
        cost = MPPI_FORESEE.cost_goal_coeff * ((robot_state[0:2]-goal).T @ (robot_state[0:2]-goal))[0,0]
        cost = cost + MPPI_FORESEE.cost_safety_coeff / jnp.max(  jnp.array([human_dist, 0.01 ]) )
        cost = cost + MPPI_FORESEE.cost_safety_coeff / jnp.max(jnp.array([jnp.min(robot_obstacle_dists), 0.01])  )
        return cost
    
    @staticmethod
    @jit
    def rollout_control(init_state, actions):
        states = jnp.zeros((MPPI_FORESEE.robot_n, MPPI_FORESEE.horizon+1))
        states = states.at[:,0].set(init_state[:,0])
        def body(i, inputs):
            states = inputs
            states = states.at[:,i+1].set( MPPI_FORESEE.robot_dynamics_step(states[:,[i]], actions[i,:].reshape(-1,1))[:,0] )
            return states
        states = lax.fori_loop(0, MPPI_FORESEE.horizon, body, states)
        return states
        
    @staticmethod
    @jit
    def weighted_sum(U, perturbation, costs):#weights):
        costs = costs - jnp.min(costs)
        costs = costs / jnp.max(costs)
        lambd = MPPI_FORESEE.costs_lambda #300
        weights = jnp.exp( - 1.0/lambd * costs )   
        normalization_factor = jnp.sum(weights)
        def body(i, inputs):
            U = inputs
            U = U + perturbation[i] * weights[i] / normalization_factor
            return U
        return lax.fori_loop( 0, MPPI_FORESEE.samples, body, (U) )
    
    @staticmethod
    @jit
    def true_func(robot_x, robot_u, initial_state):

        robot_xdot = MPPI_FORESEE.robot_xdot(robot_x, robot_u)
        robot_social_state = jnp.array([[ robot_x[0,0], robot_x[1,0], robot_xdot[0,0], robot_xdot[1,0], 100.0, 100.0 ]]) # far away goal somewhere
        initial_state = jnp.append( initial_state, robot_social_state, axis=0 )
        return initial_state
        
    @staticmethod
    @jit
    def false_func(robot_x, robot_u, initial_state):
        return initial_state
    
    @staticmethod
    @jit
    def true_func_obstacle(u_human, obstacle_x, robot_x):
        # u_human = u_human - (robot_x - mu_x) / jnp.clip(jnp.linalg.norm( mu_x-robot_x ), 0.01, None) * ( 2.0 * jnp.tanh( 1.0 / jnp.linalg.norm( mu_x-robot_x ) ) )
        u_human = u_human - (robot_x[0:2] - obstacle_x) / jnp.clip(jnp.linalg.norm( obstacle_x-robot_x[0:2] ), 0.01, None) * ( MPPI_FORESEE.human_repulsion_gain * jnp.tanh( 1.0 / jnp.clip( jnp.linalg.norm( obstacle_x-robot_x[0:2] )-MPPI_FORESEE.obstacle_radius, 0.01, None ) ) )
        return u_human
    
    @staticmethod
    @jit
    def multi_human_dynamics_sfm(human_x, other_human_x, human_goal, other_human_goal, robot_x, robot_u, obstaclesX, aware):
        # easiest but ineffecient way to replace PF with SFM for now

        initial_state = jnp.append( human_x.T, human_goal.T, axis=1 )
        if MPPI_FORESEE.humans_interact:
            other_human_social_state = jnp.append( other_human_x.T, other_human_goal.T, axis=1 )
            initial_state = jnp.append( initial_state, other_human_social_state, axis=0 )

        # if MPPI_FORESEE.obstacles_interact:
        #     # single integrator

        # robot repulsion
        # initial_state = lax.cond( aware, MPPI_FORESEE.true_func, MPPI_FORESEE.false_func, robot_x, robot_u, initial_state  )
        robot_xdot = MPPI_FORESEE.robot_xdot(robot_x, robot_u)
        robot_social_state = jnp.array([[ robot_x[0,0], robot_x[1,0], robot_xdot[0,0], robot_xdot[1,0], 100.0, 100.0 ]]) # far away goal somewhere
        initial_state = jnp.append( initial_state, robot_social_state, axis=0 )
        
        tau = 0.5
        tau = tau * jnp.ones(initial_state.shape[0])
        social_state = jnp.concatenate((initial_state, jnp.expand_dims(tau, -1)), axis=-1)

        mu, cov = MPPI_FORESEE.sfm_step_(social_state, MPPI_FORESEE.initial_speeds, MPPI_FORESEE.max_speeds, MPPI_FORESEE.dt)
        return mu, cov
    
    @staticmethod
    @jit
    def multi_human_dynamics_sfm_actual(human_x, human_goal, robot_x, robot_u, obstaclesX, aware):
        # easiest but ineffecient way to replace PF with SFM for now

        

        initial_state = jnp.append( human_x.T, human_goal.T, axis=1 )
        
        # if MPPI_FORESEE.obstacles_interact:
        #     # single integrator

        # robot repulsion
        # initial_state = lax.cond( aware, MPPI_FORESEE.true_func, MPPI_FORESEE.false_func, robot_x, robot_u, initial_state  )
        robot_xdot = MPPI_FORESEE.robot_xdot(robot_x, robot_u)
        robot_social_state = jnp.array([[ robot_x[0,0], robot_x[1,0], robot_xdot[0,0], robot_xdot[1,0], 100.0, 100.0 ]]) # far away goal somewhere
        initial_state = jnp.append( initial_state, robot_social_state, axis=0 )
        
        tau = 0.5
        tau = tau * jnp.ones(initial_state.shape[0])
        social_state = jnp.concatenate((initial_state, jnp.expand_dims(tau, -1)), axis=-1)

        mu, cov = MPPI_FORESEE.sfm_step_actual_(social_state, MPPI_FORESEE.initial_speeds, MPPI_FORESEE.max_speeds, MPPI_FORESEE.dt)
        return mu, cov
    
    @staticmethod
    @jit
    def multi_human_sigma_point_expand(sigma_points, weights, other_human_mus, human_goal, other_human_goal, robot_state, robot_input, obstaclesX, aware):

        new_points = jnp.zeros((MPPI_FORESEE.human_n*(2*MPPI_FORESEE.human_n+1), 2*MPPI_FORESEE.human_n+1 ))
        new_weights = jnp.zeros((2*MPPI_FORESEE.human_n+1, 2*MPPI_FORESEE.human_n+1))

        # loop over sigma points
        def body(i, inputs):
            new_points, new_weights = inputs
            mu, cov = MPPI_FORESEE.multi_human_dynamics_sfm( sigma_points[:,[i]], other_human_mus, human_goal, other_human_goal, robot_state, robot_input, obstaclesX, aware )
            root_term = get_ut_cov_root_diagonal(cov)           
            temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, jnp.zeros((MPPI_FORESEE.human_n, 1)), 1.0 )
            new_points = new_points.at[:,i].set( temp_points.reshape(-1,1, order='F')[:,0] )
            new_weights = new_weights.at[:,i].set( temp_weights.reshape(-1,1, order='F')[:,0] * weights[:,i] )   
            return new_points, new_weights
        new_points, new_weights = lax.fori_loop( 0, 2*MPPI_FORESEE.human_n+1, body, (new_points, new_weights) )


        # def body_vmap(  ):
        #     mu, cov = MPPI_FORESEE.multi_human_dynamics_sfm( sigma_points[:,[i]], other_human_mus, human_goal, other_human_goal, robot_state, robot_input, obstaclesX, aware )
        #     root_term = get_ut_cov_root_diagonal(cov)           
        #     temp_points, temp_weights = generate_sigma_points_gaussian( mu, root_term, jnp.zeros((MPPI_FORESEE.human_n, 1)), 1.0 )
        #     new_points = new_points.at[:,i].set( temp_points.reshape(-1,1, order='F')[:,0] )
        #     new_weights = new_weights.at[:,i].set( temp_weights.reshape(-1,1, order='F')[:,0] * weights[:,i] )   

        return new_points.reshape( (MPPI_FORESEE.human_n, (2*MPPI_FORESEE.human_n+1)**2), order='F' ), new_weights.reshape(( 1,(2*MPPI_FORESEE.human_n+1)**2 ), order='F')

    @staticmethod
    @jit
    def single_sample_rollout(goal, robot_states_init, perturbed_control, human_sigma_points_init, human_sigma_weights_init, human_mus_init, human_covs_init, obstaclesX, aware, human_goals):
        
        # Initialize variables
        robot_states = jnp.zeros( ( MPPI_FORESEE.robot_n, MPPI_FORESEE.horizon) )
        human_sigma_points = jnp.zeros( ((2*MPPI_FORESEE.human_n+1)*MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_sigma_weights = jnp.zeros( (2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_mus = jnp.zeros( (MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_covs = jnp.zeros( (MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )

        # Set inital value
        robot_states = robot_states.at[:,0].set(robot_states_init)
        human_sigma_points = human_sigma_points.at[:,:,0].set(human_sigma_points_init)
        human_sigma_weights  = human_sigma_weights.at[:,:,0].set(human_sigma_weights_init)
        human_mus = human_mus.at[:,:,0].set(human_mus_init)
        human_covs = human_covs.at[:,:,0].set(human_covs_init)

        # loop over humans? but symmetric so there should be a better way to do this
        @jit
        def loop_over_humans(j, inputs_humans):
            i, cost_sample, robot_state, human_sigma_state, human_sigma_weight, human_sigma_points, human_sigma_weights, human_mus, human_covs, human_goals, robot_input, obstaclesX = inputs_humans

            # get cost for collision avoidance with humans
            cost_sample = cost_sample + MPPI_FORESEE.single_sample_state_cost( robot_state, human_sigma_state[:,:,j], human_sigma_weight[:,[j]].T,  goal, obstaclesX) / MPPI_FORESEE.num_humans

            # Expand this human's state
            expanded_states, expanded_weights = MPPI_FORESEE.multi_human_sigma_point_expand( human_sigma_state[:,:,j], human_sigma_weight[:,[j]].T, human_mus[:, MPPI_FORESEE.hindex_list[j,:], i], human_goals[:,[j]], human_goals[:,MPPI_FORESEE.hindex_list[j,:]], robot_state, robot_input, obstaclesX, aware )
            mu_temp, cov_temp, compressed_states, compressed_weights = sigma_point_compress( expanded_states, expanded_weights )
            human_sigma_points = human_sigma_points.at[:, j, i+1].set( compressed_states.T.reshape(-1,1)[:,0] )
            human_sigma_weights = human_sigma_weights.at[:, j, i+1].set( compressed_weights.T[:,0] )
            human_mus = human_mus.at[:, j, i+1].set( mu_temp[:,0] )   
            human_covs = human_covs.at[:, j, i+1].set( jnp.diag(cov_temp) )
            return i, cost_sample, robot_state, human_sigma_state, human_sigma_weight, human_sigma_points, human_sigma_weights, human_mus, human_covs, human_goals, robot_input, obstaclesX
        
        # loop over horizon
        cost_sample = 0
        @jit
        def body(i, inputs):
            cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs, obstaclesX = inputs
            human_sigma_state = human_sigma_points[:,:,i].reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.num_humans), order='F')
            human_sigma_weight = human_sigma_weights[:,:,i]
            robot_state = robot_states[:,[i]]

            # Get goal cost
            cost_sample = cost_sample + MPPI_FORESEE.cost_goal_coeff * ((robot_state[0:2]-goal).T @ (robot_state[0:2]-goal))[0,0]

            # Update robot states
            robot_states = robot_states.at[:,i+1].set( MPPI_FORESEE.robot_dynamics_step( robot_states[:,[i]], perturbed_control[:, [i]] )[:,0] )
            
            # Update human states -> OR based on nearest human, obstacle only???? atleast in gazbeo its based on nearest human/object only so should be fine
            _, cost_sample, robot_state, human_sigma_state, human_sigma_weight, human_sigma_points, human_sigma_weights, human_mus, human_covs, _, _, _ = lax.fori_loop(0, MPPI_FORESEE.num_humans, loop_over_humans, (i, cost_sample, robot_state, human_sigma_state, human_sigma_weight, human_sigma_points, human_sigma_weights, human_mus, human_covs, human_goals, perturbed_control[:, [i]], obstaclesX))            

            return cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs, obstaclesX
        cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs, _ = lax.fori_loop( 0, MPPI_FORESEE.horizon-1, body, (cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs, obstaclesX) )
        
        # Update with cost for final state
        human_sigma_state = human_sigma_points[:,:,MPPI_FORESEE.horizon-1].reshape((MPPI_FORESEE.human_n, 2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.num_humans), order='F')
        human_sigma_weight = human_sigma_weights[:,:, MPPI_FORESEE.horizon-1]
        robot_state = robot_states[:,[MPPI_FORESEE.horizon-1]]
        cost_sample = cost_sample + MPPI_FORESEE.cost_goal_coeff * ((robot_state[0:2]-goal).T @ (robot_state[0:2]-goal))[0,0] # goal cost

        @jit
        def update_final_cost(j, inputs_humans):
            cost_sample, robot_state, human_sigma_state, human_sigma_weight, obstaclesX = inputs_humans
            cost_sample = cost_sample + MPPI_FORESEE.single_sample_state_cost( robot_state, human_sigma_state[:,:,j], human_sigma_weight[:,[j]].T,  goal, obstaclesX) / MPPI_FORESEE.num_humans
            return cost_sample, robot_state, human_sigma_state, human_sigma_weight, obstaclesX
        cost_sample, _, _, _, _ = lax.fori_loop(0, MPPI_FORESEE.num_humans, update_final_cost, (cost_sample, robot_state, human_sigma_state, human_sigma_weight, obstaclesX) )

        return cost_sample, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs

    @staticmethod
    @jit
    def human_point_init(i, inputs):
        points_init, weights_init, human_init_state_mu, human_init_state_cov = inputs
        points, weights = generate_sigma_points_gaussian( human_init_state_mu[:,[i]], jnp.diag(jnp.sqrt(human_init_state_cov[:,i])), jnp.zeros((MPPI_FORESEE.human_n,1)), 1.0 )
        points_init = points_init.at[:,i].set( points.T.reshape(-1,1)[:,0] )
        weights_init = weights_init.at[:,i].set( weights.T[:,0] )
        return points_init, weights_init, human_init_state_mu, human_init_state_cov

    @staticmethod
    @jit
    def rollout_states_foresee(subkey, robot_init_state, perturbed_control, previous_control, goal, human_init_state_mu, human_init_state_cov, obstaclesX, aware, human_goals):

        # Human
        human_sigma_points = jnp.zeros( (MPPI_FORESEE.samples,(2*MPPI_FORESEE.human_n+1)*MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_sigma_weights = jnp.zeros( (MPPI_FORESEE.samples,2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_mus = jnp.zeros( (MPPI_FORESEE.samples,MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_covs = jnp.zeros( (MPPI_FORESEE.samples,MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        
        ##### Initialize
        points_init = jnp.zeros(((2*MPPI_FORESEE.human_n+1)*MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans))
        weights_init = jnp.zeros((2*MPPI_FORESEE.human_n+1, MPPI_FORESEE.num_humans))
        points_init, weights_init, _, _ = lax.fori_loop(0, MPPI_FORESEE.num_humans, MPPI_FORESEE.human_point_init, (points_init, weights_init, human_init_state_mu, human_init_state_cov))

        human_sigma_points = human_sigma_points.at[:,:,:,0].set( jnp.tile(points_init.reshape((1,-1,MPPI_FORESEE.num_humans)), (MPPI_FORESEE.samples,1,1)) )
        human_sigma_weights = human_sigma_weights.at[:,:,:,0].set( jnp.tile( weights_init.reshape((1,-1,MPPI_FORESEE.num_humans)), (MPPI_FORESEE.samples,1,1) ) )
        human_mus = human_mus.at[:,:,:,0].set( jnp.tile(human_init_state_mu.reshape((1,MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans)), (MPPI_FORESEE.samples,1,1)) )
        human_covs = human_covs.at[:,:,:,0].set( jnp.tile(human_init_state_cov.reshape((1,MPPI_FORESEE.human_n, MPPI_FORESEE.num_humans)), (MPPI_FORESEE.samples,1,1)) )
        
        # Robot
        robot_states = jnp.zeros( (MPPI_FORESEE.samples, MPPI_FORESEE.robot_n, MPPI_FORESEE.horizon) )
        robot_states = robot_states.at[:,:,0].set( jnp.tile( robot_init_state.T, (MPPI_FORESEE.samples,1) ) )

        # Cost
        cost_total = jnp.zeros(MPPI_FORESEE.samples)

        if MPPI_FORESEE.use_gpu:
            @jit
            def body_sample(robot_states_init, perturbed_control_sample, human_sigma_points_init, human_sigma_weights_init, human_mus_init, human_covs_init): #, human_speed, obstaclesX):
                cost_sample, robot_states_sample, human_sigma_points_sample, human_sigma_weights_sample, human_mus_sample, human_covs_sample = MPPI_FORESEE.single_sample_rollout(goal, robot_states_init, perturbed_control_sample.T, human_sigma_points_init, human_sigma_weights_init, human_mus_init, human_covs_init, obstaclesX, aware, human_goals )
                return cost_sample, robot_states_sample, human_sigma_points_sample, human_sigma_weights_sample, human_mus_sample, human_covs_sample
            batched_body_sample = jax.vmap( body_sample, in_axes=0 )
            cost_total, robot_states, human_sigma_points, human_sigma_weights, human_mus, human_covs = batched_body_sample( robot_states[:,:,0], perturbed_control, human_sigma_points[:,:,:,0], human_sigma_weights[:,:,:,0], human_mus[:,:,:,0], human_covs[:,:,:,0])#, human_speed, obstaclesX  )
        else:
            @jit
            def body_samples(i, inputs):
                robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs, obstaclesX = inputs     

                # Get cost
                cost_sample, robot_states_sample, human_sigma_points_sample, human_sigma_weights_sample, human_mus_sample, human_covs_sample = MPPI_FORESEE.single_sample_rollout(goal, robot_states[i,:,0], perturbed_control[i,:,:].T, human_sigma_points[i,:,:,0], human_sigma_weights[i,:,:,0], human_mus[i,:,:,0], human_covs[i,:,:,0], obstaclesX, aware, human_goals )
                cost_total = cost_total.at[i].set( cost_sample )
                robot_states = robot_states.at[i,:,:].set( robot_states_sample )
                human_sigma_points= human_sigma_points.at[i,:,:].set( human_sigma_points_sample )
                human_sigma_weights= human_sigma_weights.at[i,:,:].set( human_sigma_weights_sample )
                human_mus = human_mus.at[i,:,:].set( human_mus_sample )
                human_covs = human_covs.at[i,:,:].set( human_covs_sample )
                return robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs, obstaclesX  
            robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs, obstaclesX = lax.fori_loop( 0, MPPI_FORESEE.samples, body_samples, (robot_states, human_sigma_points, human_sigma_weights, cost_total, human_mus, human_covs, obstaclesX) )
        return robot_states, cost_total, human_sigma_points, human_sigma_weights, human_mus, human_covs#, perturbation
        
    @staticmethod
    @jit
    def compute_perturbed_control(subkey, control_mu, control_cov, control_bound, U):
        perturbation = multivariate_normal( subkey, control_mu, control_cov, shape=( MPPI_FORESEE.samples, MPPI_FORESEE.horizon ) ) # K x T x nu
        
        # perturbation = jnp.clip( perturbation, -0.8, 0.8 ) #0.3
        perturbation = jnp.clip( perturbation, -1.0, 1.0 ) #0.3
        perturbed_control = U + perturbation

        perturbed_control = jnp.clip( perturbed_control, -control_bound, control_bound )
        perturbation = perturbed_control - U
        return U, perturbation, perturbed_control
    
    def compute_rollout_costs( self, init_state, goal, human_states_mu, human_states_cov, obstaclesX, aware, human_goals ):

        self.key, subkey = jax.random.split(self.key)
        self.U, perturbation, perturbed_control = MPPI_FORESEE.compute_perturbed_control(subkey, self.control_mu, self.control_cov, self.control_bound, self.U)


        t0 = time.time()
        sampled_robot_states, costs, human_sigma_points, human_sigma_weights, human_mus, human_covs = MPPI_FORESEE.rollout_states_foresee(subkey, init_state, perturbed_control, self.U, goal, human_states_mu, human_states_cov, obstaclesX, aware, human_goals)
        # sampled_robot_states, costs, human_sigma_points, human_sigma_weights, human_mus, human_covs, perturbation = MPPI_FORESEE.rollout_states_foresee(subkey, init_state, self.U, goal, human_states_mu, human_states_cov)
        tf = time.time()        
        print(f"time: {tf-t0}")                       
        # print(f"costs: min: {jnp.min(costs)}, max: {jnp.max(costs)}")
        self.U = MPPI_FORESEE.weighted_sum( self.U, perturbation, costs) #weights )

        states_final = MPPI_FORESEE.rollout_control(init_state, self.U)              
        action = self.U[0,:].reshape(-1,1)
        self.U = jnp.append(self.U[1:,:], self.U[[-1],:], axis=0)

        sampled_robot_states = sampled_robot_states.reshape(( MPPI_FORESEE.robot_n*MPPI_FORESEE.samples, MPPI_FORESEE.horizon ))
        human_mus = human_mus.reshape( (MPPI_FORESEE.human_n*MPPI_FORESEE.samples, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
        human_covs = human_covs.reshape( (MPPI_FORESEE.human_n*MPPI_FORESEE.samples, MPPI_FORESEE.num_humans, MPPI_FORESEE.horizon) )
   
        return sampled_robot_states, states_final, action, human_mus, human_covs