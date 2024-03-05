import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from robot_models.single_integrator import single_integrator
from robot_models.unicycle import unicycle
from robot_models.humans import humans
from utils import *
from mppi_foresee_multi_humans import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from crowd import crowd
from humansocialforce import *
from obstacles import rectangle, circle

# from jax import config
# config.update("jax_enable_x64", True)


# Simulatiojn Parameters
num_humans = 10
d_human = 0.4
use_GPU = False

# Simulation parameters
human_noise_cov = 4.0 # 4.0 #4.0 #0.5
human_noise_mean = 0
human_localization_noise = 0.05
dt = 0.05 #0.05
T = 50 # simulation steps
control_bound = 4 #7 #4
kx = 4.0
sensing_radius = 2
factor = 2.0 # no of standard deviations
choice = 0
samples = 500 #100
horizon = 40 #80 #50 #100 #50
human_ci_alpha = 0.05 #0.005

# cost terms
human_nominal_speed = jnp.array([3.0,0]).reshape(-1,1)
human_repulsion_gain = 2.0
costs_lambda = 0.03 #300 #0.05 #300
cost_goal_coeff = 0.2 #1.0
cost_safety_coeff = 10.0 #10.0

# Good cases
# 1. bound =7, samples=5000, horizon=80, lambda=300, costs coeff: 1, 5

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-9,6])
ax.set_ylim([-7,4])
# ax.axis('equal')
ax.set_aspect(1)
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.legend(loc='upper right')

# Initialize robot
# robot = single_integrator( ax, pos=np.array([5.0,-1.0]), dt=dt )
robot = single_integrator( ax, pos=np.array([2.0,1.3]), dt=dt )
robot = unicycle( ax, pos=np.array([2.0,1.3, -np.pi/2]), dt=dt )
robot_goal = np.array([-3.0, -2.0]).reshape(-1,1)
ax.scatter(robot_goal[0,0], robot_goal[1,0], c='g', s=70)

# Initialize robot


# Setup environment
# Initialize obstacles
obstacles = []
obstacles.append(circle(ax, pos=np.array([1.5, 1.0]), radius=1.0))
obstacles.append(circle(ax, pos=np.array([1.5, -3.5]), radius=1.0))
# obstacles.append( rectangle( ax, pos = np.array([0.5,1.0]), width = 2.5 ) )        
# obstacles.append( rectangle( ax, pos = np.array([-0.75,-4.5]), width = 6.0 ) )
# obstacles.append( rectangle( ax, pos = np.array([-1.28,3.3]), height = 4.0 ) )
# obstacles.append( rectangle( ax, pos = np.array([-4.0,1.0]), height = 12.0 ) )
num_obstacles = len(obstacles)
obstacleX = None
for i in range(num_obstacles):
    if i==0:
        obstaclesX = np.copy(obstacles[0].X)
    else:
        obstaclesX = np.append(obstaclesX, obstacles[i].X, axis=1)

# Initialize humans
horizon_human = horizon
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_humans, dt = dt, horizon = horizon_human, paths_file = [])#social-navigation/
h_curr_humans = np.zeros(num_humans)

# hard code positions and goals
humans.X[0,0] = -6.7; humans.X[1,0] = -1.5;
humans.X[0,1] = -2.7; humans.X[1,1] = -0.7#-1.0;
humans.X[0,2] = -1.2; humans.X[1,2] = -1.6;
humans.X[0,3] = -1.2; humans.X[1,3] = -0.6;
humans.X[0,4] = -1.2; humans.X[1,4] = -1.9;
humans.goals[0,0] =  4.0; humans.goals[1,0] = -1.5;
humans.goals[0,1] =  4.0; humans.goals[1,1] = -2.4#-1.0;
humans.goals[0,2] =  4.0; humans.goals[1,2] = -1.6;
humans.goals[0,3] =  4.0; humans.goals[1,3] = -0.6;
humans.goals[0,4] =  4.0; humans.goals[1,4] = -1.9;

humans.X[0,5] = -0.7; humans.X[1,5] = -1.5;
humans.X[0,6] = -0.7; humans.X[1,6] = -0.7#-1.0;
humans.X[0,7] = -0.2; humans.X[1,7] = -1.6;
humans.X[0,8] = -0.2; humans.X[1,8] = -0.6;
humans.X[0,9] = -0.2; humans.X[1,9] = -1.9;
humans.goals[0,5] =  4.0; humans.goals[1,5] = -1.5;
humans.goals[0,6] =  4.0; humans.goals[1,6] = -2.4#-1.0;
humans.goals[0,7] =  4.0; humans.goals[1,7] = -1.6;
humans.goals[0,8] =  4.0; humans.goals[1,8] = -0.6;
humans.goals[0,9] =  4.0; humans.goals[1,9] = -1.9;

humans.U = np.tile( human_nominal_speed, (1,num_humans) )
# human_nominal_speed = jnp.copy(humans.U)


humans.render_plot(humans.X)

human_mus = humans.X
human_covs = human_localization_noise * jnp.ones((2,num_humans))

control_init_ratio = (robot_goal[1,0]-robot.X[1,0])/(robot_goal[0,0]-robot.X[0,0])

#generate initial guess
u_guess = -1.0 * jnp.ones((horizon, 2))
u_guess = jnp.zeros((horizon, 2))
# robot_x = jnp.copy(robot.X)
# for i in range(horizon):
#     u_robot = jnp.clip( kx * ( robot_goal - robot_x ), -control_bound, control_bound)
#     u_guess = u_guess.at[i,:].set(u_robot[:,0])
#     robot_x = robot_x + u_robot * dt
# u_guess = None
mppi = MPPI_FORESEE(horizon=horizon, samples=samples, input_size=2, dt=dt, sensing_radius=sensing_radius, human_noise_cov=human_noise_cov, std_factor=factor, control_bound=control_bound, control_init_ratio=control_init_ratio, u_guess=u_guess, human_nominal_speed=jnp.copy(humans.U), human_repulsion_gain=human_repulsion_gain, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, num_humans=num_humans, num_obstacles = num_obstacles, use_GPU=use_GPU)


plot_num_samples = 4
sample_plot = []
ax.plot([0,0], [0,0], 'r*')
for i in range(plot_num_samples):
    sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'g', alpha=0.2) )
sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'b') )

human_sample_plot = []
human_sample_cov_plot = []
for i in range(plot_num_samples):
    for j in range(num_humans):
        human_sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'y', alpha=1.0) )
        human_sample_cov_plot.append( plt.fill_between( [1], [0.5], [0.5], facecolor='r', alpha=human_ci_alpha ) )

# Human plot
robot_action = 0

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=7, metadata=metadata)
movie_name = 'mppi_multi_human_zero_u.mp4'

with writer.saving(fig, movie_name, 100): 

    for t in range(T):

        if t>0:
            robot.step( robot_action )

            # expanded_states, expanded_weights = MPPI_FORESEE.human_sigma_point_expand_actual( human_sigma_points, human_sigma_weights, robot.X )
            # human_mu, human_cov, human_sigma_points,  human_sigma_weights = sigma_point_compress( expanded_states, expanded_weights )
            # human_cov = jnp.diag(human_cov).reshape(-1,1)
            humans.controls = jnp.zeros((2,num_humans))
            for j in range(num_humans):
                u_mu, u_cov = MPPI_FORESEE.multi_human_dynamics_actual( humans.X[:,[j]], humans.X[:,MPPI_FORESEE.hindex_list[j,:]], robot.X, humans.U[:,[j]], obstaclesX )
                humans.controls = humans.controls.at[:,j].set(u_mu[:,0])

            humans.step_using_controls(dt)
            human_positions = np.copy(humans.X)
            humans.render_plot(human_positions)

            human_mus = jnp.copy(humans.X)

        t0 = time.time()
        robot_sampled_states, robot_chosen_states, robot_action, human_mus_traj, human_covs_traj = mppi.compute_rollout_costs(robot.X, robot_goal, human_mus, human_covs, jnp.copy(humans.U), obstaclesX)
        print(f"time: {time.time()-t0}")

        for i in range(plot_num_samples):
            sample_plot[i][0].set_xdata( robot_sampled_states[2*i, :] )
            sample_plot[i][0].set_ydata( robot_sampled_states[2*i+1, :] )

            # Human Prediction
            for j in range(num_humans):
                human_sample_cov_plot[i*plot_num_samples+j].remove()
                if 0: #i==0:
                    human_sample_cov_plot[i*plot_num_samples+j] = plt.fill_between( human_mus_traj[2*i,j,:], human_mus_traj[2*i+1,j,:]-factor*np.sqrt(human_covs_traj[2*i+1,j,:]), human_mus_traj[2*i+1,j,:]+factor*np.sqrt(human_covs_traj[2*i+1,j,:]), facecolor='r', alpha=human_ci_alpha, label='State Prediction Uncertainty' )
                else:
                    human_sample_cov_plot[i*plot_num_samples+j] = plt.fill_between( human_mus_traj[2*i,j,:], human_mus_traj[2*i+1,j,:]-factor*np.sqrt(human_covs_traj[2*i+1,j,:]), human_mus_traj[2*i+1,j,:]+factor*np.sqrt(human_covs_traj[2*i+1,j,:]), facecolor='r', alpha=human_ci_alpha )
                human_sample_plot[i*plot_num_samples+j][0].set_xdata( human_mus_traj[2*i,j,:] )
                human_sample_plot[i*plot_num_samples+j][0].set_ydata( human_mus_traj[2*i+1,j,:] )

        sample_plot[-1][0].set_xdata( robot_chosen_states[0, :] )
        sample_plot[-1][0].set_ydata( robot_chosen_states[1, :] )
        ax.legend(loc='lower left')

        fig.canvas.draw()
        fig.canvas.flush_events()

        writer.grab_frame()

        t = t + dt

