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

from jax import random

import pdb

# from jax import config
# config.update("jax_enable_x64", True)


# Simulatiojn Parameters
num_humans = 10
d_human = 0.4
use_GPU = True #False
visualize = False #False
num_iters = 20
name = "media/case1"

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
samples = 1000# 1000 #200 #100
horizon = 40 #80 #50 #100 #50
human_ci_alpha = 0.05 #0.005

# cost terms
human_nominal_speed = jnp.array([3.0,0]).reshape(-1,1)
human_repulsion_gain = 2.0 #2.0
costs_lambda = 0.03 #300 #0.05 #300
cost_goal_coeff = 0.2 #1.0
cost_safety_coeff = 10.0 #10.0

humans_interact = True
obstacles_interact = True

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



robot_goal = np.array([-3.0, -2.0]).reshape(-1,1)
def run(aware=True):
    jax.clear_caches()
    key = random.key(42)
    costs_list = []
    costs = []
    costs_array = np.zeros((1,T))
    robot_pos_array_s = np.zeros((1,2,T-1))
    human_pos_array_s = np.zeros((1,2*num_humans,T-1))
    safety_obstacle_array_s = np.zeros((1,T-1))
    safety_human_array_s = np.zeros((1,T-1))
    for iter in range(num_iters):
        print(f"Iter : {iter}")
        cost_list = [0]
        robot_pos_array = np.zeros((2,1))
        human_pos_array = np.zeros((2*num_humans,1))
        safety_obstacle_list = []
        safety_human_list = []
        # Initialize robot
        robot = single_integrator( ax, pos=np.array([2.7,1.3]), dt=dt )
        
        ax.scatter(robot_goal[0,0], robot_goal[1,0], c='g', s=70)

        # Initialize robot

        # Setup environment
        # Initialize obstacles
        obstacles = []
        obstacles.append(circle(ax, pos=np.array([1.5, 1.0]), radius=1.0))
        obstacles.append(circle(ax, pos=np.array([1.5, -3.5]), radius=1.0))
        num_obstacles = len(obstacles)
        obstaclesX = None #100*jnp.ones((2,1))#None
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
        # humans.X[0,0] = 4.0; humans.X[1,0] = 4.0;
        # humans.goals[0,0] =  -4.0; humans.goals[1,0] = 4.0;

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


        # main theme: if u know that humans are reactuve, then much better performance
        # if wrong about the assumption, then still satisfying the risk

        humans.render_plot(humans.X)

        human_mus = humans.X
        human_covs = human_localization_noise * jnp.ones((2,num_humans))

        control_init_ratio = (robot_goal[1,0]-robot.X[1,0])/(robot_goal[0,0]-robot.X[0,0])

        #generate initial guess
        u_guess = -1.0 * jnp.ones((horizon, 2))

        # robot_x = jnp.copy(robot.X)
        # u_guess = jnp.zeros((horizon, 2))
        # for i in range(horizon):
        #     u_robot = jnp.clip( kx * ( robot_goal - robot_x ), -control_bound, control_bound)
        #     u_guess = u_guess.at[i,:].set(u_robot[:,0])
        #     robot_x = robot_x + u_robot * dt
        # u_guess = None
        mppi = MPPI_FORESEE(horizon=horizon, samples=samples, input_size=2, dt=dt, sensing_radius=sensing_radius, human_noise_cov=human_noise_cov, std_factor=factor, control_bound=control_bound, control_init_ratio=control_init_ratio, u_guess=u_guess, human_nominal_speed=jnp.copy(humans.U), human_repulsion_gain=human_repulsion_gain, costs_lambda=costs_lambda, cost_goal_coeff=cost_goal_coeff, cost_safety_coeff=cost_safety_coeff, num_humans=num_humans, num_obstacles = num_obstacles, use_GPU=use_GPU, aware=aware, humans_interact=humans_interact, obstacles_interact=obstacles_interact)

        if visualize:
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

        # metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
        # writer = FFMpegWriter(fps=7, metadata=metadata)
        # movie_name = 'mppi_multi_human_zero_u.mp4'
        if 1: #with writer.saving(fig, movie_name, 100): 

            for t in range(T):

                if t>0:
                    robot.step( robot_action )

                    humans.controls = jnp.zeros((2,num_humans))
                    for j in range(num_humans):
                        u_mu, u_cov = MPPI_FORESEE.multi_human_dynamics_actual( humans.X[:,[j]], humans.X[:,MPPI_FORESEE.hindex_list[j,:]], robot.X, humans.U[:,[j]], obstaclesX )
                        # sample control input
                        key, subkey = random.split(key)
                        control = u_mu + random.normal(subkey, (2,1)) * jnp.sqrt(u_cov)
                        control = jnp.clip( control, -4, 4 )
                        humans.controls = humans.controls.at[:,j].set(control[:,0])
                        # humans.controls = humans.controls.at[:,j].set(u_mu[:,0])

                    humans.step_using_controls(dt)
                    human_positions = np.copy(humans.X)
                    humans.render_plot(human_positions)

                    human_mus = jnp.copy(humans.X)

                    cost_list.append(cost_list[-1] + MPPI_FORESEE.state_cost( robot.X, humans.X, robot_goal, obstaclesX ).item())

                    robot_pos_array = np.append(robot_pos_array, robot.X, axis=1)
                    human_pos_array = np.append(human_pos_array, human_mus.reshape(-1,1, order='F'), axis=1)
                    safety_obstacle_list.append( np.min( np.linalg.norm(robot.X[0:2]-obstaclesX, axis=0) ) )
                    safety_human_list.append( np.min( np.linalg.norm(robot.X[0:2]-human_mus, axis=0) ) )

                robot_sampled_states, robot_chosen_states, robot_action, human_mus_traj, human_covs_traj = mppi.compute_rollout_costs(robot.X, robot_goal, human_mus, human_covs, jnp.copy(humans.U), obstaclesX, aware)

                if visualize:
                    for i in range(plot_num_samples):
                        sample_plot[i][0].set_xdata( robot_sampled_states[2*i, :] )
                        sample_plot[i][0].set_ydata( robot_sampled_states[2*i+1, :] )

                        # Human Prediction
                        for j in range(num_humans):
                            # print(f"{i*plot_num_samples+j}")
                            # index = i*plot_num_samples+j
                            index = i*num_humans+j
                            human_sample_cov_plot[index].remove()
                            if 0: #i==0:
                                human_sample_cov_plot[index] = plt.fill_between( human_mus_traj[2*i,j,:], human_mus_traj[2*i+1,j,:]-factor*np.sqrt(human_covs_traj[2*i+1,j,:]), human_mus_traj[2*i+1,j,:]+factor*np.sqrt(human_covs_traj[2*i+1,j,:]), facecolor='r', alpha=human_ci_alpha, label='State Prediction Uncertainty' )
                            else:
                                human_sample_cov_plot[index] = plt.fill_between( human_mus_traj[2*i,j,:], human_mus_traj[2*i+1,j,:]-factor*np.sqrt(human_covs_traj[2*i+1,j,:]), human_mus_traj[2*i+1,j,:]+factor*np.sqrt(human_covs_traj[2*i+1,j,:]), facecolor='r', alpha=human_ci_alpha )
                            human_sample_plot[index][0].set_xdata( human_mus_traj[2*i,j,:] )
                            human_sample_plot[index][0].set_ydata( human_mus_traj[2*i+1,j,:] )

                    sample_plot[-1][0].set_xdata( robot_chosen_states[0, :] )
                    sample_plot[-1][0].set_ydata( robot_chosen_states[1, :] )
                    ax.legend(loc='lower left')

                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # writer.grab_frame()

                t = t + dt
            print(f"cost: {cost_list[-1]}")
            costs_list.append(cost_list)
            costs_array = np.append( costs_array, np.asarray(cost_list).reshape(1,-1), axis=0 )
            costs.append( cost_list[-1] )
            robot_pos_array = robot_pos_array[:,1:]
            human_pos_array = human_pos_array[:,1:]

            robot_pos_array_s = np.append( robot_pos_array_s, robot_pos_array.reshape(1,robot_pos_array.shape[0],robot_pos_array.shape[1]), axis=0)  #np.zeros((2,1,1))
            human_pos_array_s = np.append( human_pos_array_s, human_pos_array.reshape(1,human_pos_array.shape[0],human_pos_array.shape[1]), axis=0)  #np.zeros((2*num_humans,1))
            safety_obstacle_array_s = np.append( safety_obstacle_array_s, np.asarray(safety_obstacle_list).reshape(1,-1), axis=0 )
            safety_human_array_s = np.append( safety_human_array_s, np.asarray( safety_human_list ).reshape(1,-1), axis=0 )

    costs_array = costs_array[1:,:]
    robot_pos_array_s = robot_pos_array_s[1:,:,:]
    human_pos_array_s = human_pos_array_s[1:,:,:]
    safety_obstacle_array_s = safety_obstacle_array_s[1:,:]
    safety_human_array_s = safety_human_array_s[1:,:]
    print("done")

    costs = np.asarray(costs)
    costs_mu, costs_cov = np.mean(costs), np.std(costs)
    print(f"mu, cov: {costs_mu}, {costs_cov}")

    costs_means = np.mean( costs_array, axis=0 )
    costs_stds = np.std( costs_array, axis=0 )
    mppi.true_func.clear_cache()

    # pos
    robot_pos_mus = np.mean( robot_pos_array_s, axis=0 )
    human_pos_mus = np.mean( human_pos_array_s, axis=0 )
    robot_pos_covs = np.std( robot_pos_array_s, axis=0 )
    human_pos_covs = np.std( human_pos_array_s, axis=0 )

    safety_human_mus = np.mean(safety_human_array_s, axis=0)
    safety_obstacle_mus = np.mean(safety_obstacle_array_s, axis=0)
    safety_human_covs = np.std(safety_human_array_s, axis=0)
    safety_obstacle_covs = np.std(safety_obstacle_array_s, axis=0)




    return costs_means, costs_stds, (robot_pos_mus, robot_pos_covs), (human_pos_mus, human_pos_covs), (safety_human_mus, safety_human_covs), (safety_obstacle_mus, safety_obstacle_covs)

costs_means_aware, costs_stds_aware, robot_pos_aware, human_pos_aware, safety_human_aware, safety_obstacle_aware = run(aware=True)
costs_means_unaware, costs_stds_unaware, robot_pos_unaware, human_pos_unaware, safety_human_unaware, safety_obstacle_unaware = run(aware=False)

plt.ioff()
fig2, ax2 = plt.subplots()

index = np.linspace(0,T-1,T, dtype=int) * dt



# trajectories
fig3, ax3 = plt.subplots()
ax3.set_title("Dynamics Aware")
ax3.set_xlim([-9,6])
ax3.set_ylim([-7,4])
# ax.axis('equal')
ax3.set_aspect(1)
ax3.set_xlabel("X")
ax3.set_ylabel("Y")
plt.legend(loc='upper right')
obstacles = []
circle(ax3, pos=np.array([1.5, 1.0]), radius=1.0)
circle(ax3, pos=np.array([1.5, -3.5]), radius=1.0)
ax3.plot( robot_pos_aware[0][0,:], robot_pos_aware[0][1,:], 'g' )
ax3.fill_between( robot_pos_aware[0][0,:], robot_pos_aware[0][1,:]-1.96*robot_pos_aware[1][1,:], robot_pos_aware[0][1,:]+1.96*robot_pos_aware[1][1,:], facecolor='g', alpha=0.2 )
for j in range(num_humans):
    ax3.plot( human_pos_aware[0][2*j,:], human_pos_aware[0][2*j+1,:], 'r' )
    ax3.fill_between( human_pos_aware[0][2*j,:], human_pos_aware[0][2*j+1,:]-1.96*human_pos_aware[1][2*j+1,:], human_pos_aware[0][2*j+1,:]+1.96*human_pos_aware[1][2*j+1,:], facecolor='r', alpha=0.2 )
ax3.scatter(robot_goal[0,0], robot_goal[1,0], edgecolors='g', s=70, facecolors='none')
fig3.savefig(name+"_aware.png")
fig3.savefig(name+"_aware.eps")

fig3u, ax3u = plt.subplots()
ax3u.set_title("Dynamics Unaware")
ax3u.set_xlim([-9,6])
ax3u.set_ylim([-7,4])
# ax.axis('equal')
ax3u.set_aspect(1)
ax3u.set_xlabel("X")
ax3u.set_ylabel("Y")
plt.legend(loc='upper right')
obstacles = []
circle(ax3u, pos=np.array([1.5, 1.0]), radius=1.0)
circle(ax3u, pos=np.array([1.5, -3.5]), radius=1.0)
ax3u.plot( robot_pos_unaware[0][0,:], robot_pos_unaware[0][1,:], 'g' )
ax3u.fill_between( robot_pos_unaware[0][0,:], robot_pos_unaware[0][1,:]-1.96*robot_pos_unaware[1][1,:], robot_pos_unaware[0][1,:]+1.96*robot_pos_unaware[1][1,:], facecolor='g', alpha=0.2 )
for j in range(num_humans):
    ax3u.plot( human_pos_aware[0][2*j,:], human_pos_aware[0][2*j+1,:], 'r' )
    ax3u.fill_between( human_pos_unaware[0][2*j,:], human_pos_unaware[0][2*j+1,:]-1.96*human_pos_unaware[1][2*j+1,:], human_pos_unaware[0][2*j+1,:]+1.96*human_pos_unaware[1][2*j+1,:], facecolor='r', alpha=0.2 )
ax3u.scatter(robot_goal[0,0], robot_goal[1,0], edgecolors='g', s=70, facecolors='none')
fig3u.savefig(name+"_aware.png")
fig3u.savefig(name+"_aware.eps")


def plot_costs(ax, costs_means, costs_stds, mu_color='g', std_color='y', label='Dynamics Aware'):
    ax.plot(index[1:], costs_means, mu_color, label=label)
    ax.fill_between( index[1:], costs_means - 1.96 * costs_stds, costs_means + 1.96 * costs_stds, facecolor=std_color, alpha=0.2)




plot_costs( ax2, costs_means_aware[1:], costs_stds_aware[1:], mu_color='g', std_color='y', label='Dynamics Aware' )
plot_costs( ax2, costs_means_unaware[1:], costs_stds_unaware[1:], mu_color='r', std_color='m', label='Dynamics Unaware' )
ax2.legend()
ax2.set_xlabel("time (s)")
ax2.set_ylabel("Accumulated Cost")
fig2.savefig(name+"_costs.png")
fig2.savefig(name+"_costs.eps")
# Safety Human
fig4, ax4 = plt.subplots()
plot_costs( ax4, safety_human_aware[0], safety_human_aware[1], mu_color='g', std_color='y', label='Dynamics Aware' )
plot_costs( ax4, safety_human_unaware[0], safety_human_unaware[1], mu_color='r', std_color='m', label='Dynamics Unaware' )
ax4.legend()
ax4.set_xlabel("time (s)")
ax4.set_ylabel("Min Distance to Humans")
fig4.savefig(name+"_human_dists.png")
fig4.savefig(name+"_human_dists.eps")

# Safety Obstacle
fig5, ax5 = plt.subplots()
plot_costs( ax5, safety_obstacle_aware[0], safety_obstacle_aware[1], mu_color='g', std_color='y', label='Dynamics Aware' )
plot_costs( ax5, safety_obstacle_unaware[0], safety_obstacle_unaware[1], mu_color='r', std_color='m', label='Dynamics Unaware' )
ax5.legend()
ax5.set_xlabel("time (s)")
ax5.set_ylabel("Min Distance to obstacles")
fig5.savefig(name+"_obstacle_dists.eps")
fig5.savefig(name+"_obstacle_dists.png")






plt.show()