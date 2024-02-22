import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, grad, value_and_grad
import cvxpy as cp
from robot_models.single_integrator import single_integrator
from robot_models.humans import humans
# from mpc_utils import *
from utils import *
from mppi_foresee import *
from mppi import *

from jax import config
config.update("jax_enable_x64", True)

# Simulation parameters
human_noise_cov = 4.0 # 4.0 #4.0 #0.5
human_noise_mean = 0
dt = 0.05 #0.05
T = 30 # simulation steps
control_bound = 4 #7 #4
kx = 4.0
sensing_radius = 2
factor = 2.0 # no of standard deviations
choice = 0
samples = 50
horizon = 50 #100 #50
human_ci_alpha = 0.005

fig, ax = plt.subplots()
ax.set_xlim([-6,5])
ax.set_ylim([-2,9])
# ax.axis('equal')
ax.set_aspect(1)

# Initialize robot
robot = single_integrator( ax, pos=np.array([0.0,0.0]), dt=dt )
robot_goal = np.array([2.0, 6.0]).reshape(-1,1)
ax.scatter(robot_goal[0,0], robot_goal[1,0], c='g', s=70)

# Initialize human
human = humans(ax, pos=np.array([4.0,4.0]), dt=dt)
human_mu = human.X
human_cov = jnp.zeros((2,1))

# Human motion functions
def human_input(tau=0):
    u = jnp.array([-3.0,0]).reshape(-1,1)
    return u

@jit
def human_step_noisy(mu_x,cov_x,u,dt):
    return mu_x.reshape(-1,1) + u*dt, cov_x.reshape(-1,1) + human_noise_cov*jnp.ones((2,1))*dt*dt

# Initialize robot controller

if choice:
    mppi = MPPI(horizon=horizon, samples=samples, input_size=2, dt=dt)
else:
    control_init_ratio = (robot_goal[1,0]-robot.X[1,0])/(robot_goal[0,0]-robot.X[0,0])

    #generate initial guess
    u_guess = jnp.zeros((horizon, 2))
    robot_x = jnp.copy(robot.X)
    for i in range(horizon):
        u_robot = jnp.clip( kx * ( robot_goal - robot_x ), -control_bound, control_bound)
        u_guess = u_guess.at[i,:].set(u_robot[:,0])
        robot_x = robot_x + u_robot * dt
    # u_guess = None
    mppi = MPPI_FORESEE(horizon=horizon, samples=samples, input_size=2, dt=dt, sensing_radius=sensing_radius, human_noise_cov=human_noise_cov, std_factor=factor, control_bound=control_bound, control_init_ratio=control_init_ratio, u_guess=u_guess)

sample_plot = []
ax.plot([0,0], [0,0], 'r*')
for i in range(mppi.samples):
    sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'g', alpha=0.2) )
sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'b') )

human_sample_plot = []
for i in range(mppi.samples):
    human_sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'y', alpha=1.0) )
human_sample_cov_plot = []
for i in range(mppi.samples):
    human_sample_cov_plot.append( plt.fill_between( [1], [0.5], [0.5], facecolor='r', alpha=human_ci_alpha ) )
# Human plot

robot_action = 0

confidence_ellipse(human_mu, np.eye(2) * (sensing_radius**2), ax, n_std=1, edgecolor = 'green', label='Sensing Radius')
ax.legend(loc='lower left')

# Run Simulation
for t in range(30):

    # Move humans
    # human.step( human_input(t*dt) )
    # robot.step( jnp.zeros((2,1)) )

    u_human = human_input(t*dt)

    if t>0:
        robot.step( robot_action )
        human_mu, human_cov = human_step_noisy(human_mu, human_cov, u_human, dt)    

    
    
    if choice:
        # robot_sampled_states, robot_chosen_states, robot_action = mppi.compute_rollout_costs(robot.X, robot_goal, human_mu, u_human)
        robot_sampled_states, robot_chosen_states, robot_action = mppi.compute_rollout_costs_chance_constraint(robot.X, robot_goal, human_mu, human_cov, u_human)
    else:
        robot_sampled_states, robot_chosen_states, robot_action, human_mus, human_covs = mppi.compute_rollout_costs(robot.X, robot_goal, human_mu, human_cov)

      
    human.render_predictions(1, human_mu, human_cov, factor)
    confidence_ellipse(human_mu, np.eye(2) * (sensing_radius**2), ax, n_std=1, edgecolor = 'green', label='Sensing Radius')

    for i in range(mppi.samples):
        sample_plot[i][0].set_xdata( robot_sampled_states[2*i, :] )
        sample_plot[i][0].set_ydata( robot_sampled_states[2*i+1, :] )

        # Human Prediction
        human_sample_cov_plot[i].remove()
        if 0: #i==0:
            human_sample_cov_plot[i] = plt.fill_between( human_mus[2*i,:], human_mus[2*i+1,:]-factor*np.sqrt(human_covs[2*i+1,:]), human_mus[2*i+1,:]+factor*np.sqrt(human_covs[2*i+1,:]), facecolor='r', alpha=human_ci_alpha, label='State Prediction Uncertainty' )
        else:
            human_sample_cov_plot[i] = plt.fill_between( human_mus[2*i,:], human_mus[2*i+1,:]-factor*np.sqrt(human_covs[2*i+1,:]), human_mus[2*i+1,:]+factor*np.sqrt(human_covs[2*i+1,:]), facecolor='r', alpha=human_ci_alpha )
        human_sample_plot[i][0].set_xdata( human_mus[2*i, :] )
        human_sample_plot[i][0].set_ydata( human_mus[2*i+1, :] )
        

    sample_plot[-1][0].set_xdata( robot_chosen_states[0, :] )
    sample_plot[-1][0].set_ydata( robot_chosen_states[1, :] )
    ax.legend(loc='lower left')

    

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
        

plt.ioff()
plt.show()