import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from crowd import crowd
from humansocialforce import *
from obstacles import rectangle
from mppi import *


# Simulatiojn Parameters
num_people = 5
t = 0
tf = 15.0
dt = 0.1 #0.05
dt_human = dt
tf_human = tf
d_human = 0.4
goal = np.array([-3, -2]).reshape(-1,1)


# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-9,7), ylim=(-7,9))
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.legend(loc='upper right')
p_robot = ax.scatter([0],[0],s=50, c='g')
ax.scatter( goal[0,0], goal[1,0], c='g' )

# Initialize robot
robot = np.array([3.0,-1.0]).reshape(-1,1)
p_robot.set_offsets([robot[0,0], robot[1,0]])


# Initialize obstacles
obstacles = []
obstacles.append( rectangle( ax, pos = np.array([0.5,1.0]), width = 2.5 ) )        
obstacles.append( rectangle( ax, pos = np.array([-0.75,-4.5]), width = 6.0 ) )
obstacles.append( rectangle( ax, pos = np.array([-1.28,3.3]), height = 4.0 ) )
obstacles.append( rectangle( ax, pos = np.array([-4.0,1.0]), height = 12.0 ) )

# Initialize humans
horizon_human = int(tf_human/dt_human)
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt_human, horizon = horizon_human, paths_file = [])#social-navigation/
h_curr_humans = np.zeros(num_people)

# hard code positions and goals
humans.X[0,0] = -1.7; humans.X[1,0] = -1.5;
humans.X[0,1] = -1.7; humans.X[1,1] = -0.7#-1.0;
humans.X[0,2] = -2.2; humans.X[1,2] = -1.6;
humans.X[0,3] = -2.2; humans.X[1,3] = -0.6;
humans.X[0,4] = -2.2; humans.X[1,4] = -1.9;
humans.goals[0,0] =  4.0; humans.goals[1,0] = -1.5;
humans.goals[0,1] =  4.0; humans.goals[1,1] = -2.4#-1.0;
humans.goals[0,2] =  4.0; humans.goals[1,2] = -1.6;
humans.goals[0,3] =  4.0; humans.goals[1,3] = -0.6;
humans.goals[0,4] =  4.0; humans.goals[1,4] = -1.9;
humans.render_plot(humans.X)

socialforce_initial_state = np.append( np.append( np.copy( humans.X.T ), 0*np.copy( humans.X.T ) , axis = 1 ), humans.goals.T, axis=1   )

#attach robot state
# robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.X[3,0]*np.cos(robot.X[2,0]), robot.X[3,0]*np.sin(robot.X[2,0]) , goal[0,0], goal[1,0]]).reshape(1,-1)
# socialforce_initial_state = np.append( socialforce_initial_state, robot_social_state, axis=0 )

humans_socialforce = socialforce.Simulator( socialforce_initial_state, delta_t = dt )

mppi = MPPI(horizon=50, samples=10, input_size=2, dt=dt)

sample_plot = []
ax.plot([0,0], [0,0], 'r*')
for i in range(mppi.samples):
    sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'g', alpha=0.5) )
sample_plot.append( ax.plot(jnp.ones(mppi.horizon), 0*jnp.ones(mppi.horizon), 'r') )
# plt.show()    
def robot_step(x,u,dt):
    return x + u*dt

while t<tf:

    # robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.X[3,0]*np.cos(robot.X[2,0]), robot.X[3,0]*np.sin(robot.X[2,0]) , goal[0,0], goal[1,0]])
    # humans_socialforce.state[-1,0:6] = robot_social_state
    # humans.controls = humans_socialforce.step().state.copy()[:-1,2:4].copy().T # When robot included

    humans.controls = humans_socialforce.step().state.copy()[:,2:4].copy().T # when robot not included

    humans.step_using_controls(dt)
    human_positions = np.copy(humans.X)
    humans.render_plot(human_positions)

    robot_sampled_states, robot_chosen_states, robot_action = mppi.compute_rollout_costs(robot, goal, human_positions, humans.controls)
    for i in range(mppi.samples):
        sample_plot[i][0].set_xdata( robot_sampled_states[2*i, :] )
        sample_plot[i][0].set_ydata( robot_sampled_states[2*i+1, :] )
    sample_plot[-1][0].set_xdata( robot_chosen_states[0, :] )
    sample_plot[-1][0].set_ydata( robot_chosen_states[1, :] )

    # robot = robot_step(robot, jnp.array([ [-1], [0] ]), dt)
    robot = robot_step(robot, robot_action, dt)
    p_robot.set_offsets([robot[0,0], robot[1,0]])

    fig.canvas.draw()
    fig.canvas.flush_events()

    t = t + dt

