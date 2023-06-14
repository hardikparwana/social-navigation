import numpy as np
import cvxpy as cp
import polytope as pt
import matplotlib.pyplot as plt

from bicycle_new import bicycle
from single_integrator import single_integrator_square
from polytope_utils import plot_polytope_lines
from obstacles import circle
from matplotlib.animation import FFMpegWriter
from crowd import crowd
from humansocialforce import *


# Sim parameters
t = 0
dt = 0.03
tf = 15
alpha = 2#6#5#2.0#3.0#20
alpha1 = 0.5#2#20#4.0#0.5#50
control_bound = 2.0
goal = np.array([-3.0,-1.0]).reshape(-1,1)
num_people = 5
num_obstacles = 4
k_v = 1.5
######### holonomic controller
n = 4 + num_obstacles + num_people # number of constraints
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref ) )
A2 = cp.Parameter((n,2))
b2 = cp.Parameter((n,1))
const2 = [A2 @ u2 >= b2]
controller2 = cp.Problem( objective2, const2 )
##########

plt.ion()
fig1, ax1 = plt.subplots( 1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [5, 5, 2]})# )#, gridspec_kw={'height_ratios': [1, 1]} )
ax1[0].set_xlim([-3,5])
ax1[0].set_ylim([-3,5])
offset = 3.0
ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
ax1[1].set_ylim([-control_bound-offset, control_bound+offset])

# Obstacles
obstacles = []
obstacles.append( circle( ax1[0], pos = np.array([-1.0,-0.6]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([0.0,-0.6]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([-1.0,2.2]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([0.0,2.2]), radius = 0.5 ) )

# Robot
# robot = single_integrator_square( ax1[0], pos = np.array([ 0, 0 ]), dt = dt, plot_polytope=False )
robot = bicycle( ax1[0], pos = np.array([ 1.0, 1.0, np.pi, 1.3 ]), dt = dt, plot_polytope=False )
control_input_limit_points = np.array([ [control_bound, control_bound], [-control_bound, control_bound], [-control_bound, -control_bound], [control_bound, -control_bound] ])
control_bound_polytope = pt.qhull( control_input_limit_points )
ax1[0].scatter( goal[0], goal[1], edgecolors ='g', facecolors='none' )

# Humans

humans = crowd(ax1[0], crowd_center = np.array([0,0]), num_people = num_people, paths_file = [])#social-navigation/
humans.X[0,0] = -1.7; humans.X[1,0] = 0.5;
humans.X[0,1] = -1.7; humans.X[1,1] = 1.3#-1.0;
humans.X[0,2] = -2.2; humans.X[1,2] = 0.4;
humans.X[0,3] = -2.2; humans.X[1,3] = 1.4;
humans.X[0,4] = -2.2; humans.X[1,4] = 0.1;
humans.goals[0,0] =  4.0; humans.goals[1,0] = 0.5;
humans.goals[0,1] =  4.0; humans.goals[1,1] = -0.4#-1.0;
humans.goals[0,2] =  4.0; humans.goals[1,2] = 0.4;
humans.goals[0,3] =  4.0; humans.goals[1,3] = 1.4;
humans.goals[0,4] =  4.0; humans.goals[1,4] = 0.1;
humans.render_plot(humans.X)

socialforce_initial_state = np.append( np.append( np.copy( humans.X.T ), 0*np.copy( humans.X.T ) , axis = 1 ), humans.goals.T, axis=1   )
robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.X[3,0]*np.cos(robot.X[2,0]), robot.X[3,0]*np.sin(robot.X[2,0]) , goal[0,0], goal[1,0]]).reshape(1,-1)
socialforce_initial_state = np.append( socialforce_initial_state, robot_social_state, axis=0 )
humans_socialforce = socialforce.Simulator( socialforce_initial_state, delta_t = dt )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

volume = []
# if 1:
with writer.saving(fig1, 'Videos/DU_test_feasible_space.mp4', 100): 
    while t < tf:

        robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.X[3,0]*np.cos(robot.X[2,0]), robot.X[3,0]*np.sin(robot.X[2,0]) , goal[0,0], goal[1,0]])
        humans_socialforce.state[-1,0:6] = robot_social_state
        humans.controls = humans_socialforce.step().state.copy()[:-1,2:4].copy().T
        humans.step_using_controls(dt)

        # desired input
        u2_ref.value = robot.nominal_controller( goal, k_v = k_v )

        # barrier function
        A = np.zeros((1,2)); b = np.zeros((1,1))
        for i in range(len(obstacles)):
            h, dh_dx, _ = robot.barrier( obstacles[i], d_min = 0.5, alpha1 = alpha1 )
            A = np.append( A, dh_dx @ robot.g(), axis = 0 )
            b = np.append( b, - alpha * h - dh_dx @ robot.f(), axis = 0 )
        for i in range(num_people):
            h, dh_dx, _ = robot.barrier_humans( humans.X[:,i].reshape(-1,1), humans.controls[:,i].reshape(-1,1), d_min = 0.5, alpha1 = alpha1 )
            A = np.append( A, dh_dx @ robot.g(), axis = 0 )
            b = np.append( b, - alpha * h - dh_dx @ robot.f(), axis = 0 )
        A2.value = np.append( A[1:], -control_bound_polytope.A, axis=0 )
        b2.value = np.append( b[1:], -control_bound_polytope.b.reshape(-1,1), axis=0 )

        ax1[1].clear()
        ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
        ax1[1].set_ylim([-control_bound-offset, control_bound+offset])
        hull = pt.Polytope( -A2.value, -b2.value )
        hull_plot = hull.plot(ax1[1], color = 'g')
        plot_polytope_lines( ax1[1], hull, control_bound )

        volume.append(pt.volume( hull, nsamples=50000 ))
        ax1[2].plot( volume, 'r' )
        ax1[2].set_title('Polytope Volume')

        controller2.solve()
        if controller2.status == 'infeasible':
            print(f"QP infeasible")
            exit()
        robot.step( u2.value )
        robot.render_plot()
        humans.render_plot(humans.X)

        ax1[1].set_xlabel('Linear Acceleration'); ax1[1].set_ylabel('Angular Velocity')
        # ax1[1].set_xlabel(r'$u_x$'); ax1[1].set_ylabel(r'$u_y$')
        ax1[1].scatter( u2.value[0,0], u2.value[1,0], c = 'r', label = 'CBF-QP chosen control' )
        ax1[1].legend()
        ax1[1].set_title('Feasible Space for Control')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        t = t + dt
        
        writer.grab_frame()