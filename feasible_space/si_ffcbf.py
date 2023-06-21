import numpy as np
import cvxpy as cp
import polytope as pt
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad, value_and_grad
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt

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
alpha = 2#5#2.0#3.0#20#6
control_bound = 1.0
goal = np.array([-3.0,-1.0]).reshape(-1,1)
num_people = 5
num_obstacles = 4
kx = 30.0

######### holonomic controller
n = 4 + num_obstacles + num_people + 1# number of constraints
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1))
alpha_qp = cp.Variable((num_people,1))
alpha_qp_nominal = cp.Parameter((num_people,1), value = alpha*np.ones((num_people,1)))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref ) + 1 * cp.sum_squares( alpha_qp - alpha_qp_nominal ) )
A2_u = cp.Parameter((n,2), value=np.zeros((n,2))) 
b2 = cp.Parameter((n,1))
const2 = [A2_u @ u2 >= b2]
const2 += [alpha_qp >= 0]
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
# obstacles.append( circle( ax1[0], pos = np.array([0.0,-0.6]), radius = 0.5 ) )  
# obstacles.append( circle( ax1[0], pos = np.array([-1.0,2.2]), radius = 0.5 ) )  
# obstacles.append( circle( ax1[0], pos = np.array([0.0,2.2]), radius = 0.5 ) )
obstacle_states = np.append( obstacles[0].X, np.array([[obstacles[0].radius]]), axis=0 )
for i in range(1,len(obstacles)):
    state = np.append( obstacles[i].X, np.array([[obstacles[i].radius]]), axis=0 )
    obstacle_states = np.append(obstacle_states, state, axis=1  )
obstacle_states = jnp.asarray(obstacle_states)
obstacle_states_dot = 0 * obstacle_states

# Robot
robot = single_integrator_square( ax1[0], pos = np.array([ 1.0, 1.0 ]), dt = dt, plot_polytope=False )
control_input_limit_points = np.array([ [control_bound, control_bound], [-control_bound, control_bound], [-control_bound, -control_bound], [control_bound, -control_bound] ])
control_bound_polytope = pt.qhull( control_input_limit_points )
ax1[0].scatter( goal[0], goal[1], edgecolors ='g', facecolors='none' )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

n  = 2
m = 2
def f_robot(t, y, args):
    return robot.xdot_jax( y[:-m].reshape(-1,1), y[m:].rehape(-1,1) )
term = ODETerm(f_robot)
solver = Dopri5()

@jit
def robot_future_positions(t0 =0, t1 = 2.0, dt0 = 0.1, y0 = 0, saveat=jnp.linspace(0,2,3)):
    solution = diffeqsolve( term, solver, t0=t0, t1=t1, dt0=dt0, y0=0, saveat = SaveAt(ts=saveat) )
    return solution.ys

# @jit 
def barrier_from_state_series(robot_states, target_states):
    dist = robot_states[0:2, :] - target_states[0:2,:]
    h = jnp.sum( dist * dist, axis=0 )
    return jnp.min(h)
barrier_from_state_series_value_grad = jit( value_and_grad(barrier_from_state_series, argnums=(0,1)) )

@jit
def compute_ff_barrier( robot_state, robot_input, obstacle_states, obstacle_states_dot, tf, dt ):
    saveat = jnp.linspace(0, tf, int(tf/dt)+1)    
    init_state = jnp.append( robot_state, robot_input, axis=0 )
    robot_future_positions = robot_future_positions( t0 = 0.0, t1=tf, dt0=0.05, y0 = init_state, saveat=saveat )
    
    A_u = np.zeros((1,2)); b = np.zeros((1,1))
    for i in range(len(obstacles)):
        obstacle_future_positions = obstacle_states[:,i].reshape(-1,1) + obstacle_states_dot * saveat
        h, dh_dx = barrier_from_state_series_value_grad( robot_future_positions, obstacle_future_positions )
        dh_dx_robot = dh_dx[0]
        dh_dx_obstacle = dh_dx[1]
        A_u = jnp.append( A_u, dh_dx_robot @ robot.g_jax(robot_state), axis=0 )
        b_u = jnp.append( b_u, - alpha * h, dh_dx_robot @ robot.f_jax(robot_state) - dh_dx_obstacle @ obstacle_states_dot[:,i].reshape(-1,1) )
    return A_u, b_u

volume = []
if 1:
# with writer.saving(fig1, 'Videos/DU_test_feasible_space.mp4', 100): 
    while t < tf:

        # desired input
        u2_ref.value = robot.nominal_controller( goal, kx = kx )

        # barrier function standard
        # A_u = np.zeros((1,2)); A_alpha = np.zeros((1,num_people)); b = np.zeros((1,1))
        # for i in range(len(obstacles)):
        #     h, dh_dx, _ = robot.barrier( obstacles[i], d_min = 0.5 )
        #     A_u = np.append( A_u, dh_dx @ robot.g(), axis = 0 )
        #     A_alpha = np.append( A_alpha, np.zeros((1,num_people)), axis=0 )
        #     b = np.append( b, - alpha * h - dh_dx @ robot.f(), axis = 0 )

        # barrier function future focused
        # use robot's ;ast control input for now
        A_u, b_u = compute_ff_barrier( jnp.asarray(robot.X), jnp.asarray(robot.U), obstacle_states, obstacle_states_dot, tf, dt )
        A2_u.value = np.append( A_u[1:], -control_bound_polytope.A, axis=0 )
        b2.value = np.append( b_u[1:], -control_bound_polytope.b.reshape(-1,1), axis=0 )

        controller2.solve()
        if controller2.status == 'infeasible':
            print(f"QP infeasible")
            exit()
        robot.step( u2.value )
        robot.render_plot()
        humans.render_plot(humans.X)
        print(f"alpha: {alpha_qp.value.T}")
        
        ax1[1].clear()
        ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
        ax1[1].set_ylim([-control_bound-offset, control_bound+offset])
        hull = pt.Polytope( -A2_u.value, -b2.value )
        hull_plot = hull.plot(ax1[1], color = 'g')
        plot_polytope_lines( ax1[1], hull, control_bound )
        volume.append(pt.volume( hull, nsamples=50000 ))
        ax1[2].plot( volume, 'r' )
        ax1[2].set_title('Polytope Volume')

        ax1[1].set_xlabel(r'$u_x$'); ax1[1].set_ylabel(r'$u_y$')
        ax1[1].scatter( u2.value[0,0], u2.value[1,0], c = 'r', label = 'CBF-QP chosen control' )
        ax1[1].legend()
        ax1[1].set_title('Feasible Space for Control')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        t = t + dt
        
        # writer.grab_frame()