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
alpha1 = 0.5#5#2.0#3.0#20#6
alpha2 = 6
control_bound = 1.0
goal = np.array([-3.0,-1.0]).reshape(-1,1)
# goal = np.array([-1.0,-1.0]).reshape(-1,1)
num_people = 0
num_obstacles = 1
kv = 1.5
tf_horizon = 0.2

######### holonomic controller
n = 4 + num_obstacles + num_people# + 1# number of constraints
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1))
# alpha_qp = cp.Variable((num_people,1))
# alpha_qp_nominal = cp.Parameter((num_people,1), value = alpha*np.ones((num_people,1)))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref ) )#+ 1 * cp.sum_squares( alpha_qp - alpha_qp_nominal ) )
A2_u = cp.Parameter((n,2), value=np.zeros((n,2))) 
b2 = cp.Parameter((n,1))
const2 = [A2_u @ u2 >= b2]
# const2 += [alpha_qp >= 0]
controller2 = cp.Problem( objective2, const2 )
##########

######### holonomic controller
n = 4 + num_obstacles + num_people# + 1# number of constraints
u2_copy = cp.Variable((2,1))
u2_ref_copy = cp.Parameter((2,1))
# alpha_qp = cp.Variable((num_people,1))
# alpha_qp_nominal = cp.Parameter((num_people,1), value = alpha*np.ones((num_people,1)))
objective2_copy = cp.Minimize( cp.sum_squares( u2_copy - u2_ref_copy ) )#+ 1 * cp.sum_squares( alpha_qp - alpha_qp_nominal ) )
A2_u_copy = cp.Parameter((n,2), value=np.zeros((n,2))) 
b2_copy = cp.Parameter((n,1))
const2_copy = [A2_u_copy @ u2_copy >= b2_copy]
# const2 += [alpha_qp >= 0]
controller2_copy = cp.Problem( objective2_copy, const2_copy )
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
robot = bicycle( ax1[0], pos = np.array([ 1.0, 1.0, -2*np.pi/3, 0.1 ]), dt = dt, plot_polytope=False )
control_input_limit_points = np.array([ [control_bound, control_bound], [-control_bound, control_bound], [-control_bound, -control_bound], [control_bound, -control_bound] ])
control_bound_polytope = pt.qhull( control_input_limit_points )
ax1[0].scatter( goal[0], goal[1], edgecolors ='g', facecolors='none' )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

n  = 4
m = 2
def f_robot(t, y, args):
    control_input = robot.nominal_controller_jax(y[:-m].reshape(-1,1), goal, k_omega = 3.0, k_v = kv)
    return jnp.append( robot.xdot_jax( y[:-m].reshape(-1,1), control_input ), y[-m:].reshape(-1,1), axis=0)
    # return jnp.append( robot.xdot_jax( y[:-m].reshape(-1,1), y[-m:].reshape(-1,1) ), y[-m:].reshape(-1,1), axis=0)
term = ODETerm(f_robot)
solver = Dopri5()

# @jit
def compute_robot_future_positions(t0 =0, t1 = 2.0, dt0 = 0.1, y0 = 0, saveat=jnp.linspace(0,2,3)):
    solution = diffeqsolve( term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, saveat = SaveAt(ts=saveat) )
    return solution.ys

@jit 
def barrier_from_state_series(robot_state, robot_input, target_state, target_state_dot, saveat):
    init_state = jnp.append( robot_state, robot_input, axis=0 )
    robot_future_positions = compute_robot_future_positions( t0 = 0.0, t1=tf_horizon, dt0=dt, y0 = init_state, saveat=saveat )
    obstacle_future_positions = target_state + target_state_dot * saveat
    dist = robot_future_positions[:, 0:2, 0].T - obstacle_future_positions[0:2,:]
    h = jnp.sum( dist * dist, axis=0 ) - jnp.square(target_state[2,0]) # radius of circular obstacle
    return jnp.min(h)
barrier_from_state_series_value_grad = jit( value_and_grad(barrier_from_state_series, argnums=(0,2)) )

@jit
def compute_ff_barrier( robot_state, robot_input, obstacle_states, obstacle_states_dot ):
    saveat = jnp.linspace(0, tf_horizon, int(tf_horizon/dt)+1)    
    A_u = jnp.zeros((1,2)); b_u = jnp.zeros((1,1))
    for i in range(len(obstacles)):        
        h, dh_dx = barrier_from_state_series_value_grad( robot_state, robot_input, obstacle_states[:,i].reshape(-1,1), obstacle_states_dot[:,i].reshape(-1,1), saveat )
        dh_dx_robot = dh_dx[0].T
        dh_dx_obstacle = dh_dx[1].T
        A_u = jnp.append( A_u, dh_dx_robot @ robot.g_jax(robot_state), axis=0 )
        b_u = jnp.append( b_u, - alpha2 * h - dh_dx_robot @ robot.f_jax(robot_state) - dh_dx_obstacle @ obstacle_states_dot[:,i].reshape(-1,1), axis=0 )
    return A_u[1:], b_u[1:]

@jit
def compute_barrier( robot_state, obstacle_states, obstacle_states_dot ):
    A_u = jnp.zeros((1,2)); b_u = jnp.zeros((1,1))
    for i in range(len(obstacles)):        
        h, dh_dx_robot, dh_dx_obstacle = robot.barrier_jax( robot_state, obstacle_states[0:2,i].reshape(-1,1), obstacle_states[2,i], alpha1 = alpha1 )
        A_u = jnp.append( A_u, dh_dx_robot @ robot.g_jax(robot_state), axis=0 )
        b_u = jnp.append( b_u, - alpha2 * h - dh_dx_robot @ robot.f_jax(robot_state) - dh_dx_obstacle @ obstacle_states_dot[0:2,i].reshape(-1,1), axis=0 )
    return A_u[1:], b_u[1:]

volume = []
if 1:
# with writer.saving(fig1, 'Videos/DU_test_feasible_space.mp4', 100): 
    while t < tf:

        # desired input
        u2_ref.value = robot.nominal_controller( goal, k_v = kv )
        
        A_u, b_u = compute_barrier( jnp.asarray(robot.X), obstacle_states, obstacle_states_dot )
        A2_u.value = np.append( np.asarray(A_u), -control_bound_polytope.A, axis=0 )
        b2.value = np.append( np.asarray(b_u), -control_bound_polytope.b.reshape(-1,1), axis=0 )

        controller2.solve()#solver=cp.GUROBI)
        if controller2.status == 'infeasible':
            print(f"QP infeasible")
            exit()
            
        # Now do FF-CBF with this input
        A_u, b_u = compute_ff_barrier( jnp.asarray(robot.X), jnp.zeros((2,1)), obstacle_states, obstacle_states_dot )
        A_u, b_u = compute_ff_barrier( jnp.asarray(robot.X), jnp.asarray(u2.value), obstacle_states, obstacle_states_dot )
        A2_u_copy.value = np.append( np.asarray(A_u), -control_bound_polytope.A, axis=0 )
        b2_copy.value = np.append( np.asarray(b_u), -control_bound_polytope.b.reshape(-1,1), axis=0 )
        u2_ref_copy.value = u2_ref.value
        controller2_copy.solve()#solver=cp.GUROBI)            
        
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
        
        # robot.step( u2.value )
        robot.step( u2_copy.value )
        robot.render_plot()
        
        # robot.step( np.clip(u2_ref.value, -control_bound, control_bound) )
        # robot.render_plot()

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        t = t + dt
        
        # writer.grab_frame()