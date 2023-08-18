import time
import numpy as np
import cvxpy as cp
import polytope as pt
import matplotlib.pyplot as plt

from bicycle_new import bicycle
from single_integrator import single_integrator_square
from polytope_utils import *
from obstacles import circle
from matplotlib.animation import FFMpegWriter
from crowd import crowd
from humansocialforce import *
import jax.numpy as jnp

# Sim parameters
t = 0
dt = 0.03
tf = 15
alpha2 =6.0#5#2.0#3.0#20#6
alpha1 = 2.0#20#4.0#0.5#50#2
alpha1_obstacle = 2
alpha2_obstacle = 6
control_bound = 2.0
goal = np.array([-3.0,-1.0]).reshape(-1,1)
num_people = 5
num_obstacles = 4
k_x = 1.0#0.5#30.0
k_v = 1.0#1.5
d_min_human = 0.5
d_min_obstacle = 0.5
use_ellipse = True#False
plot_ellipse = True#False
use_circle = False#True
plot_circle = False#True
alpha_polytope_user = 1.0
min_polytope_volume_ellipse = -2.7#-0.5
min_polytope_volume_circle = 0.0

######### holonomic controller
n = 4 + num_obstacles + num_people # number of constraints
u2_base = cp.Variable((2,1))
u2_base_ref = cp.Parameter((2,1))
alpha_qp_base = cp.Variable((num_people,1))
alpha_qp_base_nominal = cp.Parameter((num_people,1), value = alpha1*np.ones((num_people,1)))
objective2_base = cp.Minimize( 10*cp.sum_squares( u2_base - u2_base_ref ) + 1 * cp.sum_squares( alpha_qp_base - alpha_qp_base_nominal ) )
A2_base_u = cp.Parameter((n,2), value=np.zeros((n,2))) 
A2_base_alpha = cp.Parameter((n,num_people), value = np.zeros((n,num_people)))
b2_base = cp.Parameter((n,1))
const2_base = [A2_base_u @ u2_base + A2_base_alpha @ alpha_qp_base>= b2_base]
const2_base += [alpha_qp_base >= 0]
const2_base += [alpha_qp_base <= alpha2/2.0*np.ones((num_people,1))]
# const2_base += [alpha_qp_base == alpha_qp_base_nominal]
controller2_base = cp.Problem( objective2_base, const2_base )

n = 4 + num_obstacles + num_people + 1 # number of constraints
u2 = cp.Variable((2,1))
alpha_polytope = cp.Variable((1,1))
alpha_polytope_nominal = cp.Parameter((1,1), value=alpha_polytope_user*np.ones((1,1)))
u2_ref = cp.Parameter((2,1))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref ) + 100 * cp.sum_squares(alpha_polytope - alpha_polytope_nominal) )
A2 = cp.Parameter((n,2))
A2_polytope = cp.Parameter((n,1), value = np.zeros((n,1)))
b2 = cp.Parameter((n,1))
const2 = [A2 @ u2 + A2_polytope @ alpha_polytope >= b2]
# const2 = [alpha_polytope == alpha_polytope_nominal]
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
obstacle_states = []
obstacles.append( circle( ax1[0], pos = np.array([-1.0,-0.6]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([0.0,-0.6]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([-1.0,2.2]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([0.0,2.2]), radius = 0.5 ) )
obstacle_states = np.append( obstacles[0].X, np.array([[obstacles[0].radius]]), axis=0 )
for i in range(1,len(obstacles)):
    state = np.append( obstacles[i].X, np.array([[obstacles[i].radius]]), axis=0 )
    obstacle_states = np.append(obstacle_states, state, axis=1  )
obstacle_states = jnp.asarray(obstacle_states)

# exit()0
# Robot
robot = bicycle( ax1[0], pos = np.array([ 1.0, 1.0, np.pi, 0.3 ]), dt = dt, plot_polytope=False )
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
robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.U[0,0], robot.U[1,0] , goal[0,0], goal[1,0]]).reshape(1,-1)
socialforce_initial_state = np.append( socialforce_initial_state, robot_social_state, axis=0 )
humans_socialforce = socialforce.Simulator( socialforce_initial_state, delta_t = dt )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)
# exit()
volume = []
volume2 = []
volume_ellipse = []
volume_ellipse2 = []
volume_circle2 = []

@jit
def construct_alpha_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1 = alpha1, alpha2=alpha2 ):         
            # barrier function
            A = jnp.zeros((1,2)); b = jnp.zeros((1,1)); A_alpha = jnp.zeros((1,num_people))
            # for i in range(obstacle_states.shape[1]):
            for i in range(len(obstacles)):
                h, dh_dx, _ = robot.barrier_jax( robot_state, obstacle_states[:,i].reshape(-1,1), d_min = d_min_human, alpha1 = alpha1_obstacle )
                A = jnp.append( A, dh_dx @ robot.g_jax(robot_state), axis = 0 )
                A_alpha = jnp.append( A_alpha, jnp.zeros((1,num_people)), axis=0 )
                b = jnp.append( b, - alpha2_obstacle * h - dh_dx @ robot.f_jax(robot_state), axis = 0 )
            for i in range(humans.X.shape[1]):
                # h, dh_dx, dh_dx_human = robot.barrier_humans_jax( robot_state, humans_states[:,i].reshape(-1,1), human_states_dot[:,i].reshape(-1,1), d_min = d_min_obstacle, alpha1 = alpha1 )
                dh_dot_dx1, dh_dot_dx2, h_dot, h = robot.barrier_humans_alpha_jax( robot_state, humans_states[:,i].reshape(-1,1), human_states_dot[:,i].reshape(-1,1), d_min = d_min_obstacle)
                A = jnp.append( A, dh_dot_dx1 @ robot.g_jax(robot_state), axis = 0 )
                A_temp = jnp.zeros((1,num_people))
                A_temp = A_temp.at[0,i].set(h_dot[0,0] + alpha2*h[0,0])
                A_alpha = jnp.append( A_alpha, A_temp, axis=0 )
                b = jnp.append( b, - dh_dot_dx1 @ robot.f_jax(robot_state) - dh_dot_dx2 @ humans.controls[:,i].reshape(-1,1) - alpha2 * h_dot, axis = 0 )
            return A[1:], A_alpha[1:], b[1:]

@jit
def construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1 = alpha1, alpha2=alpha2 ):         
            # barrier function
            A = jnp.zeros((1,2)); b = jnp.zeros((1,1))
            # for i in range(obstacle_states.shape[1]):
            for i in range(len(obstacles)):
                h, dh_dx, _ = robot.barrier_jax( robot_state, obstacle_states[:,i].reshape(-1,1), d_min = d_min_human, alpha1 = alpha1_obstacle )
                A = jnp.append( A, dh_dx @ robot.g_jax(robot_state), axis = 0 )
                b = jnp.append( b, - alpha2_obstacle * h - dh_dx @ robot.f_jax(robot_state), axis = 0 )
            # for i in range(humans_states.shape[1]):
            for i in range(humans.X.shape[1]):
                dh_dot_dx1, dh_dot_dx2, h_dot, h = robot.barrier_humans_alpha_jax( robot_state, humans_states[:,i].reshape(-1,1), human_states_dot[:,i].reshape(-1,1), d_min = d_min_obstacle)
                A = jnp.append( A, dh_dot_dx1 @ robot.g_jax(robot_state), axis = 0 )
                b = jnp.append( b, - dh_dot_dx1 @ robot.f_jax(robot_state) - dh_dot_dx2 @ humans.controls[:,i].reshape(-1,1) - alpha2 * h_dot - alpha1[i] * (h_dot + alpha2*h), axis = 0 )
            return A[1:], b[1:]

@jit       
def ellipse_volume( B, d ):
     return jnp.log( jnp.linalg.det(B) )

@jit
def circle_volume( r, c ):
    return jnp.pi * jnp.square(r)

def compute_ellipse_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1=alpha1, alpha2=alpha2):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1=alpha1, alpha2=alpha2 )
    A2 = jnp.append( -A, control_A, axis=0 )
    b2 = jnp.append( -b, control_b, axis=0 )
    solution = ellipse_cvxpylayer( A2, b2 )
    B = solution[0]
    d = solution[1]
    return B, d, ellipse_volume( B, d )

def compute_circle_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1=alpha1, alpha2=alpha2):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1=alpha1, alpha2=alpha2 )
    A2 = jnp.append( -A, control_A, axis=0 )
    A2_root = jnp.linalg.norm( A2, axis=1 )
    b2 = jnp.append( -b, control_b, axis=0 )
    solution = circle_cvxpylayer( A2, A2_root, b2 )
    r = solution[0]
    c = solution[1]
    return r, c, circle_volume( r, c )

def polytope_ellipse_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1=alpha1, alpha2=alpha2):
    return compute_ellipse_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1=alpha1, alpha2=alpha2)[2]

def polytope_circle_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1=alpha1, alpha2=alpha2):
    return compute_circle_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1=alpha1, alpha2=alpha2)[2]
 
polytope_ellipse_volume_from_states_grad = grad( polytope_ellipse_volume_from_states, argnums=(0,1,2,3) )
# polytope_ellipse_volume_from_states_grad2 = grad( polytope_ellipse_volume_from_states, argnums=(0) )

polytope_circle_volume_from_states_grad = grad( polytope_circle_volume_from_states, argnums=(0,1,2,3) )
    
if 1:
# with writer.saving(fig1, 'Videos/DU_fs_alpha_v1.mp4', 100): 
    while t < tf:

        robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.U[0,0], robot.U[1,0] , goal[0,0], goal[1,0]]).reshape(1,-1)
        humans_socialforce.state[-1,0:6] = robot_social_state
        humans.controls = humans_socialforce.step().state.copy()[:-1,2:4].copy().T
        humans.step_using_controls(dt)

        # desired input
        u2_base_ref.value = robot.nominal_controller( goal, k_x = k_x, k_v = k_v )
        u2_ref.value = u2_base_ref.value#  robot.nominal_controller( goal, k_x = k_x, k_v = k_v )
       
        A, A_alpha, b = construct_alpha_barrier_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls) )
        A = np.append( np.asarray(A), -control_bound_polytope.A, axis=0 )
        A_alpha = np.append( np.asarray(A_alpha), np.zeros( (4, num_people) ), axis=0 )
        b = np.append( np.asarray(b), -control_bound_polytope.b.reshape(-1,1), axis=0 )
        
        A2_base_u.value = A
        A2_base_alpha.value = A_alpha
        b2_base.value = b
        controller2_base.solve()
        if controller2_base.status == 'infeasible':
            print(f"QP base infeasible")
        
        A, b = construct_barrier_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), alpha1=jnp.asarray(alpha_qp_base.value) )
        A = np.append( np.asarray(A), -control_bound_polytope.A, axis=0 )
        b = np.append( np.asarray(b), -control_bound_polytope.b.reshape(-1,1), axis=0 )
        
        ax1[1].clear()
        ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
        ax1[1].set_ylim([-control_bound-offset, control_bound+offset])
        hull = pt.Polytope( -A, -b )
        hull_plot = hull.plot(ax1[1], color = 'g')
        plot_polytope_lines( ax1[1], hull, control_bound )

        volume.append(pt.volume( hull, nsamples=50000 ))
        volume2.append(np.array(mc_polytope_volume( jnp.array(hull.A), jnp.array(hull.b.reshape(-1,1)), bounds = control_bound)))
        ax1[2].plot( volume, 'r' )
        ax1[2].plot( volume2, 'g' )
        ax1[2].set_title('Polytope Volume')
    
        # print(f"GRAD : { mc_polytope_volume_grad( jnp.array(hull.A), jnp.array(hull.b.reshape(-1,1)), bounds = control_bound, num_samples=50000 ) } ")

        
        if use_ellipse:
            ellipse_B2, ellipse_d2, volume_new = compute_ellipse_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1=jnp.asarray(alpha_qp_base.value) )
            if plot_ellipse:
                angles   = np.linspace( 0, 2 * np.pi, 100 )
                ellipse_inner  = (ellipse_B2 @ np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )) + ellipse_d2# * np.ones( 1, noangles );
                ellipse_outer  = (2* ellipse_B2 @ np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )) + ellipse_d2
                volume_ellipse2.append(volume_new)
                ax1[2].plot( volume_ellipse2, 'g--' )
                ax1[1].plot( ellipse_inner[0,:], ellipse_inner[1,:], 'c--', label='Jax Inner Ellipse' )
                ax1[1].plot( ellipse_outer[0,:], ellipse_outer[1,:], 'c--', label='Jax Outer Ellipse' )

            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU = polytope_ellipse_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1=jnp.asarray(alpha_qp_base.value)  )
            # t0 = time.time()
            # volume_grad_robot2 = polytope_ellipse_volume_from_states_grad2(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)))
            # print(f"time 2 : {time.time() - t0}")
            # print(f" polytope_volume_from_states_grad: {volume_grad_robot} ")
            
            h_polytope = volume_new - min_polytope_volume_ellipse
            dh_polytope_dx = volume_grad_robot.T
            A_polytope = np.asarray(dh_polytope_dx @ robot.g())
            # b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - alpha_polytope_user * h_polytope - np.sum(volume_grad_humansX * humans.controls ))
            b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - np.sum(volume_grad_humansX * humans.controls ))
            A2.value = np.append( A, np.asarray(A_polytope), axis=0 )
            A2_polytope.value[-1,0] = h_polytope
            b2.value = np.append( b, np.asarray(b_polytope), axis=0 )
        elif use_circle:
            circle_r2, circle_c2, volume_new = compute_circle_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1=jnp.asarray(alpha_qp_base.value) )
            if plot_circle:
                angles   = np.linspace( 0, 2 * np.pi, 100 )
                circle_inner = circle_c2 + circle_r2 * np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )
                volume_circle2.append(volume_new)
                ax1[2].plot( volume_circle2, 'g--' )
                ax1[1].plot( circle_inner[0,:], circle_inner[1,:], 'c--', label='Inner Circle' )
            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU = polytope_circle_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1=jnp.asarray(alpha_qp_base.value)  )
            # print(f" polytope_volume_from_states_grad: {volume_grad_robot} ")
            
            h_polytope = volume_new - min_polytope_volume_circle
            dh_polytope_dx = volume_grad_robot.T
            A_polytope = np.asarray(dh_polytope_dx @ robot.g())
            # b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - alpha_polytope_user * h_polytope - np.sum(volume_grad_humansX * humans.controls ))
            b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - np.sum(volume_grad_humansX * humans.controls ))
            A2.value = np.append( A, np.asarray(A_polytope), axis=0 )
            A2_polytope.value[-1,0] = h_polytope
            b2.value = np.append( b, np.asarray(b_polytope), axis=0 )
     
        controller2.solve()
        if controller2.status == 'infeasible':
            print(f"QP infeasible, trying to relax polytope volume constraint ")
            
        print(f"alpha: {alpha_qp_base.value.T}, alpha_polytope: {alpha_polytope.value}")
        #     exit()
        # robot.step( u2_base.value )
        # ax1[1].scatter( u2_base.value[0,0], u2_base.value[1,0], c = 'r', label = 'CBF-QP chosen control' )
        
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
        
        # writer.grab_frame()

    # def polytope_barrier(  )




