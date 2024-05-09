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
name = 'Videos/case1_ellipse_barrier.mp4'
t = 0
dt = 0.03
tf = 9#15
alpha = 6#5#2.0#3.0#20#6
alpha1 = 2#20#4.0#0.5#50#2
control_bound = 2.0
goal = np.array([-3.0,-1.0]).reshape(-1,1)
num_people = 5
num_obstacles = 4
k_v = 1.5 #1.0
use_ellipse = False#False
plot_ellipse = True#False
use_circle = False#True
plot_circle = True#False#
use_smooth = True
alpha_polytope = 0.8#1.0
alpha_polytope_smooth = 0.01 #0.1#1.0
min_polytope_volume_ellipse = -0.5
min_polytope_volume_circle = 0.0

######### holonomic controller
n = 4 + num_obstacles + num_people + 1 # number of constraints
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
# ax1[0].set_xlim([-3,5])
# ax1[0].set_xlim([-3,5])
ax1[0].set_xlim([-3.2,2.5])
ax1[0].set_ylim([-2,3.2])
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
# exit()
volume = []
volume2 = []
volume_ellipse = []
volume_ellipse2 = []
volume_circle2 = []

pos_x = [robot.X[0,0]]
pos_y = [robot.X[1,0]]
path, = ax1[0].plot(pos_x, pos_y, 'g')

@jit
def construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot ):         
            # barrier function
            A = jnp.zeros((1,2)); b = jnp.zeros((1,1))
            # for i in range(obstacle_states.shape[1]):
            for i in range(len(obstacles)):
                h, dh_dx, _ = robot.barrier_jax( robot_state, obstacle_states[:,i].reshape(-1,1), d_min = 0.5, alpha1 = alpha1 )
                A = jnp.append( A, dh_dx @ robot.g_jax(robot_state), axis = 0 )
                b = jnp.append( b, - alpha * h - dh_dx @ robot.f_jax(robot_state), axis = 0 )
            # for i in range(humans_states.shape[1]):
            for i in range(humans.X.shape[1]):
                h, dh_dx, _ = robot.barrier_humans_jax( robot_state, humans_states[:,i].reshape(-1,1), human_states_dot[:,i].reshape(-1,1), d_min = 0.5, alpha1 = alpha1 )
                A = jnp.append( A, dh_dx @ robot.g_jax(robot_state), axis = 0 )
                b = jnp.append( b, - alpha * h - dh_dx @ robot.f_jax(robot_state), axis = 0 )
            return A[1:], b[1:]
construct_barrier_from_states_jit = jit(construct_barrier_from_states)

@jit       
def ellipse_volume( B, d ):
     return jnp.log( jnp.linalg.det(B) )

@jit
def circle_volume( r, c ):
    return jnp.pi * jnp.square(r)

def compute_ellipse_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot )
    A2 = jnp.append( -A, control_A, axis=0 )
    b2 = jnp.append( -b, control_b, axis=0 )
    solution = ellipse_cvxpylayer( A2, b2 )
    B = solution[0]
    d = solution[1]
    return B, d, ellipse_volume( B, d )

def compute_circle_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot )
    A2 = jnp.append( -A, control_A, axis=0 )
    A2_root = jnp.linalg.norm( A2, axis=1 )
    b2 = jnp.append( -b, control_b, axis=0 )
    solution = circle_cvxpylayer( A2, A2_root, b2 )
    r = solution[0]
    c = solution[1]
    return r, c, circle_volume( r, c )

@jit
def compute_smooth_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, lb, ub):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot )
    A2 = jnp.append( -A, control_A, axis=0 )
    b2 = jnp.append( -b, control_b, axis=0 )
    vol = mc_polytope_volume( A2, b2, lb=lb[:,0], ub=ub[:,0] ) #bounds=control_bound,
    return vol

@jit
def compute_smooth_volume_for_grad_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, samples, total_volume):
    A, b = construct_barrier_from_states_jit(robot_state, obstacle_states, humans_states, human_states_dot )
    A2 = jnp.append( -A, control_A, axis=0 )
    b2 = jnp.append( -b, control_b, axis=0 )
    vol = mc_polytope_volume_about_lines( A2, b2, samples, total_volume )
    return vol

def compute_bounds_from_polytope(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot )
    A2 = jnp.append( -A, control_A, axis=0 )
    b2 = jnp.append( -b, control_b, axis=0 )
    hull = pt.Polytope( np.asarray(A2.primal), np.asarray(b2.primal) )
    lb, ub = pt.bounding_box(hull)
    return lb, ub

def polytope_ellipse_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b):
    return compute_ellipse_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b)[2]

def polytope_circle_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b):
    return compute_circle_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b)[2]
 
polytope_ellipse_volume_from_states_grad = grad( polytope_ellipse_volume_from_states, argnums=(0,1,2,3) )
# polytope_ellipse_volume_from_states_grad2 = grad( polytope_ellipse_volume_from_states, argnums=(0) )

polytope_circle_volume_from_states_grad = grad( polytope_circle_volume_from_states, argnums=(0,1,2,3) )

polytope_smooth_volume_from_states_grad = jit(grad( compute_smooth_volume_from_states, argnums=(0,1,2,3) ))
compute_smooth_volume_for_grad_from_states_grad = jit(grad(compute_smooth_volume_for_grad_from_states, argnums=(0,1,2,3)))
    
# if 1:
with writer.saving(fig1, name, 100): 
    while t < tf:

        robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.X[3,0]*np.cos(robot.X[2,0]), robot.X[3,0]*np.sin(robot.X[2,0]) , goal[0,0], goal[1,0]])
        humans_socialforce.state[-1,0:6] = robot_social_state
        humans.controls = humans_socialforce.step().state.copy()[:-1,2:4].copy().T
        humans.step_using_controls(dt)

        # desired input
        u2_ref.value = robot.nominal_controller( goal, k_v = k_v )
    
        A, b = construct_barrier_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls) )
        A = np.append( np.asarray(A), -control_bound_polytope.A, axis=0 )
        b = np.append( np.asarray(b), -control_bound_polytope.b.reshape(-1,1), axis=0 )
        # lb, ub = compute_bounds_from_polytope(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)))
        ax1[1].clear()
        ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
        ax1[1].set_ylim([-control_bound-offset, control_bound+offset])
        hull = pt.Polytope( -A, -b )
        # hull = pt.Polytope( np.asarray(A2.primal), np.asarray(b2.primal) )
        lb, ub = pt.bounding_box(hull)
        lb = lb - 0.5
        ub = ub + 0.5
        hull_plot = hull.plot(ax1[1], color = 'g')
        plot_polytope_lines( ax1[1], hull, control_bound )

        volume.append(pt.volume( hull ))#, nsamples=50000 ))
        volume2.append(np.array(mc_polytope_volume( jnp.array(hull.A), jnp.array(hull.b.reshape(-1,1)), lb=lb, ub=ub ))) # bounds = control_bound)))
        # ax1[2].plot( volume, 'r' )
        ax1[2].plot( volume2, 'g' )
        ax1[2].set_title('Polytope Volume')
        # print(f"GRAD : { mc_polytope_volume_grad( jnp.array(hull.A), jnp.array(hull.b.reshape(-1,1)), bounds = control_bound, num_samples=50000 ) } ")

        A2.value = np.append( A, np.zeros((1,2)), axis=0 )
        b2.value = np.append( b, np.zeros((1,1)), axis=0 )
        
        if use_ellipse:
            ellipse_B2, ellipse_d2, volume_new = compute_ellipse_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)) )
            if plot_ellipse:
                angles   = np.linspace( 0, 2 * np.pi, 100 )
                ellipse_inner  = (ellipse_B2 @ np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )) + ellipse_d2# * np.ones( 1, noangles );
                ellipse_outer  = (2* ellipse_B2 @ np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )) + ellipse_d2
                # volume_ellipse2.append(volume_new)
                volume_ellipse2.append(  np.pi * np.exp(volume_new)  )
                ax1[2].plot( volume_ellipse2, 'g--' )
                ax1[1].plot( ellipse_inner[0,:], ellipse_inner[1,:], 'c--', label='Inscribing Ellipse' )
                # ax1[1].plot( ellipse_outer[0,:], ellipse_outer[1,:], 'c--', label='Jax Outer Ellipse' )

            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU = polytope_ellipse_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)))
            print(f"hello vol:{volume_new}, grad: {volume_grad_robot.T}, grad_obs: {volume_grad_obstacles.T}, grad_humans: {volume_grad_humansX.T}")
            # t0 = time.time()
            # volume_grad_robot2 = polytope_ellipse_volume_from_states_grad2(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)))
            # print(f"time 2 : {time.time() - t0}")
            # print(f" polytope_volume_from_states_grad: {volume_grad_robot} ")
            
            h_polytope = volume_new - min_polytope_volume_ellipse
            dh_polytope_dx = volume_grad_robot.T
            A_polytope = np.asarray(dh_polytope_dx @ robot.g())
            b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - alpha_polytope * h_polytope - np.sum(volume_grad_humansX * humans.controls ))
            A2.value = np.append( A, np.asarray(A_polytope), axis=0 )
            b2.value = np.append( b, np.asarray(b_polytope), axis=0 )

            # volume_new_smooth = compute_smooth_volume_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)) )
            # volume_grad_robot_smooth, volume_grad_obstacles_smooth, volume_grad_humansX_smooth, volume_grad_humansU_smooth = polytope_smooth_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)) )
            # print(f"smooth vol:{volume_new_smooth}, grad: {volume_grad_robot_smooth.T}, grad_obs: {volume_grad_obstacles_smooth.T}, grad_humans: {volume_grad_humansX_smooth.T}")

        elif use_circle:
            circle_r2, circle_c2, volume_new = compute_circle_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)) )
            if plot_circle:
                angles   = np.linspace( 0, 2 * np.pi, 100 )
                circle_inner = circle_c2 + circle_r2 * np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )
                volume_circle2.append(volume_new)
                ax1[2].plot( volume_circle2, 'g--' )
                ax1[1].plot( circle_inner[0,:], circle_inner[1,:], 'c--', label='Inner Circle' )
            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU = polytope_circle_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)))
            print(f" polytope_volume_from_states_grad: {volume_grad_robot} ")
            
            h_polytope = volume_new - min_polytope_volume_circle
            dh_polytope_dx = volume_grad_robot.T
            A_polytope = np.asarray(dh_polytope_dx @ robot.g())
            b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - alpha_polytope * h_polytope - np.sum(volume_grad_humansX * humans.controls ))
            A2.value = np.append( A, np.asarray(A_polytope), axis=0 )
            b2.value = np.append( b, np.asarray(b_polytope), axis=0 )

        elif use_smooth:
            # print(f"smooth")

            sampled_pts = 0
            init = 0 
            total_volume = 0
            for i in range(hull.A.shape[0]):
                pts = get_intersection_points( hull.A[i,:], hull.b[i], lb , ub )
                if len(pts)>0:
                    plt.plot([ pts[0][0], pts[1][0] ], [ pts[0][1], pts[1][1] ]  )
                else:
                    continue
                new_pts, temp_volume = generate_points_about_line( pts ) #, num_line_points, num_normal_points, increment )
                total_volume = total_volume + temp_volume    
                if init==0:
                    sampled_pts = new_pts
                    init = 1
                else:
                    sampled_pts = jnp.append( sampled_pts, new_pts, axis=1 )
                ax1[1].scatter(new_pts[0,::3], new_pts[1,::3], s=3, alpha=0.1)
            samples = sampled_pts 

            volume_new = compute_smooth_volume_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), lb, ub)

            volume_circle2.append(volume_new)
            ax1[2].plot( volume_circle2, 'g--' )
            
            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU = compute_smooth_volume_for_grad_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), samples, total_volume)
            # volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU = polytope_smooth_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), lb, ub )

            # circle_r2, circle_c2, volume_new_circle = compute_circle_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)) )
            volume_grad_robot_circle, volume_grad_obstacles_circle, volume_grad_humansX_circle, volume_grad_humansU_circle = polytope_circle_volume_from_states_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)))
            
            # print(f"hello vol:{volume_new}, vol ellipse: {volume_new_circle} grad: {volume_grad_robot.T}, grad_ell: {volume_grad_robot_circle.T}")
            print(f"hello vol:{volume_new}, grad: {volume_grad_robot.T}, grad_obs: {volume_grad_obstacles.T}, grad_humans: {volume_grad_humansX.T} *** circle: robot:{volume_grad_robot_circle.T}, obs: {volume_grad_obstacles_circle.T}, humans: {volume_grad_humansX_circle.T} ")


            h_polytope = volume_new
            dh_polytope_dx = volume_grad_robot.T
            A_polytope = np.asarray(dh_polytope_dx @ robot.g())
            b_polytope = np.asarray(- dh_polytope_dx @ robot.f() - alpha_polytope_smooth * h_polytope - np.sum(volume_grad_humansX * humans.controls ))
            A2.value = np.append( A, np.asarray(A_polytope), axis=0 )
            b2.value = np.append( b, np.asarray(b_polytope), axis=0 )
    
        controller2.solve()
        if controller2.status == 'infeasible':
            print(f"QP infeasible")
            break
            # exit()
        robot.step( u2.value )
        robot.render_plot()
        humans.render_plot(humans.X)
        
        pos_x.append(robot.X[0,0])
        pos_y.append(robot.X[1,0])
        path.set_xdata(pos_x)
        path.set_ydata(pos_y)
        
        ax1[1].set_xlabel('Linear Acceleration'); ax1[1].set_ylabel('Angular Velocity')
        # ax1[1].set_xlabel(r'$u_x$'); ax1[1].set_ylabel(r'$u_y$')
        ax1[1].scatter( u2.value[0,0], u2.value[1,0], c = 'r', label = 'CBF-QP chosen control' )
        ax1[1].scatter( u2_ref.value[0,0], u2_ref.value[1,0], edgecolors='r', facecolors='none', label = 'Nominal Input' )
        # print(f"u: {u2.value.T}, ref:{u2_ref.value.T}")
        ax1[1].legend(loc='upper right')
        ax1[1].set_title('Feasible Space for Control')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        t = t + dt
        
        writer.grab_frame()


plt.ioff()
fig1.savefig(name+'.png')
fig1.savefig(name+'.eps')

    # def polytope_barrier(  )




