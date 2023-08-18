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
# alpha2 =6.0#5#2.0#3.0#20#6
# alpha1 = 2.0#20#4.0#0.5#50#2
num_people = 5
num_obstacles = 4
alpha1_obstacle = 2*np.ones(num_obstacles)
alpha2_obstacle = 6*np.ones(num_obstacles)
alpha1_human = 2*np.ones(num_people)
alpha2_human = 6*np.ones(num_people)
control_bound = 2.0
# goal = np.array([-3.0,-1.0]).reshape(-1,1)
goal = np.array([-3.0,1.0]).reshape(-1,1)
k_x = 1.0#0.5#30.0
k_v = 1.5
d_min_human = 0.2#0.5
d_min_obstacle = 0.5
use_ellipse = False#True#False
plot_ellipse = False#True#False
use_circle = True#False#True
plot_circle = True#False#True
alpha_polytope_user = 0.5#1.0
min_polytope_volume_ellipse = -2.7#-0.5
min_polytope_volume_circle = 0.0

######### holonomic controller
n = 4 + num_obstacles + num_people + 1 # number of constraints

##########
# cvxpylayer base controller
n_base = 4 + num_obstacles + num_people
u2_base = cp.Variable((2,1))
u2_ref_base = cp.Parameter((2,1))
A2_base = cp.Parameter((n_base,2))
b2_base = cp.Parameter((n_base,1))
const2_base = [A2_base @ u2_base >= b2_base]
objective2_base = cp.Minimize( cp.norm(u2_base-u2_ref_base) )
controller2_base = cp.Problem( objective2_base, const2_base )
controller2_base_layer = CvxpyLayer(controller2_base, parameters=[u2_ref_base, A2_base, b2_base], variables=[u2_base])
##########

p1_human = cp.Variable((num_people,1))
p2_human = cp.Variable((num_people,1))
p1_obstacle = cp.Variable((num_obstacles,1))
p2_obstacle = cp.Variable((num_obstacles,1))
A1_human = cp.Parameter((1,num_people), value=np.zeros((1,num_people)))
A2_human = cp.Parameter((1,num_people), value=np.zeros((1,num_people)))
A1_obstacle = cp.Parameter((1,num_obstacles), value=np.zeros((1,num_obstacles)))
A2_obstacle = cp.Parameter((1,num_obstacles), value=np.zeros((1,num_obstacles)))
b1 = cp.Parameter((1,1), value=np.array([[0.0]]))
obj = cp.Minimize( cp.sum_squares(p1_human-1) + cp.sum_squares(p2_human-1) + cp.sum_squares(p1_obstacle-1) + cp.sum_squares(p2_obstacle-1) )
const = [ A1_human @ p1_human + A2_human @ p2_human + A1_obstacle @ p1_obstacle + A2_obstacle @ p2_obstacle >= b1 ]
alpha_dot_controller = cp.Problem( obj, const )


##################

plt.ion()
fig1, ax1 = plt.subplots( 1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [5, 5, 2]})# )#, gridspec_kw={'height_ratios': [1, 1]} )
ax1[0].set_xlim([-3,5])
ax1[0].set_ylim([-3,5])
offset = 3.0
ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
ax1[1].set_ylim([-control_bound-offset, control_bound+offset])

# plot alphas live
fig2, ax2 = plt.subplots(2, 2)
ax2[0,0].set_title('Human')
ax2[0,1].set_title('Obstacle')
ax2[0,0].set_ylabel(r'$\alpha_1$')
ax2[0,1].set_ylabel(r'$\alpha_1$')
ax2[1,0].set_ylabel(r'$\alpha_2$')
ax2[1,1].set_ylabel(r'$\alpha_2$')

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

# exit()
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
def construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle ):         
            # barrier function
            A = jnp.zeros((1,2)); b = jnp.zeros((1,1))
            for i in range(len(obstacles)):
                dh_dot_dx1, dh_dot_dx2, h_dot, h = robot.barrier_alpha_jax( robot_state, obstacle_states[:,i].reshape(-1,1), d_min = d_min_obstacle)
                A = jnp.append( A, dh_dot_dx1 @ robot.g_jax(robot_state), axis = 0 )
                b = jnp.append( b, - dh_dot_dx1 @ robot.f_jax(robot_state) - alpha1_obstacle[i] * h_dot - alpha2_obstacle[i] * (h_dot + alpha1_obstacle[i]*h), axis = 0 )
            for i in range(humans.X.shape[1]):
                dh_dot_dx1, dh_dot_dx2, h_dot, h = robot.barrier_humans_alpha_jax( robot_state, humans_states[:,i].reshape(-1,1), human_states_dot[:,i].reshape(-1,1), d_min = d_min_human)
                A = jnp.append( A, dh_dot_dx1 @ robot.g_jax(robot_state), axis = 0 )
                b = jnp.append( b, - dh_dot_dx1 @ robot.f_jax(robot_state) - dh_dot_dx2 @ humans.controls[:,i].reshape(-1,1) - alpha1_human[i] * h_dot - alpha2_human[i] * (h_dot + alpha1_human[i]*h), axis = 0 )
            return A[1:], b[1:]

# @jit
def get_next_step_circle_volume( robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle ):
     A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle)
     u_ref = robot.nominal_controller_jax( robot_state, goal, k_x = k_x, k_v = k_v )
     A = jnp.append( A, -control_A, axis=0 )
     b = jnp.append( b, -control_b, axis=0 )
     solution = controller2_base_layer(u_ref, A, b)
     robot_next_state = robot_state + robot.xdot_jax(robot_state, solution[0]) * dt
     humans_next_state = humans_states + human_states_dot * dt
     return compute_circle_from_states( robot_next_state, obstacle_states, humans_next_state, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )[2]
get_next_step_circle_volume_grad = grad( get_next_step_circle_volume, argnums=(0,1,2,3,6,7,8,9) )

def get_next_step_ellipse_volume( robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle ):
     A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle)
     A = jnp.append( A, -control_A, axis=0 )
     b = jnp.append( b, -control_b, axis=0 )
     u_ref = robot.nominal_controller_jax( robot_state, goal, k_x = k_x, k_v = k_v )
     solution = controller2_base_layer(u_ref, A, b)
     robot_next_state = robot_state + robot.xdot_jax(robot_state, solution[0]) * dt
     humans_next_state = humans_states + human_states_dot * dt
     return compute_ellipse_from_states( robot_next_state, obstacle_states, humans_next_state, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )[2]
get_next_step_ellipse_volume_grad = grad( get_next_step_ellipse_volume, argnums=(0,1,2,3,6,7,8,9) )

@jit       
def ellipse_volume( B, d ):
     return jnp.log( jnp.linalg.det(B) )

@jit
def circle_volume( r, c ):
    return jnp.pi * jnp.square(r)

def compute_ellipse_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
    A2 = jnp.append( -A, control_A, axis=0 )
    b2 = jnp.append( -b, control_b, axis=0 )
    solution = ellipse_cvxpylayer( A2, b2 )
    B = solution[0]
    d = solution[1]
    return B, d, ellipse_volume( B, d )

def compute_circle_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle):
    A, b = construct_barrier_from_states(robot_state, obstacle_states, humans_states, human_states_dot, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
    A2 = jnp.append( -A, control_A, axis=0 )
    A2_root = jnp.linalg.norm( A2, axis=1 )
    b2 = jnp.append( -b, control_b, axis=0 )
    solution = circle_cvxpylayer( A2, A2_root, b2 )
    r = solution[0]
    c = solution[1]
    return r, c, circle_volume( r, c )

def polytope_ellipse_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human, alpha2_human, alpha1_obstacle, alpha2_obstacle):
    return compute_ellipse_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle)[2]

def polytope_circle_volume_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human, alpha2_human, alpha1_obstacle, alpha2_obstacle):
    return compute_circle_from_states(robot_state, obstacle_states, humans_states, human_states_dot, control_A, control_b, alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle)[2]
 
polytope_ellipse_volume_from_states_grad = grad( polytope_ellipse_volume_from_states, argnums=(0,1,2,3,6,7,8,9) )
polytope_circle_volume_from_states_grad = grad( polytope_circle_volume_from_states, argnums=(0,1,2,3,6,7,8,9) )
    
# if 1:
with writer.saving(fig1, 'Videos/DU_fs_alpha_dot_adapt.mp4', 100): 
    while t < tf:

        robot_social_state = np.array([ robot.X[0,0], robot.X[1,0], robot.U[0,0], robot.U[1,0] , goal[0,0], goal[1,0]]).reshape(1,-1)
        humans_socialforce.state[-1,0:6] = robot_social_state
        humans.controls = humans_socialforce.step().state.copy()[:-1,2:4].copy().T
        humans.step_using_controls(dt)
        
        if use_ellipse:
            ellipse_B2, ellipse_d2, volume_new = compute_ellipse_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
            if plot_ellipse:
                angles   = np.linspace( 0, 2 * np.pi, 100 )
                ellipse_inner  = (ellipse_B2 @ np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )) + ellipse_d2# * np.ones( 1, noangles );
                ellipse_outer  = (2* ellipse_B2 @ np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )) + ellipse_d2
                volume_ellipse2.append(volume_new)
                ax1[2].plot( volume_ellipse2, 'g--' )
                ax1[1].plot( ellipse_inner[0,:], ellipse_inner[1,:], 'c--', label='Jax Inner Ellipse' )
                ax1[1].plot( ellipse_outer[0,:], ellipse_outer[1,:], 'c--', label='Jax Outer Ellipse' )

            next_step_volume = get_next_step_ellipse_volume(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU, alpha1_human_grad, alpha2_human_grad, alpha1_obstacle_grad, alpha2_obstacle_grad = get_next_step_ellipse_volume_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1_human, alpha2_human, alpha1_obstacle, alpha2_obstacle  )
            

            h_polytope = volume_new - min_polytope_volume_ellipse
        elif use_circle:            
            circle_r2, circle_c2, volume_new = compute_circle_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
            if plot_circle:
                angles   = np.linspace( 0, 2 * np.pi, 100 )
                circle_inner = circle_c2 + circle_r2 * np.append(np.cos(angles).reshape(1,-1) , np.sin(angles).reshape(1,-1), axis=0 )
                volume_circle2.append(volume_new)
                ax1[2].plot( volume_circle2, 'g--' )
                # ax1[1].plot( circle_inner[0,:], circle_inner[1,:], 'c--', label='Inner Circle' )
            next_step_volume = get_next_step_circle_volume(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
            volume_grad_robot, volume_grad_obstacles, volume_grad_humansX, volume_grad_humansU, alpha1_human_grad, alpha2_human_grad, alpha1_obstacle_grad, alpha2_obstacle_grad = get_next_step_circle_volume_grad(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), jnp.asarray(control_bound_polytope.A), jnp.asarray(control_bound_polytope.b.reshape(-1,1)), alpha1_human, alpha2_human, alpha1_obstacle, alpha2_obstacle  )
            
            h_polytope = volume_new - min_polytope_volume_circle

        A1_human.value[0,:] = alpha1_human_grad
        A2_human.value[0,:] = alpha2_human_grad
        A1_obstacle.value[0,:] = alpha1_obstacle_grad
        A2_obstacle.value[0,:] = alpha2_obstacle_grad
        b1.value[0] = - alpha_polytope_user * h_polytope
        alpha_dot_controller.solve()
        if alpha_dot_controller.status == 'infeasible':
            print(f"Alpha dot QP infeasible")
        alpha1_human += p1_human.value[:,0] * dt
        alpha2_human += p2_human.value[:,0] * dt
        alpha1_obstacle += p1_obstacle.value[:,0] * dt
        alpha2_obstacle += p2_obstacle.value[:,0] * dt
        
        
        ax2[0,0].scatter(t*np.ones(num_people),alpha1_human, c=np.arange(num_people))
        ax2[0,1].scatter(t*np.ones(num_obstacles),alpha1_obstacle, c=np.arange(num_obstacles))
        ax2[1,0].scatter(t*np.ones(num_people),alpha2_human, c=np.arange(num_people))
        ax2[1,1].scatter(t*np.ones(num_obstacles),alpha2_obstacle, c=np.arange(num_obstacles))

        A, b = construct_barrier_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls), alpha1_human = alpha1_human, alpha2_human=alpha2_human, alpha1_obstacle = alpha1_obstacle, alpha2_obstacle=alpha2_obstacle )
        A2_base.value = np.append( np.asarray(A), -control_bound_polytope.A, axis=0 )
        b2_base.value = np.append( np.asarray(b), -control_bound_polytope.b.reshape(-1,1), axis=0 )
        u2_ref_base.value = robot.nominal_controller( goal, k_x = k_x, k_v = k_v )
        controller2_base.solve()
        # print(f"human alpha1: {p1_human.value.T}, alpha2: {p2_human.value.T}, obs: alpha1:{p1_obstacle.value.T}, alpha2:{p2_obstacle.value.T}")
        print(f"human alpha1: {alpha1_human}, alpha2: {alpha2_human}, obs: alpha1:{alpha1_obstacle}, alpha2:{alpha2_obstacle}")

        # Plot polytope       
        ax1[1].clear()
        ax1[1].set_xlim([-control_bound-offset, control_bound+offset])
        ax1[1].set_ylim([-control_bound-offset, control_bound+offset])
        hull = pt.Polytope( -A2_base.value, -b2_base.value )
        hull_plot = hull.plot(ax1[1], color = 'g')
        ax1[1].plot( circle_inner[0,:], circle_inner[1,:], 'c--', label='Inner Circle' )
        plot_polytope_lines( ax1[1], hull, control_bound )

        volume.append(pt.volume( hull, nsamples=50000 ))
        ax1[2].plot( volume, 'r' )
        ax1[2].set_title('Polytope Volume')


        # robot.step(u2_ref_base.value)
        robot.step( u2_base.value )
        
        ax1[1].scatter( u2_base.value[0,0], u2_base.value[1,0], c = 'r', label = 'CBF-QP chosen control' )
        robot.render_plot()
        humans.render_plot(humans.X)

        ax1[1].set_xlabel('Linear Acceleration'); ax1[1].set_ylabel('Angular Velocity')
        ax1[1].legend()
        ax1[1].set_title('Feasible Space for Control')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        t = t + dt
        
        writer.grab_frame()




