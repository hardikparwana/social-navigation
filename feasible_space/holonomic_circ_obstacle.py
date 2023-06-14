import numpy as np
import cvxpy as cp
import polytope as pt
import matplotlib.pyplot as plt

from bicycle_new import bicycle
from single_integrator import single_integrator_square
from polytope_utils import plot_polytope_lines
from obstacles import circle
from matplotlib.animation import FFMpegWriter

######### holonomic controller
n = 7 # number of constraints
u2 = cp.Variable((2,1))
u2_ref = cp.Parameter((2,1))
objective2 = cp.Minimize( cp.sum_squares( u2 - u2_ref ) )
A2 = cp.Parameter((n,2))
b2 = cp.Parameter((n,1))
const2 = [A2 @ u2 >= b2]
controller2 = cp.Problem( objective2, const2 )
##########

plt.ion()

# fig1  =plt.figure()
xlim = (-1,5); ylim = (-1,5)
fig1, ax1 = plt.subplots( 1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': [5, 5, 2]})# )#, gridspec_kw={'height_ratios': [1, 1]} )
ax1[0].set_xlim([-1,5])
ax1[0].set_ylim([-1,5])

control_bound = 3.0
ax1[1].set_xlim([-control_bound-1, control_bound+1])
ax1[1].set_ylim([-control_bound-1, control_bound+1])

t = 0
dt = 0.03
tf = 15
goal = np.array([3,4]).reshape(-1,1)
ax1[0].scatter( goal[0], goal[1], edgecolors ='g', facecolors='none' )
alpha = 1.0#3.0
obstacles = []
obstacles.append( circle( ax1[0], pos = np.array([2.0,2.0]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([1.0,3.0]), radius = 0.5 ) )  
obstacles.append( circle( ax1[0], pos = np.array([2.7,3.0]), radius = 0.5 ) )  

# robot = single_integrator_square( ax1[0], pos = np.array([ 0, 0 ]), dt = dt, plot_polytope=False )
robot = bicycle( ax1[0], pos = np.array([ 0, 0, np.pi/3, 2.0 ]), dt = dt, plot_polytope=False )
control_bound = 2.0
control_input_limit_points = np.array([ [control_bound, control_bound], [-control_bound, control_bound], [-control_bound, -control_bound], [control_bound, -control_bound] ])
control_bound_polytope = pt.qhull( control_input_limit_points )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

# if 1:
volume = []
with writer.saving(fig1, 'Videos/DU_limit_init_feasible_space.mp4', 100): 
    while t < tf:

        # desired input
        u2_ref.value = robot.nominal_controller( goal )

        # barrier function
        A = np.zeros((1,2)); b = np.zeros((1,1))
        for i in range(len(obstacles)):
            h, dh_dx, _ = robot.barrier( obstacles[i], d_min = 0.5, alpha1 = 0.5 )
            A = np.append( A, dh_dx @ robot.g(), axis = 0 )
            b = np.append( b, - alpha * h - dh_dx @ robot.f(), axis = 0 )

        A2.value = np.append( A[1:], -control_bound_polytope.A, axis=0 )
        b2.value = np.append( b[1:], -control_bound_polytope.b.reshape(-1,1), axis=0 )
        controller2.solve()

        if controller2.status == 'infeasible':
            print(f"QP infeasible")
            exit()

        robot.step( u2.value )
        robot.render_plot()

        ax1[1].clear()
        ax1[1].set_xlim( [-control_bound-1, control_bound+1] )
        ax1[1].set_ylim( [-control_bound-1, control_bound+1] )
        hull = pt.Polytope( -A2.value, -b2.value )
        hull_plot = hull.plot(ax1[1], color = 'g')
        volume.append(pt.volume( hull, nsamples=50000 ))
        plot_polytope_lines( ax1[1], hull, control_bound )

        ax1[1].set_xlabel('Linear Acceleration'); ax1[1].set_ylabel('Angular Velocity')
        # ax1[1].set_xlabel(r'$u_x$'); ax1[1].set_ylabel(r'$u_y$')
        ax1[1].scatter( u2.value[0,0], u2.value[1,0], c = 'r', label = 'CBF-QP chosen control' )
        ax1[1].legend()
        ax1[1].set_title('Feasible Space for Control')

        ax1[2].plot( volume, 'r' )
        ax1[2].set_title('Polytope Volume')

        fig1.canvas.draw()
        fig1.canvas.flush_events()

        t = t + dt
        
        writer.grab_frame()