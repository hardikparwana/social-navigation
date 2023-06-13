import numpy as np
import cvxpy as cp
import polytope as pt
import matplotlib.pyplot as plt

from bicycle import bicycle
from single_integrator import single_integrator_square
from obstacles import circle

######### holonomic controller
n = 5 # number of constraints
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
fig1, ax1 = plt.subplots( 1, 2, figsize=(12, 6))#, gridspec_kw={'height_ratios': [1, 1]} )
ax1[0].set_xlim([-1,5])
ax1[0].set_ylim([-1,5])
# ax[0].axis
control_bound = 2.0
ax1[1].set_xlim([-control_bound-1, control_bound+1])
ax1[1].set_ylim([-control_bound-1, control_bound+1])
# ax1.set_xlabel("X")
# ax1.set_ylabel("Y")

# fig1  =plt.figure()
# ax1 = plt.axes( xlim = (-1,5), ylim = (-1,5) )
# ax1.set_xlabel("X")
# ax1.set_ylabel("Y")

# fig2 = plt.figure()
# ax2 = plt.axes( xlim=(-3,3), ylim=(-3,3) )
# ax2.set_xlabel("X")
# ax2.set_ylabel("Y")

t = 0
dt = 0.05
tf = 10
goal = np.array([3,4]).reshape(-1,1)
ax1[0].scatter( goal[0], goal[1], edgecolors ='g', facecolors='none' )
alpha = 1.0
obstacles = []
obstacles.append( circle( ax1[0], pos = np.array([2.0,2.0]), radius = 0.5 ) )  

robot = single_integrator_square( ax1[0], pos = np.array([ 0, 0 ]), dt = dt, plot_polytope=False )
control_bound = 2.0
control_input_limit_points = np.array([ [control_bound, control_bound], [-control_bound, control_bound], [-control_bound, -control_bound], [control_bound, -control_bound] ])
control_bound_polytope = pt.qhull( control_input_limit_points )

while t < tf:

    # desired input
    u2_ref.value = robot.nominal_controller( goal )

    # barrier function
    h, dh_dx, _ = robot.barrier( obstacles[0], d_min = 0.5 )
    A = dh_dx 
    b = - alpha * h

    A2.value = np.append( A, -control_bound_polytope.A, axis=0 )
    b2.value = np.append( b, -control_bound_polytope.b.reshape(-1,1), axis=0 )
    controller2.solve()

    if controller2.status == 'infeasible':
        print(f"QP infeasible")
        exit()

    robot.step( u2.value )
    robot.render_plot()

    ax1[1].clear()
    ax1[1].set_xlim( [-control_bound-1, control_bound+1] )
    ax1[1].set_ylim( [-control_bound+1, control_bound+1] )
    hull_plot = pt.Polytope( -A2.value, -b2.value ).plot(ax1[1], color = 'g')

    fig1.canvas.draw()
    fig1.canvas.flush_events()

    t = t + dt

