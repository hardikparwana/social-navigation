import sys; sys.version
import sys
sys.path.append('/home/dasc/hardik/social-navigation/refineCBF')

import refine_cbfs
from cbf_opt import ControlAffineDynamics, ControlAffineCBF, ControlAffineASIF
import matplotlib.pyplot as plt
import hj_reachability as hj
import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
import seaborn as sns
from experiment_wrapper import RolloutTrajectory, TimeSeriesExperiment, StateSpaceExperiment
from dubins import nominal_hjr_control

import matplotlib
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': '80',
    'text.usetex': False,   # Toggle to true for official LaTeX output
    'pgf.rcfonts': False,
    'lines.linewidth': 6.,
})
import matplotlib.animation as anim

params = {'axes.labelsize': 28,'axes.titlesize':28, 'font.size': 28, 'legend.fontsize': 28, 
          'xtick.labelsize': 28, 'ytick.labelsize': 28, 'lines.linewidth': 5}
matplotlib.rcParams.update(params)


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

## Setup Problem (dynamics, environment and CBF)
### Parameter values
# Dynamics properties

dubins_vel = 1.0  # m / s 
umin = np.array([-0.5])  # rad / s
umax = np.array([0.5])  # rad / s 
dt = 0.05 # s

### Dynamics
#Dynamics of the Dubins car problem (control affine), with fixed velocity $v$ in the format of `cbf_opt`
#`DubinsJNPDynamics` is required for usage with `hj_reachability` module

class DubinsDynamics(ControlAffineDynamics):
    STATES = ['X', 'Y', 'THETA']
    CONTROLS = ['OMEGA']
    
    def __init__(self, params, test = False, **kwargs):
        params['n_dims'] = 3
        params['control_dims'] = 1
        params["periodic_dims"] = [2]
        self.v = params["v"]
        super().__init__(params, test, **kwargs)

    def open_loop_dynamics(self, state, time=0.0):
        f = np.zeros_like(state)
        f[..., 0] = self.v * np.cos(state[..., 2])
        f[..., 1] = self.v * np.sin(state[..., 2])
        return f

    def control_matrix(self, state, time=0.0):
        B = np.repeat(np.zeros_like(state)[..., None], self.control_dims, axis=-1)
        B[..., 2, 0] = 1
        return B

    def disturbance_jacobian(self, state, time=0.0):
        return np.repeat(np.zeros_like(state)[..., None], 1, axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        J = np.repeat(np.zeros_like(state)[..., None], self.n_dims, axis=-1)
        J[..., 0, 2] = -self.v * np.sin(state[..., 2])
        J[..., 1, 2] = self.v * np.cos(state[..., 2])
        return J
        
class DubinsJNPDynamics(DubinsDynamics):
    def open_loop_dynamics(self, state, time=0.0):
        return jnp.array([self.v * jnp.cos(state[2]), self.v * jnp.sin(state[2]), 0])

    def control_matrix(self, state, time=0.0):
        return jnp.expand_dims(jnp.array([0, 0, 1]), axis=-1)

    def disturbance_jacobian(self, state, time=0.0):
        return jnp.expand_dims(jnp.zeros(3), axis=-1)

    def state_jacobian(self, state, control, time=0.0):
        return jnp.array([
            [0, 0, -self.v * jnp.sin(state[2])],
            [0, 0, self.v * jnp.cos(state[2])], 
            [0, 0, 0]])
    
dyn = DubinsDynamics({"v": dubins_vel, "dt": dt}, test=True)
dyn_jnp = DubinsJNPDynamics({"v": dubins_vel, "dt": dt}, test=True)

### Environment
# Defining the discretized state space and the location of obstacles

dyn_hjr = refine_cbfs.dynamics.HJControlAffineDynamics(dyn_jnp, control_space=hj.sets.Box(jnp.array(umin), jnp.array(umax)))

state_domain = hj.sets.Box(lo=jnp.array([0., 0., -jnp.pi]), hi=jnp.array([4., 4., jnp.pi]))
grid_resolution = (101, 101, 81)
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(state_domain, grid_resolution, periodic_dims=2)

@jit
def obstacle_barrier(robotX, obsX, d_min):
        h = (robotX[0] - obsX[0])*(robotX[0] - obsX[0]) + (robotX[1] - obsX[1])*(robotX[1] - obsX[1]) - d_min**2
        
        h_dot = 2*(robotX[0] - obsX[0])*dubins_vel*jnp.cos(robotX[2]) + 2*(robotX[1] - obsX[1])*dubins_vel*jnp.sin(robotX[2])
        dh_dot_dx = jnp.array( [ 2*dubins_vel*jnp.cos(robotX[2]), 2*dubins_vel*jnp.sin(robotX[2]), 2*(robotX[0] - obsX[0])*dubins_vel*(-jnp.sin(robotX[2]))+2*(robotX[1] - obsX[1])*dubins_vel*(jnp.cos(robotX[2])) ] )
        # dh_dx = 2 * jnp.array([robotX[0] - obsX[0], robotX[1] - obsX[1], 0.0 ])
        # print(f"h: {h}, dh_dot_dx:{dh_dot_dx}")
        return h, h_dot, dh_dot_dx.reshape(1,-1)

def robot_f(robotX):
    return jnp.array([ dubins_vel*jnp.cos(robotX[2]), dubins_vel*jnp.sin(robotX[2]), 0.0 ]).reshape(-1,1)

def robot_g(robotX):
    return jnp.array([ 0.0, 0.0, 1.0  ]).reshape(-1,1)

alpha1 = 1.0
alpha2 = 2.0
key = jax.random.PRNGKey(10)
num_samples=10000
samples = jax.random.uniform( key, shape=(1,num_samples), minval=umin, maxval=umax )#A.shape[1]   

@jit
def mc_polytope_volume(A, b):
    aux = A @ samples - b    
    aux = -aux
    aux = jnp.min(aux, axis=0)
    aux = (jnp.tanh( aux / 0.0000001 ) + 1.0)/2.0
    aux = jnp.sum( aux )
    vol = (umax-umin) * (aux / num_samples)
    return vol

control_bound_polytope_A = jnp.array([ 1.0, -1.0 ]).reshape(-1,1)
control_bound_polytope_b = jnp.array([ umax, umax ]).reshape(-1,1)

# # obstacles
obs1 = np.array([2.0,2.0])
obs2 = np.array([1.0,3.0])
obs3 = np.array([2.7,3.0])

@jit
def constraint_set(state):
    """A real-valued function s.t. the zero-superlevel set is the safe set

    Args:
        state : An unbatched (!) state vector, an array of shape `(4,)` containing `[y, v_y, phi, omega]`.

    Returns:
        A scalar, positive iff the state is in the safe set
    """
    # CBF conditions
    h1, h1_dot, dh1_dot_dx = obstacle_barrier( state, obs1, d_min = 0.5)
    h2, h2_dot, dh2_dot_dx = obstacle_barrier( state, obs2, d_min = 0.5)
    h3, h3_dot, dh3_dot_dx = obstacle_barrier( state, obs3, d_min = 0.5)

    # Robot f and g matrices
    gx = robot_g(state)
    fx = robot_f(state)
    
    # Ax >= b 
    A1 = dh1_dot_dx @ gx
    b1 = - dh1_dot_dx @ fx  - (alpha2 + alpha1) * h1_dot - alpha1 * alpha2 * h1

    A2 = dh2_dot_dx @ gx
    b2 = - dh2_dot_dx @ fx  - (alpha2 + alpha1) * h2_dot - alpha1 * alpha2 * h2

    A3 = dh3_dot_dx @ gx
    b3 = - dh3_dot_dx @ fx  - (alpha2 + alpha1) * h3_dot - alpha1 * alpha2 * h3

    A = jnp.concatenate([-A1, -A2, -A3, control_bound_polytope_A], axis=0)
    b = jnp.concatenate([-b1, -b2, -b3, control_bound_polytope_b], axis=0)

    violation = jnp.sign( jnp.min(jnp.array([h1, h2, h3])) )
    
    volume = mc_polytope_volume(A, b) 

    # return jnp.min(jnp.array([h1, h2, h3])), volume[0] * violation
    return volume[0] * violation

constraint_set( np.array([1.0, 1.0, 0.0]) )
heading = 0.0
x = np.linspace(0,4 ,100)
y = np.linspace(0,4 ,100)
X, Y = np.meshgrid(x,y)
volumes = np.zeros((100,100))
for j in range(100):
    for k in range(100):
        volumes[j,k] = constraint_set( np.array([ X[j,k], Y[j,k], heading ]) )


fig3, ax3 = plt.subplots()
im3 = ax3.scatter( X.reshape(-1,1), Y.reshape(-1,1), c = volumes.reshape(-1,1), cmap='jet', vmin=-1, vmax=1.0 )
fig3.colorbar(im3, ax=ax3)
theta = np.linspace(-np.pi, np.pi, 360)
ax3.plot( obs1[0]+0.5*np.cos(theta), obs1[1]+0.5*np.sin(theta), 'k--', linewidth=1.5 )
ax3.plot( obs2[0]+0.5*np.cos(theta), obs2[1]+0.5*np.sin(theta), 'k--', linewidth=1.5)
ax3.plot( obs3[0]+0.5*np.cos(theta), obs3[1]+0.5*np.sin(theta), 'k--', linewidth=1.5 )

fig4, ax4 = plt.subplots()
im4 = ax4.scatter( X.reshape(-1,1), Y.reshape(-1,1), c = hs.reshape(-1,1), cmap='jet', vmin=-1, vmax=3.0 )
fig4.colorbar(im4, ax=ax4)
theta = np.linspace(-np.pi, np.pi, 360)
ax4.plot( obs1[0]+0.5*np.cos(theta), obs1[1]+0.5*np.sin(theta), 'k--', linewidth=1.5 )
ax4.plot( obs2[0]+0.5*np.cos(theta), obs2[1]+0.5*np.sin(theta), 'k--', linewidth=1.5)
ax4.plot( obs3[0]+0.5*np.cos(theta), obs3[1]+0.5*np.sin(theta), 'k--', linewidth=1.5 )