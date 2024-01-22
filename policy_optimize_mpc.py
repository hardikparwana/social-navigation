import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, grad, value_and_grad
import cvxpy as cp

from utils import *
import pdb
from mpc_utils import *

noise_cov = 0.01
noise_mean = 0

N = 20 # horizon

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-1,4])
ax.set_ylim([-1,4])
dt = 0.05
T = 200

def robot_step(x,u,dt):
    return x + u*dt
def dynamics(x, u):
    return x + u * dt

def dynamics_step_nominal(x,u,dt):
    return x + u*dt
def dynamics_step(x,u,dt):
    return x + (u + np.random.normal(noise_mean, np.sqrt(noise_cov), size=(2,1)))*dt
@jit
def dynamics_step_noisy(mu_x,cov_x,u,dt):
    return mu_x.reshape(-1,1) + u*dt, cov_x.reshape(-1,1) + noise_cov*jnp.ones((2,1))

# @jit
def qp_sol(Q, G, A, b):
    alpha = 1.0

    Q_inv = np.diag( 1.0/np.diag(Q) )
    psi = b + A @ Q_inv @ G
    # u = Q_inv @ ( A.T / jnp.linalg.norm(A @ np.sqrt(Q_inv)) @ (jnp.tanh(10*psi)+1.0)/2.0  - G )
    if psi<0:
        u = Q_inv @ ( A.T / jnp.linalg.norm(A @ np.sqrt(Q_inv)) @ ( psi ) - G )
    else: # constraint is psi>0
        u = - Q_inv @ G
    return u

@jit
def human_predict(t,x):
    mu_x = jnp.zeros((2,N))
    cov_x = jnp.zeros((2,N))
    mu_x = mu_x.at[:,0].set(x[:,0])
    def body(i, inputs):
        tau, mu_x, cov_x = inputs
        u = jnp.array([0.05*tau, 0.4*jnp.sin(0.5 * tau)]).reshape(-1,1)
        mu_x_temp, cov_x_temp = dynamics_step_noisy(mu_x[:,i], cov_x[:,i], u, dt)
        mu_x = mu_x.at[:,i+1].set( mu_x_temp[:,0] )
        cov_x = cov_x.at[:,i+1].set( cov_x_temp[:,0] )
        return (tau+dt), mu_x, cov_x    
    return lax.fori_loop( 0, N-1, body, (t, mu_x, cov_x) )


human = np.array([0,0]).reshape(-1,1)
robot = np.array([0.0,3]).reshape(-1,1)
robot_init_state = np.copy(robot)
robot_copy = np.array([0.0,3]).reshape(-1,1)
robot_x_des = np.array([2, 0]).reshape(-1,1)
ax.scatter( robot_x_des[0], robot_x_des[1], c='g', s=100, marker='x' )
p = ax.scatter(human[0,0],human[1,0],s=50, c='r')
p_robot = ax.scatter(robot[0,0],robot[1,0],s=50, c='g', label='With adaptation')
p_robot_copy = ax.scatter(robot_copy[0,0],robot_copy[1,0],s=50, c='k', label='Without adaptation')
ax.legend()



def p_controller(k1, k2, robotX, humanX):
    u = -k1 * ( robotX[0:2] - robot_x_des ) + k2 * ( robotX[0:2] - humanX ) / jnp.linalg.norm( robotX[0:2] - humanX ) * jnp.tanh( 1.0 / jnp.linalg.norm(( robotX[0:2] - humanX )) )
    return u

@jit
def robot_predict(k1, k2, robotX, human_pred_mu, human_pred_cov ):
    x = jnp.zeros((2,N))
    x = x.at[:,i].set(robotX[:,0])
    def body(i, inputs):
        x = inputs
        vd = p_controller(k1, k2, x[:,i].reshape(-1,1), human_pred_mu[:,i].reshape(-1,1)) #  - k1 * ( robot[0:2] - robot_x_des )
        x = x.at[:,i+1].set(x[:,i] + vd[:,0] * dt)
        return x    
    xs = lax.fori_loop( 0, N, body, (x) )
    return xs

def predict_reward(final_x):
    return jnp.sum( jnp.square( final_x - robot_x_des ) )

def predict_collision(final_x, human_pred_mu, human_pred_cov, factor=1.96):
    dist_barrier_mu = jnp.sum( jnp.square( final_x - human_pred_mu), axis=0  )
    dist_barrier_cov = jnp.sum(2 * jnp.square(final_x) * human_pred_cov, axis=0) #approx
    min_dist = dist_barrier_mu - factor * jnp.sqrt( dist_barrier_cov )
    violation = jnp.min(min_dist)
    return min_dist

collision_grad = jit( value_and_grad( lambda k1, k2, robotX, human_pred_mu, human_pred_cov: predict_collision( robot_predict(k1, k2, robotX, human_pred_mu, human_pred_cov), human_pred_mu, human_pred_cov ), (0,1) ) )
reward_grad = jit( value_and_grad( lambda k1, k2, robotX, human_pred_mu, human_pred_cov:  predict_reward( robot_predict(k1, k2, robotX, human_pred_mu, human_pred_cov) ), (0,1) ) )

k1 = 0.2
k2 = 0.2
k1_org = 0.2
k2_org = 0.2

adapt = True

# Frame optimization problem
d = cp.Variable((2,1))
Q = cp.Parameter((1,2), np.zeros((1,2)))
A0 = cp.Parameter((1,1), value=np.zeros((1,1)))
A1 = cp.Parameter((1,2), np.zeros((1,2)))
cons = [A0 + A1 @ d >= 0]
cons += [ cp.abs(d) <= 0.05 ]
obj = cp.Minimize( Q @ d )
prob = cp.Problem(obj, cons)

#################

def mpc_stage_cost( X, u ):
    return 4 * jnp.sum( jnp.square( X[:,0] - robot_x_des[:,0] ) ) + 2.0 * jnp.sum( jnp.square( u ) )
def dynamics(X, u):
    return X + u*dt
def state_inequality_cons(X):
    dist_barrier_mu = jnp.sum( jnp.square( X - human_pred_mu), axis=0  )
    dist_barrier_cov = jnp.sum(2 * jnp.square(final_x) * human_pred_cov, axis=0) #approx
    min_dist = dist_barrier_mu - factor * jnp.sqrt( dist_barrier_cov )
    violation = jnp.min(min_dist)
    return jnp.sum( jnp.square( X[:,0] - jnp.array([ 0, 0.5 ]) ) ) - 0.3**2
def control_cons(U):
    return jnp.append( U[:,0] + 1.0, 1.0 - U[:,0] )

objective, objective_grad = mpc_cost_setup( horizon, n, m, mpc_stage_cost )
equality_constraint, equality_constraint_grad = equality_constraint_grad( horizon, n, m, dynamics, robot_init_state )
inequality_constraint, inequality_constraint_grad = inequality_constraint_grad( horizon, n, m,  )



# for t in range(T):

#     # Predict
#     _, pred_mu, pred_cov = human_predict( t*dt, jnp.asarray(human) )

#     # Show live animation
#     fig.canvas.draw()
#     fig.canvas.flush_events()

plt.ioff()
plt.show()
    





