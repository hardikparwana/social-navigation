import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, grad

from utils import *

noise_cov = 0.01
noise_mean = 0

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-1,4])
ax.set_ylim([-1,4])
dt = 0.1
T = 400

def robot_step(x,u,dt):
    return x + u*dt
    return x + np.array([x[2,0], x[3,0],u[0,0],u[1,0]]).reshape(-1,1) * dt

def dynamics_step_nominal(x,u,dt):
    return x + u*dt

def dynamics_step(x,u,dt):
    return x + (u + np.random.normal(noise_mean, np.sqrt(noise_cov), size=(2,1)))*dt

@jit
def dynamics_step_noisy(mu_x,cov_x,u,dt):
    return mu_x.reshape(-1,1) + u*dt, cov_x.reshape(-1,1) + noise_cov*jnp.ones((2,1))

def get_qp_control( robot_x, agent_x ):
    h = barrier1(robot_x, agent_x)
    dh_dx_robot, dh_dx_agent = barrier1_grad(robot_x, agent_x)



def cbf_qp(Q, G, A, b):
    alpha = 1.0

    Q_inv = 1 / Q
    psi = b + A @ Q_inv @ G

    u = Q_inv @ ( A.T / jnp.linalg.norm(A @ np.sqrt(Q_inv)) @ (jnp.tanh(10*psi)+1.0)/2.0  - G )
    return u

    # analytic solution
    # if psi<0:
    #     Q_inv @ ( A.T / jnp.linalg.norm(A @ np.sqrt(Q_inv)) @ ( psi ) - G )
    #     grad_b = Q_inv @ A.T / np.linalg.norm(A @ np.sqrt(Q_inv)) 
    # else:
    #     u = - Q_inv @ G
    #     grad_b = np.zeros((u.shape))
    # return u, grad_b

@jit
def barrier1(x1, x2): # distance
    return (x1[0:2]-x2[0:2]).T @ (x1[0:2]-x2[0:2])
barrier1_grad = jit(grad(barrier1, argnums=(0,1)))

def barrier2(x1, x2): # velocity
    v_max = 1.0
    return v_max**2 - (x1[2:4] - x2[2:4]).T @ (x1[2:4] - x2[2:4])

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
robot = np.array([0,3]).reshape(-1,1)
robot_x_des = np.array([2, 0]).reshape(-1,1)
p = ax.scatter(human[0,0],human[1,0],s=50, c='r')
p_robot = ax.scatter(robot[0,0],robot[1,0],s=50, c='g')

N = 20

for t in range(T):

    # Predict
    _, pred_mu, pred_cov = human_predict( t*dt, jnp.asarray(human) )

    # Robot control
    vd = -0.2 * ( robot[0:2] - robot_x_des )
    # ad = -1.0 * ( robot[2:4] - vd )
    # control = 
    robot = robot_step(robot, vd, dt)
    p_robot.set_offsets([robot[0,0], robot[1,0]])

    # Plot
    [p.remove() for p in reversed(ax.patches)] # ax.patches.clear() # ci_ellipse[i].remove() # ci_ellipse[i] = 
    for i in range(N):
        confidence_ellipse(np.asarray(pred_mu[:,i]).reshape(-1,1), np.diag(np.asarray(pred_cov[:,i])), ax, n_std=2.0, edgecolor = 'red')
    p.set_offsets([human[0,0], human[1,0]])

    # Step human
    u = jnp.array([0.05*t*dt, 0.4*jnp.sin(0.5 * t*dt)]).reshape(-1,1)
    human = dynamics_step(human, u, dt)

    # Show live animation
    fig.canvas.draw()
    fig.canvas.flush_events()

plt.ioff()
plt.show()
    





