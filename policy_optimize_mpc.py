import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, grad, value_and_grad
import cvxpy as cp

from utils import *
import pdb

noise_cov = 0.01
noise_mean = 0

plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-1,4])
ax.set_ylim([-1,4])
dt = 0.05
T = 200

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
robot = np.array([0.0,3]).reshape(-1,1)
robot_copy = np.array([0.0,3]).reshape(-1,1)
robot_x_des = np.array([2, 0]).reshape(-1,1)
ax.scatter( robot_x_des[0], robot_x_des[1], c='g', s=100, marker='x' )
p = ax.scatter(human[0,0],human[1,0],s=50, c='r')
p_robot = ax.scatter(robot[0,0],robot[1,0],s=50, c='g', label='With adaptation')
p_robot_copy = ax.scatter(robot_copy[0,0],robot_copy[1,0],s=50, c='k', label='Without adaptation')
ax.legend()

N = 20
H = 10

def p_controller(k1, k2, robotX, humanX):
    u = -k1 * ( robotX[0:2] - robot_x_des ) + k2 * ( robotX[0:2] - humanX ) / jnp.linalg.norm( robotX[0:2] - humanX ) * jnp.tanh( 1.0 / jnp.linalg.norm(( robotX[0:2] - humanX )) )
    return u

@jit
def robot_predict(k1, k2, robotX, human_pred_mu, human_pred_cov ):
    def body(i, inputs):
        x = inputs
        vd = p_controller(k1, k2, robotX, human_pred_mu[:,i].reshape(-1,1)) #  - k1 * ( robot[0:2] - robot_x_des )
        x = x + vd * dt
        return x    
    final_x = lax.fori_loop( 0, H, body, (jnp.copy(robotX)) )
    return final_x

def predict_reward(final_x):
    return jnp.sum( jnp.square( final_x - robot_x_des ) )

def predict_collision(final_x, human_pred_mu, human_pred_cov):
    dist_barrier_mu = jnp.sum( (final_x[:,0] - human_pred_mu[:,-1])**2 )
    dist_barrier_cov = jnp.sum(human_pred_cov[:,-1]**2) + 2 * jnp.sum(human_pred_mu[:,-1]**2 * human_pred_cov[:,-1])
    dist_ci = dist_barrier_mu - 1.96 * dist_barrier_cov
    # violation = jnp.min( jnp.array([dist_ci, 0.0]) ) # want violation to be greater than 0 no matter what 
    violation = dist_ci  
    return violation

# pred_grad = jit(grad(predict_collision, (0, 1)))

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


for t in range(T):

    # Predict
    _, pred_mu, pred_cov = human_predict( t*dt, jnp.asarray(human) )

    if adapt:
        robot_final_x = robot_predict( k1, k2, robot, pred_mu, pred_cov )
        # predict_collision( robot_final_x, pred_mu, pred_cov )
        # reward( robot_final_x, pred_mu, pred_cov )
        collision, (grads_k1, grads_k2) = collision_grad( k1, k2, robot, pred_mu, pred_cov )
        # print(f"grads k1: {grads_k1}, k2: {grads_k2}")
        # k1 = np.clip(k1 + 0.001 * grads_k1, 0, None)
        # k2 = np.clip(k2 + 0.001 * grads_k2, 0, None)
        if collision < 0:
            print(f"Constraint violated, barrier value: {collision}")
            exit()
        
        reward, (reward_grads_k1, reward_grads_k2) = reward_grad( k1, k2, robot, pred_mu, pred_cov )
        # print(f"collision: {collision}, collision grads: {grads_k1}, {grads_k2}")
        # k1 = np.clip(k1 - 0.001 * reward_grads_k1, 0, None)
        # k2 = np.clip(k2 - 0.001 * reward_grads_k2, 0, None)

        Q = np.eye(2)
        G = 1.0 * np.array([[reward_grads_k1, reward_grads_k2]]).T
        # G = - 1.0 * np.array([[ grads_k1, grads_k2 ]]).T # improve constraint satisfaction
        A = np.array([[ grads_k1, grads_k2 ]])
        b = -collision
        dir = np.clip(qp_sol( Q, G, A, b), -0.02, 0.02)

        # print(f"update dir: {dir.T}")
        k1 = np.clip( k1 + dir[0,0], 0, None )
        k2 = np.clip( k2 + dir[1,0], 0, None )
        print(f"params: k1: {k1}, k2: {k2}, G: {G.T}")

        # Q.value = np.array([[reward_grads_k1, reward_grads_k2]])
        # A0.value[0,0] = collision
        # A1.value = np.array([[ grads_k1, grads_k2 ]])
        # prob.solve(solver=cp.GUROBI)
        # if prob.status != 'optimal':
        #     print(f"ERROR")
        #     # pdb.set_trace()
        #     exit()
        # print(f"grads: {d.value.T}")
        # k1 = np.clip( k1 + d.value[0,0], 0, None )
        # k2 = np.clip( k2 + d.value[1,0], 0, None )

    # Robot control
    vd = p_controller( k1, k2, robot, human )
    robot = robot_step(robot, vd, dt)
    p_robot.set_offsets([robot[0,0], robot[1,0]])

    vd_copy = p_controller(k1_org, k2_org, robot_copy, human)
    robot_copy = robot_step(robot_copy, vd_copy, dt)
    p_robot_copy.set_offsets([robot_copy[0,0], robot_copy[1,0]])

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
    





