import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, lax, grad, value_and_grad
import cvxpy as cp
from robot_models.single_integrator import single_integrator
from robot_models.humans import humans
# from mpc_utils import *

from jax import config
config.update("jax_enable_x64", True)

# Simulation parameters
human_noise_cov = 4.0 #4.0 #0.5
human_noise_mean = 0
dt = 0.05
# N = 27 # prediction source v  
# T = N #1 #200 # Simulation steps
factor = 4.0 #2.0 # no of standard deviations
print(f"Nominal Risk factor: 2")

# Set figure
# plt.ion()
fig, ax = plt.subplots()
ax.set_xlim([-1,8])
ax.set_ylim([-1,8])

# Initialize robot
robot = single_integrator( ax, pos=np.array([0.0,0.0]), dt=dt )
robot_goal = np.array([2.0, 6.0]).reshape(-1,1)
ax.scatter(robot_goal[0,0], robot_goal[1,0], c='g', s=70)

# Initialize human
human = humans(ax, pos=np.array([4.0,4.0]), dt=dt)

# Human helper functions
def human_input(tau=0):
    # u = jnp.array([0.2*tau, 0.4*jnp.sin(0.5 * tau)]).reshape(-1,1)
    u = jnp.array([-3.0,0]).reshape(-1,1)
    return u

@jit
def human_step_noisy(mu_x,cov_x,u,dt):
    return mu_x.reshape(-1,1) + u*dt, cov_x.reshape(-1,1) + human_noise_cov*jnp.ones((2,1))*dt*dt

@jit
def human_predict(t,x, robotX, robotU):
    mu_x = jnp.zeros((2,N))
    cov_x = jnp.zeros((2,N))
    mu_x = mu_x.at[:,0].set(x[:,0])
    def body(i, inputs):
        tau, mu_x, cov_x, robotX = inputs
        u = human_input(tau)
        # if jnp.linalg.norm( x-robotX )<2.0:            
        #     u = u - (robotX - x) / jnp.linalg.norm( x-robotX ) * ( 1.0 * jnp.tanh( 1.0 / jnp.linalg.norm( x-robotX ) ) )
        mu_x_temp, cov_x_temp = human_step_noisy(mu_x[:,i], cov_x[:,i], u, dt)
        mu_x = mu_x.at[:,i+1].set( mu_x_temp[:,0] )
        cov_x = cov_x.at[:,i+1].set( cov_x_temp[:,0] )
        robotX = robotX + robotU * dt
        return tau+dt, mu_x, cov_x, robotX    
    return lax.fori_loop( 0, N-1, body, (t, mu_x, cov_x, robotX) )

# Prepare MPC controller
# for MPC, append robot and human state vectors
# dt = 0.05
horizon = 30 #50
N = horizon
T = N
n = 7 # robot x,y, human x, y mu and cov
m = 2
control_bound = 4 #4

# Run Simulation
# for t in range(T):

#     # Predict
#     _, pred_mu, pred_cov, _ = human_predict( t*dt, jnp.asarray(human.X), robot.X, robot.U )

#     # Move humans
#     human.step( human_input(t*dt) )
#     human.render_predictions(N, pred_mu, pred_cov, factor)

#     # Plan robot path
#     # solve nonlinear MPC   

#     fig.canvas.draw()
#     fig.canvas.flush_events()

# lb = -100 * jnp.ones((N)*(n+m)+n)
# ub =  100 * jnp.ones((N)*(n+m)+n)

robot_init_state = jnp.concatenate((robot.X, human.X, 0.01 * jnp.ones((2,1)), jnp.array([[factor]]) ), axis=0) #jnp.array([0,0]).reshape(-1,1)
# X_guess = jnp.copy(robot_init_state)
X_guess = jnp.ones((n,N+1))
X_guess = X_guess.at[:,0].set( robot_init_state[:,0] )
U_guess = jnp.ones((m,N))
for t in range(T):
    u = jnp.clip( -1.0 * ( X_guess[0:2,[t]] - robot_goal ), -control_bound, control_bound )
    u_human = human_input()
    mu_x_next, cov_x_next = human_step_noisy(X_guess[2:4,[t]], X_guess[4:6,[t]], u_human, dt)
    x_next = jnp.concatenate( ( X_guess[0:2,[t]]+u*dt, mu_x_next, cov_x_next, jnp.array([[factor]]) ), axis=0 )

    U_guess = U_guess.at[:,t].set(u[:,0])
    X_guess = X_guess.at[:,t+1].set(x_next[:,0])

plt.plot(X_guess[0,:], X_guess[1,:], 'c*')

mpc_X = jnp.concatenate( (X_guess.T.reshape(-1,1), U_guess.T.reshape(-1,1)), axis=0 )[:,0] # has to be a 1D array for ipopt

X_guess_lb = jnp.concatenate( (-10*jnp.ones((2,N+1)), -10*jnp.ones((2,N+1)), 0.001*jnp.ones((2,N+1)),   0*jnp.ones((1,N+1)) ), axis=0)
X_guess_ub = jnp.concatenate( ( 20*jnp.ones((2,N+1)),  20*jnp.ones((2,N+1)),   100*jnp.ones((2,N+1)),  10*jnp.ones((1,N+1)) ), axis=0)
U_guess_lb = -10*jnp.ones((m,N))
U_guess_ub =  10*jnp.ones((m,N))
lb = jnp.concatenate( (X_guess_lb.T.reshape(-1,1), U_guess_lb.T.reshape(-1,1)), axis=0 )[:,0]
ub = jnp.concatenate( (X_guess_ub.T.reshape(-1,1), U_guess_ub.T.reshape(-1,1)), axis=0 )[:,0]
 
from mpc_utils import *

def mpc_stage_cost( X, u ):
    return 2 * jnp.sum( jnp.square( X[0:2,0] - robot_goal[:,0] ) ) + jnp.sum( jnp.square( X[6,0]-factor ) )#+ 1.0 * jnp.sum( jnp.square( u ) )

def true_func(u_human, mu_x, robot_x):
    u_human = u_human - (robot_x - mu_x) / jnp.clip(jnp.linalg.norm( mu_x-robot_x ), 0.01, None) * ( 3.0 * jnp.tanh( 1.0 / jnp.linalg.norm( mu_x-robot_x ) ) )
    return u_human

def false_func(u_human, mu_x, robot_x):
    return u_human

def robot_human_dynamics(X, u):
    robot_x = X[0:2]
    robot_next_x = X[0:2] + u * dt
    mu_x = X[2:4]
    cov_x = X[4:6]
    factor_state = X[6].reshape(-1,1)

    u_human = human_input()
    u_human = lax.cond( jnp.linalg.norm( mu_x-robot_x )<2.0, true_func, false_func, u_human, mu_x, robot_x)
    mu_x_next, cov_x_next = human_step_noisy(mu_x, cov_x, u_human, dt)
    factor_state_next = factor_state
    return jnp.concatenate( ( robot_next_x, mu_x_next, cov_x_next, factor_state_next ), axis=0 )

def state_inequality_cons(X):
    robot_x = X[0:2]
    mu_x = X[2:4]
    cov_x = X[4:6]
    factor_state = X[6]
    std = factor_state * jnp.sqrt( 0.01 + cov_x )
    # # dist = (robot_x[0]-mu_x[0])**2 * std[1]**2 + (robot_x[0]-mu_x[0])**2 * std[1]**2 - factor*std[0]**2 * std[1]**2
    dist = jnp.sum( jnp.square( robot_x - mu_x ) ) -  std[0,0]**2# * std[1]**2
    # dist = jnp.sum( jnp.square( robot_x - mu_x ) ) -  jnp.square(jnp.max(std))
    # dist = jnp.sum( jnp.square( robot_x - mu_x ) ) -  jnp.square(jnp.mean(std))
    # dist = robot_x[0,0]*robot_x[0,0]*std[1,0]*std[1,0] + robot_x[1,0]*robot_x[1,0]*std[0,0]*std[0,0] - std[0,0]*std[0,0]*std[1,0]*std[1,0]
    # cc = jnp.array([[2],[4]])
    # dist = (robot_x[0:2]-cc).T @ (robot_x[0:2]-cc) - 1
    return  dist

def control_cons(U):
    return jnp.append( U[:,0] + control_bound, control_bound - U[:,0] )

max_cost = 790
objective, objective_grad = mpc_cost_setup(horizon, n, m,mpc_stage_cost)
equality_constraint, equality_constraint_grad = equality_constraint_setup(horizon, n, m, robot_human_dynamics, robot_init_state)
inequality_constraint, inequality_constraint_grad = inequality_constraint_setup(horizon, n, m, state_inequality_cons, control_cons, objective_func=objective, max_cost=max_cost)

cons = ( {'type': 'eq', 'fun': equality_constraint, 'jac': equality_constraint_grad},
        {'type': 'ineq', 'fun': inequality_constraint, 'jac': inequality_constraint_grad} )

# res = minimize(objective, mpc_X, method='SLSQP', jac=objective_grad, constraints=cons, options={'gtol': 1e-6, 'disp': True, 'maxiter': 10000}) # mpc_X is the initial guess
# print(res.message)
# sol_X = res.x[0:n*(N+1)].reshape(n,N+1, order='F')
# sol_U = res.x[-m*N:].reshape(m,N, order='F')

nlp = make_ipopt_solver( mpc_X, objective, objective_grad, equality_constraint, equality_constraint_grad, inequality_constraint, inequality_constraint_grad, lb, ub )

t1 = time.time()
x, info = nlp.solve(mpc_X) # mpc_X is the initial guess
sol_X = x[0:n*(N+1)].reshape(n,N+1, order='F')
sol_U = x[-m*N:].reshape(m,N, order='F')

print(f"solved problem in :{time.time()-t1} with cost: {objective(x)} and risk: {sol_X[6,0]}")



robot_x = sol_X[0:2]
human_mu = sol_X[2:4]
human_cov = sol_X[4:6]

ax.plot( robot_x[0,:], robot_x[1,:], 'b' )
ax.axis('equal')

human.render_predictions(N, human_mu, human_cov, factor)

# ax[1].plot(sol_U[0,:], 'b', label='u1')
# ax[1].plot(sol_U[1,:], 'r*', label='u2')
# ax[1].legend()
# pdb.set_trace()
# plt.show()

plt.ion()

fig2, ax2 = plt.subplots(2,1)
ax2[0].set_xlim([0, T])
ax2[1].set_xlim([0, T])
ax2[0].set_ylim([-control_bound, control_bound])
ax2[1].set_ylim([-control_bound, control_bound])
ax2[0].scatter(t, sol_U[0,0], c = 'r', label='X velocity')
ax2[1].scatter(t, sol_U[1,0], c = 'c', label='Y velocity')   
ax2[0].legend()
ax2[1].legend()

from matplotlib.animation import FFMpegWriter
metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=7, metadata=metadata)
movie_name = 'test.mp4'

with writer.saving(fig, movie_name, 100): 
    # Run Simulation
    for t in range(T):

        # Move humans
        # human.step( human_input(t*dt) )
        robot.step( sol_U[:,t] )
        human.render_predictions(1, human_mu[:,[t+1]], human_cov[:,[t+1]], factor)

        # Plan robot path
        # solve nonlinear MPC 
        ax2[0].scatter(t, sol_U[0,t], c = 'r')
        ax2[1].scatter(t, sol_U[1,t], c = 'c')    

        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        plt.pause(0.1)
        

plt.ioff()
plt.show()
pdb.set_trace()

# config 1: u_human = u_human - (robot_x - mu_x) / jnp.clip(jnp.linalg.norm( mu_x-robot_x ), 0.01, None) * ( 2.0 * jnp.tanh( 1.0 / jnp.linalg.norm( mu_x-robot_x ) ) )
# config 2: u_human = u_human - (robot_x - mu_x) / jnp.clip(jnp.linalg.norm( mu_x-robot_x ), 0.01, None) * ( 3.0 * jnp.tanh( 1.0 / jnp.linalg.norm( mu_x-robot_x ) ) )