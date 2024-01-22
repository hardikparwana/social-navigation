import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import jax.numpy as jnp

from jax import jit, grad, jacfwd, jacrev
# import cyipopt
# Scipy optimize
from scipy.optimize import minimize

import pdb

run_example = False


def mpc_cost_setup(horizon, n, m, objective_func):
    N = horizon
    @jit
    def body(mpc_X):
        X = mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
        U = mpc_X[-m*N:].reshape(m,N, order='F')
        cost = 0
        for i in range(horizon):
            cost = cost + objective_func( X[:,i].reshape(-1,1), U[:,i].reshape(-1,1) )
        return cost
    
    body_grad = jit(grad(body, 0))
    return body, body_grad


def equality_constraint_setup(horizon, n, m, step, robot_init_state):
    N = horizon
    @jit
    def body(mpc_X):
        '''
            Assume of form g(x) = 0
            Returns g(x) as 1D array
        '''
        # return jnp.array([0.0])
        X = mpc_X[0:n*(N+1)].reshape(n,N+1, order='F') 
        U = mpc_X[-m*N:].reshape(m,N, order='F')
        const = jnp.zeros(1)
        
        # Initial state constraint
        init_state_error = X[:,0] - robot_init_state[:,0]
        const = jnp.append(const, init_state_error)
        # return const

        # Dynamics Constraint
        for i in range(horizon):
            dynamics_const = X[:,i+1] - step( X[:,i].reshape(-1,1), U[:,i].reshape(-1,1) )[:,0]
            const  = jnp.append( const, dynamics_const )
        return const[1:]
    
    body_grad = jit(jacrev(body, 0))
    return body, body_grad

def inequality_constraint_setup(horizon, n, m, state_constraint=None, control_constraint=None):
    N = horizon
    @jit
    def body(mpc_X):
        '''
            Assume of form g(x) >= 0
            Retruns g(x) as 1D array
        '''
        X = mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
        U = mpc_X[-m*N:].reshape(m,N, order='F')
        const = jnp.zeros(1)

        if state_constraint!=None:
            # state constraint
            for i in range(N+1):
                const = jnp.append( const, state_constraint(X[:,i].reshape(-1,1)) )

        # Control input constraint
        if control_constraint!=None:
            for i in range(N):
                const = jnp.append( const, control_constraint(U[:,i].reshape(-1,1) ) )
            return const[1:]

    body_grad = jit( jacrev( body, 0 ) )

    return body, body_grad



################## EXAMPLE ###############################

######## Sim parameters #########
if run_example:
    dt = 0.05
    horizon = 50
    N = horizon
    n = 2
    m = 2

    lb = -100 * jnp.ones((N)*(n+m)+n)
    ub =  100 * jnp.ones((N)*(n+m)+n)

    robot_init_state = jnp.array([0,0]).reshape(-1,1)
    X_guess = jnp.zeros((n,N+1))
    U_guess = jnp.zeros((m,N))
    mpc_X = jnp.concatenate( (X_guess.T.reshape(-1,1), U_guess.T.reshape(-1,1)), axis=0 )[:,0] # has to be a 1D array for ipopt

    def mpc_stage_cost( X, u ):
        return jnp.sum( jnp.square( X[:,0] - jnp.array([0.4,1.3]) ) ) + 2.0 * jnp.sum( jnp.square( u ) )
    def dynamics(X, u):
        return X + u*dt
    def state_inequality_cons(X):
        return jnp.sum( jnp.square( X[:,0] - jnp.array([ 0, 0.5 ]) ) ) - 0.3**2
    def control_cons(U):
        return jnp.append( U[:,0] + 1.0, 1.0 - U[:,0] )

    objective, objective_grad = mpc_cost_setup(horizon, n, m,mpc_stage_cost)
    equality_constraint, equality_constraint_grad = equality_constraint_setup(horizon, n, m, dynamics, robot_init_state)
    inequality_constraint, inequality_constraint_grad = inequality_constraint_setup(horizon, n, m, state_inequality_cons, control_cons)

    cons = ( {'type': 'eq', 'fun': equality_constraint, 'jac': equality_constraint_grad},
            {'type': 'ineq', 'fun': inequality_constraint, 'jac': inequality_constraint_grad} )
    res = minimize(objective, mpc_X, method='SLSQP', jac=objective_grad, constraints=cons, options={'gtol': 1e-6, 'disp': True, 'maxiter': 1000})
    print(res.message)

    sol_X = res.x[0:n*(N+1)].reshape(n,N+1, order='F')
    sol_U = res.x[-m*N:].reshape(m,N, order='F')

    fig, ax = plt.subplots(2)
    ax[0].plot( sol_X[0,:], sol_X[1,:] )
    circ = plt.Circle((0,0.5),0.3,linewidth = 1, edgecolor='k',facecolor='k')
    ax[0].add_patch(circ)
    ax[0].axis('equal')

    ax[1].plot(sol_U[0,:], 'b', label='u1')
    ax[1].plot(sol_U[1,:], 'r*', label='u2')
    ax[1].legend()

    plt.show()


