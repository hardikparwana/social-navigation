import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import jax.numpy as jnp
import time

from jax import jit, grad, jacfwd, jacrev, lax
from scipy.optimize import minimize
import cyipopt

import pdb

run_scipy_example = False
run_ipopt_example = False

def mpc_cost_setup(horizon, n, m, objective_func):
    N = horizon
    @jit
    def body(mpc_X):
        X = mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
        U = mpc_X[-m*N:].reshape(m,N, order='F')
        cost = 0
        # for i in range(horizon):
        #     cost = cost + objective_func( X[:,[i]], U[:,[i]] )
        # cost = cost + objective_func(X[:,[horizon+1]], U[:,[i]]) # correct this

        cost = objective_func(X[:,[horizon+1]], U[:,[horizon]])
        def body_loop(i, inputs):
            cost = inputs
            cost = cost + objective_func( X[:,[i]], U[:,[i]] )
            return cost
        cost = cost + lax.fori_loop(0, horizon, body_loop, cost)
        # cost = cost + objective_func(X[:,[horizon+1]], U[:,[i]]) # correct this

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
        init_state_error = X[0:6,0] - robot_init_state[0:6,0]
        const = jnp.append(const, init_state_error)
        # return const

        # Dynamics Constraint
        # for i in range(horizon):
        #     dynamics_const = X[:,i+1] - step( X[:,[i]], U[:,[i]] )[:,0]
        #     const  = jnp.append( const, dynamics_const )

        const_dynamics = jnp.zeros((n,horizon))
        def body_loop(i, inputs):
            const = inputs
            dynamics_const = X[:,i+1] - step( X[:,[i]], U[:,[i]] )[:,0]
            const = const.at[:,i].set( dynamics_const )
            return const
        const_dynamics = lax.fori_loop(0, horizon, body_loop, const_dynamics).T.reshape(-1,1)[:,0]
        const = jnp.append( const, const_dynamics )
        return const[1:]
    
    body_grad = jit(jacrev(body, 0))
    return body, body_grad

def inequality_constraint_setup(horizon, n, m, state_constraint=None, control_constraint=None, objective_func=None, max_cost=790):
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
            # for i in range(N+1):
            #     const = jnp.append( const, state_constraint(X[:,[i]]) )
            const_state = jnp.zeros(N+1)
            def state_loop(i, inputs):
                const = inputs
                const = const.at[i].set( state_constraint(X[:,[i]]) )
                return const
            const_state = lax.fori_loop( 0, N+1, state_loop, const_state )
            const = jnp.append(const, const_state)

        # Control input constraint
        if control_constraint!=None:
            # for i in range(N):
            #     const = jnp.append( const, control_constraint(U[:,[i]]) )
            const_input = jnp.zeros((4,N))
            def control_loop(i, inputs):
                const = inputs
                const = const.at[:,i].set( control_constraint(U[:,[i]]))
                return const
            const_input = lax.fori_loop( 0, N, control_loop, const_input ).T.reshape(-1,1)[:,0]
            const = jnp.append(const, const_input)

        # mpc cost constraint # jnp.array([[factor]]) best cost.. without any state inequality constraint
        if objective_func!=None:
            objective_cost = objective_func( mpc_X )
            const = jnp.append(const, max_cost-objective_cost)


        return const[1:]

    body_grad = jit( jacrev( body, 0 ) )

    return body, body_grad


def make_ipopt_solver(mpc_X, objective, objective_grad, equality_constraint, equality_constraint_grad, inequality_constraint, inequality_constraint_grad, lb, ub):

    class CYIPOPT_Wrapper():
        '''
            Based on native interface https://cyipopt.readthedocs.io/en/stable/tutorial.html#scipy-compatible-interface
            Note: the above page has 2 interfaces: one based on scipy and one based on a direct wrapper(scroll down on the above webpage). Scipy interface is slower so I use the direct wrapper
        '''

        def objective(self, x):
            """Returns the scalar value of the objective given x."""
            return objective(x)

        def gradient(self, x):
            """Returns the gradient of the objective with respect to x."""
            return objective_grad(x)

        def constraints(self, x):
            """Returns the constraints."""
            # return jnp.zeros(1)
            return jnp.append( equality_constraint(x), inequality_constraint(x) )

        def jacobian(self, x):
            # return jnp.zeros(x.size)
            """Returns the Jacobian of the constraints with respect to x."""
            return jnp.append( equality_constraint_grad(x), inequality_constraint_grad(x), axis=0 )
        
    cl_equality = jnp.zeros( equality_constraint(mpc_X).size )
    cu_equality = cl_equality
    cl_inequality = jnp.zeros( inequality_constraint(mpc_X).size )
    cu_inequality = 2.0e19 * jnp.ones( cl_inequality.size )
    cl = jnp.append( cl_equality, cl_inequality )
    cu = jnp.append( cu_equality, cu_inequality )

    t0 = time.time()

    nlp = cyipopt.Problem(
    n=mpc_X.size,
    m=len(cl),
    problem_obj=CYIPOPT_Wrapper(),
    lb=lb,
    ub=ub,
    cl=cl,
    cu=cu,
    )
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('tol', 1e-10) #1e-6)
    nlp.add_option('linear_solver', 'ma97')
    nlp.add_option('print_level', 3)
    nlp.add_option('max_iter', 20000)

    t1 = time.time()
    print(f"set IPOPT problem in :{t1-t0}")

    return nlp



################## EXAMPLE ###############################

######## Sim parameters #########
if run_scipy_example:
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
    res = minimize(objective, mpc_X, method='SLSQP', jac=objective_grad, constraints=cons, options={'gtol': 1e-6, 'disp': True, 'maxiter': 1000}) # mpc_X is the initial guess
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

if run_ipopt_example:
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

    nlp = make_ipopt_solver( mpc_X, objective, objective_grad, equality_constraint, equality_constraint_grad, inequality_constraint, inequality_constraint_grad, lb, ub )

    t1 = time.time()
    x, info = nlp.solve(mpc_X) # mpc_X is the initial guess
    print(f"solved problem in :{time.time()-t1}")

    sol_X = x[0:n*(N+1)].reshape(n,N+1, order='F')
    sol_U = x[-m*N:].reshape(m,N, order='F')

    fig, ax = plt.subplots(2)
    ax[0].plot( sol_X[0,:], sol_X[1,:] )
    circ = plt.Circle((0,0.5),0.3,linewidth = 1, edgecolor='k',facecolor='k')
    ax[0].add_patch(circ)
    ax[0].axis('equal')

    ax[1].plot(sol_U[0,:], 'b', label='u1')
    ax[1].plot(sol_U[1,:], 'r*', label='u2')
    ax[1].legend()

    plt.show()


