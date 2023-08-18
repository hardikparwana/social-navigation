import jax
import time
import jax.numpy as jnp
from jax import jit, grad, jacfwd, jacrev
import cyipopt
import matplotlib.pyplot as plt
from obstacles import circle
jax.config.update("jax_enable_x64", True)
# import numpy as jnp

# Control Hyperparameters
n = 2  # Dimension of state
m = 2  # Dimension of control input
tf = 6.0
dt = 0.05
N = int(tf/dt) # MPC horizon
d_min = 0.3
control_bound = [3,3]

# Declare Variables
X = 2*jnp.zeros((n,N+1))
U = jnp.zeros((m,N))
robot_init_state = jnp.array([-0.5,-0.5]).reshape(-1,1)

# append all to one array
mpc_X = jnp.concatenate( (X.T.reshape(-1,1), U.T.reshape(-1,1)), axis=0 )[:,0] # has to be a 1D array for ipopt

# Declare Parameters
obstacle1X = jnp.array([0.7,0.7]).reshape(-1,1)
obstacle2X = jnp.array([1.5,1.9]).reshape(-1,1)
goal = jnp.array([2.0,2.0]).reshape(-1,1)

@jit
def step(x,u,dt):
    return x+u*dt

@jit
def objective(mpc_X):
    X = mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
    U = mpc_X[-m*N:].reshape(m,N, order='F')
    cost = 0
    for i in range(N):
        cost = cost + jnp.sum( 100*jnp.square(X[:,i] - goal[:,0]) ) + jnp.sum( jnp.square( U[:,i] ) )
    # cost = cost + jnp.sum( jnp.square(X[:,N] - goal[:,0]) )
    return cost
objective_grad = jit(grad(objective, 0))
# objective(mpc_X)

@jit
def equality_constraint(mpc_X):
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
    
    # Dynamic Constraint
    for i in range(N):
        dynamics_const = X[:,i+1] - step( X[:,i].reshape(-1,1), U[:,i].reshape(-1,1), dt )[:,0]
        const  = jnp.append( const, dynamics_const )
    return const[1:]
equality_constraint_grad = jit(jacrev(equality_constraint, 0))

@jit
def inequality_constraint(mpc_X):
    '''
        Assume of form g(x) >= 0
        Retruns g(x) as 1D array
    '''
    # return jnp.array([0.0])
    X = mpc_X[0:n*(N+1)].reshape(n,N+1, order='F')
    U = mpc_X[-m*N:].reshape(m,N, order='F')
    const = jnp.zeros((1,1))
    
    # Collision avoidance
    for i in range(N+1):
        barrier = jnp.sum(jnp.square(X[0:2,i] - obstacle1X[:,0])) - d_min**2
        const = jnp.append( const, barrier )
        barrier = jnp.sum(jnp.square(X[0:2,i] - obstacle2X[:,0])) - d_min**2
        const = jnp.append( const, barrier )
        
    # Control input bounds
    for i in range(N):
        const = jnp.append( const, control_bound[0]*control_bound[0] - U[0,i]*U[0,i] )
        const = jnp.append( const, control_bound[1]*control_bound[1] - U[1,i]*U[1,i] )
    
    return const[1:]
# inequality_constraint(0.64645*jnp.ones(22))
# inequality_constraint(mpc_X)
inequality_constraint_grad = jit(jacrev(inequality_constraint, 0))



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
        # return jnp.zeros(1,1)
        return jnp.append( equality_constraint(x), inequality_constraint(x) )

    def jacobian(self, x):
        # return jnp.zeros(x.size)
        """Returns the Jacobian of the constraints with respect to x."""
        return jnp.append( equality_constraint_grad(x), inequality_constraint_grad(x), axis=0 )

# Upper and lower bounds on optimization variables
# lb = jnp.concatenate((-20*jnp.ones(X.size), -control_bound[0]*jnp.ones(N), -control_bound[1]*jnp.ones(N)) )
# ub = jnp.concatenate(( 20*jnp.ones(X.size),  control_bound[0]*jnp.ones(N),  control_bound[1]*jnp.ones(N)) )

lb = -100 * jnp.ones((N)*(n+m)+n)
ub =  100 * jnp.ones((N)*(n+m)+n)

# Equality and inequality constraint function upper and lower bound
# Equality g(x) == 0
cl_equality = jnp.zeros( equality_constraint(mpc_X).size )
cu_equality = cl_equality
# Inequality  Infinity>=g(x)>=0 0
cl_inequality = jnp.zeros( inequality_constraint(mpc_X).size )
cu_inequality = 2.0e19 * jnp.ones( cl_inequality.size )
# Combine all in one
cl = jnp.append( cl_equality, cl_inequality )
cu = jnp.append( cu_equality, cu_inequality )

# run once to do grad and jit
equality_constraint_grad(mpc_X)
inequality_constraint_grad(mpc_X)
t0 = time.time()
print(f"Starting: ")

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
nlp.add_option('tol', 1e-3)
nlp.add_option('linear_solver', 'ma57')
nlp.add_option('print_level', 0)
nlp.add_option('max_iter', 20)
t1 = time.time()
print(f"set problem in :{t1-t0}")

x, info = nlp.solve(mpc_X)
print(f"solved problem in :{time.time()-t1}")
sol_X = x[0:n*(N+1)].reshape(n,N+1, order='F')
sol_U = x[-m*N:].reshape(m,N, order='F')
print(f"sol_X: {sol_X}")
print(f"sol_U: {sol_U}")

t2 = time.time()
x, info = nlp.solve(x)
print(f"second solve time: {time.time()-t2}")

fig, ax = plt.subplots(1,1)
ax.plot(sol_X[0,:], sol_X[1,:], 'g*')
circ = plt.Circle((obstacle1X[0],obstacle1X[1]),d_min,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
circ2 = plt.Circle((obstacle2X[0],obstacle2X[1]),d_min,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ2)
plt.show()

print(f"final cost: {objective(x)}")
