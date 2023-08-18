import time
import numpy as np
import casadi as cd
import matplotlib.pyplot as plt


n = 2  # Dimension of state
m = 2  # Dimension of control input
tf = 6.0
dt = 0.05
N = int(tf/dt) # MPC horizon
d_min = 0.3
control_bound = np.asarray([3,3])
robot_init_state = np.array([-0.5,-0.5]).reshape(-1,1)
obstacle1X = np.array([0.7,0.7]).reshape(-1,1)
obstacle2X = np.array([1.5,1.9]).reshape(-1,1)
goal = np.array([2.0,2.0]).reshape(-1,1)

def step(X,U,dt):
    return X+U*dt
t0 = time.time()

opti = cd.Opti()
X = opti.variable(n, N+1)
U = opti.variable(m ,N)

# Initial Condition
opti.subject_to( X[:,0] == robot_init_state )

# Objective
cost = 0
for i in range(N):
    cost = cost + 100 * cd.mtimes( (X[:,i]-goal).T, (X[:,i]-goal) ) + cd.mtimes( U[:,i].T , U[:,i] )
cost += 100 * cd.mtimes( (X[:,i]-goal).T, (X[:,i]-goal) )
opti.minimize(cost)

# Dynamics
for i in range(N):
    opti.subject_to( X[:,i+1] == step( X[:,i], U[:,i], dt ) )

# Collision avoidance
for i in range(N):
    dist = X[:,i] - obstacle1X
    opti.subject_to( cd.mtimes(dist.T, dist) >= d_min**2)
    dist = X[:,i] - obstacle2X 
    opti.subject_to( cd.mtimes(dist.T, dist) >= d_min**2)

#Input bounds
for i in range(N):
    opti.subject_to(U[0,i]*U[0,i] <= control_bound[0]**2)
    opti.subject_to(U[1,i]*U[1,i] <= control_bound[1]**2)

option_mpc = {"verbose": False, "ipopt.print_level": 1, "print_time": 1, "ipopt.linear_solver":"ma57"}#, 'linear_solver':'m27'}
opti.solver("ipopt", option_mpc)

print(f"casadi built problem in :{time.time()-t0}")

t1 = time.time()
sol = opti.solve()
print(f"casadi solved problem in :{time.time()-t1}")
print(f"X: {sol.value(X)}")

t2 = time.time()
sol = opti.solve()
print(f"second casadi solved problem in :{time.time()-t2}")

fig, ax = plt.subplots(1,1)
ax.plot(sol.value(X[0,:]), sol.value(X[1,:]), 'g*')
circ = plt.Circle((obstacle1X[0],obstacle1X[1]),d_min,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ)
circ2 = plt.Circle((obstacle2X[0],obstacle2X[1]),d_min,linewidth = 1, edgecolor='k',facecolor='k')
ax.add_patch(circ2)

plt.show()
