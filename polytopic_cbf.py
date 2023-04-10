import numpy as np
import cvxpy as cp  
from single_integrator import single_integrator_square
from obstacles import rectangle
import matplotlib.pyplot as plt

alpha_cbf_nominal = 1.0

## Find barrier function
y_o = cp.Variable((2,1))
y_r = cp.Variable((2,1))
A_o = cp.Parameter((2,2))
A_r = cp.Parameter((2,2))
b_o = cp.Parameter((2,1))
b_r = cp.Parameter((2,1))

objective_barrier = cp.Minimize( cp.sum_squares( y_o - y_r ) )
const_barrier = [ A_o @ y_o <= b_o, A_r @ y_r <= b_r ]
problem_barrier = cp.Problem( objective_barrier, const_barrier )

# Solving CBF optimization problem
lambda_o = cp.Variable((1,2))
lambda_r = cp.Variable((1,2))
b_r_f = cp.Parameter((2,1))
b_r_g = cp.Parameter((2,2))
alpha_cbf = cp.Variable()
U = cp.Variable((2,1))
h = cp.Parameter()
U_ref = cp.Parameter((2,1))

const_cbf = []
const_cbf += [ - lambda_o @ b_o - lambda_r @ b_r_f - lambda_r @ b_r_g @ U >= alpha_cbf * h ]
const_cbf += [ lambda_o @ A_o + lambda_r @ A_r == 0  ]
const_cbf += [ cp.norm( lambda_o @ A_o) <= 1  ]
const_cbf += [ lambda_o >= 0, lambda_r >= 0 ]
objective_cbf = cp.Minimize( cp.sum_squares(U) + 0.1 * cp.sum_squares( alpha_cbf - alpha_cbf_nominal ) )
problem_cbf = cp.Problem( objective_cbf, const_cbf )

fig = plt.figure()
ax = plt.axes()
ax.set_xlabel("X")
ax.set_ylabel("Y")

t = 0
tf = 4.0
dt = 0.05

robot = single_integrator_square(ax, pos = np.array([0,0]), dt = dt)
obs = rectangle( pos = np.array([1,1])  )





while t < tf:
    
    A_r.value, b_r.value = robot.polytopic_location()
    A_o.value, b_o.value = obs.polytopic_location()
    
    # find barrier function
    problem_barrier.solve()
    h.value = problem_barrier.value  
    
    # find control input
    U_ref.value = np.array([1,0]).reshape(-1,1)
    b_r_f.value, b_r_g.value = P.polytopic_location_next_state(  )
    problem_cbf.solve()
    if problem_cbf.status != 'optimal':
        print("CBF problem infeasible")
        exit()
    
    # propagate dynamics
    robot.step(U.value)
    
    t = t + dt
    
    
    
    
    
    
    



