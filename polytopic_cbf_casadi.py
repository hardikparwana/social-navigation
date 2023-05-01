import numpy as np
import cvxpy as cp  
import casadi as cd
from single_integrator import single_integrator_square
from obstacles import rectangle
import matplotlib.pyplot as plt

alpha_cbf_nominal = 100.0

opti = cd.Opti()


## Find barrier function
y_o = cp.Variable((2,1))
y_r = cp.Variable((2,1))
A_o = cp.Parameter((4,2))
A_r = cp.Parameter((4,2))
b_o = cp.Parameter((4,1))
b_r = cp.Parameter((4,1))

objective_barrier = cp.Minimize( cp.sum_squares( y_o - y_r ) )
const_barrier = [ A_o @ y_o <= b_o, A_r @ y_r <= b_r ]
problem_barrier = cp.Problem( objective_barrier, const_barrier )

# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes()
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Simulation Parameters
t = 0
tf = 4.0
dt = 0.05

robot = single_integrator_square(ax, pos = np.array([0,0]), dt = dt)
obs = rectangle( ax, pos = np.array([3,3])  )

while t < tf:
    
    opti = cd.Opti();
    # Solving CBF optimization problem
    lambda_o = opti.variable(1,4)
    lambda_r = opti.variable(1,4)
    alpha_cbf = opti.variable()
    U = opti.variable(2,1)
    X_next = opti.variable(2,1)
    
    A_r.value, b_r.value = robot.polytopic_location()
    A_o.value, b_o.value = obs.polytopic_location()
    
    # find barrier function
    problem_barrier.solve()
    h = np.sqrt(problem_barrier.value)
    
    # find control input
    U_ref = np.array([1,0]).reshape(-1,1)
    Anext, bnext_r_f, bnext_r_g = robot.polytopic_location_next_state()
    
    U_error = U - U_ref
    objective = 100 * cd.mtimes( U.T, U ) + 1.0 * ( alpha_cbf - alpha_cbf_nominal )
    opti.minimize(objective)
    
    # Next state polytopic location
    Rot = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
            ])
    Rot_dot = np.array([
        [0.0, 0.0],
        [0.0, 0.0]
        ])

    A_next = robot.A @ Rot.T
    b_next = cd.mtimes(robot.A @ Rot.T, X_next) + robot.b
    
    opti.subject_to( alpha_cbf >= 0 )
    opti.subject_to(  - cd.mtimes(lambda_o, b_o.value) - cd.mtimes(lambda_r, b_next) >= alpha_cbf_nominal * h )
    opti.subject_to(  cd.mtimes(lambda_o, A_o.value) + cd.mtimes(lambda_r, A_next) == 0  )
    temp = cd.mtimes( lambda_o, A_o.value )
    opti.subject_to(  cd.mtimes( temp, temp.T ) <= 1  )
    opti.subject_to( lambda_o >= 0 ) 
    opti.subject_to( lambda_r >= 0 )
    opti.subject_to(  X_next == robot.X + robot.f()*dt + cd.mtimes(robot.g()*dt, U) )
    
    opti.solver('ipopt');
    sol = opti.solve();
    
    print(f" h:{h} alpha:{sol.value(alpha_cbf)}, U:{sol.value(U).T} ")
    # propagate dynamics
    robot.step(sol.value(U))
    robot.render_plot()
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    t = t + dt
    
    
    
    
    
    
    



