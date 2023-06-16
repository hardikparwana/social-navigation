import numpy as np
# import cvxpy as cp  
import casadi as cd
from unicycle import unicycle
from obstacles import rectangle
import matplotlib.pyplot as plt

alpha_cbf_nominal = 0.9
h_offset = 0.05
# higher: more conservative
# lower: less conservative

opti = cd.Opti()

# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-1,10), ylim=(-5,2))
ax.set_xlabel("X")
ax.set_ylabel("Y")


# Simulation Parameters
t = 0
tf = 8.0
dt = 0.05

robot = unicycle(ax, pos = np.array([0.0,0.5,0.0]), dt = dt)
obs = rectangle( ax, pos = np.array([1,0.5])  )

while t < tf:
    
    # Find barrier function value first
    opti = cd.Opti()
    A_r, b_r = robot.polytopic_location()
    A_o, b_o = obs.polytopic_location()
    y_o = opti.variable(2,1)
    y_r = opti.variable(2,1)
    const1 = cd.mtimes(A_r, y_r) <= b_r
    const2 = cd.mtimes(A_o, y_o) <= b_o
    opti.subject_to( const1 )
    opti.subject_to( const2 )
    dist_vec = y_o - y_r
    cost = cd.mtimes(dist_vec.T, dist_vec)
    opti.minimize(cost)
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", option)
    opt_sol = opti.solve()
    # minimum distance & dual variables
    h = opt_sol.value(cost) #opt_sol.value(cd.norm_2(dist_vec))
    if h > 0:
        lamb_r = opt_sol.value(opti.dual(const1)) #/ (2 * h)
        lamb_o = opt_sol.value(opti.dual(const2)) #/ (2 * h)
    else:
        lamb_r = np.zeros(shape=(2,))
        lamb_o = np.zeros(shape=(2,))
  
    opti = cd.Opti();
    # Solving CBF optimization problem
    lambda_o = opti.variable(1,4)
    lambda_r = opti.variable(1,4)
    alpha_cbf = opti.variable()
    U = opti.variable(2,1)
    X_next = opti.variable(3,1)
        
    # find control input
    U_ref = np.array([0.5,-0.7]).reshape(-1,1)
    Anext, bnext_r_f, bnext_r_g = robot.polytopic_location_next_state()
    
    U_error = U - U_ref
    objective = 10 * cd.mtimes( U_error.T, U_error ) + 100.0 * ( alpha_cbf - alpha_cbf_nominal )**2
    opti.minimize(objective)
    
    # Next state polytopic location
    Rot = cd.hcat(
        [
            cd.vcat(
                [
                    cd.cos(X_next[2,0]), cd.sin(X_next[2,0])
                ]
            ),
            cd.vcat(
                [
                    -cd.sin(X_next[2,0]), cd.cos(X_next[2,0])
                ]
            )
        ]
    )
  
    A_next = robot.A @ Rot
    b_next = cd.mtimes(cd.mtimes(robot.A, Rot), X_next[0:2]) + robot.b
    lambda_bound = max(1.0, 2*h)
    opti.subject_to( alpha_cbf >= 0 )
    opti.subject_to( alpha_cbf <= 0.99 )
    opti.subject_to(  - cd.mtimes(lambda_o, b_o) - cd.mtimes(lambda_r, b_next) >= alpha_cbf * h + h_offset )
    opti.subject_to(  cd.mtimes(lambda_o, A_o) + cd.mtimes(lambda_r, A_next) == 0  )
    temp = cd.mtimes( lambda_o, A_o )
    opti.subject_to(  cd.mtimes( temp, temp.T ) <= lambda_bound  )
    opti.subject_to( lambda_o >= 0 ) 
    opti.subject_to( lambda_r >= 0 )
    opti.subject_to(  X_next == robot.X + robot.f()*dt + cd.mtimes(robot.g()*dt, U) )
    
    opti.set_initial(lambda_o,lamb_o )
    opti.set_initial(lambda_r,lamb_r )
    
    # opti.solver('ipopt');
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti.solver("ipopt", option)
    sol = opti.solve();
    
    if h>1.32:
        print("Monitor from here")
    
    print(f" h:{h} alpha:{sol.value(alpha_cbf)}, U:{sol.value(U).T}, alpha*h :{alpha_cbf_nominal * h} ")
    # propagate dynamics
    robot.step(sol.value(U))
    robot.render_plot()
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    t = t + dt
    
    
    
    
    
    
    



