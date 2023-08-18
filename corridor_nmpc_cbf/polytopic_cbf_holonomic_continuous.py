import numpy as np
import cvxpy as cp  
# import casadi as cd
from holonomic_car import holonomic_car
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

robot = holonomic_car(ax, pos = np.array([0.0,0.5,np.pi/3]), dt = dt)
obs = rectangle( ax, pos = np.array([1,0.5])  )

while t < tf:
    
    # Find barrier function value first
    A_r, b_r = robot.polytopic_location()
    A_o, b_o = obs.polytopic_location()
    y_o = cp.Variable((2,1))
    y_r = cp.Variable((2,1))
    const = []
    const += [A_r @ y_r <= b_r ]
    const += [A_o @ y_o <= b_o ]
    dist_vec = y_o - y_r
    objective = cp.Minimize( cp.sum_squares(dist_vec) )
    prob = cp.Problem( objective, const )
    prob.solve()
    # minimum distance & dual variables
    h = dist_vec.value.T @ dist_vec.value #opt_sol.value(cd.norm_2(dist_vec))
    if h > 0:
        lamb_r = const[0].dual_value
        lamb_o = const[1].dual_value
    else:
        lamb_r = np.zeros(shape=(2,))
        lamb_o = np.zeros(shape=(2,))

    # find control input
    U_ref = np.array([0.5,-0.3, 0.1]).reshape(-1,1)

    u = cp.Variable((3,1))
    lambda_r_dot = cp.Variable((1,4))
    lambda_o_dot = cp.Variable((1,4))
    alpha_u = 1.0
    epsilon1 = 0.05
    epsilon2 = 0.05
    M = 1000
    objective_u = cp.Minimize( cp.sum_squares( u - U_ref ) )
    const_u = []
    if lambda_r < epsilon2:
        const_u += [ lambda_r_dot >= 0 ]
    if lambda_o < epsilon2:
        const_u += [ lambda_o_dot >= 0 ]
    const_u += [ lambda_r_dot <= M ]
    const_u += [ lambda_o_dot <= M ]
    const_u += [ lambda_o_dot ]
    A_r_dot = Lf_A_r + Lg_A_r @ u
    A_o_dot = np.zeros((4,4))
    b_r_dot = 
    b_o_dot = np.zeros((2,1))
    const_u += [ lambda_r_dot @ A_r + lambda_r @ A_r_dot + lambda_o_dot @ A_o + lambda_o @ A_o_dot == 0 ]
    L_dot = 1/2 * lambda_r @ A_r @ A_r.T @ lambda_r_dot.T - 1/2 * lambda_r @ A_r @ A_r_dot.T @ lambda_r.T - lambda_r_dot @ b_r - lambda_r @ b_r_dot - lambda_o_dot @ b_o - lambda_o @ b_o_dot
    const_u += [ L_dot >= -alpha_u * (h - epsilon1**2) ]
    
    

    opti = cd.Opti();
    # Solving CBF optimization problem
    lambda_o = opti.variable(1,4)
    lambda_r = opti.variable(1,4)
    alpha_cbf = opti.variable()
    U = opti.variable(3,1)
    X_next = opti.variable(3,1)
        
    
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
    
    # opti.set_initial(lambda_o,lamb_o )
    # opti.set_initial(lambda_r,lamb_r )
    
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
    
    
    
    
    
    
    



