import numpy as np
# import cvxpy as cp  
import casadi as cd
from holonomic_car import holonomic_car
from obstacles import rectangle
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

alpha_cbf_nominal = 0.9
h_offset = 0.05
# higher: more conservative
# lower: less conservative

movie_name = 'holonomic_take1.mp4'

opti = cd.Opti()

# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-1,4), ylim=(-3,2))
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Simulation Parameters
t = 0
tf = 8.0
dt = 0.05

robot = holonomic_car(ax, pos = np.array([0.0,0.5,np.pi/3]), dt = dt)
obstacles = []
h = [0, 0] # barrier functions
obstacles.append( rectangle( ax, pos = np.array([1,0.5]) ) )
obstacles.append( rectangle( ax, pos = np.array([1,-1.5]) ) )

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

with writer.saving(fig, movie_name, 100): 
    while t < tf:
        
        # Find barrier function value first
        for i in range(len(obstacles)):
            opti = cd.Opti()
            A_r, b_r = robot.polytopic_location()
            A_o, b_o = obstacles[i].polytopic_location()
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
            h[i] = opt_sol.value(cost) #opt_sol.value(cd.norm_2(dist_vec))
    
        opti = cd.Opti();
        # Solving CBF optimization problem
        lambda_o = opti.variable(len(obstacles),4)
        lambda_r = opti.variable(len(obstacles),4)
        alpha_cbf = opti.variable(len(obstacles))
        U = opti.variable(3,1)
        X_next = opti.variable(3,1)
            
        # find control input
        U_ref = np.array([0.5,-0.3, 0.1]).reshape(-1,1)
        Anext, bnext_r_f, bnext_r_g = robot.polytopic_location_next_state()
        
        U_error = U - U_ref
        objective = 10 * cd.mtimes( U_error.T, U_error ) + 100.0 *(  (alpha_cbf[0]-alpha_cbf_nominal)**2 + (alpha_cbf[1]-alpha_cbf_nominal)**2   )
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
        
        for i in range(len(obstacles)):
            A_r, b_r = robot.polytopic_location()
            A_o, b_o = obstacles[i].polytopic_location()
            lambda_bound = max(1.0, 2*h[i])
            opti.subject_to( alpha_cbf[i] >= 0 )
            opti.subject_to( alpha_cbf[i] <= 0.99 )
            opti.subject_to(  - cd.mtimes(lambda_o[i,:], b_o) - cd.mtimes(lambda_r[i,:], b_next) >= alpha_cbf[i] * h[i] + h_offset )
            opti.subject_to(  cd.mtimes(lambda_o[i,:], A_o) + cd.mtimes(lambda_r[i,:], A_next) == 0  )
            temp = cd.mtimes( lambda_o[i,:], A_o )
            opti.subject_to(  cd.mtimes( temp, temp.T ) <= lambda_bound  )
            opti.subject_to( lambda_o[i,:] >= 0 ) 
            opti.subject_to( lambda_r[i,:] >= 0 )
        opti.subject_to(  X_next == robot.X + robot.f()*dt + cd.mtimes(robot.g()*dt, U) )
        
        # opti.solver('ipopt');
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve();
        
        print(f" h1:{h[0]}, h2:{h[1]} alpha:{sol.value(alpha_cbf[0])}, {sol.value(alpha_cbf[1])}, U:{sol.value(U).T}")
        # propagate dynamics
        robot.step(sol.value(U))
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        
        t = t + dt
    
    
    
    
    
    
    



