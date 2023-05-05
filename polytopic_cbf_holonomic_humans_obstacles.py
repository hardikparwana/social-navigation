import numpy as np
# import cvxpy as cp  
import casadi as cd
from holonomic_car import holonomic_car
from obstacles import rectangle
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from crowd import crowd

alpha_cbf_nominal = 0.2
h_offset = 0.07
# higher: more conservative
# lower: less conservative

movie_name = 'holonomic_humans_take_test.mp4'

opti = cd.Opti()

# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-10,10), ylim=(-10,10))
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Simulation Parameters
t = 0
tf = 15.0
dt = 0.05
U_ref = np.array([-0.5,0.5, 0.4]).reshape(-1,1)

robot = holonomic_car(ax, pos = np.array([2.5,-2.5,0]), dt = dt)
obstacles = []
# h = [0, 0] # barrier functions
# obstacles.append( rectangle( ax, pos = np.array([0,0.5]) ) )
# obstacles.append( rectangle( ax, pos = np.array([0,-1.5]) ) )

h = [0, 0, 0, 0] # barrier functions
obstacles.append( rectangle( ax, pos = np.array([0,0.5]), width = 2.5 ) )        
obstacles.append( rectangle( ax, pos = np.array([-0.75,-2.0]), width = 4.0 ) )
obstacles.append( rectangle( ax, pos = np.array([-1.28,2.0]), height = 4.0 ) )
obstacles.append( rectangle( ax, pos = np.array([-3.2,1.0]), height = 7.0 ) )
# plt.show()
dt_human = 0.5
tf_human = 10.0
horizon_human = int(tf_human/dt_human)
num_people = 10
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt_human, horizon = horizon_human, paths_file = 'paths.npy')#social-navigation/

human_frequency = int(dt_human/dt)
human_counter = 0

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

with writer.saving(fig, movie_name, 100): 
    human_positions = humans.current_position(t, dt)
    while t < tf:
        human_counter = int(1.001*t/dt_human)
        # if ( (t>0) and (t % dt_human < dt) ):
        #     human_counter += 1        
        
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
        alpha_human = opti.variable(num_people)
        U = opti.variable(3,1)
        X_next = opti.variable(3,1)
            
        # find control input        
        U_error = U - U_ref            
        objective = 10 * cd.mtimes( U_error.T, U_error ) + 100.0 *(  (alpha_cbf[0]-alpha_cbf_nominal)**2 + (alpha_cbf[1]-alpha_cbf_nominal)**2 )
        for i in range(num_people):
            objective += 100 * ( alpha_human[i]-alpha_cbf_nominal )**2
        
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
        
        human_positions = humans.current_position(t, dt)#humans.paths[:,human_counter*num_people: (human_counter+1)*num_people] # 2 x num_humans
        for i in range(num_people):
            dist = X_next[0:2] - human_positions[:,i] 
            
            # normal distance constraint        
            # opti.subject_to( cd.mtimes(dist.T, dist) >= 0.3 )
            
            # CBF constraint
            h_curr = np.linalg.norm(robot.X[0:2] - human_positions[:,i])**2 - 0.5**2
            print(f"i:{i}, h:{h_curr}")
            h_next = cd.mtimes(dist.T , dist) - 0.5**2
            opti.subject_to( h_next >= alpha_human[i] * h )
            opti.subject_to( alpha_human[i] >= 0 )
            
        opti.subject_to(  X_next == robot.X + robot.f()*dt + cd.mtimes(robot.g()*dt, U) )
        
        # opti.solver('ipopt');
        option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve();
        
        # print(f" h1:{h[0]}, h2:{h[1]} alpha:{sol.value(alpha_cbf[0])}, {sol.value(alpha_cbf[1])}, U:{sol.value(U).T}")
        # print(f" U:{sol.value(U).T}")
        # propagate dynamics
        robot.step(sol.value(U))
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        
        humans.render_plot(human_positions)
        
        t = t + dt
        
        
    
    
    
    
    
    
    



