import numpy as np
# import cvxpy as cp  
import casadi as cd
from holonomic_car import holonomic_car
from obstacles import rectangle
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from crowd import crowd

alpha_cbf_nominal1 = 0.2#0.2
alpha_cbf_nominal2 = 0.8
h_offset = 0.07#0.07
# higher: more conservative
# lower: less conservative

movie_name = 'social-navigation/videos/humans_2alphas_demo1.mp4'
paths_file = 'social-navigation/paths.npy'
# paths_file = 'social-navigation/paths_n20_tf40_v1.npy'

num_people = 10


# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-10,10), ylim=(-10,10))
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Simulation Parameters
t = 0
tf = 40.0
dt = 0.05
U_ref = np.array([-0.5,0.5, 0.0]).reshape(-1,1)
d_human = 0.5#0.5
mpc_horizon = 5

robot = holonomic_car(ax, pos = np.array([4.0,-4.0,0]), dt = dt, color = 'red', alpha_nominal = alpha_cbf_nominal1*np.ones(num_people), plot_label='less conservative')#2.5,-2.5,0
robot_nominal = holonomic_car(ax, pos = np.array([4.0,-4.0,0]), dt = dt, color='blue', alpha_nominal = alpha_cbf_nominal2*np.ones(num_people), plot_label='more conservative')#2.5,-2.5,0
obstacles = [1]
plt.legend(loc='upper right')
# h = [0, 0] # barrier functions
# obstacles.append( rectangle( ax, pos = np.array([0,0.5]) ) )
# obstacles.append( rectangle( ax, pos = np.array([0,-1.5]) ) )

h_curr_obstacles = [0, 0, 0, 0] # barrier functions
# obstacles.append( rectangle( ax, pos = np.array([0,0.5]), width = 2.5 ) )        
# obstacles.append( rectangle( ax, pos = np.array([-0.75,-2.0]), width = 4.0 ) )
# obstacles.append( rectangle( ax, pos = np.array([-1.28,2.0]), height = 4.0 ) )
# obstacles.append( rectangle( ax, pos = np.array([-3.2,1.0]), height = 7.0 ) )
# plt.show()
dt_human = 0.5 #0.2
tf_human = 10#40.0
horizon_human = int(tf_human/dt_human)
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt_human, horizon = horizon_human, paths_file = paths_file)#social-navigation/
h_curr_humans = np.zeros(num_people)

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

with writer.saving(fig, movie_name, 100): 
    
    ## BUILD MPC here #############################################################################################
    opti_mpc = cd.Opti();
    
    # Parameters to set inside time loop
    humans_state = opti_mpc.parameter(2,(mpc_horizon+1)*num_people)
    h_human = opti_mpc.parameter(num_people)
    h_obstacles = opti_mpc.parameter(len(obstacles))
    robot_current_state = opti_mpc.parameter( robot.X.shape[0],1 )
    robot_input_ref = opti_mpc.parameter(robot.U.shape[0], robot.U.shape[1])
    alpha_nominal_humans = opti_mpc.parameter(num_people)
    
    # Variables to solve for
    robot_states = opti_mpc.variable(robot.X.shape[0], mpc_horizon+1)
    robot_inputs = opti_mpc.variable(robot.U.shape[0], mpc_horizon)
    alpha_obstacle = opti_mpc.variable(len(obstacles))
    alpha_human = opti_mpc.variable(num_people)
    
    # alpha constraints
    opti_mpc.subject_to( alpha_human >= np.zeros(num_people) )
    opti_mpc.subject_to( alpha_human <= 0.99*np.ones(num_people) )
    opti_mpc.subject_to( alpha_obstacle >= np.zeros(len(obstacles)) )
    opti_mpc.subject_to( alpha_obstacle <= 0.99*np.ones(len(obstacles)) )
        
    ## Initial state constraint 
    opti_mpc.subject_to( robot_states[:,0] == robot_current_state )
    
    ## Time Loop
    objective  = 0.0
    for k in range(mpc_horizon+1): # +1 For loop over time horizon
        
        ################ Dynamics ##########################
        if (k < mpc_horizon):
            opti_mpc.subject_to(  robot_states[:,k+1] == robot_states[:,k] + robot.f_casadi(robot_states[:,k])*dt + cd.mtimes(robot.g_casadi(robot_states[:,k])*dt, robot_inputs[:,k]) )
            
            # current state-input contribution to objective ####
            U_error = robot_inputs[:,k] - robot_input_ref 
            objective += 10 * cd.mtimes( U_error.T, U_error )
        
        if (k > 0):
            ################ Collision avoidance with humans
            human_states_horizon = humans_state[0:2, k*num_people:(k+1)*num_people]
            for i in range(num_people):
                dist = robot_states[0:2,k] - human_states_horizon[0:2,i]  # take horizon step into account               
                h = cd.mtimes(dist.T , dist) - d_human**2
                if k>0:
                    opti_mpc.subject_to( h >= alpha_human[i]**    k * h_human[i] ) # CBF constraint # h_human is based on current state
                    opti_mpc.subject_to( h >= 0.0 ) # normal distance constraint   # 0.3
            
            ################ Collision avoidance with polytopic obstacles          
            lambda_o = opti_mpc.variable(len(obstacles),4)
            lambda_r = opti_mpc.variable(len(obstacles),4)
            
            # Robot Polytopic location at state at time k
            Rot = cd.hcat( [  
                cd.vcat( [ cd.cos(robot_states[2,k]), cd.sin(robot_states[2,k]) ] ),
                cd.vcat( [-cd.sin(robot_states[2,k]), cd.cos(robot_states[2,k]) ] )
                ] )        
            A_r = robot.A @ Rot
            b_r = cd.mtimes(cd.mtimes(robot.A, Rot), robot_states[0:2,k]) + robot.b
        
            
    # find control input ###############################          
    alpha_obstacle_diff = alpha_obstacle-alpha_cbf_nominal1*np.ones(len(obstacles))
    alpha_humans_diff = alpha_human-alpha_nominal_humans
    objective += 10.0 *(  cd.mtimes( alpha_obstacle_diff.T, alpha_obstacle_diff ) + cd.mtimes( alpha_humans_diff.T, alpha_humans_diff ) )                
    opti_mpc.minimize(objective)
        
    option_mpc = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti_mpc.solver("ipopt", option_mpc)
        
        ############################################################################################
    
    while t < tf:
        
        human_positions = humans.current_position(t, dt)
        human_future_positions = humans.get_future_states(t,dt,mpc_horizon)
        
        # Non-adaptive 
        for i in range(num_people):
            dist = robot_nominal.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
            h_curr_humans[i] = (dist.T @ dist - d_human**2)[0,0]
            h_curr_humans[i] = max(h_curr_humans[i], 0.01) # to account for numerical issues
        
        # Find control input
        opti_mpc.set_value(robot_current_state, robot_nominal.X)
        opti_mpc.set_value(humans_state, human_future_positions)
        opti_mpc.set_value(h_human, h_curr_humans)
        opti_mpc.set_value(robot_input_ref, U_ref)
        opti_mpc.set_value(alpha_nominal_humans, robot_nominal.alpha_nominal)
    
        # mpc_sol = opti_mpc.solve();
        try:
            mpc_sol = opti_mpc.solve();
        except Exception as e:
            print(e)
            u_temp = np.array([[100],[100],[0]])
            opti_mpc.set_value(robot_input_ref, u_temp)
            opti_mpc.set_initial( robot_inputs, np.repeat( u_temp, mpc_horizon, 1 ) ) 
            mpc_sol = opti_mpc.solve();
        
        robot_nominal.step(mpc_sol.value(robot_inputs[:,0]))
        # print(f"t: {t} U: {robot.U.T}, human_dist:{ np.min(h_curr_humans)}, obs_dist: {np.min(h_curr_obstacles)} alpha_human:{mpc_sol.value(alpha_human)}")
        robot_nominal.render_plot()
        
        # Adaptive
        for i in range(num_people):
            dist = robot.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
            h_curr_humans[i] = (dist.T @ dist - d_human**2)[0,0]
            h_curr_humans[i] = max(h_curr_humans[i], 0.01) # to account for numerical issues
        
        # Find control input
        opti_mpc.set_value(robot_current_state, robot.X)
        opti_mpc.set_value(humans_state, human_future_positions)
        opti_mpc.set_value(h_human, h_curr_humans)
        opti_mpc.set_value(robot_input_ref, U_ref)
        opti_mpc.set_value(alpha_nominal_humans, robot.alpha_nominal)
    
        # mpc_sol = opti_mpc.solve();
        try:
            mpc_sol = opti_mpc.solve();
        except Exception as e:
            print(e)
            u_temp = np.array([[100],[100],[0]])
            opti_mpc.set_value(robot_input_ref, u_temp)
            opti_mpc.set_initial( robot_inputs, np.repeat( u_temp, mpc_horizon, 1 ) ) 
            mpc_sol = opti_mpc.solve();
        robot.step(mpc_sol.value(robot_inputs[:,0]))
        # print(f"t: {t} U: {robot.U.T}, human_dist:{ np.min(h_curr_humans)}, obs_dist: {np.min(h_curr_obstacles)} alpha_human:{mpc_sol.value(alpha_human)}")
        robot.render_plot()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        
        humans.render_plot(human_positions)
        
        t = t + dt
        
        
    
    
    
    
    
# Lessons learned
# 1. initial stata may end up violating by a small negative number si ignore that check.
# 2. also when computing current h barriers, have them always > 0 otherwise if they are even small negative, it allows barriers to get more negative
# 3. NLP may show local infeasibility. this can be because
#     1. initial state violates constraints
#     2. the initial control input guess is very bad leading to solver infeasibility
#     3. the objective function reference might be bad
    
    


