import numpy as np
# import cvxpy as cp  
import casadi as cd
from holonomic_car import holonomic_car
from obstacles import rectangle
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from crowd import crowd

alpha_cbf_nominal = 0.2
h_offset = 0.0#0.07
# higher: more conservative
# lower: less conservative

movie_name = 'holonomic_humans_take_test.mp4'

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
U_ref = np.array([-0.5,0.5, 0.0]).reshape(-1,1)
d_min = 1.0#0.5
mpc_horizon = 2     

robot = holonomic_car(ax, pos = np.array([2.5,-1.0,0]), dt = dt)#2.5,-2.5,0
obstacles = [1]
# h = [0, 0] # barrier functions
# obstacles.append( rectangle( ax, pos = np.array([0,0.5]) ) )
# obstacles.append( rectangle( ax, pos = np.array([0,-1.5]) ) )

h_curr_obstacles = [0, 0, 0, 0] # barrier functions
# obstacles.append( rectangle( ax, pos = np.array([0,0.5]), width = 2.5 ) )        
# obstacles.append( rectangle( ax, pos = np.array([-0.75,-2.0]), width = 4.0 ) )
# obstacles.append( rectangle( ax, pos = np.array([-1.28,2.0]), height = 4.0 ) )
# obstacles.append( rectangle( ax, pos = np.array([-3.2,1.0]), height = 7.0 ) )
# plt.show()
dt_human = 0.5
tf_human = 10.0
horizon_human = int(tf_human/dt_human)
num_people = 10
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt_human, horizon = horizon_human, paths_file = 'social-navigation/paths.npy')#social-navigation/
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
    for k in range(mpc_horizon+1): # For loop over time horizon
        
        ################ Collision avoidance with humans
        human_states_horizon = humans_state[0:2, k*num_people:(k+1)*num_people]

        for i in range(num_people):
            dist = robot_states[0:2,k] - human_states_horizon[0:2,i]  # take horizon step into account               
            h = cd.mtimes(dist.T , dist) - d_min**2
            # if k==0:
            #     opti_mpc.subject_to( h >= alpha_human[i]**k * h_human[i] - 0.1 ) # CBF constraint # h_human is based on current state
            # else:
            if k>0:
                opti_mpc.subject_to( h >= alpha_human[i]**k * h_human[i] )
            # opti_mpc.subject_to( h >= 0.0 ) # normal distance constraint   # 0.3
        
        ################ Collision avoidance with polytopic obstacles            
        # lambda_o = opti_mpc.variable(len(obstacles),4)
        # lambda_r = opti_mpc.variable(len(obstacles),4)
        
        # # Robot Polytopic location at state at time k
        # Rot = cd.hcat( [  
        #        cd.vcat( [ cd.cos(robot_states[2,k]), cd.sin(robot_states[2,k]) ] ),
        #        cd.vcat( [-cd.sin(robot_states[2,k]), cd.cos(robot_states[2,k]) ] )
        #       ] )        
        # A_r = robot.A @ Rot
        # b_r = cd.mtimes(cd.mtimes(robot.A, Rot), robot_states[0:2,k]) + robot.b
    
        # Form polytopic CBF constraints
        # for i in range(len(obstacles)):
        #     A_o, b_o = obstacles[i].polytopic_location()
        #     lambda_bound = cd.fmax(1.0, 2*h_obstacles[i])                
        #     opti_mpc.subject_to(  - cd.mtimes(lambda_o[i,:], b_o) - cd.mtimes(lambda_r[i,:], b_r) >= alpha_obstacle[i]**k * h_obstacles[i] + h_offset )
        #     opti_mpc.subject_to(  cd.mtimes(lambda_o[i,:], A_o) + cd.mtimes(lambda_r[i,:], A_r) == 0  )
        #     temp = cd.mtimes( lambda_o[i,:], A_o )
        #     opti_mpc.subject_to(  cd.mtimes( temp, temp.T ) <= lambda_bound  )
        #     opti_mpc.subject_to( lambda_o[i,:] >= 0 ) 
        #     opti_mpc.subject_to( lambda_r[i,:] >= 0 )
            
        ################ Dynamics ##########################
        if (k < mpc_horizon):
            opti_mpc.subject_to(  robot_states[:,k+1] == robot_states[:,k] + robot.f_casadi(robot_states[:,k])*dt + cd.mtimes(robot.g_casadi(robot_states[:,k])*dt, robot_inputs[:,k]) )
        
            # current state-input contribution to objective ####
            U_error = robot_inputs[:,k] - U_ref 
            objective += 10 * cd.mtimes( U_error.T, U_error )
            
    # find control input ###############################          
    alpha_obstacle_diff = alpha_obstacle-alpha_cbf_nominal*np.ones(len(obstacles))
    alpha_humans_diff = alpha_human-alpha_cbf_nominal*np.ones(num_people)
    objective += 10.0 *(  cd.mtimes( alpha_obstacle_diff.T, alpha_obstacle_diff ) + cd.mtimes( alpha_humans_diff.T, alpha_humans_diff ) )                
    opti_mpc.minimize(objective)
        
    option_mpc = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti_mpc.solver("ipopt", option_mpc)
        
        ############################################################################################
    
    
    ## opt for finding barrier function  ###################################################
    # opti_barrier = cd.Opti()
    
    # # Parameters
    # A_r_barrier = opti_barrier.parameter(4,2)
    # b_r_barrier = opti_barrier.parameter(4,1)
    # A_o_barrier = opti_barrier.parameter(4,2)
    # b_o_barrier = opti_barrier.parameter(4,1)
    
    # # Variables
    # y_o = opti_barrier.variable(2,1)
    # y_r = opti_barrier.variable(2,1)
    # const1 = cd.mtimes(A_r_barrier, y_r) <= b_r_barrier
    # const2 = cd.mtimes(A_o_barrier, y_o) <= b_o_barrier
    # opti_barrier.subject_to( const1 )
    # opti_barrier.subject_to( const2 )
    # dist_vec = y_o - y_r
    # cost_barrier = cd.mtimes(dist_vec.T, dist_vec)
    # opti_barrier.minimize(cost_barrier)
    # option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    # opti_barrier.solver("ipopt", option)
    ######################################################################################
    
    while t < tf:

        human_positions = humans.current_position(t, dt)
        human_future_positions = humans.get_future_states(t,dt,mpc_horizon)
 
        # Find barrier function value first
        # for i in range(len(obstacles)):
        #     A_r_temp, b_r_temp = robot.polytopic_location()
        #     A_o_temp, b_o_temp = obstacles[i].polytopic_location()
        #     opti_barrier.set_value( A_r_barrier, A_r_temp ); opti_barrier.set_value( b_r_barrier, b_r_temp ); opti_barrier.set_value( A_o_barrier, A_o_temp ); opti_barrier.set_value( b_o_barrier, b_o_temp )
        #     opt_sol = opti_barrier.solve()
        #     h_curr_obstacles[i] = opt_sol.value(cost_barrier) 
            
        for i in range(num_people):
            dist = robot.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
            h_curr_humans[i] = (dist.T @ dist - d_min**2)[0,0]
            h_curr_humans[i] = max(h_curr_humans[i], 0.01)
        if ( np.min(h_curr_humans)<=0 ):
            print(f"********************* ERROR *************************")
        # Find control input
        opti_mpc.set_value(robot_current_state, robot.X)
        opti_mpc.set_value(humans_state, human_future_positions)
        opti_mpc.set_value(h_human, h_curr_humans)
        # opti_mpc.set_value(h_obstacles, h_curr_obstacles)
    
        mpc_sol = opti_mpc.solve();
        
        robot.step(mpc_sol.value(robot_inputs[:,0]))
        print(f"t: {t} U: {robot.U.T}, human_dist:{ np.min(h_curr_humans) }")
        robot.render_plot()
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        
        humans.render_plot(human_positions)
        
        t = t + dt
        
        
    
    
    
    
    
    
    



