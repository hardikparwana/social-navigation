# first order barrier: works very bad!! mpc cannot find a solution
# let's try some other barrier: 2nd order barrier

import numpy as np
# import cvxpy as cp  
import casadi as cd
from bicycle import bicycle
import matplotlib.pyplot as plt
from obstacles import rectangle
from matplotlib.animation import FFMpegWriter
from crowd import crowd
from trust_utils import compute_trust

alpha_cbf_nominal_adaptive = 0.9#10.0#0.9#0.2#0.2 # red
alpha_cbf_nominal_fixed = 0.9#1.0#0.9            # blue
alpha_obstacle_nominal = 0.2
h_offset = 0.07#0.07
adapt_params = True

# Trust parameters
alpha_der_max = 2.0#0.5
h_min = 4.0 # 6.0 # 1.0
min_dist = 2.0 # 2.0 # 1.0

movie_name = 'social-navigation/Videos/bicycle_corridor_test.mp4'
paths_file = []#'social-navigation/paths.npy'
# paths_file = 'social-navigation/paths_n20_tf40_v1.npy'

# demo 1: 1st order CBF
# demo2: without h1>=0 constraint

num_people = 5

# Set Figure
plt.ion()
fig = plt.figure(figsize=(10,10))
ax = plt.axes(xlim=(-8,6), ylim=(-6,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")

obstacles = [] #[1]
h_curr_obstacles = [0, 0, 0, 0] # barrier functions
obstacles.append( rectangle( ax, pos = np.array([0,0.5]), width = 2.5 ) )        
obstacles.append( rectangle( ax, pos = np.array([-0.75,-3.0]), width = 4.0 ) )
obstacles.append( rectangle( ax, pos = np.array([-1.28,2.3]), height = 4.0 ) )
obstacles.append( rectangle( ax, pos = np.array([-3.2,1.0]), height = 7.0 ) )

# Simulation Parameters
t = 0
tf = 15.0
dt = 0.05#0.05
U_ref = np.array([2.0, 0.0]).reshape(-1,1)
control_bound = np.array([2000.0, 20.0]).reshape(-1,1) # works if control input bound very large
d_human = 0.3#0.5#0.5
mpc_horizon = 6
goal = np.array([-2.0, -1.3]).reshape(-1,1)

pos_init = np.array([2.0,-2.0,np.pi/2, 0])
# pos_init = np.array([4.0,-4.0,np.pi/2, 0]) # and kv = 0.2 and 
pos_init = np.array([2.0,-1.3,-np.pi, -0.1])
# pos_init = np.array([4.0,-1.3,-np.pi, -0.1])

robot_nominal = bicycle(ax, pos = pos_init, dt = dt, color='blue', alpha_nominal = alpha_cbf_nominal_fixed*np.ones(num_people), alpha_nominal_obstacles = alpha_obstacle_nominal, plot_label='Fixed Params')#2.5,-2.5,0
robot = bicycle(ax, pos = pos_init, dt = dt, color = 'red', alpha_nominal = alpha_cbf_nominal_adaptive*np.ones(num_people), alpha_nominal_obstacles = alpha_obstacle_nominal, plot_label='Adaptive')#2.5,-2.5,0

plt.legend(loc='upper right')

dt_human = 0.5 #0.2
tf_human = 10#40.0
horizon_human = int(tf_human/dt_human)
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt_human, horizon = horizon_human, paths_file = paths_file)#social-navigation/
h_curr_humans = np.zeros(num_people)

# hard code positions and speeds
humans.X[0,0] = -1.7; humans.X[1,0] = -1.5;
humans.X[0,1] = -1.7; humans.X[1,1] = -0.7#-1.0;
humans.X[0,2] = -2.2; humans.X[1,2] = -1.6;
humans.X[0,3] = -2.2; humans.X[1,3] = -0.6;
humans.X[0,4] = -2.2; humans.X[1,4] = -1.9;

humans.goals[0,0] =  4.0; humans.goals[1,0] = -1.5;
humans.goals[0,1] =  4.0; humans.goals[1,1] = -0.9#-1.0;
humans.goals[0,2] =  4.0; humans.goals[1,2] = -1.6;
humans.goals[0,3] =  4.0; humans.goals[1,3] = -0.6;
humans.goals[0,4] =  4.0; humans.goals[1,4] = -1.9;

humans.render_plot(humans.X)

# plt.show()
# exit()

humans.controls = np.zeros((2,num_people))
humans.controls[0,0] = 0.0; humans.controls[1,0] = 1.0;
humans.controls[0,1] = 0.0; humans.controls[1,1] = 0.5;
humans.controls[0,2] = 0.0; humans.controls[1,2] = 0.5;
humans.controls[0,3] = 0.0; humans.controls[1,3] = 1.0;

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

with writer.saving(fig, movie_name, 100): 
    
    ## BUILD MPC here #############################################################################################
    opti_mpc = cd.Opti()
    
    # Parameters to set inside time loop
    humans_state = opti_mpc.parameter(2,(mpc_horizon+1)*num_people)
    # humans_state_dot = opti_mpc.parameter(2,(mpc_horizon+1)*num_people)
    h_human = opti_mpc.parameter(num_people)
    h_obstacles = opti_mpc.parameter(len(obstacles))
    robot_current_state = opti_mpc.parameter( robot.X.shape[0],1 )
    robot_input_ref = opti_mpc.parameter(robot.U.shape[0], robot.U.shape[1])
    alpha_nominal_humans = opti_mpc.parameter(num_people)
    alpha_nominal_obstacles = opti_mpc.parameter(len(obstacles))
    
    # Variables to solve for
    robot_states = opti_mpc.variable(robot.X.shape[0], mpc_horizon+1)
    robot_inputs = opti_mpc.variable(robot.U.shape[0], mpc_horizon)
    alpha_human = opti_mpc.variable(num_people)
    alpha_obstacle = opti_mpc.variable(len(obstacles))
    alpha1_human = opti_mpc.variable(num_people)
    alpha2_human = opti_mpc.variable(num_people)
    
    # alpha constraints
    opti_mpc.subject_to( alpha_human >= np.zeros(num_people) )
    opti_mpc.subject_to( alpha_human <= 0.99*np.ones(num_people) )
    opti_mpc.subject_to( alpha1_human >= np.zeros(num_people) )
    opti_mpc.subject_to( alpha2_human >= np.zeros(num_people) )
        
    ## Initial state constraint 
    opti_mpc.subject_to( robot_states[:,0] == robot_current_state )
    
    ## Time Loop
    objective  = 0.0
    for k in range(mpc_horizon+1): # +1 For loop over time horizon
        
        ################ Dynamics ##########################
        if (k < mpc_horizon):
            opti_mpc.subject_to(  robot_states[:,k+1] == robot_states[:,k] + robot.f_casadi(robot_states[:,k])*dt + cd.mtimes(robot.g_casadi(robot_states[:,k])*dt, robot_inputs[:,k]) )
            opti_mpc.subject_to( robot_inputs[:,k] <= control_bound*np.ones((2,1)) )
            opti_mpc.subject_to( robot_inputs[:,k] >= -control_bound*np.ones((2,1)) )

            opti_mpc.subject_to( robot_states[3,k] >= -1.2)#-control_bound*np.ones((2,1)) )
            opti_mpc.subject_to( robot_states[3,k] <= 1.2)#control_bound*np.ones((2,1)) )
            # current state-input contribution to objective ####
            U_error = robot_inputs[:,k] - robot_input_ref 
            objective += 1 * cd.mtimes( U_error.T, U_error )

            goal_error = robot_states[0:2,k] - goal
            objective += 10*cd.mtimes( goal_error.T, goal_error )

        if 1:#(k > 0):
            ################ Collision avoidance with humans
            human_states_horizon = humans_state[0:2, k*num_people:(k+1)*num_people]
            # human_states_dot_horizon = humans_state_dot[0:2, k*num_people:(k+1)*num_people]

            if (k < mpc_horizon):
                humans_state_horizon_next = humans_state[0:2, (k+1)*num_people:(k+2)*num_people]
                humans_state_horizon_prev = humans_state[0:2, (k-1)*num_people:(k)*num_people]
                human_states_dot_horizon = (humans_state_horizon_next - human_states_horizon)/dt
            for i in range(2): # TODOs.. it fails with just 2 humans????
                a = 1.0
                dist = robot_states[0:2,k] - human_states_horizon[0:2,i]  # take horizon step into account  
                dist[0,0] = dist[0,0] / a
                h = cd.mtimes(dist.T , dist) - d_human**2

                if (k < mpc_horizon) and (k>0): 
                    print(f" k:{k}, i:{i} ")
                    # First order CBF condition
                    # opti_mpc.subject_to( h >= alpha_human[i]**k * h_human[i] ) # CBF constraint # h_human is based on current state
                    # opti_mpc.subject_to( h >= 1.0 * h_human[i] ) # CBF constraint # h_human is based on current state

                    dist_prev = robot_states[0:2,k-1] - humans_state_horizon_prev[0:2,i] 
                    h_prev = cd.mtimes(dist_prev.T , dist_prev) - d_human**2
                    opti_mpc.subject_to( h >= alpha_human[i] * h_prev )
                                                            
                    # Direct state constraint                                        
                    # opti_mpc.subject_to( h >= 0.0 ) # normal distance constraint   # 0.3

                    # Second order CBF condition in discrete time
                    # robot_state_dot = (robot_states[:,k+1] - robot_states[:,k])/dt
                    # robot_state_dot = robot.f_casadi(robot_states[:,k]) + cd.mtimes( robot.g_casadi(robot_states[:,k]), robot_inputs[:,k]  )
                    # dist_dot = robot_state_dot[0:2] - human_states_dot_horizon[0:2,i]
                    # dist_dot[0,0] = dist_dot[0,0] / a
                    # dist_ddot = (robot.f_xddot_casadi(robot_states[:,k]) + cd.mtimes(robot.g_xddot_casadi(robot_states[:,k]), robot_inputs[:,k] ))[0:2,0]
                    # dist_ddot[0,0] = dist_ddot[0,0]/a
                    # h_dot  = 2*cd.mtimes(dist.T, dist_dot )
                    # h_ddot = 2 * cd.mtimes( dist.T, dist_ddot ) + 2 * cd.mtimes( dist_dot.T, dist_dot )
                    # h1 = h_dot + alpha1_human[i]**k * h_human[i]
                    # h1_dot = h_ddot
                    # opti_mpc.subject_to( h1_dot >= - alpha2_human[i] * h1 )
                    # opti_mpc.subject_to( h1 >= 0 )

            ################ Collision avoidance with polytopic obstacles    
            if (k>0):      
                lambda_o = opti_mpc.variable(len(obstacles),4)
                lambda_r = opti_mpc.variable(len(obstacles),4)
                
                # Robot Polytopic location at state at time k
                Rot = cd.hcat( [  
                    cd.vcat( [ cd.cos(robot_states[2,k]), cd.sin(robot_states[2,k]) ] ),
                    cd.vcat( [-cd.sin(robot_states[2,k]), cd.cos(robot_states[2,k]) ] )
                    ] )        
                A_r = robot.A @ Rot
                b_r = cd.mtimes(cd.mtimes(robot.A, Rot), robot_states[0:2,k]) + robot.b
            
                # Form polytopic CBF constraints
                for i in range(len(obstacles)):
                    A_o, b_o = obstacles[i].polytopic_location()
                    lambda_bound = cd.fmax(1.0, 2*h_obstacles[i])                
                    opti_mpc.subject_to(  - cd.mtimes(lambda_o[i,:], b_o) - cd.mtimes(lambda_r[i,:], b_r) >= alpha_obstacle[i]**k * h_obstacles[i] + h_offset )
                    opti_mpc.subject_to(  cd.mtimes(lambda_o[i,:], A_o) + cd.mtimes(lambda_r[i,:], A_r) == 0  )
                    temp = cd.mtimes( lambda_o[i,:], A_o )
                    opti_mpc.subject_to(  cd.mtimes( temp, temp.T ) <= lambda_bound  )
                    opti_mpc.subject_to( lambda_o[i,:] >= 0 ) 
                    opti_mpc.subject_to( lambda_r[i,:] >= 0 )
            
            
    # find control input ###############################          
    alpha_obstacle_diff = alpha_obstacle-alpha_nominal_obstacles
    alpha_humans_diff = alpha_human-alpha_nominal_humans
    alpha1_humans_diff = alpha1_human-alpha_nominal_humans
    alpha2_humans_diff = alpha2_human-20*alpha_nominal_humans
    objective += 10.0 *(  cd.mtimes( alpha_humans_diff.T, alpha_humans_diff ) )  + 10.0 *(  cd.mtimes( alpha1_humans_diff.T, alpha1_humans_diff ) )  + 10.0 *(  cd.mtimes( alpha2_humans_diff.T, alpha2_humans_diff ) ) + 1.0 *(  cd.mtimes( alpha_obstacle_diff.T, alpha_obstacle_diff ) ) 
    # objective += 1.0 *(  cd.mtimes( alpha_humans_diff.T, alpha_humans_diff ) )  + 1.0 *(  cd.mtimes( alpha1_humans_diff.T, alpha1_humans_diff ) )  + 1.0 *(  cd.mtimes( alpha2_humans_diff.T, alpha2_humans_diff ) ) + 1.0 *(  cd.mtimes( alpha_obstacle_diff.T, alpha_obstacle_diff ) ) 
    opti_mpc.minimize(objective)
        
    option_mpc = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti_mpc.solver("ipopt", option_mpc)
        
        ############################################################################################

    ## opt for finding barrier function  ###################################################
    opti_barrier = cd.Opti()
    
    # Parameters
    A_r_barrier = opti_barrier.parameter(4,2)
    b_r_barrier = opti_barrier.parameter(4,1)
    A_o_barrier = opti_barrier.parameter(4,2)
    b_o_barrier = opti_barrier.parameter(4,1)
    
    # Variables
    y_o = opti_barrier.variable(2,1)
    y_r = opti_barrier.variable(2,1)
    const1 = cd.mtimes(A_r_barrier, y_r) <= b_r_barrier
    const2 = cd.mtimes(A_o_barrier, y_o) <= b_o_barrier
    opti_barrier.subject_to( const1 )
    opti_barrier.subject_to( const2 )
    dist_vec = y_o - y_r
    cost_barrier = cd.mtimes(dist_vec.T, dist_vec)
    opti_barrier.minimize(cost_barrier)
    option = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti_barrier.solver("ipopt", option)
    ######################################################################################

    nominal_input_prev = np.zeros((2,mpc_horizon))
    adaptive_input_prev = np.zeros((2,mpc_horizon))

    nominal_sim = True
    adaptive_sim = True

    max_speed = 0.5
    human_goals = np.copy(humans.goals)
    while t < tf:
        
        # Control Humans
        for i in range(num_people):
            force  = 0
            force += humans.attractive_potential( humans.X[:,i].reshape(-1,1), humans.goals[:,i].reshape(-1,1) )
            for j in range(num_people):
                if j==i:
                    continue
                force += humans.repulsive_potential( humans.X[:,i].reshape(-1,1), humans.X[:,i].reshape(-1,1) )
            if np.linalg.norm(force)>max_speed:
                force = force / np.linalg.norm(force) * max_speed
            humans.controls[0,i] = force[0,0]
            humans.controls[1,i] = force[1,0]

        human_positions = np.copy(humans.X)
        human_future_positions = humans.get_future_states_with_input(dt,mpc_horizon)
        human_speeds = np.copy(humans.controls)

        humans.step_using_controls(dt)

        # Adaptive / more conservative
        if nominal_sim:

            for i in range(len(obstacles)):
                A_r_temp, b_r_temp = robot_nominal.polytopic_location()
                A_o_temp, b_o_temp = obstacles[i].polytopic_location()
                opti_barrier.set_value( A_r_barrier, A_r_temp ); opti_barrier.set_value( b_r_barrier, b_r_temp ); opti_barrier.set_value( A_o_barrier, A_o_temp ); opti_barrier.set_value( b_o_barrier, b_o_temp )
                opt_sol = opti_barrier.solve()
                h_curr_obstacles[i] = opt_sol.value(cost_barrier) 
                h_curr_obstacles[i] = max( h_curr_obstacles[i], 0.01 )

            for i in range(num_people):

                dist = robot_nominal.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
                h_curr_humans[i] = (dist.T @ dist - d_human**2)[0,0]
                if h_curr_humans[i]<-0.01:
                    print(f"***************************** Adaptive safety violated ********************************")
                h_curr_humans[i] = max(h_curr_humans[i], 0.01) # to account for numerical issues

            # Find control input
            U_ref = robot_nominal.nominal_controller( goal )
            # U_ref = np.array([0.0,0.0]).reshape(-1,1)
            # print(f"U_ref:{ U_ref }")
            
            opti_mpc.set_value(robot_current_state, robot_nominal.X)
            opti_mpc.set_value(humans_state, human_future_positions)
            opti_mpc.set_value(h_human, h_curr_humans)
            opti_mpc.set_value(h_obstacles, h_curr_obstacles)
            opti_mpc.set_value(robot_input_ref, U_ref)
            opti_mpc.set_value(alpha_nominal_humans, robot_nominal.alpha_nominal)
            opti_mpc.set_value(alpha_nominal_obstacles, robot_nominal.alpha_nominal_obstacles)
        
            # mpc_sol = opti_mpc.solve();
            try:
                mpc_sol = opti_mpc.solve();
                robot_nominal.step(mpc_sol.value(robot_inputs[:,0]))
                # print(f" U_ref:{ U_ref.T }, U:{ robot_nominal.U.T } ")
            except Exception as e:
                print(e)
                print(f"********************************* Fixed: MPC Failed ********************************")
                nominal_sim = False
                # # exit()
                # u_temp = np.array([[-20],[0]])
                # opti_mpc.set_value(robot_input_ref, u_temp)
                # # print(f" Set ref value to : {} ")
                # opti_mpc.set_initial( robot_inputs, np.repeat( u_temp, mpc_horizon, 1 ) ) 
                # try:
                #     mpc_sol = opti_mpc.solve();
                #     robot_nominal.step(mpc_sol.value(robot_inputs[:,0]))
                # except Exception as e:
                    # print(f"********************************* Fixed: MPC Failed ********************************")
                #     # robot_nominal.step(mpc_sol.value(robot_inputs[:,0]))
                    # nominal_sim = False
                

                
        # Adaptive / more conservative
        if adaptive_sim:

            for i in range(len(obstacles)):
                A_r_temp, b_r_temp = robot.polytopic_location()
                A_o_temp, b_o_temp = obstacles[i].polytopic_location()
                opti_barrier.set_value( A_r_barrier, A_r_temp ); opti_barrier.set_value( b_r_barrier, b_r_temp ); opti_barrier.set_value( A_o_barrier, A_o_temp ); opti_barrier.set_value( b_o_barrier, b_o_temp )
                opt_sol = opti_barrier.solve()
                h_curr_obstacles[i] = opt_sol.value(cost_barrier) 
                h_curr_obstacles[i] = max( h_curr_obstacles[i], 0.01 )

            for i in range(num_people):

                dist = robot.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
                h_curr_humans[i] = (dist.T @ dist - d_human**2)[0,0]
                if h_curr_humans[i]<-0.01:
                    print(f"***************************** Adaptive safety violated ********************************")
                h_curr_humans[i] = max(h_curr_humans[i], 0.01) # to account for numerical issues

                if adapt_params:
                    # Do trust adaptation here: no best case right now. just use last velocity for now
                    dh_dx_robot = 2 * dist.T 
                    dh_dx_human = - 2 * dist.T
                    dx_dt_human = human_speeds[0:2,i]
                    dx_dt_robot = robot.U[0:2]

                    alpha = (1-robot.alpha_nominal[i])/dt

                    # dx_dt_human_nominal = dx_dt_human
                    dx_dt_human_nominal = (human_goals[0:2,i] - human_positions[0:2,i]).reshape(-1,1)

                    A = dh_dx_human
                    b = - alpha * h_curr_humans[i] - dh_dx_robot @ dx_dt_robot
                    trust, asserted = compute_trust( A, b, dx_dt_human, dx_dt_human_nominal, h_curr_humans[i], min_dist = min_dist, h_min = h_min )
                    alpha = max(0,alpha + alpha_der_max * trust)
                    robot.alpha_nominal[i] = max( 0, 1-alpha*dt )
                    print(f"alphas: {robot.alpha_nominal}")
                    print(f"alpha: { np.min(robot.alpha_nominal) }, {np.max( robot.alpha_nominal )}")
            
            # Find control input
            U_ref = robot.nominal_controller( goal )
            # U_ref = np.array([0.0,0.0]).reshape(-1,1)
            # print(f"U_ref:{ U_ref }")
            
            opti_mpc.set_value(robot_current_state, robot.X)
            opti_mpc.set_value(humans_state, human_future_positions)
            opti_mpc.set_value(h_human, h_curr_humans)
            opti_mpc.set_value(h_obstacles, h_curr_obstacles)
            opti_mpc.set_value(robot_input_ref, U_ref)
            opti_mpc.set_value(alpha_nominal_humans, robot.alpha_nominal)
            opti_mpc.set_value(alpha_nominal_obstacles, robot.alpha_nominal_obstacles)
        
            # mpc_sol = opti_mpc.solve();
            try:
                mpc_sol = opti_mpc.solve();
                robot.step(mpc_sol.value(robot_inputs[:,0]))
                # print(f" U_ref:{ U_ref.T }, U:{ robot.U.T } ")
            except Exception as e:
                print(e)
                print(f"********************************* Adaptive: MPC Failed ********************************")
                adaptive_sim = False
                # exit()
                # adaptive_sim = False
                # u_temp = np.array([[-20],[0]])
                # opti_mpc.set_value(robot_input_ref, u_temp)
                # # print(f" Set ref value to : {} ")
                # opti_mpc.set_initial( robot_inputs, np.repeat( u_temp, mpc_horizon, 1 ) ) 
                # try:
                #     mpc_sol = opti_mpc.solve();
                #     robot.step(mpc_sol.value(robot_inputs[:,0]))
                # except Exception as e:
                #     print(f"********************************* Adaptive: MPC Failed ********************************")
                #     # robot.step(mpc_sol.value(robot_inputs[:,0]))
                #     adaptive_sim = False
                
            
            # print(f"t: {t} U: {robot.U.T}, human_dist:{ np.min(h_curr_humans)}, obs_dist: {np.min(h_curr_obstacles)} alpha_human:{mpc_sol.value(alpha_human)}")
            robot.render_plot()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        
        # humans.render_plot(human_positions)
        humans.render_plot_trust(human_positions, robot.alpha_nominal)
        
        t = t + dt
        
    
# Lessons learned
# 1. initial stata may end up violating by a small negative number si ignore that check.
# 2. also when computing current h barriers, have them always > 0 otherwise if they are even small negative, it allows barriers to get more negative
# 3. NLP may show local infeasibility. this can be because
#     1. initial state violates constraints
#     2. the initial control input guess is very bad leading to solver infeasibility
#     3. the objective function reference might be bad
    

# initial state important
# with HOCBF, if horizon < 3, has no effect! not a valid formulation
# d_human and obstacle also very important
# alpha - alpha_nominal weight also important. most times it ends up choosing a value far away from aloha_nominal. always very relaxed to be able to go towards the goal
# sometimes it fails only because of bad U_ref -> local infeasibility fir IPOPT., it is stupid though
# MPC with h>=0 constraint only and long time horizon takes it in expected motion