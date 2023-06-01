# first order barrier: works very bad!! mpc cannot find a solution
# let's try some other barrier: 2nd order barrier

import numpy as np
# import cvxpy as cp  
import casadi as cd
from bicycle import bicycle
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from crowd import crowd
from trust_utils import compute_trust

alpha_cbf_nominal_adaptive = 0.9#0.2#0.2 # red
alpha_cbf_nominal_fixed = 0.9            # blue
alpha_obstacle_nominal = 0.2
h_offset = 0.07#0.07
adapt_params = True
# higher: more conservative in Discrete time
# lower: less conservative in Discrete time

# Trust parameters
alpha_der_max = 2.0#0.5
h_min = 10.0 # 6.0 # 1.0
min_dist = 3.0 # 2.0 # 1.0

movie_name = 'social-navigation/Videos/bicycle_humans_trust_2_6_10.mp4'
paths_file = 'social-navigation/paths.npy'
# paths_file = 'social-navigation/paths_n20_tf40_v1.npy'

# demo 1: 1st order CBF
# demo2: without h1>=0 constraint

num_people = 10

# Set Figure
plt.ion()
fig = plt.figure()
ax = plt.axes(xlim=(-8,6), ylim=(-6,8))
ax.set_xlabel("X")
ax.set_ylabel("Y")

# Simulation Parameters
t = 0
tf = 15.0
dt = 0.05#0.05
U_ref = np.array([2.0, 0.0]).reshape(-1,1)
control_bound = np.array([2000.0, 2000.0]).reshape(-1,1) # works if control input bound very large
d_human = 0.5#0.5
mpc_horizon = 6
goal = np.array([-7.5, 7.5]).reshape(-1,1)

pos_init = np.array([2.0,-2.0,np.pi/2, 0])
# pos_init = np.array([4.0,-4.0,np.pi/2, 0]) # and kv = 0.2 and 

robot_nominal = bicycle(ax, pos = pos_init, dt = dt, color='blue', alpha_nominal = alpha_cbf_nominal_fixed*np.ones(num_people), plot_label='Fixed Params')#2.5,-2.5,0
robot = bicycle(ax, pos = pos_init, dt = dt, color = 'red', alpha_nominal = alpha_cbf_nominal_adaptive*np.ones(num_people), plot_label='Adaptive')#2.5,-2.5,0

plt.legend(loc='upper right')

dt_human = 0.5 #0.2
tf_human = 10#40.0
horizon_human = int(tf_human/dt_human)
humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt_human, horizon = horizon_human, paths_file = paths_file)#social-navigation/
h_curr_humans = np.zeros(num_people)

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

with writer.saving(fig, movie_name, 100): 
    
    ## BUILD MPC here #############################################################################################
    opti_mpc = cd.Opti()
    
    # Parameters to set inside time loop
    humans_state = opti_mpc.parameter(2,(mpc_horizon+1)*num_people)
    # humans_state_dot = opti_mpc.parameter(2,(mpc_horizon+1)*num_people)
    h_human = opti_mpc.parameter(num_people)
    robot_current_state = opti_mpc.parameter( robot.X.shape[0],1 )
    robot_input_ref = opti_mpc.parameter(robot.U.shape[0], robot.U.shape[1])
    alpha_nominal_humans = opti_mpc.parameter(num_people)
    
    # Variables to solve for
    robot_states = opti_mpc.variable(robot.X.shape[0], mpc_horizon+1)
    robot_inputs = opti_mpc.variable(robot.U.shape[0], mpc_horizon)
    alpha_human = opti_mpc.variable(num_people)
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
            opti_mpc.subject_to( robot_inputs[:,k] <= control_bound )
            opti_mpc.subject_to( robot_inputs[:,k] >= -control_bound )
            # current state-input contribution to objective ####
            U_error = robot_inputs[:,k] - robot_input_ref 
            objective += 10 * cd.mtimes( U_error.T, U_error )
        
        if 1:#(k > 0):
            ################ Collision avoidance with humans
            human_states_horizon = humans_state[0:2, k*num_people:(k+1)*num_people]
            # human_states_dot_horizon = humans_state_dot[0:2, k*num_people:(k+1)*num_people]

            if (k < mpc_horizon):
                humans_state_horizon_next = humans_state[0:2, (k+1)*num_people:(k+2)*num_people]
                human_states_dot_horizon = (humans_state_horizon_next - human_states_horizon)/dt
            for i in range(num_people):
                dist = robot_states[0:2,k] - human_states_horizon[0:2,i]  # take horizon step into account               
                h = cd.mtimes(dist.T , dist) - d_human**2
                if (k < mpc_horizon) and (k>0): 

                    # First order CBF condition
                     opti_mpc.subject_to( h >= alpha_human[i]**k * h_human[i] ) # CBF constraint # h_human is based on current state
                    
                    # # Second order CBF condition in continuous time
                    # robot_state_dot = robot.f_casadi(robot_states[:,k]) + cd.mtimes( robot.g_casadi(robot_states[:,k]), robot_inputs[:,k]  )
                    # dist_dot = robot_state_dot[0:2] - human_states_dot_horizon[0:2,i]
                    # dist_ddot = (robot.f_xddot_casadi(robot_states[:,k]) + cd.mtimes(robot.g_xddot_casadi(robot_states[:,k]), robot_inputs[:,k] ))[0:2,0]
                    # h_dot  = 2*cd.mtimes(dist.T, dist_dot )
                    # h_ddot = 2 * cd.mtimes( dist.T, dist_ddot ) + 2 * cd.mtimes( dist_dot.T, dist_dot )
                    # h1 = h_dot + alpha1_human[i]**k * h_human[i]
                    # opti_mpc.subject_to( alpha1_human[i] == alpha_nominal_humans[i] )
                    # h1_dot = h_ddot
                    # opti_mpc.subject_to( h1_dot >= - alpha2_human[i] * h1 )
                    
                    # if (k>0):
                    #     opti_mpc.subject_to( h1 >= 0 )

                    # # Second order CBF condition in discrete time
                    # robot_state_dot = (robot_states[:,k+1] - robot_states[:,k])/dt
                    # dist_dot = robot_state_dot[0:2] - human_states_dot_horizon[0:2,i]
                    # dist_ddot = (robot.f_xddot_casadi(robot_states[:,k]) + cd.mtimes(robot.g_xddot_casadi(robot_states[:,k]), robot_inputs[:,k] ))[0:2,0]
                    # h_dot  = 2*cd.mtimes(dist.T, dist_dot )
                    # h_ddot = 2 * cd.mtimes( dist.T, dist_ddot ) + 2 * cd.mtimes( dist_dot.T, dist_dot )
                    # h1 = h_dot + alpha1_human[i]**k * h_human[i]
                    # h1_dot = h_ddot
                    # opti_mpc.subject_to( h1_dot >= - alpha2_human[i] * h1 )

                    # opti_mpc.subject_to( h1 >= 0 )
                                        
                    # Direct state constraint                                        
                    # opti_mpc.subject_to( h >= 0.0 ) # normal distance constraint   # 0.3
        
            
    # find control input ###############################          
    alpha_humans_diff = alpha_human-alpha_nominal_humans
    alpha1_humans_diff = alpha1_human-alpha_nominal_humans
    alpha2_humans_diff = alpha2_human-30*alpha_nominal_humans
    objective += 1.0 *(  cd.mtimes( alpha_humans_diff.T, alpha_humans_diff ) )  + 1.0 *(  cd.mtimes( alpha1_humans_diff.T, alpha1_humans_diff ) )  + 1.0 *(  cd.mtimes( alpha2_humans_diff.T, alpha2_humans_diff ) ) 
    opti_mpc.minimize(objective)
        
    option_mpc = {"verbose": False, "ipopt.print_level": 0, "print_time": 0}
    opti_mpc.solver("ipopt", option_mpc)
        
        ############################################################################################
    nominal_input_prev = np.zeros((2,mpc_horizon))
    adaptive_input_prev = np.zeros((2,mpc_horizon))
    human_position_prev = humans.current_position(t, dt)

    nominal_sim = True
    adaptive_sim = True

    human_goals = humans.paths[:,-humans.num_people:]
    while t < tf:
        
        human_positions = humans.current_position(t, dt)
        human_speeds = (human_positions - human_position_prev)/dt
        human_future_positions = humans.get_future_states(t,dt,mpc_horizon)
        
        if nominal_sim:

            # Non-adaptive 
            for i in range(num_people):
                dist = robot_nominal.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
                h_curr_humans[i] = (dist.T @ dist - d_human**2)[0,0]
                if h_curr_humans[i]<-0.01:
                    print(f"Nominal safety violated")
                h_curr_humans[i] = max(h_curr_humans[i], 0.01) # to account for numerical issues
            
            # Find control input
            U_ref = robot_nominal.nominal_controller( goal )
            opti_mpc.set_value(robot_current_state, robot_nominal.X)
            opti_mpc.set_value(humans_state, human_future_positions)
            # opti_mpc.set_value(humans_state_dot, human_speeds)
            opti_mpc.set_value(h_human, h_curr_humans)
            opti_mpc.set_value(robot_input_ref, U_ref)
            opti_mpc.set_value(alpha_nominal_humans, robot_nominal.alpha_nominal)
        
            # mpc_sol = opti_mpc.solve();
            try:
                mpc_sol = opti_mpc.solve();
                robot_nominal.step(mpc_sol.value(robot_inputs[:,0]))
            except Exception as e:
                print(e)
                u_temp = np.array([[-10000],[0]])
                opti_mpc.set_value(robot_input_ref, u_temp)
                opti_mpc.set_initial( robot_inputs, np.repeat( u_temp, mpc_horizon, 1 ) ) 
                try:
                    mpc_sol = opti_mpc.solve();
                    robot_nominal.step(mpc_sol.value(robot_inputs[:,0]))
                except Exception as e:
                    print(f"************************** Nominal: MPC failed ********************************")
                    nominal_sim = False
            
            # print(f"t: {t} U: {robot.U.T}, human_dist:{ np.min(h_curr_humans)}, obs_dist: {np.min(h_curr_obstacles)} alpha_human:{mpc_sol.value(alpha_human)}")
            robot_nominal.render_plot()
        
        # Adaptive / more conservative
        if adaptive_sim:
            for i in range(num_people):

                dist = robot.X[0:2] - human_positions[0:2,i].reshape(-1,1)                 
                h_curr_humans[i] = (dist.T @ dist - d_human**2)[0,0]
                if h_curr_humans[i]<-0.01:
                    print(f"Adaptive safety violated")
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
                    print(f"alpha: { np.min(robot.alpha_nominal) }, {np.max( robot.alpha_nominal )}")
        
            
            # Find control input
            U_ref = robot.nominal_controller( goal )
            opti_mpc.set_value(robot_current_state, robot.X)
            opti_mpc.set_value(humans_state, human_future_positions)
            opti_mpc.set_value(h_human, h_curr_humans)
            opti_mpc.set_value(robot_input_ref, U_ref)
            opti_mpc.set_value(alpha_nominal_humans, robot.alpha_nominal)
        
            # mpc_sol = opti_mpc.solve();
            try:
                mpc_sol = opti_mpc.solve();
                robot.step(mpc_sol.value(robot_inputs[:,0]))
            except Exception as e:
                print(e)
                u_temp = np.array([[-100],[0]])
                opti_mpc.set_value(robot_input_ref, u_temp)
                opti_mpc.set_initial( robot_inputs, np.repeat( u_temp, mpc_horizon, 1 ) ) 
                try:
                    mpc_sol = opti_mpc.solve();
                    robot.step(mpc_sol.value(robot_inputs[:,0]))
                except Exception as e:
                    print(f"********************************* Adaptive: MPC Failed ********************************")
                    adaptive_sim = False
                
            
            # print(f"t: {t} U: {robot.U.T}, human_dist:{ np.min(h_curr_humans)}, obs_dist: {np.min(h_curr_obstacles)} alpha_human:{mpc_sol.value(alpha_human)}")
            robot.render_plot()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        writer.grab_frame()
        
        # humans.render_plot(human_positions)
        humans.render_plot_trust(human_positions, robot.alpha_nominal)
        human_position_prev = np.copy(human_positions)
        
        t = t + dt
        
        
    
    
    
    
    
# Lessons learned
# 1. initial stata may end up violating by a small negative number si ignore that check.
# 2. also when computing current h barriers, have them always > 0 otherwise if they are even small negative, it allows barriers to get more negative
# 3. NLP may show local infeasibility. this can be because
#     1. initial state violates constraints
#     2. the initial control input guess is very bad leading to solver infeasibility
#     3. the objective function reference might be bad
    
    



