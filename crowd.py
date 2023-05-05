import numpy as np
import matplotlib.pyplot as plt
import casadi as cd
import time

class crowd:
    
    def __init__(self, ax, crowd_center = np.array([0,0]), num_people = 10, dt = 0.01, horizon = 100, paths_file = []):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        self.num_people = num_people
        self.X0 = 40*( np.random.rand(2,num_people) - np.array([[0.5],[0.5]]) ) + crowd_center.reshape(-1,1)
        self.X = np.copy(self.X0)
        self.U = np.zeros((2,num_people))
        self.dt = dt
        self.horizon = horizon
        self.ax = ax
        self.t = 0
        self.counter = 0
        
        # Set goals for each human
        self.goals = -np.copy(self.X0)
        
        if paths_file != []:
            with open(paths_file,'rb') as f:
                self.paths = np.load(f)
            self.X = self.paths[0,0: self.num_people], self.paths[1,0: self.num_people] + 1.0
            
            # Animate trajectories
            self.body = self.ax.scatter(self.paths[0,0: self.num_people], self.paths[1,0: self.num_people],c='g',alpha=0.5,s=70)
            self.plot_counter = 1
        
    # def render_plot(self):
    #     self.body.set_offsets(self.paths[:,self.plot_counter*self.num_people: (self.plot_counter+1)*self.num_people].T)
    #     self.plot_counter = self.plot_counter + 1
        
    def current_position(self, t, dt):
        if t<=(self.horizon*self.dt):
            if t>=self.counter*self.dt:
                self.counter += 1
        counter = self.counter - 1
        # counter = int(1.001*t/self.dt)
        U = (self.paths[:,(counter+1)*self.num_people: (counter+2)*self.num_people] - self.paths[:,counter*self.num_people: (counter+1)*self.num_people])/self.dt
        self.X = self.X + U * dt
        # current_pos = self.paths[:,counter*self.num_people: (counter+1)*self.num_people] + U * (t-counter*self.dt)
        # print(f"t:{t}, counter:{counter}, change in time: {(t-counter*self.dt)}")
        # return current_pos
        return np.copy(self.X)
        
    def render_plot(self, current_pos):# t, dt):
        # counter = int(1.001*t/self.dt)
        # U = self.paths[:,(counter+1)*self.num_people: (counter+2)*self.num_people] - self.paths[:,counter*self.num_people: (counter+1)*self.num_people]
        # # U = self.paths[:,(counter+1)*self.num_people: (counter+2)*self.num_people] - self.paths[:,counter*self.num_people: (counter+1)*self.num_people]
        # current_pos = self.paths[:,counter*self.num_people: (counter+1)*self.num_people] + U * (t-counter*self.dt)
        self.body.set_offsets(current_pos.T)
        # self.plot_counter = self.plot_counter + 1
        
    
    def plan_paths(self, obstacles):
        # Use MPC to plan paths for all humans in centralized manner
        opti = cd.Opti()
        
        X = opti.variable(2,self.num_people*(self.horizon+1))
        U = opti.variable(2,self.num_people*self.horizon)
        
        # Initial state
        initial_state_error = X[:,0:self.num_people] -  self.X0
        
        for i in range(self.horizon):
            
            # Dynamics
            opti.subject_to( X[:,(i+1)*self.num_people:(i+2)*self.num_people] == X[:,i*self.num_people:(i+1)*self.num_people] + U[:,i*self.num_people:(i+1)*self.num_people]*self.dt )
            
            # Collision with obstacles: Incorrect. Needs to be chaanged
            # for j in range(len(obstacles)):
            #     A, b = obstacles[j].polytopic_location()
            #     opti.subject_to( cd.vec(cd.mtimes( A, X[:,i*self.num_people:(i+1)*self.num_people] )) <=  cd.vec(np.repeat(b, self.num_people, axis=1)) ) # Collision avoidance constraint
            
            for j in range(self.num_people):
                # Input constraints
                control_input = U[:,i*self.num_people+j]
                opti.subject_to( cd.mtimes( control_input.T, control_input ) <= 2 )
                
                # Collision avoidance with other people
                for k in range(j+1,self.num_people):
                    dist = X[:,i*self.num_people+j] - X[:,i*self.num_people+k]
                    opti.subject_to( cd.mtimes(dist.T, dist) >= 0.3 )
                    
        # Goal location
        final_state_error = X[:,(self.horizon)*self.num_people:(self.horizon+1)*self.num_people] - self.goals
        
        cost = 100*cd.norm_fro(initial_state_error)**2 + cd.norm_fro(final_state_error)**2
        opti.minimize(cost)
        option = {"verbose": True, "ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", option)
        sol = opti.solve();
        return sol.value(X)
        
if 0:    
    # Set Figure
    plt.ion()
    fig = plt.figure()
    ax = plt.axes(xlim=(-20,20), ylim=(-20,20))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    dt = 0.5
    tf = 20.0
    horizon = int(tf/dt)
    num_people = 20
    
    obstacles = []
    # obstacles.append( rectangle( ax, pos = np.array([1,0.5]) ) )
    # obstacles.append( rectangle( ax, pos = np.array([1,-1.5]) ) )

    humans = crowd(ax, crowd_center = np.array([0,0]), num_people = num_people, dt = dt, horizon = horizon)
    
    paths = humans.plan_paths(obstacles)
    
    # Save Data
    with open('paths_n20_tf20_v2.npy','wb') as f:
        np.save(f, paths)
    
    # Plot trajectory
    # for i in range(num_people):
    #     plt.plot(paths[0,i::num_people], paths[1,i::num_people])

    # Animate trajectories
    body = ax.scatter(paths[0,0: num_people], paths[1,0: num_people],c='g',alpha=0.5,s=70)
    for i in range(horizon-1):
        body.set_offsets(paths[:,i*num_people: (i+1)*num_people].T)
        time.sleep(0.1)
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.ioff()       
        
    plt.show()
    
# paths.npy
# dt = 0.5
# tf = 10.0
# horizon = int(tf/dt)
# num_people = 10

# paths2.npy
