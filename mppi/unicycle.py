import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

def wrap_angle(theta):
    return np.arctan2( np.sin(theta), np.cos(theta) )

class Unicycle2D:
    
    def __init__(self,X0,dt,ax):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'Unicycle2D'
        
        self.X = X0.reshape(-1,1)
        self.X_nominal = np.copy(self.X)
        self.dt = dt
        
        self.U = np.array([0,0]).reshape(-1,1)
        
        self.radii = 1.0
        
        self.body = ax.scatter([],[],s=60,facecolors=self.color,edgecolors=self.color) #facecolors='none'
        self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])], color=self.color)
        self.render_plot()

    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [ np.cos(self.X[3,0]), 0],
                          [ np.sin(self.X[3,0]), 0],
                          [0, 1] ]) 
       

    def step(self,U): 
        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.X[2,0] = wrap_angle(self.X[2,0])
        return self.X
    
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0],self.X[2,0]])
        self.body._offsets3d = ([[x[0]],[x[1]],[x[2]]])
        self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[3,0])])
        self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[3,0])] )
        # self.axis[0].set_3d_properties( [self.X[2,0],self.X[2,0]] )

    def nominal_input(self,G, d_min = 0.3):
        G = np.copy(G.reshape(-1,1))
        k_omega = 2.0
        k_v = 4.0
        distance = max(np.linalg.norm( self.X[0:2,0]-G[0:2,0] ) - d_min,0)
        theta_d = np.arctan2(G[1,0]-self.X[1,0],G[0,0]-self.X[0,0])
        error_theta = wrap_angle( theta_d - self.X[2,0] )

        omega = k_omega*error_theta   
        v = k_v*( distance )*np.cos( error_theta )
        return np.array([v, omega]).reshape(-1,1)
    
    def sigma(self,s):
        k1 = 0.5 # 2.0
        k2 = 4.0 # 1.0
        return k2 * (np.exp(k1-s)-1)/(np.exp(k1-s)+1)
    
    def sigma_der(self,s):
        k1 = 0.5
        k2 = 4.0    
        return - k2 * np.exp(k1-s)/( 1+np.exp( k1-s ) ) * ( 1 - self.sigma(s)/k2 )
    
    def obstacle_barrier(self, agent, d_min):
        
        beta = 1.01
        theta = self.X[2,0]
        
        h = np.linalg.norm( self.X[0:3] - agent.X[0:3] )**2 - beta*d_min**2   
        s = ( self.X[0:3] - agent.X[0:3]).T @ np.array( [np.cos(theta),np.sin(theta),0] ).reshape(-1,1)
        h = h - self.sigma(s)
        
        der_sigma = self.sigma_der(s)
        dh_dx = np.append( 2*( self.X[0:3] - agent.X[0:3] ).T - der_sigma * ( np.array([ [np.cos(theta), np.sin(theta), 0] ]) ),  - der_sigma * ( -np.sin(theta)*( self.X[0,0]-agent.X[0,0] ) + np.cos(theta)*( self.X[1,0] - agent.X[1,0] ) ) , axis=1)
            
        return h, dh_dx
    

if 0:

    # Set figure
    plt.ion()
    fig = plt.figure()
    ax = plt.axes( xlim = (-2,2), ylim = (-2,2) )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect(1)
    
    dt = 0.01
    tf = 20
    num_steps = int(tf/dt)

    robot = Unicycle2D( np.array([0,0,0]).reshape(-1,1), dt, a)
    obsX = [0.5, 0.5]
    obs2X = [1.5, 1.9]
    targetX = np.array([1, 1]).reshape(-1,1)
    d_min = 0.3
    obs1 = circle2D(obsX[0], obsX[1], d_min, ax, 0)
    obs2 = circle2D(obs2X[0], obs2X[1], d_min, ax, 0)




    fig.canvas.draw()
    fig.canvas.flush_events()