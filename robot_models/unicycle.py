
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import polytope as pt

def wrap_angle(theta):
    return np.arctan2( np.sin(theta), np.cos(theta) )

class unicycle:
    
    def __init__(self, ax, pos = np.array([0,0,0]), dt = 0.01):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        id: robot id
        '''
        
        self.type = 'SingleIntegrator2D'        
        
        self.X0 = pos.reshape(-1,1)
        self.X = np.copy(self.X0)
        self.U = np.array([0,0]).reshape(-1,1)
        self.dt = dt
        self.ax = ax
        
        # Plot handles
        self.body = ax.scatter([],[],c='g',alpha=1.0,s=70, label='Robot')
        self.radii = 0.25
        self.axis = ax.plot([self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])],[self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
        self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def f_jax(self,X):
        return jnp.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [np.cos(self.X[2,0]), 0],[np.sin(self.X[2,0]), 0], [0, 1] ])
    
    def g_jax(self,X):
        return jnp.array([ [jnp.cos(self.X[2,0]), 0],[jnp.sin(self.X[2,0]), 0], [0, 1] ])
    
    def xdot_jax(self,X, U):
        return self.f_jax(X) + self.g_jax(X) @ U
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X

    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        self.body.set_offsets([x[0],x[1]])
        self.axis[0].set_ydata([self.X[1,0],self.X[1,0]+self.radii*np.sin(self.X[2,0])])
        self.axis[0].set_xdata( [self.X[0,0],self.X[0,0]+self.radii*np.cos(self.X[2,0])] )

    def nominal_controller(self, targetX):
        k_omega = 2.0 
        k_v = 0.1
        distance = np.linalg.norm( self.X[0:2]-targetX[0:2] )
        desired_heading = np.arctan2( targetX[1,0]-self.X[1,0], targetX[0,0]-self.X[0,0] )
        error_heading = wrap_angle( desired_heading - self.X[2,0] )

        omega = k_omega * error_heading
        v = k_v * distance * np.cos(error_heading)
        return np.array([v, omega]).reshape(-1,1)
 
        
        
        
    