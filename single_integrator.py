
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class single_integrator:
    
    def __init__(self, ax, pos = np.array([0,0]), dt = 0.01):
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
        self.body = ax.scatter([],[],c='g',alpha=1.0,s=70)
        self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def f_jax(self,X):
        return jnp.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
    
    def g_jax(self,X):
        return jnp.array([ [1.0, 0.0], [0, 1] ])
    
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
        if self.plot_polytope:
            self.rect.set_xy( (self.X[0,0]-self.width/2, self.X[1,0]-self.height/2) )
            
    def nominal_controller(self, goal, k_x = 3.0):
        error = self.X - goal
        return - k_x * error
    
    def barrier(self, target, d_min = 0.5):
        h = (self.X - target.X).T @ (self.X - target.X) - d_min**2
        dh_dx1 = 2 * (self.X - target.X).T
        dh_dx2 = - 2 * ( self.X - target.X ).T
        return h, dh_dx1, dh_dx2
    
    def barrier_jax(self, X, targetX, d_min, alpha1):
        h = (X[0:2] - targetX[0:2]).T @ (X[0:2] - targetX[0:2]) - d_min**2
        dh_dx1 = 2 * (X[0:2] - targetX[0:2]).T
        dh_dx2 = - 2 * ( X[0:2] - targetX[0:2] ).T
        return h, dh_dx1, dh_dx2
    
    def barrier_humans(self, targetX, d_min = 0.5):
        h = (self.X - targetX).T @ (self.X - targetX) - d_min**2
        dh_dx1 = 2 * (self.X - targetX).T
        dh_dx2 = - 2 * ( self.X - targetX ).T
        return h, dh_dx1, dh_dx2
    
    def barrier_humans_jax(self, X, targetX, targetXdot, d_min = 0.5, alpha1=0.5):
        h = (X - targetX).T @ (X - targetX) - d_min**2
        dh_dx1 = 2 * (X - targetX).T
        dh_dx2 = - 2 * ( X - targetX ).T
        return h, dh_dx1, dh_dx2

        
        
        
    