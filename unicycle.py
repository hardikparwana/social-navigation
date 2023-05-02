import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import polytope as pt

class single_integrator_square:
    
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
        
        self.width = 0.4
        self.height = 0.4
        self.A, self.b = self.base_polytopic_location()
        
        # Plot handles
        self.body = ax.scatter([],[],c='g',alpha=0.0,s=70)
        self.rect = Rectangle((self.X[0,0]-self.width/2,self.X[1,0]-self.height/2),self.width,self.height,linewidth = 1, edgecolor='k',facecolor='k')
        ax.add_patch(self.rect)
        self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [1, 0],[0, 1] ])
        
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
        self.rect.set_xy( (self.X[0,0]-self.width/2, self.X[1,0]-self.height/2) )
            
    def base_polytopic_location(self):
        x = np.array([0,0])
        points = np.array( [ [x[0]-self.width/2,x[1]-self.height/2], [x[0]+self.width/2,x[1]-self.height/2], [x[0]+self.width/2,x[1]+self.height/2], [x[0]-self.width/2,x[1]+self.height/2] ] )
        hull = pt.qhull(points)
        return hull.A, hull.b.reshape(-1,1)
    
    def polytopic_location(self):
        Rot = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
            ])
        # return self.A @ Rot.T, self.A @ Rot.T @ self.X[0:2].reshape(-1,1)+self.b
        return self.A @ Rot, self.A @ Rot @ self.X[0:2].reshape(-1,1)+self.b
    
    def polytopic_location_next_state(self):
        Rot = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
            ])
        Rot_dot = np.array([
            [0.0, 0.0],
            [0.0, 0.0]
            ])
    
        A, b = self.polytopic_location()
        
        b_f = np.copy(b)
        b_g = A @ Rot.T # to be multiplied with control input
        
        return A, b_f, b_g*self.dt

        
        
        
    