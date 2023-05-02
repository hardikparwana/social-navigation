import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon
import polytope as pt

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
        
        self.width = 0.4
        self.height = 0.4
        self.A, self.b = self.base_polytopic_location()
        
        # Plot handles
        self.body = ax.scatter([],[],c='g',alpha=0.0,s=70)
        points = np.array( [ [-self.width/2,-self.height/2], [self.width/2,-self.height/2], [self.width/2,self.height/2], [-self.width/2,self.height/2] ] )  
        self.patch = Polygon( points, linewidth = 1, edgecolor='k',facecolor='k' )      
        ax.add_patch(self.patch)
        self.render_plot()
        self.Xs = np.copy(self.X)
        self.Us = np.copy(self.U)
        
    def f(self):
        return np.array([0,0,0]).reshape(-1,1)
    
    def g(self):
        return np.array([ [np.cos(self.X[2,0]), 0],[np.sin(self.X[2,0]), 0], [0, 1] ])
        
    def step(self,U): #Just holonomic X,T acceleration

        self.U = U.reshape(-1,1)
        self.X = self.X + ( self.f() + self.g() @ self.U )*self.dt
        self.render_plot()
        self.Xs = np.append(self.Xs,self.X,axis=1)
        self.Us = np.append(self.Us,self.U,axis=1)
        return self.X
    
    def rot_mat(self,theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
            ])
    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        theta = self.X[2,0]
        self.body.set_offsets([x[0],x[1]])
        points = np.array( [ [-self.width/2,-self.height/2], [self.width/2,-self.height/2], [self.width/2,self.height/2], [-self.width/2,self.height/2] ] )
        R = self.rot_mat(theta)
        points = (R @ points.T).T  + x     
        self.patch.set_xy( points )
            
    def base_polytopic_location(self):
        x = np.array([0,0])
        points = np.array( [ [x[0]-self.width/2,x[1]-self.height/2], [x[0]+self.width/2,x[1]-self.height/2], [x[0]+self.width/2,x[1]+self.height/2], [x[0]-self.width/2,x[1]+self.height/2] ] )
        hull = pt.qhull(points)
        return hull.A, hull.b.reshape(-1,1)
    
    def polytopic_location(self):
        theta = self.X[2,0]
        Rot = self.rot_mat(theta)
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

        
        
        
    