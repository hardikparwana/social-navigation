import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

class rectangle:

    def __init__(self,x,y,width,height,ax,id):
        self.X = np.array([x,y]).reshape(-1,1)
        self.width = width
        self.height = height
        self.id = id
        self.ax = ax
        self.type = 'rectangle'
        
        self.rect = Rectangle((self.X[0,0],self.X[1,0]),self.width,self.height,linewidth = 1, edgecolor='k',facecolor='k')
        self.ax.add_patch(self.rect)
        
        self.render()

    def render(self):
        self.rect.set_xy((self.X[0,0],self.X[1,0]))      
        
    def polytope_location(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        points = np.array( [x[0]-self.width/2,x[1]-self.height/2], [x[0]+self.width/2,x[1]-self.height/2], [x[0]+self.width/2,x[1]+self.height/2], [x[0]-self.width/2,x[1]+self.height/2]  )
        hull = pt.qhull(points)
        return hull.A, hull.b.reshape(-1,1)


class circle:

    def __init__(self,x,y,radius,ax,id):
        self.X = np.array([x,y]).reshape(-1,1)
        self.radius = radius
        self.id = id
        self.type = 'circle'

        self.render(ax)

    def render(self,ax):
        circ = plt.Circle((self.X[0],self.X[1]),self.radius,linewidth = 1, edgecolor='k',facecolor='k')
        ax.add_patch(circ)

