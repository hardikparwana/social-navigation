import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cvxpy as cp
from matplotlib.animation import FFMpegWriter

class SingleIntegrator:

    def __init__(self,X0,dt,ax,color='r', alpha=1.0):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        '''

        self.X = X0.reshape(-1,1)
        self.dt = dt
        self.ax = ax
        self.color=color
        self.body = ax.scatter([],[],c=self.color,s=60, label=r'$\nu = $'+str(alpha))
        self.traj = ax.plot([],[],self.color)
        self.Xs = np.copy(self.X)
        ax.legend()

        self.render_plot()

    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        self.body.set_offsets([x[0],x[1]])
        self.traj.clear()
        self.traj = ax.plot( self.Xs[0,:], self.Xs[1,:], self.color )
        # self.traj.set_xdata(self.Xs[0,:])
        # self.traj.set_xdata(self.Xs[1,:])
    
    def f(self):
        return np.array([
            [0.0],
            [0.0]
            ]).reshape(-1,1)

    def g(self):
        return np.array([
            [ 1, 0 ],
            [ 0, 1 ]
        ])


    # Move the robot
    def step(self, U):
        self.X = self.X + ( self.f() + self.g() @ U ) * self.dt
        self.Xs = np.append(self.Xs, self.X, axis=1)

    def barrier(self,obstacleX,d_min):
        h = (self.X[0:2] - obstacleX[0:2]).T @ (self.X[0:2] - obstacleX[0:2]) - d_min**2
        dh_dx1 = 2 * (self.X[0:2] - obstacleX[0:2]).T
        dh_dx2 = - 2 * (2 * (self.X[0:2] - obstacleX[0:2]).T)
        return dh_dx1, dh_dx2, h

# Figure    
plt.ion()
fig = plt.figure()
ax = plt.axes( xlim = (-2,4), ylim = (-2,4) )
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

dt = 0.02
tf = 6.0
num_steps = int(tf/dt) #350

# CBF parameters
alpha1 = 1.0
alpha2 = 2.0
d_min = 0.5 # collision radius
goal = np.array([2.0, 2.0]).reshape(-1,1)
ax.scatter(goal[0,0], goal[1,0], c='c', label='Goal')

# robot
robot = SingleIntegrator( np.array([-1, -1]), dt, ax, alpha=alpha1 )
ax.plot( [robot.X[0,0], goal[0,0]], [robot.X[1,0], goal[1,0]], 'c--', alpha=0.4 )

# Obstacle location
obsX = np.array([0.7, 0.5]).reshape(-1,1)
circ = plt.Circle((obsX[0,0],obsX[1,0]),d_min,linewidth = 1, edgecolor='k',facecolor='k', alpha=0.7)
ax.add_patch(circ)

# Cvxpy controller
u = cp.Variable((2,1))
u_ref = cp.Parameter((2,1), value = np.zeros((2,1)))
A = cp.Parameter((1,2), value = np.zeros((1,2)))
b = cp.Parameter((1,1), value = np.zeros((1,1)))
objective = cp.Minimize( cp.sum_squares( u - u_ref )  ) 
const = [A @ u + b >= 0]
cbf_controller = cp.Problem( objective, const )
kx = 1.2

metadata = dict(title='Movie Test', artist='Matplotlib',comment='Movie support!')
writer = FFMpegWriter(fps=10, metadata=metadata)

with writer.saving(fig, 'Videos/example_si.mp4', 100): 
    
    for t in range(num_steps):

        # Desired input for double integrator
        u_desired = - kx * ( robot.X[0:2] - goal )
        u_ref.value = u_desired

        # CBF constraint: 
        dh_dx1, dh_dx2, h = robot.barrier( obsX, d_min )
        obstacle_speed = np.zeros((2,1))
        A.value = dh_dx1 @ robot.g()
        b.value = dh_dx1 @ robot.f() + dh_dx2 @ obstacle_speed + alpha1 * h

        cbf_controller.solve()
        if cbf_controller.status == 'infeasible':
                print(f"QP infeasible")

        robot.step(u.value)
        robot.render_plot()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        writer.grab_frame()
        
    robot2 = SingleIntegrator( np.array([-1, -1]), dt, ax, color='g', alpha=alpha2 )
    for t in range(num_steps):

        # Desired input for double integrator

        u_desired = - kx * ( robot2.X[0:2] - goal )
        u_ref.value = u_desired

        # CBF constraint: 
        dh_dx1, dh_dx2, h = robot2.barrier( obsX, d_min )
        obstacle_speed = np.zeros((2,1))
        A.value = dh_dx1 @ robot2.g()
        b.value = dh_dx1 @ robot2.f() + dh_dx2 @ obstacle_speed + alpha2 * h

        cbf_controller.solve()
        if cbf_controller.status == 'infeasible':
                print(f"QP infeasible")

        robot2.step(u.value)
        robot2.render_plot()
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        writer.grab_frame()


            