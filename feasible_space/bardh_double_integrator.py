import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cvxpy as cp

class DoubleIntegrator:

    def __init__(self,X0,dt,ax):
        '''
        X0: iniytial state
        dt: simulation time step
        ax: plot axis handle
        '''

        self.X = X0.reshape(-1,1)
        self.dt = dt
        self.ax = ax
        self.body = ax.scatter([],[],c='r',s=60)
        self.render_plot()

    def render_plot(self):
        x = np.array([self.X[0,0],self.X[1,0]])
        self.body.set_offsets([x[0],x[1]])
    
    def f(self):
        return np.array([
            [self.X[2,0]],
            [self.X[3,0]],
            [0.0],
            [0.0]
            ]).reshape(-1,1)

    def g(self):
        return np.array([
            [ 0, 0 ],
            [ 0, 0 ],
            [ 1, 0 ],
            [ 0, 1 ]
        ])
    
    def df_dx(self):
        return np.array([
            [ 0, 0, 1, 0 ],
            [ 0, 0, 0, 1 ],
            [ 0, 0, 0, 0 ],
            [ 0, 0, 0, 0 ]
        ])

    # Move the robot
    def step(self, U):
        self.X = self.X + ( self.f() + self.g() @ U ) * self.dt

    def barrier(self,obstacleX,d_min):
        h = (self.X[0:2] - obstacleX[0:2]).T @ (self.X[0:2] - obstacleX[0:2]) - d_min**2
        h_dot = 2 * (self.X[0:2] - obstacleX[0:2]).T @ ( self.f()[0:2] )
        df_dx = self.df_dx()
        dh_dot_dx1 = np.append( ( self.f()[0:2] - obstacleX[0:2] ).T, np.array([[0,0]]), axis = 1 ) + 2 * ( self.X[0:2] - obstacleX[0:2] ).T @ df_dx[0:2,:]
        dh_dot_dx2 = - 2 * ( self.f()[0:2].T -obstacleX[0:2].T )
        return dh_dot_dx1, dh_dot_dx2, h_dot, h

# Figure    
plt.ion()
fig = plt.figure()
ax = plt.axes( xlim = (-2,4), ylim = (-2,4) )
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect(1)

dt = 0.02
tf = 20
num_steps = int(tf/dt)

# robot
robot = DoubleIntegrator( np.array([-1, -1, 0, 0]), dt, ax )

# CBF parameters
alpha1 = 1.0
alpha2 = 4.0
d_min = 0.5 # collision radius
goal = np.array([2.0, 2.0]).reshape(-1,1)
ax.scatter(goal[0,0], goal[1,0], c='g', label='Goal')

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

for t in range(num_steps):

    # Desired input for double integrator
    kx = 0.7
    kv = 2.0
    velocity_desired = - kx * ( robot.X[0:2] - goal )
    u_desired = - kv * ( robot.X[2:4] - velocity_desired )
    u_ref.value = u_desired

    # CBF constraint: 
    dh_dot_dx1, dh_dot_dx2, h_dot, h = robot.barrier( obsX, d_min )
    obstacle_speed = np.zeros((2,1))
    A.value = dh_dot_dx1 @ robot.g()
    b.value = dh_dot_dx1 @ robot.f() + dh_dot_dx2 @ obstacle_speed + (alpha1 + alpha2)*h_dot + alpha1 * alpha2 * h

    cbf_controller.solve()
    if cbf_controller.status == 'infeasible':
            print(f"QP infeasible")

    robot.step(u.value)
    robot.render_plot()
    
    fig.canvas.draw()
    fig.canvas.flush_events()


        