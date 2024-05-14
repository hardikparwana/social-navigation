import numpy as np
import jax
from jax import jit, grad
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer
import polytope as pt
import cvxpy as cp

def compute_polytope_from_points(points):
    hull = pt.qhull(points)
    return hull, hull.A, hull.b

def polytope_union( polytope_1, polytope_2 ):
    nonconvex = polytope_1.union(polytope_2)  # construct union of convex polytopes
    return nonconvex

def polytope_intersect( polytope_1, polytope_2 ):
    nonconvex = polytope_1.intersect(polytope_2)  # construct union of convex polytopes
    return nonconvex

def plot_polytope_lines(ax, hull, u_bound):
    xs = np.linspace( -u_bound, u_bound, 3 )
    A, b = hull.A, hull.b
    alpha = 0.1
    for i in range(A.shape[0]):
        if np.abs(A[i,1])>0.001:
            ax.plot( xs, (b[i] - A[i,0]*xs)/A[i,1], color='k', linestyle='--', alpha = alpha )
        else:
            if np.abs(A[i,0])>0.001:
                ax.axvline( b[i]/A[i,0], color='k', linestyle='--', alpha = alpha )
            else:
                ax.axvline( 0.0, color='k', linestyle='--', alpha = alpha )

# @jit
# def mc_polytope_volume(A, b, bounds = 30, num_samples=10000):
#     key = jax.random.PRNGKey(10)
#     samples = jax.random.uniform( key, shape=(2,num_samples), minval=-bounds, maxval=bounds )  
#     aux = A @ samples - b
#     aux = jnp.nonzero(jnp.all(aux < 0, 0))[0].shape[0]
#     vol = (2*bounds)**2 * aux / num_samples
#     return vol
# mc_polytope_volume_grad = grad( mc_polytope_volume, 0 )

# @jit
def mc_polytope_volume(A, b, lb = [-2, -2], ub=[2, 2], factor1=0.001, factor2=1.0, factor3=0.05):
    # Au<=b
    # print(f"A: {A}, b:{b}, lb: {lb}, ub: {ub}")
    Anorm = jnp.linalg.norm(A, axis=1)
    A = A / Anorm.reshape(-1,1)
    b = b / Anorm.reshape(-1,1)
    key = jax.random.PRNGKey(10)
    
    num_samples=50000#500000
    # samples = jax.random.uniform( key, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]   
    key, subkey = jax.random.split(key)
    samples_x = jax.random.uniform( subkey, shape=(1,num_samples), minval=lb[0], maxval=ub[0] )
    key, subkey = jax.random.split(key)
    samples_y = jax.random.uniform( subkey, shape=(1,num_samples), minval=lb[1], maxval=ub[1] )
    samples = jnp.append( samples_x, samples_y, axis=0 )

    aux = A @ samples - b    # supposed to be negative for being inside feasible space
    aux = -aux #- factor3 # now it should be > 0
    aux = jnp.min(aux, axis=0) - factor3 #- 0.05
    aux = (jnp.tanh( aux / factor1 ) + factor2)/2.0
    aux = jnp.sum( aux )
    # vol = ((2*bounds)**2) * (aux / num_samples)
    vol = (ub[0]-lb[0])*(ub[1]-lb[1]) * (aux / num_samples)
    # print(f"volume: {vol}")
    return vol
mc_polytope_volume_grad = jit(grad(mc_polytope_volume, 0))

@jit
def mc_polytope_volume_about_lines(A, b, samples, total_volume, factor1=0.001, factor2=1.0, factor3=0.05):
    Anorm = jnp.linalg.norm(A, axis=1)
    A = A / Anorm.reshape(-1,1)
    b = b / Anorm.reshape(-1,1)
    aux = A @ samples - b    
    aux = -aux
    aux = jnp.min(aux, axis=0) - factor3 #- 0.05
    aux = (jnp.tanh( aux / factor1 ) + factor2)/2.0
    aux = jnp.sum( aux )
    vol = total_volume * (aux / samples.shape[1]) #num_samples)
    return vol
mc_polytope_volume_about_lines_grad = jit(grad(mc_polytope_volume_about_lines, 1))

num_line_points = 50 #200 #50
num_normal_points = 30 #100 #30
increment = 0.0001 #0.001 # 0.0001
# @jit
def generate_points_about_line(pts): #, num_line_pts, num_normal_points, increment):
    
    xs = jnp.linspace(0, pts[1][0,0]-pts[0][0,0], num_line_points).reshape(1,-1)    

    if jnp.abs(pts[1][0,0]-pts[0][0,0])>0.001: # Not a vertical line
        slope = (pts[1][1,0]-pts[0][1,0])/(pts[1][0,0]-pts[0][0,0])    
        ys = pts[0][1,0] + slope * xs
        theta = jnp.arctan(slope)
    else: # vertical line
        ys = jnp.linspace( pts[0][1,0], pts[1][1,0], num_line_points ).reshape(1,-1)
        theta = np.pi/2
    line_pts = jnp.append( pts[0][0,0]+xs, ys, axis=0  )
    # Now N choose points above and below perpendicular to xs, ys    
    slope_n = jnp.pi/2 + theta    
    length = jnp.linspace(0, 2 * num_normal_points, num_normal_points, dtype=int) - num_normal_points
    length = length * increment
    steps = length * jnp.array([ [jnp.cos(slope_n)], [jnp.sin(slope_n)] ])
    temp_points = line_pts.reshape((2,1,line_pts.shape[1])) + steps.reshape( (2,steps.shape[1],1) )
    new_pts = temp_points.reshape( (2, num_normal_points*num_line_points) )
    volume = jnp.linalg.norm( pts[1]-pts[0] ) * num_normal_points * increment

    return new_pts, volume

def get_intersection_points(ai, bi, lb, ub):

    bpt = []
    
    if jnp.abs(ai[1])>0.01:
        uy =  (bi - ai[0] * lb[0]) / ai[1]
        if ((uy <= ub[1]) and (uy >= lb[1])):
            bpt.append( jnp.array([ lb[0], uy ]) )

    if jnp.abs(ai[1])>0.01:
        uy =  (bi - ai[0] * ub[0]) / ai[1]
        if ((uy <= ub[1]) and (uy >= lb[1])):
            bpt.append( jnp.array([ ub[0], uy ]) )

    if jnp.abs(ai[0])>0.01:
        ux =  (bi - ai[1] * lb[1]) / ai[0]
        if ((ux <= ub[0]) and (ux >= lb[0])):
            bpt.append( jnp.array([ ux, lb[1] ]) )

    if jnp.abs(ai[0])>0.01:
        ux =  (bi - ai[1] * ub[1]) / ai[0]
        if ((ux <= ub[0]) and (ux >= lb[0])):
            bpt.append( jnp.array([ ux, ub[1] ]) )
    
    # if len(bpt)!=2:
    #     print(f"ERROR: Should have been 2 points!")
        
    if len(bpt)>2:
        bpt = [bpt[0], bpt[-1]]

    return bpt

# def mc_polytope_volume_new(A, b, bounds):
#     key = jax.random.PRNGKey(10)
#     num_samples=500000
#     samples = jax.random.uniform( key, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]   

#     # now sample more on boundary
#     for i in range(A.shape[0]):
#         # sample on Au=b but inside the hull

#         # find intersection inside the square hull
#         p1, p2  = get_intersection_points()



#     return vol

# Formulate and solve the Ellipse problem
ellipse_n = 2
ellipse_num_planes = 4 + 5 + 4
ellipse_B = cp.Variable((ellipse_n,ellipse_n), symmetric=True)
ellipse_d = cp.Variable((ellipse_n,1))
ellipse_A = cp.Parameter((ellipse_num_planes,ellipse_n))
ellipse_b = cp.Parameter((ellipse_num_planes,1))
ellipse_objective = cp.Maximize( cp.log_det( ellipse_B ) )
ellipse_const = []
for ellipse_i in range( ellipse_A.shape[0] ):
    ellipse_const += [ cp.norm( ellipse_B @ ellipse_A[ellipse_i,:]) + ellipse_A[ellipse_i,:] @ ellipse_d <= ellipse_b[ellipse_i,0] ]
ellipse_prob = cp.Problem( ellipse_objective, ellipse_const )
print(f"Ellipse DCP: {ellipse_prob.is_dgp(dpp=True)}")# # dpp=True
# ellipse_prob.solve()

# outputs (ellipse_b, ellipse_D) ellipse 
ellipse_cvxpylayer = CvxpyLayer(ellipse_prob, parameters=[ellipse_A, ellipse_b], variables=[ellipse_B, ellipse_d])

# Formulate and solve the Circle problem
# circle_n = 2
# circle_num_planes = 4 + 5 + 4
# circle_r = cp.Variable()
# circle_c = cp.Variable((2,1))
# circle_A = cp.Parameter((circle_num_planes,2))
# circle_b = cp.Parameter((circle_num_planes,1))
# circle_objective = cp.Maximize(circle_r)
# circle_const = []
# for i in range( circle_A.shape[0] ):
#     circle_const += [ circle_A[i,:] @ circle_c + cp.norm(circle_A[i,:]) @ circle_r <= circle_b[i,0] ]
# circle_prob = cp.Problem( circle_objective, circle_const )
# # circle_prob.solve()
# circle_cvxpylayer = CvxpyLayer(circle_prob, parameters=[circle_A, circle_b], variables=[circle_r, circle_c])

circle_n = 2
circle_num_planes = 4 + 5 + 4
circle_r = cp.Variable()
circle_c = cp.Variable((2,1))
circle_A = cp.Parameter((circle_num_planes,circle_n))
circle_A_root = cp.Parameter(circle_num_planes)
circle_b = cp.Parameter((circle_num_planes,1))
circle_objective = cp.Maximize(circle_r)
circle_const = []
for i in range( circle_A.shape[0] ):
    circle_const += [ circle_A[i,:] @ circle_c + circle_A_root[i] * circle_r <= circle_b[i,0] ]
circle_prob = cp.Problem( circle_objective, circle_const )
# circle_prob.solve()
circle_cvxpylayer = CvxpyLayer(circle_prob, parameters=[circle_A, circle_A_root, circle_b], variables=[circle_r, circle_c])


# A, b = construct_barrier_from_states(jnp.asarray(robot.X), obstacle_states, jnp.asarray(humans.X), jnp.asarray(humans.controls) )
# A2.value = np.append( np.asarray(A), -control_bound_polytope.A, axis=0 )
# b2.value = np.append( np.asarray(b), -control_bound_polytope.b.reshape(-1,1), axis=0 )




# x = cp.Variable()
# a = cp.Parameter(value=3) 
# const = [x <= a]
# objective = cp.Maximize(x)
# prob = cp.Problem(objective, const)
# print(f"Simple DCP: {prob.is_dgp(dpp=True)}")
# prob.solve(requires_grad=True)
# prob.backward()
# print(f"x: {x.value}, a_grad:{a.gradient}")

# x = cp.Variable()
# p = cp.Parameter()
# quadratic = cp.square(x - 2 * p)
# problem = cp.Problem(cp.Minimize(quadratic))
# p.value = 3.
# problem.solve(requires_grad=True)
# problem.backward()
# print("The gradient is {0:0.1f}.".format(p.gradient))

