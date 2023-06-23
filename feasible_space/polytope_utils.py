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

def polytope_intersect( polytope_1, polytope_2 ):
    nonconvex = polytope_1.intersect(polytope_2)  # construct union of convex polytopes

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
                ax.vline( 0.0, color='k', linestyle='--', alpha = alpha )

# @jit
# def mc_polytope_volume(A, b, bounds = 30, num_samples=10000):
#     key = jax.random.PRNGKey(10)
#     samples = jax.random.uniform( key, shape=(2,num_samples), minval=-bounds, maxval=bounds )  
#     aux = A @ samples - b
#     aux = jnp.nonzero(jnp.all(aux < 0, 0))[0].shape[0]
#     vol = (2*bounds)**2 * aux / num_samples
#     return vol
# mc_polytope_volume_grad = grad( mc_polytope_volume, 0 )

def mc_polytope_volume(A, b, bounds = 30):
    key = jax.random.PRNGKey(10)
    num_samples=50000
    samples = jax.random.uniform( key, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]   
    aux = A @ samples - b    
    aux = -aux
    aux = jnp.min(aux, axis=0)
    aux = (jnp.tanh( aux / 0.0000001 ) + 1.0)/2.0
    aux = jnp.sum( aux )
    vol = ((2*bounds)**2) * (aux / num_samples)
    return vol

# Formulate and solve the Ellipse problem
ellipse_n = 2
ellipse_num_planes = 4 + 5 + 0
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
circle_num_planes = 4 + 5 + 0
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

