import numpy as np
import polytope as pt
import cvxpy as cp

def compute_polytope_from_points(points):
    hull = pt.qhull(points)
    return hull, hull.A, hull.b

def polytope_union( polytope_1, polytope_2 ):
    nonconvex = polytope_1.union(polytope_2)  # construct union of convex polytopes

def polytope_intersect( polytope_1, polytope_2 ):
    nonconvex = polytope_1.intersect(polytope_2)  # construct union of convex polytopes

# Formulate and solve the Ellipse problem
ellipse_n = 2
ellipse_B = cp.Variable((ellipse_n,ellipse_n), symmetric=True)
ellipse_d = cp.Variable((ellipse_n,1))
ellipse_A = cp.Parameter((ellipse_n,2))
ellipse_b = cp.Parameter((ellipse_n,1))
ellipse_objective = cp.Maximize( cp.log_det( ellipse_B ) )
ellipse_const = []
for ellipse_i in range( ellipse_A.shape[0] ):
    ellipse_const += [ cp.norm( ellipse_B @ ellipse_A[ellipse_i,:].reshape(-1,1) ) + ellipse_A[ellipse_i,:].reshape(1,-1) @ ellipse_d <= ellipse_b[ellipse_i,0] ]
ellipse_prob = cp.Problem( ellipse_objective, ellipse_const )
# ellipse_prob.solve()

# Formulate and solve the Circle problem
circle_n = 2
circle_r = cp.Variable()
circle_c = cp.Variable((2,1))
circle_A = cp.Parameter((circle_n,2))
circle_b = cp.Parameter((circle_n,1))
circle_objective = cp.Maximize(circle_r)
circle_const = []
for i in range( circle_A.shape[0] ):
    circle_const += [ circle_A[i,:].reshape(1,-1) @ circle_c + np.linalg.norm(circle_A[i,:]) * circle_r <= circle_b[i,0] ]
circle_prob = cp.Problem( circle_objective, circle_const )
# circle_prob.solve()