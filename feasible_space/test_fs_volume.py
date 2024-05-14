"""How to compute a polytope's volume."""
import numpy as np
import polytope
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad
import jax
from jax import config
config.update("jax_enable_x64", True)
import time

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
    
    if len(bpt)!=2:
        print(f"ERROR: Should have been 2 points!")
        
    if len(bpt)>2:
        bpt = [bpt[0], bpt[-1]]

    return bpt

# constructing a convex polytope and computing its volume
# vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
# vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
vertices = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
# vertices = np.array([[ 0. ,  0. ],
#        [ 0.5,  1. ],
#        [ 2. ,  1.5],
#        [ 3. ,  0.5],
#        [ 1. , -0.5]])
hull = polytope.qhull(vertices)
# print(hull.volume)


print(f"A: {hull.A}, b: {hull.b}")
# constructing a nonconvex polytope and computing its volume
# vertices_1 = np.array([[0.0, 0.0], [0.0, 1.0], [2.0, 1.0]])
# vertices_2 = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
# hull_1 = polytope.qhull(vertices_1)  # convex hull of vertices in `vertices_1`
# hull_2 = polytope.qhull(vertices_2)  # convex hull of vertices in `vertices_2`
# nonconvex = hull_1.union(hull_2)  # construct union of convex polytopes
# print(polytope.volume( nonconvex ))

plt.figure()
ax = plt.axes(xlim=(-2,5), ylim=(-2,3))
hull_plot = hull.plot(ax)
ax.legend()
# plt.show()
key = jax.random.PRNGKey(20)
key, subkey = jax.random.split(key)
# num_samples=30000 #1000000 #1000000
# num_extra_pts = 200
num_samples=300000 #1000000 #1000000
num_extra_pts = 2000
# 30000000: no difference
# 300000: very different gradients
# bounds = 1.1 #1.1s
bounds = 4.0 #4.0
samples = jax.random.uniform( key, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]   

def mc_polytope_volume_org(A, b, subkey, extra_pts=None):
    samples = jax.random.uniform( subkey, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]   
    if samples!=None:
        samples = jnp.append( samples, extra_pts, axis=1 )
    aux = A @ samples - b
    aux = jnp.nonzero(jnp.all(aux < 0, 0))[0].shape[0]    
    vol = (2*bounds)**2 * aux / samples.shape[1] #num_samples
    # jax.debug.print("org vol : {vol}", vol=vol)
    return vol
mc_polytope_volume_org_grad = grad(mc_polytope_volume_org, 0)

@jit
def mc_polytope_volume(A, b, subkey, extra_pts):
    samples = jax.random.uniform( subkey, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]  
    # samples = jnp.append( samples, extra_pts, axis=1 ) 
    aux = A @ samples - b    
    aux = -aux
    aux = jnp.min(aux, axis=0)
    # jax.debug.print("aux vmap : {out}", out=aux)
    aux = (jnp.tanh( aux / 0.01 ) + 1.0)/2.0    
    aux = jnp.sum( aux )
    vol = ((2*bounds)**2) * aux / samples.shape[1] #num_samples
    # jax.debug.print("vol : {vol}", vol=vol)
    return vol
mc_polytope_volume_grad = jit(grad(mc_polytope_volume, 1))

@jit
def mc_polytope_volume_extra(A, b, subkey, extra_pts):
    samples = jax.random.uniform( subkey, shape=(2,num_samples), minval=-bounds, maxval=bounds )#A.shape[1]  
    samples = jnp.append( samples, extra_pts, axis=1 ) 
    aux = A @ samples - b    
    aux = -aux
    aux = jnp.min(aux, axis=0)
    # jax.debug.print("aux vmap : {out}", out=aux)
    aux = (jnp.tanh( aux / 0.01 ) + 1.0)/2.0    
    aux = jnp.sum( aux )
    vol = ((2*bounds)**2) * aux / samples.shape[1] #num_samples
    # jax.debug.print("vol : {vol}", vol=vol)
    return vol
mc_polytope_volume_extra_grad = jit(grad(mc_polytope_volume_extra, 1))

@jit
def mc_polytope_volume_parallel(A, b, subkey):
    subkeys = jax.random.split(subkey, num_samples)

    @jit
    def body_vmap(subkey):
        sample = jax.random.uniform( subkey, shape=(2,1), minval=-bounds, maxval=bounds )#A.shape[1]   
        aux = A @ sample - b
        aux = -aux
        # jax.debug.print("🤯 aux={aux} 🤯, {sample}, A: {A}, B:{b}", aux=aux, sample=sample, A=A, b=b)
        aux = jnp.min(aux, axis=0)
        return aux
    aux = jax.vmap( body_vmap )(subkeys)
    # jax.debug.print("aux vmap : {out}", out=aux)
    aux = (jnp.tanh( aux / 0.001 ) + 1.0)/2.0    
    aux = jnp.sum( aux )
    vol = ((2*bounds)**2) * (aux / num_samples)
    # jax.debug.print("vmap vol : {vol}", vol=vol)
    return vol
mc_polytope_volume_parallel_grad = jit(grad(mc_polytope_volume_parallel, 1))

def mc_polytope_volume_about_lines(A, b, samples, total_volume):
    aux = A @ samples - b    
    aux = -aux
    aux = jnp.min(aux, axis=0)
    aux = (jnp.tanh( aux / 0.001 ) + 1.0)/2.0    
    aux = jnp.sum( aux )
    vol = total_volume * (aux / samples.shape[1]) #num_samples)
    return vol
mc_polytope_volume_about_lines_grad = jit(grad(mc_polytope_volume_about_lines, 1))

num_line_points = 50
num_normal_points = 30
increment = 0.0001
# @jit
def generate_points_about_line(pts): #, num_line_pts, num_normal_points, increment):
    
    xs = jnp.linspace(0, pts[1][0]-pts[0][0], num_line_points).reshape(1,-1)    

    if jnp.abs(pts[1][0]-pts[0][0])>0.001: # Not a vertical line
        slope = (pts[1][1]-pts[0][1])/(pts[1][0]-pts[0][0])    
        ys = pts[0][1] + slope * xs
        theta = jnp.arctan(slope)
    else: # vertical line
        ys = jnp.linspace( pts[0][1], pts[1][1], num_line_points ).reshape(1,-1)
        theta = np.pi/2
    line_pts = jnp.append( pts[0][0]+xs, ys, axis=0  )
    # Now N choose points above and below perpendicular to xs, ys    
    slope_n = jnp.pi/2 + theta    
    length = jnp.linspace(0, 2 * num_normal_points, num_normal_points, dtype=int) - num_normal_points
    length = length * increment
    steps = length * jnp.array([ [jnp.cos(slope_n)], [jnp.sin(slope_n)] ])
    temp_points = line_pts.reshape((2,1,line_pts.shape[1])) + steps.reshape( (2,steps.shape[1],1) )
    new_pts = temp_points.reshape( (2, num_normal_points*num_line_points) )
    volume = jnp.linalg.norm( pts[1]-pts[0] ) * num_normal_points * increment

    return new_pts, volume

plt.figure()
ax = plt.axes(xlim=(-2,5), ylim=(-2,3))
hull_plot = hull.plot(ax)
lb = jnp.array([-bounds, -bounds])
ub = jnp.array([bounds, bounds])
# get_intersection_points( hull.A[0,:], hull.b[0], lb , ub )

control_input_limit_points = np.array([ [lb[0], lb[1]], [lb[0], ub[1]], [ub[0], ub[1]], [ub[0], lb[1]] ])
control_bound_polytope = polytope.qhull( control_input_limit_points )
key = jax.random.PRNGKey(10)
#plot new points
sampled_pts = []

for i in range(hull.A.shape[0]):
    pts = get_intersection_points( hull.A[i,:], hull.b[i], lb , ub )
    if len(pts)>0:
        plt.plot([ pts[0][0], pts[1][0] ], [ pts[0][1], pts[1][1] ]  )
    xs = jnp.linspace(0, pts[1][0]-pts[0][0], num_extra_pts).reshape(1,-1)
    if jnp.abs(pts[1][0]-pts[0][0])>0.001:
        slope = (pts[1][1]-pts[0][1])/(pts[1][0]-pts[0][0])
        # print(f"slope: {slope}")
        ys = pts[0][1] + slope * xs
        key, subkey = jax.random.split(key)
        ys = jnp.clip( ys + 0.2 * jax.random.normal(subkey, shape=(1,num_extra_pts)), lb[1], ub[1] )
        new_pts = jnp.append( pts[0][0]+xs, ys, axis=0  )
    else:
        ys = jnp.linspace( pts[0][1], pts[1][1], num_extra_pts ).reshape(1,-1)
        key, subkey = jax.random.split(key)
        xs = jnp.clip( pts[0][0] + xs + 0.2 * jax.random.normal(subkey, shape=(1,num_extra_pts)), lb[0], ub[0] )
        new_pts = jnp.append( xs, ys, axis=0 )  
    if i==0:
        sampled_pts = new_pts #jnp.append( pts[0][0]+xs, ys, axis=0  )
    else:
        sampled_pts = jnp.append( sampled_pts, new_pts, axis=1 )
    
    # ax.scatter(sampled_pts[0,:], sampled_pts[1,:], s=3)
# plt.show()
samples = jnp.append(samples, sampled_pts, axis=1)

mc_polytope_volume_org(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts)
mc_polytope_volume(hull.A, hull.b.reshape(-1,1), subkey , sampled_pts)
mc_polytope_volume_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts)
mc_polytope_volume_extra(hull.A, hull.b.reshape(-1,1), subkey , sampled_pts)
mc_polytope_volume_extra_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts)
# mc_polytope_volume_parallel( hull.A, hull.b.reshape(-1,1), subkey)


# exit()
key, subkey = jax.random.split(key)
print(f"old volume : { mc_polytope_volume_org(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts) }")

# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol = mc_polytope_volume(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts )
# print(f"new volume: { vol }, time: {time.time()-t0} " )
# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol = mc_polytope_volume(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts )
# print(f"new volume: { vol }, time: {time.time()-t0} " )

key, subkey = jax.random.split(key)
t0 = time.time()
vol_grad = mc_polytope_volume_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts)
print(f"new grad: { vol_grad }, time: {time.time()-t0} " )
# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol_grad = mc_polytope_volume_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts)
# print(f"new grad: { vol_grad }, time: {time.time()-t0} " )

# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol = mc_polytope_volume_parallel(hull.A, hull.b.reshape(-1,1), subkey )
# print(f"new volume vmap: { vol }, time: {time.time()-t0} " )
# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol = mc_polytope_volume_parallel(hull.A, hull.b.reshape(-1,1), subkey )
# print(f"new volume vmap: { vol }, time: {time.time()-t0} " )
# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol = mc_polytope_volume_parallel_grad(hull.A, hull.b.reshape(-1,1), subkey )
# print(f"new vmap grad: { vol }, time: {time.time()-t0} " )
# key, subkey = jax.random.split(key)
# t0 = time.time()
# vol = mc_polytope_volume_parallel_grad(hull.A, hull.b.reshape(-1,1), subkey )
# print(f"new vmap grad: { vol }, time: {time.time()-t0} " )


# vols = []
# vols_extra = []
# for i in range(100):
#     key, subkey = jax.random.split(key)
#     vols.append(mc_polytope_volume(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts ))
#     vols_extra.append(mc_polytope_volume_extra(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts ))
#     if i==0:    
#         grads = mc_polytope_volume_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts).reshape(-1,1)
#         grads_extra = mc_polytope_volume_extra_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts).reshape(-1,1)
#     else:
#         grads = jnp.append( grads, mc_polytope_volume_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts).reshape(-1,1), axis=1 )
#         grads_extra = jnp.append( grads_extra, mc_polytope_volume_extra_grad(hull.A, hull.b.reshape(-1,1), subkey, sampled_pts).reshape(-1,1), axis=1 )

# print(f"Mean vol: {jnp.mean(jnp.asarray(vols))}")
# print(f"Extra Mean vol: {jnp.mean(jnp.asarray(vols_extra))}")

# print(f"Mean grad: {jnp.mean(grads, axis=1)}")
# print(f"Extra Mean grad: {jnp.mean(grads_extra, axis=1)}")

# print(f"Std grad: {jnp.std(grads, axis=1)}")
# print(f"Extra Std grad: {jnp.std(grads_extra, axis=1)}")

# plt.show()


total_volume = 0
for i in range(hull.A.shape[0]):
    pts = get_intersection_points( hull.A[i,:], hull.b[i], lb , ub )
    if len(pts)>0:
        plt.plot([ pts[0][0], pts[1][0] ], [ pts[0][1], pts[1][1] ]  )
    else:
        continue
        
    new_pts, temp_volume = generate_points_about_line( pts ) #, num_line_points, num_normal_points, increment )
    total_volume = total_volume + temp_volume    
    if i==0:
        sampled_pts = new_pts #jnp.append( pts[0][0]+xs, ys, axis=0  )
    else:
        sampled_pts = jnp.append( sampled_pts, new_pts, axis=1 )
    plt.scatter(new_pts[0,::3], new_pts[1,::3], s=3, alpha=0.1)
# plt.scatter(sampled_pts[0,:], sampled_pts[1,:], s=3)
    # now sample points on this line
samples = sampled_pts # jnp.append(samples, sampled_pts, axis=1)
# print(f"boundary volume: { mc_polytope_volume_about_lines(hull.A, hull.b.reshape(-1,1), samples, total_volume ) } " )
# print(f"boundary grad: { mc_polytope_volume_about_lines_grad(hull.A, hull.b.reshape(-1,1), samples, total_volume) }")

vol_boundary = mc_polytope_volume_about_lines(hull.A, hull.b.reshape(-1,1), samples, total_volume )
vol_boundary_org = mc_polytope_volume_org(hull.A, hull.b.reshape(-1,1), subkey, samples)
print(f"boundary volume: { vol_boundary } " )
print(f"boundary grad: { mc_polytope_volume_about_lines_grad(hull.A, hull.b.reshape(-1,1), samples, total_volume) }")
print(f"boundary volume smooth extra grad: {mc_polytope_volume_extra_grad(hull.A, hull.b.reshape(-1,1), subkey, samples)}")
print(f"boundary + org volume: { vol_boundary_org }")
print(f"boundary vol scaled: { vol_boundary / vol_boundary_org }")
factor = vol_boundary_org / vol_boundary # (2*bounds)**2 / total_volume # 
print(f"boundary vol grad scaled: { factor * mc_polytope_volume_about_lines_grad(hull.A, hull.b.reshape(-1,1), samples, total_volume) }")

plt.show()

print(f"done")