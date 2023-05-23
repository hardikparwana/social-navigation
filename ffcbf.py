import jax.numpy as jnp
from jax import jit, jacfwd, jacrev

TFUTURE = 0.5
EPS = 1e-3
N = 7


def sigmoid_func(x, xbar):
    k = 100
    return x * (1 / 2 + 1 / 2 * jnp.tanh(k * x)) + (xbar - x) * (
        1 / 2 + 1 / 2 * jnp.tanh(k * (x - xbar))
    )


#! Circular Obstacle Avoidance
@jit
def hobst(x, y, r):
    """Obstacle avoidance constraint function. Super-level set convention.

    Args:
        x (array-like): ego state vector
        y (array-like): other state vector
        r (float): radius of obstacle

    Returns:
        ret (float): value of constraint function evaluated at time and state

    """
    xe, ye, xo, yo = x[0], x[1], y[0], y[1]
    xdot_e = 0  #! FILL IN EGO DYNAMICS
    ydot_e = 0  #! FILL IN EGO DYNAMICS
    xdot_o = 0  #! FILL IN OTHER DYNAMICS
    ydot_o = 0  #! FILL IN OTHER DYNAMICS

    # FF-CBF
    dx, dy, dvx, dvy = xe - xo, ye - yo, xdot_e - xdot_o, ydot_e - ydot_o
    tau_hat = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2 + EPS)
    tau = sigmoid_func(tau_hat, TFUTURE)

    return (dx + dvx * tau) ** 2 + (dy + dvy * tau) ** 2 - (2 * r) ** 2


@jit
def dhobstdx(x, y, r):
    return jacfwd(hobst)(x, y, r)


@jit
def d2hobstdx2(x, y, r):
    return jacrev(jacfwd(hobst))(x, y, r)


R = 1.0
h = lambda x, y: hobst(x, y, R)
dhdx = lambda x, y: dhobstdx(x, y, R)
d2hdx2 = lambda x, y: d2hobstdx2(x, y, R)