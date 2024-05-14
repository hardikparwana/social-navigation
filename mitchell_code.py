"""
qp_solver_casadi
================

This module implements a solver for quadratic programs using the Casadi library.

Functions
---------
-solve(h_mat, f_vec, g_mat, h_vec, a_mat, b_vec): calculates solution to quadratic program specified by args

Notes
-----
Quadratic Program takes the following form:
min 1/2 x.T @ h_mat @ x + f_vec @ x
subject to
g_mat @ x <= h_vec
a_mat @ x = b_vec

Examples
--------
>>> import qp_solver_casadi
>>> sol, status = qp_solver_casadi.solve(
        h_mat=jnp.eye(2),
        f_vec=jnp.ones((2,))
        g_mat=jnp.ones((2, 1))
        h_vec=jnp.array([1.0])
        a_mat=None,
        b_vec=None
    )

"""
from typing import Union, Tuple
import casadi as ca
import numpy as np
from jax import Array
import jax.numpy as jnp


def solve(
    h_mat: Array,
    f_vec: Array,
    g_mat: Union[Array, None] = None,
    h_vec: Union[Array, None] = None,
    a_mat: Union[Array, None] = None,
    b_vec: Union[Array, None] = None,
) -> Tuple[Array, int]:
    """
    Solve a quadratic program using the Casadi solver.

    Args:
        h_mat: quadratic cost matrix
        f_vec: linear cost vector
        g_mat: linear inequality constraint matrix
        b_vec: linear inequality constraint vector
        g_mat: linear equality constraint matrix
        h_vec: linear equality constraint vector

    Returns:
        solution: Solution to the QP
        status: True if optimal solution found

    """
    # Define decision variables
    n = len(f_vec)
    x = ca.MX.sym("x", n)

    # Format vectors and matrices in Casadi
    h_mat = ca.MX(np.array(h_mat))
    f_vec = ca.MX(np.array(f_vec))

    # Construct inequality constraints
    inequality_constraints = []
    if g_mat is not None and h_vec is not None:
        g_mat = ca.MX(np.array(g_mat))
        h_vec = ca.MX(np.array(h_vec))
        inequality_constraints += [h_vec - ca.mtimes(g_mat.T, x)]

    # Construct equality constraints
    equality_constraints = []
    if a_mat is not None and b_vec is not None:
        a_mat = ca.MX(np.array(a_mat))
        b_vec = ca.MX(np.array(b_vec))
        equality_constraints += [ca.mtimes(a_mat, x) - b_vec]

    # Define quadratic objective function
    objective = ca.mtimes(x.T, ca.mtimes(h_mat, x)) + ca.mtimes(f_vec.T, x)

    # Combine constraints
    constraints = inequality_constraints #+ equality_constraints

    # Define problem
    # prob = {'f': objective, 'x': x, 'g': ca.vertcat(*constraints)}
    prob = {'f': objective, 'x': x, 'g': constraints[0]}

    # # Set options for the solver
    # solver_opts = {"printLevel": "1"}

    # Solve the optimization problem
    solver = ca.qpsol("solver", "qpoases", prob)#, solver_opts)
    solution = solver()
    success = solver.stats()["success"]

    return success * jnp.array(solution["x"]).reshape((n,)), success


if __name__ == "__main__":
    sol, status = solve(
        h_mat=jnp.eye(2),
        f_vec=jnp.array([0.0, -100.0]), # f_vec=jnp.ones((2,)),
        g_mat=jnp.ones((2, 1)),
        h_vec=jnp.array([1.0]),
        a_mat=None,
        b_vec=None,
    )
    print(f"Solution: {sol}")
    print(f"Status: {status}")