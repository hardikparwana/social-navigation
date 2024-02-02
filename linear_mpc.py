import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

N = 20

dt = 0.1
A = np.eye(2) + np.eye(2)*dt
B = np.ones((2,1))*dt

def linear_dynamics(x0, u, T):
    mu = np.linalg.matrix_power( A, T )

u = cp.Variable(20)
