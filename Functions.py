from numba import njit, vectorize, jit
import numpy as np
import Constants

@njit
def update_states(v, u, a, b, I, dt):
    v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I) * dt
    v += 0.5 * (0.04 * v ** 2 + 5 * v + 140 - u + I) * dt
    u += a * (b * v - u) * dt
    return v, u
