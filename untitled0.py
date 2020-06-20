import pymc4 as pm
import numpy as np
import arviz as az

# define parametrization
J = 8
y = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)
