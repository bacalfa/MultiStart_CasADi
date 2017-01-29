import numpy as np
import multistart as ms

"""
Solve the Rosenbrock problem using multistart, formulated as the NLP:
minimize     x^2 + 100*z^2
subject to   z+(1-x)^2-y == 0

Bruno Calfa, 2016
"""

## General declarations (see documentation of multistart function)
num_variables = 3
solvername = "ipopt"
opts = {}
opts["expand"] = True
opts["verbose"] = False
opts["ipopt.max_iter"] = 1000
opts["ipopt.print_level"] = 0
msopts = {}
msopts["opts"] = opts
msopts["x0"] = np.ones((num_variables, 1))
# msopts["lbx"] = -1E5
# msopts["ubx"] = 1E5
msopts["lbg"] = 0
msopts["ubg"] = 0
numberOfSamplePoints = 20
numberOfSelectedSamplePoints = 10
threadLimit = 8 # Serial: 1, Parallel: > 1


## NLP problem
# Objective function
def f(x):
	return x[0] ** 2 + 100.0 * x[2] ** 2

# Constraints
def g(x):
	return x[2] + (1.0 - x[0])**2 - x[1]

# Call the multistart function to solve the Rosenbrock problem
res = ms.multistart(num_variables=num_variables, f=f, g=g, numberOfSamplePoints=numberOfSamplePoints,
					numberOfSelectedSamplePoints=numberOfSelectedSamplePoints, threadLimit=threadLimit,
					useInitialPoint=False, **msopts)

# Print solution
print()
print("%50s " % "Optimal cost:", res[0]["f"])
print("%50s " % "Primal solution:", res[0]["x"])
print("%50s " % "Dual solution (simple bounds):", res[0]["lam_x"])
print("%50s " % "Dual solution (nonlinear bounds):", res[0]["lam_g"])