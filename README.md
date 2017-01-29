# MultiStart CasADi
Simple multistart optimization algorithm for [CasADi](https://github.com/casadi/casadi).

Example is provided for the [Rosenbrock problem](rosenbrock.py). While a starting point is not required (since a sample of points is randomly generated using the uniform distribution), it helps to provide lower and upper bounds on the decision variables (i.e., `lbx` and `ubx`).
