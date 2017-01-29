from casadi import *
import numpy as np
import time
import itertools
from joblib import Parallel, delayed

"""
Simple multistart optimization implementation.
Given the number of total sample starting points and the number of selected starting points,
the algorithm solves multiple instances of the NLP problem and then solves additional instances
formed by the average of all the combinations of the selected starting points.

Bruno Calfa, 2016
"""



def _solveSample(num_variables, f, g, pargs, opts, x0Sample, solvername, solveropts, i):
    """
    Local function to solve an optimization problem for a given sample point when executing in parallel mode. See
    function multistart for documentation.
    :param i: Sample ID (i.e., element index in array of sample points)
    :return:
    """
    x = SX.sym("x", num_variables)
    if g is not None:
        if pargs is not None:
            nlp = {"x": x, "f": f(x, **pargs), "g": g(x, **pargs)}
        else:
            nlp = {"x": x, "f": f(x), "g": g(x)}
    else:
        if pargs is not None:
            nlp = {"x": x, "f": f(x, **pargs)}
        else:
            nlp = {"x": x, "f": f(x)}
    solver = nlpsol("solver", solvername, nlp, opts)
    res = solver(x0=x0Sample, **solveropts)
    if solver.stats()["return_status"] != "Solve_Succeeded":
        res["f"] = np.Inf
    res["sample_id"] = i
    return res


def multistart(num_variables, f, g=None, pargs=None, numberOfSamplePoints=10, numberOfSelectedSamplePoints=5,
               iterationLimit=5, timeLimit=0, threadLimit=1, numberOfBestSolutions=1, useInitialPoint=False, **kwargs):
    """
    Simple multistart algorithm where a pool of selected sample points is cross-averaged among each other to provide
    starting points for local optimizations.
    :param num_variables: Number of decision variables in optimization problem.
    :param f: Reference to the objective function (e.g., f = NameOfObjFcn)
    :param g: Reference to the constraints (Default: None. E.g., g = NameOfConstraints)
    :param pargs: Dictionary of additional arguments for the objective function and constraints (Optional. Default: None)
    :param numberOfSamplePoints: Number of sample points in each iteration (Default: 10)
    :param numberOfSelectedSamplePoints: Number of selected points for cross averaging (Default: 5)
    :param iterationLimit: Iteration limit (Default: 5)
    :param timeLimit: Time limit in seconds (Default: 0, i.e., no time limit)
    :param threadLimit: Number of threads (cores) for parallel mode (Default: 1, i.e., serial mode)
    :param numberOfBestSolutions: Number of best solutions to be returned (Default: 1, i.e., the best solution)
    :param useInitialPoint: Flag to use a starting point provided (Default: False)
    :param kwargs: Additional arguments for the optimization model and solver (e.g., lbx, ubx, solvername, etc. The
    default value for the variable solvername is "ipopt".)
    :return: A list of the first numberOfBestSolutions solutions.
    """

    ### Parse kwargs
    x0 = np.ones((num_variables, 1))
    lbx = -1E20 * x0
    ubx = -lbx
    opts = {}
    solvername = "ipopt"
    solveropts = {"lbx": lbx, "ubx": ubx}
    for k, v in kwargs.items():
        if k == "x0":
            x0 = v
        elif k == "lbx":
            lbx = v
            if np.isscalar(lbx):
                lbx *= np.ones((num_variables, 1))
            solveropts["lbx"] = lbx
        elif k == "ubx":
            ubx = v
            if np.isscalar(ubx):
                ubx *= np.ones((num_variables, 1))
            solveropts["ubx"] = ubx
        elif k == "p":
            solveropts["p"] = v
        elif k == "lbg":
            solveropts["lbg"] = v
        elif k == "ubg":
            solveropts["ubg"] = v
        elif k == "lam_x0":
            solveropts["lam_x0"] = v
        elif k == "lam_g0":
            solveropts["lam_g0"] = v
        elif k == "opts":
            opts = v
        elif k == "solvername":
            solvername = v

    ### Start algorithm
    iterCount = 1
    timeStart = time.perf_counter()
    solnSample_selected = {}

    ### Create an NLP solver
    x = SX.sym("x", num_variables)
    if g is not None:
        if pargs is not None:
            nlp = {"x": x, "f": f(x, **pargs), "g": g(x, **pargs)}
        else:
            nlp = {"x": x, "f": f(x), "g": g(x)}
    else:
        if pargs is not None:
            nlp = {"x": x, "f": f(x, **pargs)}
        else:
            nlp = {"x": x, "f": f(x)}
    solver = nlpsol("solver", solvername, nlp, opts)

    while (iterCount <= iterationLimit):
        ### Generate samples and solve local NLP problems
        x0Sample = {}
        solnSample = {}

        for i in range(numberOfSamplePoints - len(solnSample_selected)):
            x0Sample[i] = np.asarray([np.random.uniform(low=low, high=high) for low, high in zip(lbx, ubx)])
        if threadLimit == 1:
            # Serial: solve NLP for each sample point
            for i in range(numberOfSamplePoints - len(solnSample_selected)):
                solnSample[i] = solver(x0=x0Sample[i], **solveropts)
                if solver.stats()["return_status"] != "Solve_Succeeded":
                    solnSample[i]["f"] = np.Inf
        else:
            # Parallel: solve NLP for each sample point, then convert list to dict
            tmp = Parallel(n_jobs=threadLimit)(
                delayed(_solveSample)(num_variables, f, g, pargs, opts, x0Sample[i], solvername, solveropts, i) for i in
                range(numberOfSamplePoints - len(solnSample_selected)))
            for tmpsoln in tmp:
                sample_id = tmpsoln["sample_id"]
                tmpsoln.pop("sample_id", None)
                solnSample[sample_id] = tmpsoln

        if iterCount == 1 and useInitialPoint:
            i = len(x0Sample)
            solnSample[i] = solver(x0=x0, **solveropts)
            if solver.stats()["return_status"] != "Solve_Succeeded":
                solnSample[i]["f"] = np.Inf

        solnSample_selected = dict(sorted({**solnSample_selected, **solnSample}.items(), key=lambda x: x[1]["f"])[
                                   :numberOfSelectedSamplePoints])

        ### Generate new selected samples based on solution vector means and solve NLP problems
        solnSample_selected_mean = {}
        solncombs = list(map(dict, itertools.combinations(solnSample_selected.items(), 2)))
        x0Sample_selected = {}
        i = 0
        for solncomb in solncombs:
            solncomb_keys = list(solncomb.keys())
            k1 = solncomb_keys[0]
            k2 = solncomb_keys[1]
            x0Sample_selected[i] = 0.5 * (solncomb[k1]["x"] + solncomb[k2]["x"])
            i += 1
        if threadLimit == 1:
            for i in range(len(solncombs)):
                solnSample_selected_mean[i] = solver(x0=x0Sample_selected[i], **solveropts)
                if solver.stats()["return_status"] != "Solve_Succeeded":
                    solnSample_selected_mean[i]["f"] = np.Inf
        else:
            tmp = Parallel(n_jobs=threadLimit)(
                delayed(_solveSample)(num_variables, f, g, pargs, opts, x0Sample_selected[i], solvername, solveropts, i)
                for i in range(len(solncombs)))
            for tmpsoln in tmp:
                sample_id = tmpsoln["sample_id"]
                tmpsoln.pop("sample_id", None)
                solnSample_selected_mean[sample_id] = tmpsoln

        ### Update best select samples
        solnSample_selected = dict(
            sorted({**solnSample_selected, **solnSample_selected_mean}.items(), key=lambda x: x[1]["f"])[
            :numberOfSelectedSamplePoints])

        ### Stop algorithm if time limit is reached
        if (timeLimit > 0 and time.perf_counter() - timeStart > timeLimit):
            break

        iterCount += 1

    ### Return best solution(s)
    bestsolns = sorted(solnSample_selected.items(), key=lambda x: x[1]["f"])[:numberOfBestSolutions]
    return [b for (a, b) in bestsolns]
