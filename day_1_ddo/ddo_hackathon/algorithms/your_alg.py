import numpy as np
import scipy

def random_search(f, n_p, bounds_rs, iter_rs):
    """
    A naive optimization routine that randomly samples the allowed space and 
    returns the best function value found.

    Parameters
    ----------
    f : object
        The function object to be optimized. It must define a `fun_test(x)` method
        that takes a numpy array x as input and returns a scalar function value.
    n_p : int
        The dimensionality of the input space.
    bounds_rs : numpy.ndarray
        An array of shape (n_p, 2) specifying the lower and upper bounds for each
        dimension of the search space.
    iter_rs : int
        The number of random points to sample.

    Returns
    -------
    f_b : float
        The best (lowest) function value found over all sampled points.
    x_b : numpy.ndarray
        The coordinates (as a 1D array of length n_p) at which the best function 
        value is attained.
    """
    # arrays to store sampled points
    localx = np.zeros((n_p, iter_rs))  # points sampled
    localval = np.zeros(iter_rs)       # function values sampled

    # bounds
    bounds_range = bounds_rs[:, 1] - bounds_rs[:, 0]
    bounds_bias = bounds_rs[:, 0]

    for sample_i in range(iter_rs):
        # sampling
        x_trial = np.random.uniform(0, 1, n_p) * bounds_range + bounds_bias
        localx[:, sample_i] = x_trial
        localval[sample_i] = f.fun_test(x_trial)

    # choosing the best
    minindex = np.argmin(localval)
    f_b = localval[minindex]
    x_b = localx[:, minindex]

    return f_b, x_b


def your_alg(f, x_dim, bounds, iter_tot):
    """
    A demonstration optimization routine that uses random sampling to obtain
    an initial guess, then returns that guess as the final result.

    Parameters
    ----------
    f : object
        The function object to be optimized. It must define a `fun_test(x)` method
        that takes a numpy array x as input and returns a scalar function value.
    x_dim : int
        The dimensionality of the input space.
    bounds : numpy.ndarray
        An array of shape (x_dim, 2) specifying the lower and upper bounds for
        each dimension of the search space.
    iter_tot : int
        The total number of iterations (budget) allowed for the search.

    Returns
    -------
    x_best : numpy.ndarray
        The best solution found (as a 1D array of length x_dim).
    f_best : float
        The value of the function at x_best.
    team_name : list
        A list containing the team number as string(s).
    names : list
        A list containing the members' names or identifiers.
    """
    n_rs = int(min(100, max(iter_tot * 0.05, 5)))  # iterations to find good starting point
    # evaluate initial random points
    f_best, x_best = random_search(f, x_dim, bounds, n_rs)

    # Remaining budget after the random sampling
    iter_ = iter_tot - n_rs

    team_name = ["8"]
    names = ["01234567"]
    return x_best, f_best, team_name, names
