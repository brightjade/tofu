import torch

def quantile_function(qs, cws, xs):
    r""" Computes the quantile function of an empirical distribution

    Parameters
    ----------
    qs: array-like, shape (n,)
        Quantiles at which the quantile function is evaluated
    cws: array-like, shape (m, ...)
        cumulative weights of the 1D empirical distribution, if batched, must be similar to xs
    xs: array-like, shape (n, ...)
        locations of the 1D empirical distribution, batched against the `xs.ndim - 1` first dimensions

    Returns
    -------
    q: array-like, shape (..., n)
        The quantiles of the distribution
    """
    n = xs.shape[0]
    # this is to ensure the best performance for torch searchsorted
    # and avoid a warning related to non-contiguous arrays
    cws = cws.T.contiguous()
    qs = qs.T.contiguous()
    # finds the index in cumweights where qs should be inserted to maintain order
    idx = torch.searchsorted(cws, qs).T
    return torch.take_along_dim(xs, torch.clip(idx, min=0, max=n-1), dim=0)


def sliced_wasserstein_distance(W_i, W_c, n_projections=100, p=2):
    r""" Computes a Monte-Carlo approximation of the p-Sliced Wasserstein distance between two weights

    Parameters
    ----------
    W_i: array-like, shape (n, d)
        Weights of the initial parameters (frozen)
    W_c: array-like, shape (m, d)
        Weights of the current parameters
    n_projections: int, optional (default=100)
        Number of random projections
    p: int, optional (default=2)
        Power of the Wasserstein distance

    Returns
    -------
    swd: float
        Sliced Wasserstein Distance between the two weights
    """
    # Generate random projections
    projections = torch.randn(W_i.shape[1], n_projections)
    projections = projections / torch.sqrt((projections**2).sum(dim=0))

    # Project weights to lower dimension
    W_i_projections = torch.mm(W_i, projections.to(W_i))
    W_c_projections = torch.mm(W_c, projections.to(W_c))

    # Sort the weights
    u_values, _ = torch.sort(W_i_projections, dim=0)
    v_values, _ = torch.sort(W_c_projections, dim=0)

    # Compute Wasserstein distances
    wasserstein_distances = torch.abs(u_values - v_values.to(W_i))
    wasserstein_distances = (wasserstein_distances ** p).sum(dim=0)

    # Compute p-sliced SWD
    swd = wasserstein_distances.mean()

    return swd
