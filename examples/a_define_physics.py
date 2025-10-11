import numpy as np
from numba import float64, int64, njit, prange, void


@njit(
    void(float64[:, :], int64, float64),
    parallel=True,
    fastmath=True,
    cache=True,
)
def blackbox(data, steps, speed=0.01):
    """
    Advances the state of a set of 2D points through a discrete dynamical system.

    This function updates each point in `data` by iterating `n` times over a system
    defined by the equations:
        xs_new = xs + speed * ys
        ys_new = ys + speed * cos(xs + π/2)

    The updates are performed in-place using Numba for fast execution and parallelization.

    Parameters
    ----------
    data : numpy.ndarray, shape (m, 2)
        Input array containing `m` points, each with 2 coordinates (x, y).
        The array is updated in-place.
    steps : int
        Number of iterations to advance the state.
    speed : float, optional
        Step size controlling the update magnitude (default is 0.01).

    Returns
    -------
    None
        The function updates the `data` array in-place.

    Notes
    -----
    - The function uses Numba's `njit` decorator with parallel loops and fast math optimizations.
    - It modifies the input `data` array directly.
    - The system is updated `n` times per point, which may be computationally intensive for large inputs.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[0.0, 1.0], [1.0, 0.5]])
    >>> blackbox(data, 10)
    >>> print(data)

    """
    for pi in prange(data.shape[0]):
        xs = data[pi, 0]
        ys = data[pi, 1]
        for i in range(steps):
            xs += speed * ys
            ys += speed * np.cos(xs + np.pi / 2)
        data[pi, 0] = xs
        data[pi, 1] = ys
