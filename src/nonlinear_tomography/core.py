from collections import defaultdict
from math import ceil, floor
from typing import Any, Callable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from numba import njit
from scipy.spatial import Delaunay
from tqdm import tqdm


def generate_points_grid(
    x_range: Tuple[float, float] = (0.0, 1.0),
    y_range: Tuple[float, float] = (0.0, 1.0),
    num_points: int = 20,
) -> np.ndarray:
    """
    Generate a 2D grid of points within the specified x and y ranges.

    The function creates a uniform grid of approximately `num_points` total points
    by computing the square root of `num_points` to determine the number of points
    along each axis. The final grid will have (sqrt(num_points))^2 total points.

    Parameters
    ----------
    x_range : tuple of float, optional
        The (min, max) range for the x-axis. Default is (0.0, 1.0).
    y_range : tuple of float, optional
        The (min, max) range for the y-axis. Default is (0.0, 1.0).
    num_points : int, optional
        Approximate total number of points in the grid. The function will use
        the square root of this value to determine grid resolution. Default is 20.

    Returns
    -------
    points : ndarray of shape (N, 2)
        A 2D array where each row is an (x, y) coordinate on the grid.

    Notes
    -----
    - The actual number of points returned will be (sqrt(num_points))^2,
      which may differ slightly from `num_points` depending on rounding.
    """
    # Calculate number of points along each axis (assuming square grid)
    grid_size = int(np.sqrt(num_points))

    # Generate evenly spaced values for x and y axes
    x = np.linspace(*x_range, grid_size)
    y = np.linspace(*y_range, grid_size)

    # Create a meshgrid from x and y values
    xx, yy = np.meshgrid(x, y)

    # Flatten and stack the grid to get coordinate pairs
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    return points


def generate_triangle_mesh(points: np.ndarray) -> Delaunay:
    """
    Generate a triangular mesh from a set of 2D points using Delaunay triangulation.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        An array of 2D points where each row represents an (x, y) coordinate.

    Returns
    -------
    tri : scipy.spatial.Delaunay
        A Delaunay triangulation object containing the generated mesh.

    Notes
    -----
    - The Delaunay triangulation maximizes the minimum angle of all triangles,
      avoiding skinny triangles as much as possible.
    - The input points should ideally not be collinear and should span a 2D region.
    """
    # Perform Delaunay triangulation on the input 2D points
    tri = Delaunay(points)

    return tri


def plot_mesh(
    points: np.ndarray,
    triangles: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cmap: str = "viridis",
) -> None:
    """
    Plot a 2D triangular mesh with optional coloring based on triangle weights.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Array of 2D coordinates representing the mesh vertices.
    triangles : ndarray of shape (M, 3)
        Array of indices into `points`, where each row
        defines a triangle using 3 point indices.
    weights : ndarray of shape (M,), optional
        Optional array of scalar weights used to color
         each triangle. If provided,
        the triangles are colored using `tripcolor`.
        Otherwise, a wireframe mesh is plotted.
    cmap : str, optional
        Colormap used to map weights to colors. Default is "viridis".


    Notes
    -----
    - This function uses matplotlib's triangulation utilities to visualize the mesh.
    - Edge colors are shown in black for clarity.
    """
    # Create a Triangulation object from points and triangle indices
    triang = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    if weights is not None:
        # Plot the mesh with color-coded triangle weights
        plt.tripcolor(triang, facecolors=weights, edgecolors="k", cmap=cmap)
        plt.colorbar(label="Triangle Weight")  # Show color scale
    else:
        # Plot the mesh as a wireframe without coloring
        plt.triplot(triang, color="k")


@njit
def histogram(
    points: np.ndarray,
    simplices: np.ndarray,
    range_: Tuple[float, float],
    bins: int,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a weighted histogram of triangle areas along the x-axis.

    Each triangle's area is distributed across histogram bins according to how much
    its x-range overlaps with each bin. Optionally, each triangle's contribution
    can be scaled by a weight.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        2D coordinates of the mesh vertices.
    simplices : ndarray of shape (M, 3)
        Indices into `points`, where each row defines a triangle.
    range_ : tuple of float
        The (min, max) x-axis range over which to compute the histogram.
    bins : int
        Number of bins to divide the range into.
    weights : ndarray of shape (M,), optional
        Optional weights for each triangle. If not provided, all weights are set to 1.

    Returns
    -------
    hist_x : ndarray of shape (bins,)
        The x-coordinates of the bin centers.
    hist_y : ndarray of shape (bins,)
        The weighted sum of triangle area contributions in each bin.
    matrix : ndarray of shape (bins, M)
        A matrix recording the contribution of each triangle to each bin.

    Notes
    -----
    - Triangles that fall entirely outside the given range are ignored.
    - Degenerate (zero-area) triangles are skipped.
    """
    if weights is None:
        weights = np.ones(simplices.shape[0])

    # Initialize the output matrix and histogram arrays
    matrix = np.zeros((bins, simplices.shape[0]), dtype=np.float64)
    hist_y = np.zeros(bins, dtype=np.float64)

    # Calculate bin width and inverse for fast division
    step = (range_[1] - range_[0]) / bins
    inv_step = 1.0 / step

    # Bin centers for plotting
    hist_x = np.linspace(range_[0] + step / 2, range_[1] - step / 2, bins)

    for i in range(simplices.shape[0]):
        # Get the triangle vertices A, B, C
        A = points[simplices[i, 0]]
        B = points[simplices[i, 1]]
        C = points[simplices[i, 2]]

        # Compute triangle area using the cross product method
        area = 0.5 * abs(
            (B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])
        )
        if area == 0.0:
            continue  # Skip degenerate triangles

        # Apply the triangle weight
        weighted_area = area * weights[i]

        # Determine x-range of the triangle
        triangle_x = points[simplices[i, :], 0]
        idx_f = (triangle_x - range_[0]) * inv_step
        idx_start = max(0, floor(idx_f.min()))
        idx_stop = min(bins, ceil(idx_f.max()))

        if idx_stop <= idx_start:
            continue  # Triangle does not overlap histogram range

        # Distribute triangle's area contribution across overlapping bins
        for j in range(idx_start, idx_stop):
            bin_left = range_[0] + j * step
            bin_right = bin_left + step

            # Compute overlap between bin and triangle in x-direction
            overlap_left = max(triangle_x.min(), bin_left)
            overlap_right = min(triangle_x.max(), bin_right)
            overlap_width = overlap_right - overlap_left

            if overlap_width > 0.0:
                x_range = triangle_x.max() - triangle_x.min()
                fraction = overlap_width / x_range if x_range != 0.0 else 1.0
                contribution = weighted_area * fraction

                hist_y[j] += contribution
                matrix[j, i] += contribution

    return hist_x, hist_y, matrix


def algebraic_reconstruction_technique_masked(
    omegas: List[np.ndarray],
    targets: List[np.ndarray],
    mask: np.ndarray,
    num_iterations: int = 10,
    relaxation: float = 1.0,
    x0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Perform Algebraic Reconstruction Technique (ART) with a mask applied to the solution.

    ART iteratively solves a system of linear equations by sequentially projecting the
    current estimate onto hyperplanes defined by the measurement vectors. Updates
    are applied only on elements where `mask` is True, allowing selective reconstruction.

    Parameters
    ----------
    omegas : list of ndarray of shape (M_i, N)
        List of measurement matrices, each row represents a measurement vector `a_i`.
    targets : list of ndarray of shape (M_i,)
        List of target measurement vectors `b_i`, corresponding to each `omegas` matrix.
    mask : ndarray of shape (N,)
        Boolean mask indicating which elements of the solution vector `x` should be updated.
    num_iterations : int, optional
        Number of full iterations over all measurements. Default is 10.
    relaxation : float, optional
        Relaxation parameter controlling the step size of each update. Default is 1.0.
    x0 : ndarray of shape (N,), optional
        Initial guess for the solution vector. If None, starts from zero vector.

    Returns
    -------
    x : ndarray of shape (N,)
        The reconstructed solution vector after iterative updates.

    Notes
    -----
    - Non-negativity constraint is enforced after each update (values clipped to >= 0).
    - Mask controls which entries of `x` get updated during the iterative process.
    """
    n = omegas[0].shape[1]

    # Initialize solution vector x
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    for _ in tqdm(range(num_iterations), desc="ART iterations"):
        # Loop over each measurement matrix and corresponding targets
        for omega, target in zip(omegas, targets):
            for i in range(omega.shape[0]):
                a_i = omega[i]
                b_i = target[i]

                # Skip if measurement vector is zero to avoid division by zero
                a_i_norm_sq = np.dot(a_i, a_i)
                if a_i_norm_sq == 0:
                    continue

                # Compute residual between observed and predicted measurement
                residual = b_i - np.dot(a_i, x)

                # Calculate update scaled by relaxation parameter
                update = (relaxation * residual / a_i_norm_sq) * a_i

                # Update only masked elements of solution vector
                x[mask] += update[mask]

                # Enforce non-negativity constraint
                x = np.maximum(x, 0)

    return x


def compute_variable_usage(omegas: List[np.ndarray]) -> np.ndarray:
    """
    Compute the usage count of each variable across a list of measurement matrices.

    For each variable (column), counts how many measurement matrices contain at least
    one nonzero entry in that column.

    Parameters
    ----------
    omegas : list of ndarray of shape (M_i, N)
        List of measurement matrices.

    Returns
    -------
    usage : ndarray of shape (N,)
        Array where each element indicates the number of matrices in which
        the corresponding variable (column) appears (nonzero at least once).
    """
    n = omegas[0].shape[1]
    usage = np.zeros(n, dtype=int)

    for omega in omegas:
        # Boolean array indicating if each variable has any nonzero entries in this omega
        usage += np.any(omega != 0, axis=0).astype(int)

    return usage


def smooth_weights(
    simplices: np.ndarray,
    weights: np.ndarray,
    num_iterations: int = 1,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Smooth weights on a triangular mesh by iteratively averaging over neighboring triangles.

    Neighbors are defined as triangles sharing an edge. The weights of each triangle
    are updated as a weighted average between its current weight and the average
    weight of its neighbors.

    Parameters
    ----------
    simplices : ndarray of shape (M, 3)
        Array of triangle vertex indices.
    weights : ndarray of shape (M,)
        Initial weights associated with each triangle.
    num_iterations : int, optional
        Number of smoothing iterations to perform. Default is 1.
    alpha : float, optional
        Smoothing factor between 0 and 1 that controls the influence of neighbors.
        A value of 0 means no smoothing; 1 means fully replaced by neighbors' average.

    Returns
    -------
    smoothed_weights : ndarray of shape (M,)
        The smoothed weights after the specified number of iterations.

    Notes
    -----
    - The function builds triangle neighbors by shared edges using an edge-to-triangle map.
    - If a triangle has no neighbors, its weight remains unchanged during smoothing.
    """
    tri_count = len(simplices)

    # Map edges (sorted vertex pairs) to the triangles that contain them
    edge_to_triangles = defaultdict(list)
    for i, tri in enumerate(simplices):
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for u, v in edges:
            key = tuple(sorted((u, v)))
            edge_to_triangles[key].append(i)

    # Build neighbors: triangles sharing at least one edge
    triangle_neighbors: defaultdict[int, Set[int]] = defaultdict(set)
    for tris in edge_to_triangles.values():
        if len(tris) >= 2:
            for i in tris:
                for j in tris:
                    if i != j:
                        triangle_neighbors[i].add(j)

    smoothed_weights = weights.copy()

    for _ in range(num_iterations):
        new_weights = smoothed_weights.copy()
        for i in range(tri_count):
            neighbors = triangle_neighbors[i]
            if not neighbors:
                continue
            # Average weight of neighbors
            neighbor_avg = np.mean([smoothed_weights[j] for j in neighbors])
            # Weighted average between current weight and neighbors' average
            new_weights[i] = (1 - alpha) * smoothed_weights[
                i
            ] + alpha * neighbor_avg

        smoothed_weights = new_weights

    return smoothed_weights


def reconstruct(
    measurements: List[np.ndarray],
    mappings: List[Callable[[np.ndarray], None]],
) -> Tuple[np.ndarray, Any, np.ndarray]:
    """
    Perform reconstruction of weights from measurements and mappings.

    The process involves:
    1. Generating a mesh (points and triangles).
    2. Building a linear system of equations from the mesh and measurements.
    3. Deriving a mask for selective updates.
    4. Solving the system iteratively using masked algebraic reconstruction technique.
    5. Smoothing the resulting weights on the mesh.

    Parameters
    ----------
    measurements : Any
        Input measurement data used for reconstruction.
    mappings : Any
        Mapping data used to relate measurements to mesh points.

    Returns
    -------
    points : ndarray of shape (N, 2)
        Coordinates of the reconstructed mesh points.
    triangles : Any
        Mesh triangles structure (expected to have a `.simplices` attribute).
    weights : ndarray of shape (M,)
        Reconstructed and smoothed weights associated with each triangle.
    """
    # Generate the mesh points and triangles
    points, triangles = generate_mesh()

    # Create linear equation system from mappings and measurements
    omegas, targets = generate_linear_equations_system(
        mappings, measurements, points, triangles
    )

    # Derive mask indicating which variables should be updated
    mask = derive_mask(omegas)

    # Perform iterative algebraic reconstruction with masking and relaxation
    weights = algebraic_reconstruction_technique_masked(
        omegas,
        targets,
        mask=mask,
        num_iterations=100,
        relaxation=0.01,
        x0=None,
    )

    # Smooth the weights over the mesh triangles
    weights = smooth_weights(
        triangles.simplices, weights, num_iterations=3, alpha=0.5
    )

    return points, triangles, weights


def review_results(
    mappings: List[Callable[[np.ndarray], None]],
    measurements: List[np.ndarray],
    points: np.ndarray,
    triangles: Any,
    weights: np.ndarray,
) -> None:
    """
    Visualize reconstruction results by plotting the mesh with weights and
    comparing measured and reconstructed histograms.

    For a subset of measurements and corresponding mappings, applies the mapping
    to points, computes histograms of weighted triangle areas, and displays
    both the mesh and histogram comparisons.

    Parameters
    ----------
    mappings : list of callable
        List of functions that modify points in-place to simulate measurement mappings.
    measurements : list of ndarray of shape (M, 2)
        List of measurement arrays with x and y histogram data.
    points : ndarray of shape (N, 2)
        Coordinates of mesh points.
    triangles : object
        Mesh triangles structure expected to have a `.simplices` attribute.
    weights : ndarray of shape (M,)
        Weights associated with each triangle in the mesh.

    Returns
    -------
    None
    """
    plt.figure()
    # Plot the original mesh with weights
    plot_mesh(points, triangles.simplices, weights)

    n_meas = len(measurements)
    n_display = 10
    n_step = max(1, n_meas // n_display)  # Avoid division by zero

    # Loop over a subset of measurements and mappings for visualization
    for meas, map_ in zip(measurements[::n_step], mappings[::n_step]):
        # Extract measurement histogram data (downsample every second point)
        hist_x_meas, hist_y_meas = meas[::2, 0], meas[::2, 1]

        bins = len(hist_x_meas)
        step = (hist_x_meas.max() - hist_x_meas.min()) / bins
        range_ = (hist_x_meas.min() - step / 2, hist_x_meas.max() + step / 2)

        # Copy points and apply mapping function (in-place modification)
        points_tmp = points.copy()
        map_(points_tmp)

        # Compute histogram from reconstructed weights and mesh
        hist_x, hist_y, matrix = histogram(
            points_tmp, triangles.simplices, range_, bins, weights
        )

        plt.figure()

        # Plot the mapped mesh with weights
        plt.subplot(2, 1, 1)
        plot_mesh(points_tmp, triangles.simplices, weights)
        plt.xlim(*range_)

        # Plot reconstructed vs measured histograms
        plt.subplot(2, 1, 2)
        plt.plot(hist_x, hist_y, label="Reconstructed")
        plt.plot(hist_x_meas, hist_y_meas, label="Measured")
        plt.legend()

    plt.show()


def derive_mask(omegas: List[np.ndarray]) -> np.ndarray:
    """
    Derive a boolean mask for variables based on their usage frequency across measurement matrices.

    Variables used in almost all measurement matrices (above 99% of max usage) are marked True,
    others False. This mask can be used to selectively update variables during reconstruction.

    Parameters
    ----------
    omegas : list of ndarray of shape (M_i, N)
        List of measurement matrices.

    Returns
    -------
    mask : ndarray of shape (N,)
        Boolean mask indicating variables with usage greater than 99% of the maximum usage.
    """
    # Compute usage counts of each variable across omegas
    usage = compute_variable_usage(omegas)

    # Threshold set at 99% of the maximum usage
    threshold = usage.max() * 0.99

    # Variables used more than threshold times are included in the mask
    mask = usage > threshold

    return mask


def generate_linear_equations_system(
    mappings: List[Callable[[np.ndarray], None]],
    measurements: List[np.ndarray],
    points: np.ndarray,
    triangles: Any,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate the linear equation system (omegas and targets) from mappings and measurements.

    For each measurement and corresponding mapping:
    - Downsample measurement histogram.
    - Compute histogram from the mapped points and mesh triangles.
    - Collect the histogram y-values as targets and the corresponding matrices (omegas).

    Parameters
    ----------
    mappings : list of callable
        List of mapping functions that modify points in-place.
    measurements : list of ndarray of shape (M, 2)
        List of measurement arrays containing histogram data.
    points : ndarray of shape (N, 2)
        Coordinates of mesh points.
    triangles : object
        Mesh triangles structure with a `.simplices` attribute.

    Returns
    -------
    omegas : list of ndarray
        List of system matrices derived from histogram computations.
    targets : list of ndarray
        List of target measurement vectors (histogram y-values).
    """
    targets, omegas = [], []

    for i in tqdm(range(len(measurements)), desc="Generating linear system"):
        meas = measurements[i]
        map_ = mappings[i]

        # Downsample measurement histogram (take every 2nd point)
        hist_x_meas, hist_y_meas = meas[::2, 0], meas[::2, 1]

        bins = len(hist_x_meas)
        step = (hist_x_meas.max() - hist_x_meas.min()) / bins
        range_ = (hist_x_meas.min() - step / 2, hist_x_meas.max() + step / 2)

        # Apply mapping to a copy of points
        points_tmp = points.copy()
        map_(points_tmp)

        # Compute histogram from mapped points, triangles, and range
        hist_x, hist_y, matrix = histogram(
            points_tmp, triangles.simplices, range_, bins
        )

        targets.append(hist_y_meas)
        omegas.append(matrix)

    return omegas, targets


def generate_mesh() -> Tuple[np.ndarray, Any]:
    """
    Generate a 2D triangular mesh over a square grid.

    The mesh is generated by creating a grid of points within the specified ranges,
    then triangulating those points.

    Returns
    -------
    points : ndarray of shape (N, 2)
        Coordinates of the generated grid points.
    triangles : object
        Triangulation object representing the mesh triangles.
    """
    # Generate a grid of points over the specified x and y ranges
    points = generate_points_grid(
        x_range=(-1.1, 1.1), y_range=(-1.1, 1.1), num_points=4096
    )
    # Generate triangles using Delaunay triangulation on the points
    triangles = generate_triangle_mesh(points)

    return points, triangles
