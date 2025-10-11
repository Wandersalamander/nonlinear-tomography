import os
from typing import Any, Callable, List, Tuple

import numpy as np
from a_define_physics import blackbox

from nonlinear_tomography.core import reconstruct, review_results


def main() -> None:
    """
    Load measurement files, create mappings, perform reconstruction,
    and visualize the results.

    The function expects `.npy` files in `resources/lambdas` directory.
    """
    # List all measurement filenames (without .npy extension)
    files: List[str] = [
        f[:-4] for f in os.listdir("resources/lambdas") if f.endswith(".npy")
    ]

    # Load measurement data from .npy files
    measurements: List[np.ndarray] = [
        np.load(f"resources/lambdas/{f}.npy") for f in files
    ]

    # Parse parameters from filenames for blackbox mapping
    parameters: List[List[str]] = [f.split("_")[-3::2] for f in files]

    # Create list of mapping functions that apply 'blackbox' with parsed params
    mappings: List[Callable[[np.ndarray], None]] = [
        lambda x, p=p: blackbox(x, float(p[0]), float(p[1]))
        for p in parameters
    ]

    # Perform the core reconstruction
    points, triangles, weights = reconstruct(measurements, mappings)

    # Visualize and review the results
    review_results(mappings, measurements, points, triangles, weights)


if __name__ == "__main__":
    main()
