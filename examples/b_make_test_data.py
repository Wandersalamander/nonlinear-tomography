"""
Nonlinear Tomography Simulation Script
======================================

This script performs an iterative nonlinear tomography simulation, updating
a dataset over multiple iterations while visualizing and saving histogram
data at each step.

The simulation loads initial data, then repeatedly performs the following:
- Plots the current dataset.
- Computes and plots a histogram of the first dimension of the data.
- Saves the histogram data to a `.npy` file.
- Advances the data state using a physics-based step function.

Modules:
--------
- matplotlib.pyplot: For plotting data and histograms.
- numpy: For numerical operations, especially histogram computation and file saving.
- nonlinear_tomography.physics.step: Advances the simulation state by one step.
- nonlinear_tomography.resources.loader: Provides data loading and plotting utilities.

Main Loop Parameters:
---------------------
- n (int): Number of steps to advance the simulation each iteration (59).
- speed (float): Speed parameter controlling the step size (0.01).
- maxiter (int): Maximum number of iterations to perform (100).
- turn_i (int): Cumulative step counter to track simulation progress.

Usage:
------
Run this script as a standalone program. The simulation visualizes progress
in two subplots: the raw data and its histogram.

The histogram data for each iteration is saved in 'resources/lambdas/'
directory with filenames indicating the iteration and speed.

Example:
--------
    $ python simulation_script.py

Note:
-----
Ensure the 'resources/lambdas/' directory exists before running to avoid file I/O errors.

"""

import matplotlib.pyplot as plt
import numpy as np
from a_define_physics import blackbox
from resources.loader import load, plot


def main():
    """
    Run the nonlinear tomography iterative simulation.

    The function loads initial data and iteratively updates it using a physics
    step function. After each iteration, it plots the current data and its histogram,
    saves the histogram data to a file, and advances the simulation state.

    Returns
    -------
    None

    Side Effects
    ------------
    - Displays a live updating matplotlib plot with two subplots.
    - Saves histogram data as .npy files in 'resources/lambdas/'.
    - Uses `nonlinear_tomography.physics.step` to update data in place.
    """
    data = load()
    n = 59
    speed = 0.01
    maxiter = 13
    turn_i = 0
    for i in range(maxiter):
        plt.subplot(2, 1, 1)
        plt.cla()
        plot(data)
        hist_y, hist_edges = np.histogram(data[:, 0], bins=512, density=True)
        hist_x = hist_edges[:-1] + (hist_edges[1] - hist_edges[0]) / 2
        hist_data = np.concatenate(
            (hist_x[:, None], hist_y[:, None]), axis=1, dtype=float
        )
        np.save(
            f"resources/lambdas/turni_{turn_i}_speed_{speed}.npy", hist_data
        )
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.plot(hist_x, hist_y)
        plt.draw()
        plt.pause(0.1)
        blackbox(data, n, speed)
        turn_i += n
    plt.show()


if __name__ == "__main__":
    main()
