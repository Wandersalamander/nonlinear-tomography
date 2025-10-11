import os.path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

here = os.path.dirname(__file__)


def load():
    """
    Load an RGBA image, extract the alpha channel to find non-transparent pixels,
    and return their normalized and randomly jittered (x, y) coordinates.

    The coordinates are processed so that the origin (0,0) corresponds to the
    center-bottom of the image with y-axis flipped to match image coordinates.
    The x and y values are normalized by dividing by 1024 and then randomly
    perturbed to add noise.

    Returns
    -------
    numpy.ndarray
        A 2D array of shape (N, 2), where N is the number of non-transparent pixels.
        Each row corresponds to the (x, y) coordinates of a pixel, as floats.

    Notes
    -----
    - The input image is expected to be named "lambda.png" in the same directory
      as this script.
    - The alpha channel is used to identify non-transparent pixels.
    - Coordinates are centered around the mean position and scaled.
    """
    # Load the image
    image_path = os.path.join(here, "lambda.png")  # Change if needed
    img = Image.open(image_path).convert("RGBA")  # Ensure image has alpha

    # Convert to numpy array
    data = np.array(img)

    # Extract alpha channel
    alpha = data[:, :, 3]

    # Find indices where alpha is not zero (non-transparent)
    ys, xs = np.nonzero(alpha)
    xs = xs.astype(float)
    ys = ys.astype(float)

    # Optional: Flip y-axis so origin is at bottom-left like in image
    height = img.height
    ys = height - ys
    xs -= np.mean(xs)
    ys -= np.mean(ys)
    xs += (np.random.rand(len(xs)) - 0.5) * 100
    ys += (np.random.rand(len(ys)) - 0.5) * 100
    xs /= 1024
    ys /= 1024
    return np.concatenate((xs[:, None], ys[:, None]), axis=1, dtype=float)


def plot(data):
    """
    Plot 2D data points as a scatter plot.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array of shape (N, 2), where each row represents (x, y) coordinates
        of a point to be plotted.

    Returns
    -------
    None
        This function creates a scatter plot using matplotlib but does not return
        any value.
    """
    xs, ys = data[:, 0], data[:, 1]
    # Scatter plot
    plt.scatter(xs, ys, s=1, color="grey")  # small dots, grey color


if __name__ == "__main__":
    data = load()
    plot(data)
    plt.show()
