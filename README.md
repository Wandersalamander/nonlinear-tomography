
# Nonlinear Tomography

This project performs nonlinear tomographic reconstruction using algebraic techniques 
over a triangular mesh. It reconstructs weights on mesh triangles from measurement data and visualizes the results.

## Features

* Generates a 2D triangular mesh over a specified grid.
* Constructs linear equation systems from measurements and mappings.
* Solves the reconstruction problem using a masked algebraic reconstruction technique (ART).
* Smooths reconstructed weights using neighboring triangle information.
* Provides visualization tools for reviewing mesh and histogram reconstructions.
* Supports customizable mappings based on parameterized physics simulations (`blackbox`).

## Installation

Make sure you have Python 3.8+ installed. Install dependencies using pip:

```bash
git cloe https://github.com/Wandersalamander/nonlinear-tomography
cd nonlinear-tomography
pip install ./
```

## Usage
Generate the histograms for the reconstruction:
```bash
python examples/b_make_test_data.py
```

Then, run the main script:

```bash
python examples/c_perform_reconstruction.py
```

This will:

* Load measurements and parameters.
* Generate mappings between the input and measurement plane using `blackbox` function.
* Perform reconstruction to recover weights.
* Display plots comparing measured and reconstructed data.

## Code Structure

* `generate_points_grid`: Generate grid points for mesh.
* `generate_triangle_mesh`: Create triangulation of points.
* `generate_linear_equations_system`: Builds system matrices and targets from mappings and measurements.
* `algebraic_reconstruction_technique_masked`: Iteratively solves for weights with masking and relaxation.
* `smooth_weights`: Smooth weights across neighboring triangles.
* `reconstruct`: Main pipeline to generate mesh, solve linear system, and smooth weights.
* `review_results`: Visualizes mesh and histograms for qualitative assessment.

## Example

See [c_perform_reconstruction.py](examples/c_perform_reconstruction.py)



## Contact

For questions or contributions, please open an issue.

