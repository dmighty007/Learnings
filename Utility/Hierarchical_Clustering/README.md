# Hierarchical Clustering and Molecular Analysis

## Overview

This Python script is designed to perform hierarchical clustering on a molecular dynamics (MD) simulation dataset using the `MDAnalysis` library. It aims to extract and analyze the largest connected cluster of molecules in the system.

Hierarchical clustering is a data analysis technique that groups similar data points together based on their distances. In the context of molecular dynamics, this script can be useful for identifying and analyzing clusters of molecules that are closely connected in a simulation.

## Features

- **Hierarchical Clustering**: The script uses hierarchical clustering to group molecules based on their pairwise distances.

- **Connected Component Analysis**: It identifies the largest connected cluster of molecules within the simulation.

- **Visualization**: The script generates a visualization of the clustered molecules using PyMOL.

## Usage

To use this script, you need to provide the following inputs:

- `universe`: A `MDAnalysis.Universe` object or any other object containing information about the molecular system.

- `frame`: The frame number in the trajectory data to analyze.

- `num_atoms`: The total number of atoms in the system.

- `thresh1` (optional): A threshold value used in the clustering algorithm to determine the maximum distance between two atoms for them to be considered part of the same cluster.

- `thresh2` (optional): A threshold value used in the clustering algorithm to determine the minimum distance between atoms in order to consider them as part of the same cluster.

- `verbose` (optional): A boolean flag that controls whether additional information is printed during execution.

## How it works

1. The script calculates the minimum distance between pairs of atoms in the specified frame of the MD trajectory.

2. It performs hierarchical clustering on the minimum distance matrix and extracts mini clusters based on the hierarchical clustering labels.

3. The script analyzes the mini clusters, identifies broken clusters, and combines them into a large connected cluster.

4. It calculates the connectivity matrix between atoms in the large cluster and identifies bonds based on a distance threshold.

5. The largest connected cluster is extracted and its positions are stored.

6. A PyMOL visualization script is generated for visualizing the clusters.

## Example

```python
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from HC import Hierarchical_Clastering

# Provide input data
universe = mda.Universe("system.pdb", "trajectory.xtc")
frame = 0
num_atoms = len(universe.atoms)

# Initialize and run cluster analysis
cluster = FetchCluster(universe, frame, num_atoms)
cluster.create_visulization()
```

## Dependencies

- `scipy`
- `numpy`
- `MDAnalysis`

Make sure to install these dependencies before using the script.

## License

This script is provided under the [MIT License](LICENSE).

## Author

This script was developed by [Your Name]. Please feel free to reach out for any questions or improvements.

Enjoy analyzing your MD simulation data using hierarchical clustering!
