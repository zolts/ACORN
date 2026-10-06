# ACORN
**Adaptive Consensus of Repeated Clusterings**

This repository contains code used in the manuscript "ACORN: Adaptive Consensus of Repeated Clusterings identifies pyramidal neurons and interneurons in the primate motor cortex based on extracellular spike waveforms".

## Requirements
The following Python packages and their dependencies:
* numpy
* pandas
* scipy
* hdbscan
* umap-learn

The entire procedure can be performed in Python.\
The *consensus* step can be done in Python or Matlab. For the latter, 2025b is known to work; other versions most likely work too; no toolbox requirement.

## Usage overview
1. Prepare your data. It should be a 2D matrix, with multiple samples (observations) per units (neurons). In the default use case, each neuron's mean waveform should be aligned, cleaned, and normalized. This can be either a Python array **OR** a Matlab matrix called `normalizedWF` saved inside a `mat` file. ACORN does not assume that its inputs are neural waveforms in particular; in principle it could be used to cluster other types of data too.
2. Run `ACORN_generate.py` or `ACORN_generate_standardUMAP.py`. The first argument is your input (Python data or path to Matlab mat file), the second argument is the output folder. This step generates a large number of constituent clusters as csv files.
3. Run `ACORN_consensus` (Python and Matlab versions are both available) pointed at the folder containing the constituent cluster csv files. This step generates the final consensus clustering results.

## ACORN_generate.py

* **Input arguments**: a Python array or a string path to a Matlab mat file containing a matrix (normalized waveforms in the typical use case). Secondly, the desired output folder.
* **Output**: a large number of constituent clusters (1125 by default) as csv files in the designated folder.

This uses a clustering-optimized variant of UMAP (MST-min with Path Neighbors) from the paper "[Clustering with UMAP: Why and How Connectivity Matters](https://arxiv.org/abs/2108.05525)" by Ayush Dalmia and Suzanna Sia ([repo](https://github.com/adalmia96/umap-mnn)).

## ACORN_generate_standardUMAP.py

* **Input arguments**: a Python array or a string path to a Matlab mat file containing a matrix (normalized waveforms in the typical use case). Secondly, the desired output folder.
* **Output**: a large number of constituent clusters (1125 by default) as csv files in the designated folder.

Alternative to `ACORN_generate.py`; uses standard UMAP (McInnes and Healy) without Dalmia-Sia optimizations.

## ACORN_consensus

Matlab and Python versions both exist: `ACORN_consensus.m` and `ACORN_consensus.py`, respectively.

* **Input argument**: path to the folder produced by *ACORN_generate*.
* **Return values**: an array of consensus clusters (predictions), confidence values, the vote matrix, exemplar indices, and constituent clusterings.
  - *Matlab version*: consensus clusters are always returned, the rest of the return values are optional.
  - *Python version*: all the aforementioned are always returned.

## Credits
* **HDBSCAN** ([repo](https://github.com/scikit-learn-contrib/hdbscan)): McInnes L, Healy J, Astels S, "[hdbscan: Hierarchical density based clustering](https://doi.org/10.21105/joss.00205)", 2017.
* **UMAP** ([repo](https://github.com/lmcinnes/umap)): McInnes L, Healy J, Melville J, "[UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)", 2018.
* **UMAP clustering optimization** ([repo](https://github.com/adalmia96/umap-mnn)): Dalmia A, Sia S, "[Clustering with UMAP: Why and How Connectivity Matters](https://arxiv.org/abs/2108.05525)", 2021.
