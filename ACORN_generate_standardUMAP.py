#!/usr/bin/env python
# coding: utf-8

# Uses UMAP and HDBSCAN to generate an initial set of clusterings
#
# References:
# McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
# McInnes, L, Healy, J, Astels, S, hdbscan: Hierarchical density based clustering, JOSS https://doi.org/10.21105/joss.00205
#
# https://umap-learn.readthedocs.io/
# https://hdbscan.readthedocs.io/

import numpy as np
import pandas as pd
import os
import sys
import random
import math
import pathlib
import scipy
import hdbscan
from umap import umap_ as umap
from sklearn.utils import check_random_state

inputpathorarray = r"dataset/acorn-normalizedWF.csv" # may point to a csv file, an h5 file containing a key named /normalizedWF, a mat file containing a matrix named normalizedWF, or itself directly contain the normalized waveform data.
currentoutputfolder = r"ACORN-test" # will create csv files here; one for each constituent clustering. ACORN_consensus, which will create the final consensus clustering, will expect this folder as its first argument

if len(sys.argv) > 1:
    inputpathorarray = sys.argv[1]
else:
    print(f"Warning: Missing input argument. Using default location: {inputpathorarray}")
    
if len(sys.argv) > 2:
    currentoutputfolder = sys.argv[2]
else:
    print(f"Warning: Missing output argument. Using default location: {currentoutputfolder}")

if isinstance(inputpathorarray, str):
  if inputpathorarray[-4:].lower() == ".csv":
    normalizedWF = pd.read_csv(inputpathorarray, header=None)
  elif inputpathorarray[-3:].lower() == ".h5":
    import h5py
    with h5py.File(inputpathorarray, "r") as h5f:
      normalizedWF = np.transpose(h5f["/normalizedWF"][()])
  elif inputpathorarray[-4:].lower() == ".mat":
      loadedfrommatlab = scipy.io.loadmat(inputpathorarray)
      normalizedWF = loadedfrommatlab["normalizedWF"]
  else:
    raise ValueError('Unexpected input filename. Input filenames should end in ".csv" or ".h5" or ".mat"')
else:
  normalizedWF = inputpathorarray
 
pathlib.Path(currentoutputfolder).mkdir(exist_ok=True)

neuronn = normalizedWF.shape[0]

nneighrange = [round(neuronn/3), round(neuronn/4), round(neuronn/6)]
ncomprange = [4, 8, 16]
if neuronn>=32:
    ncomprange.append(32)
if neuronn>=64:
    ncomprange.append(64)
mindistrange = [0.3, 0.4, 0.5, 0.6, 0.7]
minclusterrange = [12, 16, 20]
#minsamplesrange is set later based on the actual minclusterrange at the time

randomseed = 42
random.seed(randomseed)
np.random.seed(randomseed)

i = 0
for nneigh in nneighrange:
    print('nneigh '+str(nneigh))

    for ncomp in ncomprange:
        print('ncomp '+str(ncomp))

        for mindist in mindistrange:
            print('mindist '+str(mindist))

            umapresult = umap.UMAP(n_neighbors=nneigh, n_components=ncomp, min_dist=mindist, n_jobs=1, random_state=randomseed).fit_transform(normalizedWF)
            
            for mincluster in minclusterrange:
                print('mincluster '+str(mincluster))

                minsamplesrange = [round(mincluster/1.5), round(mincluster/2), round(mincluster/2.5), round(mincluster/3), round(mincluster/4)]
                for minsamples in minsamplesrange:
                    i+=1

                    hdbscanresult = hdbscan.HDBSCAN(min_cluster_size=mincluster, min_samples=minsamples).fit_predict(umapresult)

                    iterationresults = pd.DataFrame(umapresult)
                    #iterationresults['waveform'] = list(normalizedWF)
                    iterationresults['cluster'] = hdbscanresult
                    iterationresults.to_csv(os.path.join(currentoutputfolder, 'ACORN-iteration' + str(i) + '.csv'))

print('done')

