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
import random
import math
import pathlib
import scipy
import hdbscan
from umap import umap_ as umap
from sklearn.utils import check_random_state

# SET YOUR INPUT FILE AND OUTPUT FOLDER HERE
matfiletoload = r"h:\MATLAB\waveformmatrix-new.mat" # the mat file should contain a matrix named normalizedWF
currentoutputfolder = r"ACORN-test" # will create csv files here; one for each constituent clustering. ACORN_consensus, which will create the final consensus clustering, will expect this folder as its first argument

loadedfrommatlab = scipy.io.loadmat(matfiletoload)
normalizedWF = loadedfrommatlab["normalizedWF"]
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

