#!/usr/bin/env python
# coding: utf-8

# Uses UMAP and HDBSCAN to generate an initial set of clusterings
#
# References:
# McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018
# Dalmia, A, SiaS, Clustering with UMAP: Why and How Connectivity Matters, ArXiv e-prints 2108.05525, 2021, https://github.com/adalmia96/umap-mnn
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

if len(sys.argv) > 2:
    inputpathorarray = sys.argv[1]
    currentoutputfolder = sys.argv[2]
else:
    print("Warning: Missing arguments. Using defaults.")
    inputpathorarray = r"h:\MATLAB\acorn-dataset.mat" # may either point to a mat file containing a matrix named normalizedWF, or (without Matlab) directly contain the normalized waveform data in the form of an array.
    currentoutputfolder = r"ACORN-test" # will create csv files here; one for each constituent clustering. ACORN_consensus, which will create the final consensus clustering, will expect this folder as its first argument


def min_spanning_tree(knn_indices, knn_dists, n_neighbors, threshold):
  rows = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
  cols = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.int32)
  vals = np.zeros(knn_indices.shape[0] * n_neighbors, dtype=np.float32)
  pos = 0
  for i, indices in enumerate(knn_indices):
    for j, index in enumerate(indices[:threshold]):
      if index == -1:
        continue
      rows[pos] = i 
      cols[pos] = index
      vals[pos] = knn_dists[i][j]
      pos += 1
  matrix = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
  Tcsr = scipy.sparse.csgraph.minimum_spanning_tree(matrix)
  Tcsr = scipy.sparse.coo_matrix(Tcsr)
  weights_tuples = zip(Tcsr.row, Tcsr.col, Tcsr.data)
  sorted_weights_tuples = sorted(weights_tuples, key=lambda tup: tup[2])  
  return sorted_weights_tuples

import copy
import heapq

def create_connected_graph(mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity):
  connected_mnn = copy.deepcopy(mutual_nn)
  
  if connectivity == "nearest":
    for i in range(len(knn_indices)): 
      if len(mutual_nn[i]) == 0:
        first_nn = knn_indices[i][1]
        if first_nn != -1:
          connected_mnn[i].add(first_nn) 
          connected_mnn[first_nn].add(i) 
          total_mutual_nn += 1
    return connected_mnn

            
      
  #Create graph for mutual NN
  rows = np.zeros(total_mutual_nn, dtype=np.int32)
  cols = np.zeros(total_mutual_nn, dtype=np.int32)
  vals = np.zeros(total_mutual_nn, dtype=np.float32)
  pos = 0
  for i in connected_mnn:
    for j in connected_mnn[i]:
      rows[pos] = i 
      cols[pos] = j
      vals[pos] = 1
      pos += 1
  graph = scipy.sparse.csr_matrix((vals, (rows, cols)), shape=(knn_indices.shape[0], knn_indices.shape[0]))
  
  
  #Find number of connected components
  n_components, labels = scipy.sparse.csgraph.connected_components(csgraph=graph, directed=True, return_labels=True, connection= 'strong')
  print(n_components)
  label_mapping = {i:[] for i in range(n_components)}

  for index, component in enumerate(labels):
    label_mapping[component].append(index)



  #Find the min spanning tree with KNN
  sorted_weights_tuples = min_spanning_tree(knn_indices, knn_dists, n_neighbors, n_neighbors)
  

  #Add edges until graph is connected
  for pos,(i,j,v) in enumerate(sorted_weights_tuples):

    if connectivity == "full_tree":
      connected_mnn[i].add(j)
      connected_mnn[j].add(i) 
      
      
    elif connectivity == "min_tree" and labels[i] != labels[j]:
      if len(label_mapping[labels[i]]) < len(label_mapping[labels[j]]):
        i, j = j, i
        
      connected_mnn[i].add(j)
      connected_mnn[j].add(i)
      j_pos = label_mapping[labels[j]]
      labels[j_pos] = labels[i]
      label_mapping[labels[i]].extend(j_pos)

  return connected_mnn  

#Search to find path neighbors 

def find_new_nn(knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose=False):
  
  new_knn_dists = [] 
  new_knn_indices = []
  
  for i in range(len(knn_indices)): 
    min_distances = []
    min_indices = []
    
    heap = [(0,i)]
    mapping = {}
          
    seen = set()
    heapq.heapify(heap) 
    while(len(min_distances) < n_neighbors_max and len(heap) >0):
      dist, nn = heapq.heappop(heap)
      if nn == -1:
        continue
    
      if nn not in seen:
        min_distances.append(dist)
        min_indices.append(nn)
        seen.add(nn)
        neighbor = connected_mnn[nn]
        
        for nn_nn in neighbor:
          if nn_nn not in seen:
            distance = 0
            if nn_nn in knn_indices_pos[nn]:
              pos = knn_indices_pos[nn][nn_nn]
              distance = knn_dists[nn][pos] 
            else:
              pos = knn_indices_pos[nn_nn][nn]
              distance = knn_dists[nn_nn][pos] 
            distance += dist
            if nn_nn not in mapping:
              mapping[nn_nn] = distance
              heapq.heappush(heap, (distance, nn_nn))
            elif mapping[nn_nn] > distance:
              mapping[nn_nn] = distance
              heapq.heappush(heap, (distance, nn_nn))
    
    if len(min_distances) < n_neighbors_max:
      for i in range(n_neighbors_max-len(min_distances)):
        min_indices.append(-1)
        min_distances.append(np.inf)
    
    new_knn_dists.append(min_distances)
    new_knn_indices.append(min_indices)
    
    if verbose and i % int(len(knn_dists) / 10) == 0:
      print("\tcompleted ", i, " / ", len(knn_dists), "epochs")
  return new_knn_dists, new_knn_indices



#Calculate the connected mutual nn graph
def mutual_nn_nearest(knn_indices, knn_dists, n_neighbors, n_neighbors_max, connectivity="min_tree", verbose=False):
  mutual_nn = {}
  nearest_n= {}

  knn_indices_pos = [None] * len(knn_indices)

  total = 0
  
  for i, top_vals in enumerate(knn_indices):
    nearest_n[i] = set(top_vals)
    knn_indices_pos[i] = {}
    for pos, nn in enumerate(top_vals):
      knn_indices_pos[i][nn] = pos
  
  total_mutual_nn = 0
  for i, top_vals in enumerate(knn_indices):
    mutual_nn[i] = set()
    for ind, nn in enumerate(top_vals):
      if nn != -1 and (i in nearest_n[nn] and i != nn):
        mutual_nn[i].add(nn)
        total_mutual_nn += 1

  
  connected_mnn = create_connected_graph(mutual_nn, total_mutual_nn, knn_indices, knn_dists, n_neighbors, connectivity )
  new_knn_dists, new_knn_indices = find_new_nn(knn_indices, knn_dists, knn_indices_pos, connected_mnn, n_neighbors_max, verbose)

  
  return connected_mnn, mutual_nn, np.array(new_knn_indices), np.array(new_knn_dists)  

if isinstance(inputpathorarray, str):
  loadedfrommatlab = scipy.io.loadmat(inputpathorarray)
  normalizedWF = loadedfrommatlab["normalizedWF"]
else:
  normalizedWF = inputpathorarray
pathlib.Path(currentoutputfolder).mkdir(exist_ok=True)

neuronn = normalizedWF.shape[0]

processingmetric = 'euclidean'

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

    knn_indices, knn_dists, knn_search_index = umap.nearest_neighbors(
        normalizedWF,
        n_neighbors=nneigh,
        metric = processingmetric,
        metric_kwds = {},
        angular=False,
        random_state = check_random_state(randomseed),
        low_memory=False,
        use_pynndescent=True,
        n_jobs=1,
        verbose=False
    )

    connected_mnn, mutual_nn, new_knn_indices, new_knn_dists  = mutual_nn_nearest(knn_indices, knn_dists, nneigh, nneigh, connectivity= "nearest" , verbose=False)

    # build fuzzy_simplicial_set
    P, sigmas, rhos = umap.fuzzy_simplicial_set(
        X = normalizedWF,
        n_neighbors = nneigh,
        metric = processingmetric,
        random_state = check_random_state(randomseed),
        knn_indices= new_knn_indices,
        knn_dists = new_knn_dists,
    )

    for ncomp in ncomprange:
        print('ncomp '+str(ncomp))

        for mindist in mindistrange:
            print('mindist '+str(mindist))

            a, b = umap.find_ab_params(1.0, mindist)

            umapresult, aux_data = umap.simplicial_set_embedding(
                data = normalizedWF,
                graph = P,
                n_components = ncomp,
                initial_alpha = 1.0,
                a = a,
                b = b,
                gamma = 1.0,
                negative_sample_rate = 5,
                n_epochs = None, #-1,
                init = "spectral",
                random_state = check_random_state(randomseed),
                metric = processingmetric,
                metric_kwds = {},
                densmap = False,
                densmap_kwds = {},
                output_dens = False,
                output_metric= umap.dist.named_distances_with_gradients["euclidean"],
                output_metric_kwds={},
                euclidean_output=True,
                parallel=False,
                verbose=False,
            )
            
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

