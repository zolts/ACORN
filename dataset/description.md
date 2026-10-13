## ACORN dataset

### The full `mat` file

acorn-dataset.mat contains the following variables:
 
* **originalindividualtraces**: for each neuron, each individual action potential, raw.
* **individualtraces**: for each neuron, each individual action potential, processed using `cleanspike`.
 
* **originalWF**: for each neuron, the mean spike waveform, raw.
* **cleanWF**: for each neuron, the mean spike waveform, processed using `cleanspike`.
* **normalizedWF**: for each neuron, the mean spike waveform, processed using `cleanspike` and normalized.
 
* **signaltonoise**: for each neuron, its signal-to-noise ratio.
* **spikewidths**: for each neuron, the spike width of its mean waveform, in µs.
 
* **isINT**: for each neuron, `true` if an interneuron, `false` if not.
* **isPTN**: for each neuron, `true` if a pyramidal tract neuron, `false` if not.
* **groundtruth**: for each neuron, `1` if interneuron, `2` if pyramidal tract neuron.

### The waveform `csv` file

[acorn-normalizedWF.csv](acorn-normalizedWF.csv) contains a copy of the **normalizedWF** data mentioned above, in a comma-delimited format. Each row is a different neuron; each column is a different timepoint. Neurons are INTs and PTNs, as described in the ground truth labels. Spike windows cover 2ms, presented at 100 KHz. Data points are normalized voltage values of action potential waveforms averaged across spikes from the same neuron.

### The ground truth labels `txt` file

[acorn-groundtruthlabels.txt](acorn-groundtruthlabels.txt) contains a copy of the **groundtruth** values mentioned above, in a simple text format. Each row is a different neuron. `1` denotes an interneuron (INT); `2` a pyramidal tract neuron (PTN). Neurons are in the same order in all files.
