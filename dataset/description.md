### ACORN dataset

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
