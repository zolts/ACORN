#!/usr/bin/env python
# coding: utf-8
"""
cleanspike.py -- Python port of cleanspike.m from the ACORN repository.
 
"Pattern-matched noise imputation": every sweep is aligned on its trough, and
the parts of the spike window before and after the spike itself are replaced by
stretches of that sweep's own background noise, chosen so that they join the
remaining waveform as smoothly as possible.
 
    cleaned = cleanspike(inputsweeps, stimulusindex, spikestartindex,
                         spiketroughindex, spikeendindex, samplingfrequency)
 
* inputsweeps should be a matrix of sweeps-by-samples
* stimulusindex should be, in samples, the time of the main stimulation (set it to NaN if not applicable!, e.g. when no stimulation was delivered)
* spikestartindex should be, in samples, the time when the mean spike is considered to have begun (better to err on the side of including a bit more than needed)
* spiketroughindex should be, in samples, the time when the mean spike shape is at its lowest
* spikeendindex should be in, in samples, the time when the mean spike is considered to have ended (better to err on the side of including a bit more than needed)
* samplingfrequency should be the number of samples per second
 
* outputsweeps will be a matrix of sweeps-by-samples, with spike waveforms "cleaned" through "pattern-matched noise imputation"
* the optional second return value will be a matrix of sweeps-by-samples, with spike waveforms aligned but NOT "cleaned" through "pattern-matched noise imputation"
 
The calculation follows the MATLAB file line by line (the MATLAB variable names
are kept so the two can be read side by side).  Deliberate differences:
 
* INDEXING.  Sample indices are 0-based by default, as everywhere else in
  Python.  If your indices come from MATLAB (1-based), pass ``index_base=1``;
  the numbers are then interpreted exactly as cleanspike.m interprets them.
* The second MATLAB output (nargout > 1) is requested with
  ``return_aligned=True``.
* The input array is never modified, and it is always processed in float64.
  (cleanspike.m computes in single precision if given single precision, so
  results then agree to about 7 digits only.  Integer arrays cannot hold NaN,
  which the method relies on; cleanspike.m needs floating-point input, the
  port converts.)
* Out-of-range windows raise an error instead of silently wrapping around
  (negative indices mean "count from the end" in numpy).
 
Requires only numpy.
"""
 
from __future__ import annotations
 
import math
import warnings
 
import numpy as np
 
__all__ = ["cleanspike"]
 
# --- constants, identical to cleanspike.m ------------------------------------
SPIKEWINDOWMS = 2.0            # ms, duration
TROUGHMS = +0.3                # ms, relative to spike window start
ALIGNMENTSEARCHRADIUSMS = 0.1  # ms, max for aligning troughs across sweeps
PRESTIMULUSENDMS = -8          # ms, relative to stimulusindex
MAINSTIMULUSSTARTMS = -1       # ms, relative to stimulusindex
MAINSTIMULUSENDMS = +2         # ms, relative to stimulusindex
 
 
# --- small helpers that reproduce MATLAB behaviour ---------------------------
def _matlab_round(x: float) -> int:
    """MATLAB round(): ties go away from zero (Python's round() goes to even)."""
    a = abs(x)
    r = math.floor(a)
    if a - r >= 0.5:
        r += 1
    return int(math.copysign(r, x))
 
 
def _as_index(value, name: str) -> int:
    """Accept only whole numbers as sample indices (MATLAB would error too)."""
    f = float(value)
    if not math.isfinite(f) or f != math.floor(f):
        raise ValueError(f"{name} must be a whole number of samples, got {value!r}")
    return int(f)
 
 
def _nanargmin_first(x: np.ndarray) -> tuple[int, bool]:
    """0-based position of the smallest non-NaN value (first one if tied).
 
    MATLAB's min() returns index 1 when everything is NaN; so do we (0), and
    the second return value says that this happened.  (np.argmin would return
    the position of the first NaN, np.nanargmin would raise.)
    """
    valid = ~np.isnan(x)
    if not valid.any():
        return 0, True
    return int(np.nanargmin(x)), False
 
 
def _exclude(mask: np.ndarray, first: int, last: int, name: str) -> None:
    """mask(first:last) = false with MATLAB 1-based inclusive indices.
 
    An empty range (last < first) does nothing, as in MATLAB.  It is NOT passed
    on to numpy, where e.g. mask[0:-50] would blank almost the whole array.
    """
    if last < first:
        return
    if first < 1:
        raise IndexError(f"{name}: range starts before the first sample")
    if last > mask.size:
        # MATLAB would silently grow the logical array; the net effect there is
        # the same as stopping at the last sample.
        warnings.warn(f"{name}: range ends after the last sample; clipped", stacklevel=3)
        last = mask.size
    mask[first - 1:last] = False
 
 
def _nan_counts(x: np.ndarray) -> np.ndarray:
    """c[j] = number of NaNs in x[0:j]; so x[a:b] has c[b] - c[a] NaNs."""
    return np.concatenate(([0], np.cumsum(np.isnan(x))))
 
 
# --- the function -------------------------------------------------------------
def cleanspike(inputsweeps, stimulusindex, spikestartindex, spiketroughindex,
               spikeendindex, samplingfrequency, *, index_base: int = 0,
               return_aligned: bool = False):
    """Clean spike waveforms by pattern-matched noise imputation.
 
    Parameters
    ----------
    inputsweeps : array, sweeps-by-samples
        One sweep per row.  Not modified.
    stimulusindex : number, NaN or None
        Sample of the main stimulation.  NaN/None if no stimulation was
        delivered (nothing stimulus-related is then excluded).
    spikestartindex, spikeendindex : int
        Samples where the mean spike is considered to begin / to have ended
        (better to err on the side of including a bit more than needed).
    spiketroughindex : int
        Sample where the mean spike shape is at its lowest.
    samplingfrequency : number
        Samples per second.
    index_base : 0 or 1, keyword only
        0 (default): the four indices above are Python-style, first sample = 0.
        1: they are MATLAB-style, first sample = 1, exactly as in cleanspike.m.
    return_aligned : bool, keyword only
        Also return the aligned but NOT cleaned spike windows (the optional
        second output of cleanspike.m).
 
    Returns
    -------
    outputsweeps : float64 array, sweeps-by-window samples
        Cleaned spike waveforms (window = 2 ms starting 0.3 ms before trough).
    originalindividualspikeshapes : float64 array, only if return_aligned
    """
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 (Python) or 1 (MATLAB)")
    tomatlab = 1 - index_base  # added to a user index to get the MATLAB index
 
    # MATLAB passes arrays by value; numpy passes references.  Work on a private
    # float64 copy so the caller's data is untouched and NaN can be stored.
    inputsweeps = np.array(inputsweeps, dtype=np.float64, copy=True)
    if inputsweeps.ndim == 1:
        inputsweeps = inputsweeps[np.newaxis, :]  # a single sweep
    if inputsweeps.ndim != 2:
        raise ValueError("inputsweeps must be a sweeps-by-samples matrix")
    sweepn, analysiswindown = inputsweeps.shape
 
    samplingfrequency = float(samplingfrequency)
    # From here on all indices are MATLAB indices (1-based, ranges inclusive);
    # they are turned into numpy slices only at the point of use.
    spikestartindex = _as_index(spikestartindex, "spikestartindex") + tomatlab
    spiketroughindex = _as_index(spiketroughindex, "spiketroughindex") + tomatlab
    spikeendindex = _as_index(spikeendindex, "spikeendindex") + tomatlab
    if stimulusindex is None:
        stimulusindex = math.nan
    stimulusindex = float(stimulusindex)
    if math.isinf(stimulusindex):
        raise ValueError("stimulusindex must be a number, NaN or None")
    stimulusindex += tomatlab
 
    if not 1 <= spiketroughindex <= analysiswindown:
        raise IndexError(f"spiketroughindex is outside the sweep ({analysiswindown} samples)")
 
    alignmentsamples = math.ceil(ALIGNMENTSEARCHRADIUSMS / 1000 * samplingfrequency)
 
    spikewindowstart = math.floor(spiketroughindex - TROUGHMS / 1000 * samplingfrequency)
    spikewindowend = math.ceil(spikewindowstart + SPIKEWINDOWMS / 1000 * samplingfrequency)
    spikewindown = spikewindowend - spikewindowstart + 1
 
    # in samples, relative to spike window start (1-based position in window)
    troughinwindow = 1 + _matlab_round(TROUGHMS / 1000 * samplingfrequency)
 
    if spikewindowstart < 1 or spikewindowend > analysiswindown:
        raise IndexError(
            f"the spike window (samples {spikewindowstart - tomatlab}..{spikewindowend - tomatlab}) does not "
            f"fit inside the sweep ({1 - tomatlab}..{analysiswindown - tomatlab}); cleanspike.m fails here as well")
 
    # --- align the trough of every sweep -------------------------------------
    currentlookfrom = max(spiketroughindex - alignmentsamples, 1)
    currentlookuntil = min(currentlookfrom + 2 * alignmentsamples, analysiswindown)
 
    for sweepi in range(sweepn):
        currentlocalminwhere, _ = _nanargmin_first(inputsweeps[sweepi, currentlookfrom - 1:currentlookuntil])
        currentlocalminwhere += 1  # MATLAB position within the search range
        displacementneeded = spiketroughindex - (currentlookfrom + currentlocalminwhere - 1)
        if displacementneeded > 0:
            inputsweeps[sweepi, displacementneeded:] = inputsweeps[sweepi, :-displacementneeded].copy()
            inputsweeps[sweepi, :displacementneeded] = np.nan
        elif displacementneeded < 0:
            inputsweeps[sweepi, :displacementneeded] = inputsweeps[sweepi, -displacementneeded:].copy()
            inputsweeps[sweepi, displacementneeded:] = np.nan
 
    # .copy(): a numpy slice is a view, a MATLAB slice is a copy
    individualspikeshapes = inputsweeps[:, spikewindowstart - 1:spikewindowend].copy()
    originalindividualspikeshapes = individualspikeshapes.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column -> NaN, silently as in MATLAB
        originalmeanspikeshape = np.nanmean(individualspikeshapes, axis=0)
 
    potentialnoisesweeps = inputsweeps.copy()
 
    ispotentialnoise = np.ones(analysiswindown, dtype=bool)
    # exclude our spike from being considered as a potential source for imputation
    _exclude(ispotentialnoise, spikestartindex, spikeendindex, "spikestartindex:spikeendindex")
 
    if not math.isnan(stimulusindex):  # if it's NaN, nothing stimulus-related is excluded
        # exclude the prestimulus interval
        prestimulusendindex = min(math.ceil(stimulusindex + PRESTIMULUSENDMS / 1000 * samplingfrequency),
                                  analysiswindown)
        _exclude(ispotentialnoise, 1, prestimulusendindex, "prestimulus interval")
        # exclude the main stimulus interval
        mainstimulusstartindex = max(math.floor(stimulusindex + MAINSTIMULUSSTARTMS / 1000 * samplingfrequency), 1)
        mainstimulusendindex = min(math.ceil(stimulusindex + MAINSTIMULUSENDMS / 1000 * samplingfrequency),
                                   analysiswindown)
        _exclude(ispotentialnoise, mainstimulusstartindex, mainstimulusendindex, "main stimulus interval")
 
    # --- exclude "probable spikes", now on a per-sweep basis ------------------
    for sweepi in range(sweepn):
        currentpotentialnoise = potentialnoisesweeps[sweepi, :].copy()
        currentpotentialnoise[~ispotentialnoise] = np.nan  # exclude stimulation artifacts from the noise estimate
        validn = int(np.count_nonzero(~np.isnan(currentpotentialnoise)))
        if validn == 0:
            currentspikethreshold = math.nan
        else:
            # per Quiroga, Nadasdy & Ben-Shaul (2004), after Donoho & Johnstone (1994)
            currentspikethreshold1 = 3 * np.nanmedian(np.abs(currentpotentialnoise)) / 0.6745
            # MATLAB std() normalises by N-1 (numpy's default is N) and gives 0 for a single value
            currentpotentialnoisestd = np.nanstd(currentpotentialnoise, ddof=1) if validn > 1 else 0.0
            currentspikethreshold2 = 0 + 3 * currentpotentialnoisestd  # grand mean should be 0 (highpass)
            # the more conservative of the two thresholds (which will exclude more)
            # (np.fmin ignores a NaN operand like MATLAB's min; Python's built-in min() does not)
            currentspikethreshold = float(np.fmin(currentspikethreshold1, currentspikethreshold2))
        with np.errstate(invalid="ignore"):
            exclude = (~ispotentialnoise
                       | (currentpotentialnoise > currentspikethreshold)
                       | (currentpotentialnoise < -currentspikethreshold))
        potentialnoisesweeps[sweepi, exclude] = np.nan
 
    windowpositions = np.arange(1, spikewindown + 1)  # MATLAB positions within the spike window
    with np.errstate(invalid="ignore"):
        isnegative = originalmeanspikeshape < 0
        # [m(1:end-1)-m(2:end) < 0, false]
        risesnext = np.append(originalmeanspikeshape[:-1] - originalmeanspikeshape[1:] < 0, False)
        # [false, m(2:end)-m(1:end-1) < 0]
        fellbefore = np.append(False, originalmeanspikeshape[1:] - originalmeanspikeshape[:-1] < 0)
 
    def _warn_no_noise(sweepi: int, part: str) -> None:
        warnings.warn(
            f"cleanspike: sweep {sweepi + index_base} has no usable noise stretch for the part {part} "
            "the trough; the first samples of the sweep were used instead (as cleanspike.m does), "
            "so this sweep may contain NaN", stacklevel=3)
 
    # --- Processing BEFORE the trough ----------------------------------------
    # before the trough AND negative AND negative-direction
    found = np.flatnonzero((windowpositions <= troughinwindow - 3) & isnegative & risesnext)
    if found.size > 0:
        dipbackbelowzeroindex = int(found[-1]) + 1  # find(..., 1, 'last')
        needtoreplaceuntil = dipbackbelowzeroindex
        needtoreplacen = dipbackbelowzeroindex
        for swi in range(sweepn):
            noise = potentialnoisesweeps[swi, :]
            # patterndifferences holds, at each index, the weighted sum of squared distances across a
            # 3 datapoint "overlap" immediately AFTER the interval (index = LAST sample of the candidate)
            patterndifferences = np.full(analysiswindown, np.nan)
            n = analysiswindown
            if n > 3:
                patterndifferences[:n - 3] = (
                    (noise[1:n - 2] - individualspikeshapes[swi, needtoreplaceuntil]) ** 2.0 * 1.0
                    + (noise[2:n - 1] - individualspikeshapes[swi, needtoreplaceuntil + 1]) ** 2.0 * 0.5
                    + (noise[3:n] - individualspikeshapes[swi, needtoreplaceuntil + 2]) ** 2.0 * 0.25)
            # exclude intervals that would stretch outside of the analysis window or include excluded timepoints
            nancount = _nan_counts(noise)
            lastsample = np.arange(n)                       # 0-based last sample of the candidate
            firstsample = lastsample - needtoreplacen + 1   # 0-based first sample of the candidate
            invalid = firstsample < 0
            ok = ~invalid
            invalid[ok] = (nancount[lastsample[ok] + 1] - nancount[firstsample[ok]]) > 0
            patterndifferences[invalid] = np.nan
 
            relativelybestwhere, nonefound = _nanargmin_first(patterndifferences)
            takefrom = relativelybestwhere - needtoreplacen + 1
            if takefrom < 0:
                raise ValueError(
                    f"cleanspike: sweep {swi + index_base} has no usable noise stretch of {needtoreplacen} "
                    "samples for the part before the trough (cleanspike.m stops with an indexing error "
                    "in the same situation)")
            if nonefound:
                _warn_no_noise(swi, "before")
            individualspikeshapes[swi, :needtoreplaceuntil] = noise[takefrom:relativelybestwhere + 1]
 
    # --- Processing AFTER the trough -----------------------------------------
    # after the trough AND negative AND negative-direction
    found = np.flatnonzero((windowpositions > troughinwindow + 3) & isnegative & fellbefore)
    if found.size > 0:
        returntozeroindex = int(found[0]) + 1  # find(..., 1, 'first')
        needtoreplacefrom = returntozeroindex
        needtoreplacen = spikewindown - needtoreplacefrom + 1
        for swi in range(sweepn):
            noise = potentialnoisesweeps[swi, :]
            # ... across a 3 datapoint "overlap" immediately BEFORE the interval
            # (index = FIRST sample of the candidate)
            patterndifferences = np.full(analysiswindown, np.nan)
            n = analysiswindown
            if n > 3:
                patterndifferences[3:] = (
                    (noise[2:n - 1] - individualspikeshapes[swi, needtoreplacefrom - 2]) ** 2.0 * 1.0
                    + (noise[1:n - 2] - individualspikeshapes[swi, needtoreplacefrom - 3]) ** 2.0 * 0.5
                    + (noise[0:n - 3] - individualspikeshapes[swi, needtoreplacefrom - 4]) ** 2.0 * 0.25)
            nancount = _nan_counts(noise)
            firstsample = np.arange(n)
            pastend = firstsample + needtoreplacen          # one past the 0-based last sample
            invalid = pastend > n
            ok = ~invalid
            invalid[ok] = (nancount[pastend[ok]] - nancount[firstsample[ok]]) > 0
            patterndifferences[invalid] = np.nan
 
            relativelybestwhere, nonefound = _nanargmin_first(patterndifferences)
            if nonefound:
                _warn_no_noise(swi, "after")
            individualspikeshapes[swi, needtoreplacefrom - 1:] = \
                noise[relativelybestwhere:relativelybestwhere + needtoreplacen]
 
    # final baseline compensation (the median may have slightly changed as a result of the imputation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        meanspikeshape = np.nanmean(individualspikeshapes, axis=0)
        # plain median on purpose: as in cleanspike.m, a NaN in the mean shape makes the whole output NaN
        finalbaselinecompensation = np.median(meanspikeshape)
    outputsweeps = individualspikeshapes - finalbaselinecompensation
 
    if return_aligned:
        return outputsweeps, originalindividualspikeshapes
    return outputsweeps
