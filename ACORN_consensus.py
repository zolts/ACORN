#!/usr/bin/env python
# coding: utf-8
"""
ACORN_consensus.py -- Python port of ACORN_consensus.m from the ACORN repository.
 
Reads the ACORN-iteration*.csv files written by ACORN_generate.py (or
ACORN_generate_standardUMAP.py) and builds the consensus clustering.
 
    from ACORN_consensus import ACORN_consensus
    result = ACORN_consensus("ACORN-test")            # tolerance 2.0
    result.predictions, result.confidence, result.votes, ...
 
or from the command line, after ACORN_generate.py has finished:
 
    python ACORN_consensus.py ACORN-test --output consensus.csv
 
The calculation follows the MATLAB file step by step (MATLAB variable names are
kept).  Deliberate differences:
 
* INDEXING.  By default everything is 0-based, like the HDBSCAN labels in the
  csv files: consensus clusters are 0..K-1 (-1 = uncertain), exemplarindices
  are row numbers into normalizedWF, and cluster k is column k of votes.
  With ``index_base=1`` the numbers are exactly those ACORN_consensus.m returns
  (clusters 1..K, exemplars counted from 1).
* All five outputs are always returned, as one named tuple (MATLAB: nargout).
* The csv files are processed in the order of their iteration number.
  ACORN_consensus.m sorts them by modification time instead (files with equal
  times stay in directory order: 1, 10, 11, ..., 2), and that order changes
  when a folder is copied, unzipped or synced.  The order cannot change which
  exemplars are chosen; it changes the row order of constituentclusters and
  the last digits of votes.  ``sort_by="mtime"`` sorts by time as MATLAB does.
 
Requires numpy and pandas (both already needed by ACORN_generate.py).
"""
 
from __future__ import annotations
 
import math
import re
import warnings
from pathlib import Path
from typing import NamedTuple
 
import numpy as np
import pandas as pd
 
__all__ = ["ACORN_consensus", "consensus_from_clusters", "ACORNResult"]
 
CONST_UNCLASSIFIED = -1
CONST_MISSING = -2  # internal marker for an empty/NaN cluster entry in a csv file
 
 
class ACORNResult(NamedTuple):
    """Outputs of ACORN_consensus, in the order of the MATLAB return values."""
    predictions: np.ndarray          # (neurons,) int; -1 = uncertain
    confidence: np.ndarray           # (neurons,) float; highest / second-highest vote
    votes: np.ndarray                # (neurons, K+1) float; last column = "uncertain"
    exemplarindices: np.ndarray      # (K,) int; neuron index of each cluster's exemplar
    constituentclusters: np.ndarray  # (iterations, neurons) int; the HDBSCAN labels read from the csv files
                                     # (always exactly as in the files: -1 unclustered, 0, 1, ...)
 
 
def _iteration_files(csvfolder, sort_by: str) -> list[Path]:
    folder = Path(csvfolder)
    if not folder.is_dir():
        raise NotADirectoryError(f"{csvfolder!r} is not a folder")
    files = sorted(p for p in folder.glob("ACORN-iteration*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no ACORN-iteration*.csv files in {csvfolder!r}")
 
    if sort_by == "mtime":
        # what ACORN_consensus.m does (dir().datenum); ties keep alphabetical order
        files.sort(key=lambda p: p.stat().st_mtime)
        return files
    if sort_by != "number":
        raise ValueError('sort_by must be "number" or "mtime"')
 
    numbered, other = [], []
    for p in files:
        m = re.fullmatch(r"ACORN-iteration(\d+)\.csv", p.name, flags=re.IGNORECASE)
        if m:
            numbered.append((int(m.group(1)), p))
        else:
            other.append(p)
    if other:
        warnings.warn(
            "files that match ACORN-iteration*.csv but have no plain iteration number are included "
            "(as ACORN_consensus.m would): " + ", ".join(p.name for p in other), stacklevel=3)
    return [p for _, p in sorted(numbered, key=lambda t: t[0])] + other
 
 
def _read_clusters(files: list[Path]) -> np.ndarray:
    """iterations-by-neurons matrix of the 'cluster' columns (NaN where an entry is empty)."""
    HDBSCANclusters = None
    for iterationi, path in enumerate(files):
        # the column is found by name, case-insensitively, as strcmpi() does
        pythontable = pd.read_csv(path, usecols=lambda name: str(name).strip().lower() == "cluster")
        if pythontable.shape[1] != 1:
            raise ValueError(f"{path.name}: expected exactly one 'cluster' column, found {pythontable.shape[1]}")
        try:
            column = pd.to_numeric(pythontable.iloc[:, 0], errors="raise").to_numpy(dtype=np.float64)
        except (ValueError, TypeError) as err:
            raise ValueError(f"{path.name}: the 'cluster' column is not numeric ({err})") from None
 
        if HDBSCANclusters is None:
            neuronn = column.size
            HDBSCANclusters = np.full((len(files), neuronn), np.nan)
        elif column.size != HDBSCANclusters.shape[1]:
            raise ValueError(f"{path.name} has {column.size} rows, but {files[0].name} has "
                             f"{HDBSCANclusters.shape[1]}; all files must describe the same neurons")
 
        HDBSCANclusters[iterationi, :] = column
    return HDBSCANclusters
 
 
def ACORN_consensus(csvfolder, tolerance=2.0, *, index_base: int = 0, sort_by: str = "number") -> ACORNResult:
    """Consensus of the constituent clusterings in ``csvfolder``.
 
    Parameters
    ----------
    csvfolder : path
        Folder containing the ACORN-iteration*.csv files from ACORN_generate.py.
    tolerance : float or None
        Minimum ratio between the highest and the second-highest vote for a
        neuron to be assigned.  Default and recommended 2.0 (1.5 to 3.0 are
        sensible).  None or NaN forces classification of every neuron that
        received any vote at all (not recommended).  0 does not do that: a
        neuron still needs a strict winner that is not "uncertain".
    index_base : 0 or 1, keyword only
        0 (default): clusters 0..K-1 and 0-based exemplar indices.
        1: clusters 1..K and 1-based exemplar indices, i.e. MATLAB's numbers.
        "Uncertain" is -1 in both cases.
    sort_by : "number" or "mtime", keyword only
        Order in which the csv files are processed (see module docstring).
 
    Returns
    -------
    ACORNResult(predictions, confidence, votes, exemplarindices, constituentclusters)
    """
    return consensus_from_clusters(_read_clusters(_iteration_files(csvfolder, sort_by)), tolerance,
                                   index_base=index_base)
 
 
def consensus_from_clusters(HDBSCANclusters, tolerance=2.0, *, index_base: int = 0) -> ACORNResult:
    """The consensus calculation itself, for labels that are already in memory.
 
    ``HDBSCANclusters`` is an iterations-by-neurons array of HDBSCAN labels
    (-1 = unclustered, 0, 1, ... = clusters), i.e. one row per csv file.
    Everything else is as in :func:`ACORN_consensus`.
    """
    if index_base not in (0, 1):
        raise ValueError("index_base must be 0 (Python) or 1 (MATLAB)")
    tolerance = math.nan if tolerance is None else float(tolerance)
    forced = math.isnan(tolerance)  # isnan(tolerance) in the MATLAB code
 
    HDBSCANclusters = np.asarray(HDBSCANclusters, dtype=np.float64)
    if HDBSCANclusters.ndim != 2:
        raise ValueError("HDBSCANclusters must be an iterations-by-neurons array")
    missing = np.isnan(HDBSCANclusters)
    present = HDBSCANclusters[~missing]
    if np.any(present != np.floor(present)) or np.any(present < CONST_UNCLASSIFIED):
        raise ValueError("cluster labels must be whole numbers >= -1")
    if missing.any():
        warnings.warn("some cluster entries are empty; they are ignored (no vote), as in ACORN_consensus.m",
                      stacklevel=3)
    # whole-number labels from here on; an empty entry gets the internal marker CONST_MISSING
    HDBSCANclusters = np.where(missing, CONST_MISSING, HDBSCANclusters).astype(np.int64)
    iterationn, neuronn = HDBSCANclusters.shape
 
    # in the tables, -1 denotes unclustered and 0 denotes a valid cluster, so the number of real
    # clusters is max+1.  (Missing entries, -2, can never be a row maximum unless the row is all missing.)
    clusternumbers = (HDBSCANclusters.max(axis=1) + 1) if neuronn > 0 else np.zeros(iterationn, dtype=np.int64)
    expectedclustern = int(max(clusternumbers.max(initial=0), 0))
 
    # per iteration: which neurons are in a real cluster, and how big each cluster is
    clustered = HDBSCANclusters >= 0
    groupsizes = [np.bincount(HDBSCANclusters[i, clustered[i]], minlength=max(int(clusternumbers[i]), 0))
                  for i in range(iterationn)]
 
    # --- choose the exemplars -------------------------------------------------
    exemplarindices: list[int] = []  # 0-based neuron indices
    for _ in range(neuronn):
        Svalue = np.zeros(neuronn, dtype=np.int64)  # whole numbers throughout, so the result is exact
        for iterationi in range(iterationn):
            labels = HDBSCANclusters[iterationi]
            toadd = groupsizes[iterationi]  # no clash: add the size of the cluster
            if exemplarindices:
                exemplarlabels = labels[exemplarindices]
                # number of already chosen exemplars inside each cluster of this iteration
                clashes = np.bincount(exemplarlabels[exemplarlabels >= 0], minlength=toadd.size)
                # MATLAB: round(-clashes * neuronn/expectedclustern/2, TieBreaker="even").
                # np.rint rounds ties to even, like that call (and unlike MATLAB's plain round()).
                penalty = np.rint(-clashes * neuronn / expectedclustern / 2).astype(np.int64) \
                    if expectedclustern > 0 else np.zeros_like(clashes)
                toadd = np.where(clashes == 0, toadd, penalty)
            members = clustered[iterationi]
            Svalue[members] += toadd[labels[members]]
        maxwhere = int(np.argmax(Svalue))  # first maximum, as MATLAB's max()
        if Svalue[maxwhere] > 0:
            exemplarindices.append(maxwhere)
        else:
            break
    exemplars = np.asarray(exemplarindices, dtype=np.int64)
    exemplarn = exemplars.size
    CONST_UNCERTAIN = exemplarn  # 0-based column of the "uncertain" votes (MATLAB: exemplarn + 1)
 
    # --- votes ----------------------------------------------------------------
    # votes include an "uncertain" (none of the above) column because confidence will be the ratio of the
    # highest and second-highest votes; example: with 2 votes for A, 1 vote for B, and 97 votes for
    # "none of the above", the result should NOT be "A with a confidence of 2.0" but "uncertain"
    votes = np.zeros((neuronn, exemplarn + 1))
    for iterationi in range(iterationn):
        labels = HDBSCANclusters[iterationi]
        votes[labels == CONST_UNCLASSIFIED, CONST_UNCERTAIN] += 1
        if exemplarn == 0:
            continue
        exemplarlabels = labels[exemplars]
        # number of exemplars present in each cluster; the votes are split equally between them
        exemplarsincluster = np.bincount(exemplarlabels[exemplarlabels >= 0], minlength=groupsizes[iterationi].size)
        members = np.flatnonzero(clustered[iterationi])
        if members.size == 0:
            continue
        memberlabels = labels[members]
        # A cluster without any exemplar gives no vote.  (MATLAB computes 1/0 = Inf there and assigns it
        # to an empty selection; in Python 1/0 would raise, hence the explicit case.)
        hasexemplar = exemplarsincluster[memberlabels] > 0
        members, memberlabels = members[hasexemplar], memberlabels[hasexemplar]
        share = 1.0 / exemplarsincluster[memberlabels]
        sameclusterasexemplar = memberlabels[:, np.newaxis] == exemplarlabels[np.newaxis, :]
        # one addition per (neuron, exemplar) and iteration, in iteration order, exactly as in the MATLAB loop
        votes[members, :exemplarn] += np.where(sameclusterasexemplar, share[:, np.newaxis], 0.0)
    # disagreement between "cell type A" and "uncertain" should hurt the consensus less than
    # disagreement between "cell type A" and "cell type B"
    votes[:, CONST_UNCERTAIN] = votes[:, CONST_UNCERTAIN] / 2
 
    # --- predictions and confidence --------------------------------------------
    predictions = np.full(neuronn, CONST_UNCLASSIFIED, dtype=np.int64)
    neurons = np.arange(neuronn)
    # MATLAB sort(...,'descend') keeps tied elements in their original order; so does a stable sort of -votes
    sortedvoteindices = np.argsort(-votes, axis=1, kind="stable")
    winner = sortedvoteindices[:, 0]
    confidence = np.zeros(neuronn)
    if exemplarn > 0:
        first = votes[neurons, winner]
        second = votes[neurons, sortedvoteindices[:, 1]]
        with np.errstate(divide="ignore", invalid="ignore"):  # x/0 -> Inf and 0/0 -> NaN, silently as in MATLAB
            confidence = np.where(winner == CONST_UNCERTAIN, 0.0, first / second)
 
    if not forced:
        # Consensus with a minimum threshold for agreement.  A confidence of _exactly 1_ is a tie for
        # first place, so it is left uncertain.  (Comparisons with a NaN confidence are False, as in MATLAB.)
        with np.errstate(invalid="ignore"):
            assign = (confidence >= tolerance) & (confidence > 1) & (winner != CONST_UNCERTAIN)
        predictions[assign] = winner[assign] + index_base
    elif exemplarn > 0:
        # Forced classification instead.  A neuron that no constituent clustering ever classified still
        # cannot be assigned to any consensus cluster.
        whichbest = np.argmax(votes[:, :exemplarn], axis=1)
        assign = votes[neurons, whichbest] > 0
        predictions[assign] = whichbest[assign] + index_base
 
    constituentclusters = HDBSCANclusters
    if np.any(HDBSCANclusters == CONST_MISSING):  # give empty entries back as NaN, as MATLAB does
        constituentclusters = HDBSCANclusters.astype(np.float64)
        constituentclusters[HDBSCANclusters == CONST_MISSING] = np.nan
    return ACORNResult(predictions, confidence, votes, exemplars + index_base, constituentclusters)
 
 
def _main(argv=None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="ACORN consensus step (Python port of ACORN_consensus.m)")
    parser.add_argument("csvfolder", help="folder with the ACORN-iteration*.csv files from ACORN_generate.py")
    parser.add_argument("--tolerance", default="2.0",
                        help='confidence threshold (default 2.0); "nan" forces classification')
    parser.add_argument("--index-base", type=int, choices=(0, 1), default=0,
                        help="0: clusters 0..K-1 (default); 1: clusters 1..K as in MATLAB")
    parser.add_argument("--sort-by", choices=("number", "mtime"), default="number")
    parser.add_argument("--output", help="write neuron, prediction and confidence to this csv file")
    args = parser.parse_args(argv)
 
    result = ACORN_consensus(args.csvfolder, float(args.tolerance), index_base=args.index_base,
                             sort_by=args.sort_by)
    iterationn, neuronn = result.constituentclusters.shape
    print(f"{iterationn} constituent clusterings, {neuronn} neurons, {result.exemplarindices.size} consensus clusters")
    for cluster, count in zip(*np.unique(result.predictions, return_counts=True)):
        print(f"  {'uncertain' if cluster == CONST_UNCLASSIFIED else 'cluster ' + str(cluster)}: {count}")
    if args.output:
        pd.DataFrame({"neuron": np.arange(neuronn) + args.index_base, "prediction": result.predictions,
                      "confidence": result.confidence}).to_csv(args.output, index=False)
        print("saved " + args.output)
 
 
if __name__ == "__main__":
    _main()
