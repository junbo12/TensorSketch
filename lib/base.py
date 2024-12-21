# Contains sketch base classes and helper methods

import random

import numpy as np
import numba as nb
from numba import njit
from numba.experimental import jitclass
from numba.typed import List
from typing import List, Tuple, final

# from lib.sequence import *

# A SketchedSequence contains a sequence and its sketch.
# The sketch must be a 1D array of float32s.
# @jitclass([('seq', Sequence_type), ('sketch', nb.float32[::1])])
# class SketchedSequence:
#     def __init__(self, seq: Sequence, sketch):
#         self.seq = seq
#         self.sketch = sketch


# SketchedSequence_type = SketchedSequence.class_type.instance_type

# Compute the Euclidean distance between two sketched sequences.
# @njit
# def dist(ss1: np.ndarray, ss2: np.ndarray) -> np.float32:
#     return np.linalg.norm(ss1.sketch - ss2.sketch)


# Return a sorted list of (dist, seq1, seq2).
# @njit
# def pairwise_dists(
#     seqs: List[SketchedSequence],
# ) -> List[Tuple[np.float32, SketchedSequence, SketchedSequence]]:
#     d = []
#     for j in range(len(seqs)):
#         for i in range(j):
#             d.append((dist(seqs[i], seqs[j]), seqs[i], seqs[j]))
#     d.sort(key=lambda tup: tup[0])
#     return d


sketchparams_spec = [
    ('A', nb.int32),
    ('t', nb.int32),
    ('D', nb.int32),
    ('normalize', nb.bool_),
    ('L', nb.int32),
    ('DL', nb.int32),
]


@jitclass(sketchparams_spec)
class SketchParams:
    def __init__(self, A, t, D, normalize=True, L=1):
        # Alphabet size
        self.A = A
        # Tensor Sketch tuple size
        self.t = t
        # Tensor Sketch embed dimension
        self.D = D
        # Return frequencies instead of counts
        self.normalize = normalize

        # GPU Sketch
        # Amount of work per thread, must divide D.
        # Spawn t*(D/L) instead of t*D threads when this is > 1.
        self.L = L
        assert D % L == 0
        self.DL = D // L


SketchParams_type = SketchParams.class_type.instance_type

# NOTE: Sketchers are not always jitted, since e.g. CUDA invocations do not support this.
class Sketcher:
    def __init__(self, params: SketchParams):
        self.A = params.A
        self.t = params.t
        self.D = params.D
        self.normalize = params.normalize
        self.L = params.L
        self.DL = params.DL

        # added by sayan
        self.A = 4  # reset to DNA alphabet

        self.char_list = np.full(256, -1, np.int8)
        for x in 'ACGT':
            self.char_list[ord(x)] = ((ord(x) >> 1) & 3)

    # [Optional] sketch a single sequence for all t' <= t.
    def _full_sketch(self, seq: np.array):
        assert False, "Must be overridden in derived class"

    # Sketch a single sequence.
    def sketch(self, seq: str) -> np.ndarray:
        assert False, "Must be overridden in derived class"

    # Sketch a list of sequences.
    # def sketch(self, seqs: List[Sequence]) -> List[SketchedSequence]:
    #     pass

@njit
def encode_seq(seq: str) -> np.ndarray:
    """
    Encode a sequence of DNA characters into an array of integers (A->0, C->1, T->2, G->3)
    :param seq: DNA sequence composed of A,C,G,T only (no Ns)
    :return: The encoded sequence
    """
    v = np.empty(len(seq), dtype=np.int8)
    for i, x in enumerate(seq):
        v[i] = ((ord(x) >> 1) & 3)  # maps DNA bases to {0, 1, 2, 3}. Defined only on {A,C,G.T}
    return v