# TENSOR EMBEDDING
from typing import List

from lib.base import *
from numba import jit, njit

# a_1...a_t is mapped to index  A^{t-1} a_1 + ... + A * a_{t-1} + 1 * a_t
@jitclass(sketchparams_spec + [('pow', nb.int32[:]), ('char_list', nb.int8[:])])
class TE(Sketcher):
    # https://github.com/numba/numba/issues/1694
    __init__Sketcher = Sketcher.__init__

    def __init__(self, params):
        self.__init__Sketcher(params)

        assert self.t < 16 #t larger than 16 then, 4^t > 2^32 and int32 will overflow

        self.pow = np.zeros(self.t + 1, np.int32)
        self.pow[0] = 1
        for i in range(1, self.t + 1):
            self.pow[i] = self.A * self.pow[i - 1]

    # NOTE: The sketch is stored as float64 here so counting won't overflow.
    def _empty_tensor(self):
        Ts = list()
        for l in self.pow:
            Ts.append(np.zeros(l, np.float64))
        return Ts

    # Return the sketch for the concatenation of two sequences.
    # TODO: Optimize this to modify Tr in place.
    def _join(self, Tl, Tr):
        Ts = self._empty_tensor()
        for tr in range(self.t + 1):
            for tl in range(self.t + 1 - tr):
                Ts[tl + tr] += np.kron(Tl[tl], Tr[tr])
        return Ts

    # Returns the raw 1D count sketches for all tuple sizes up to t.
    # NOTE: This returns counts, not frequencies.
    
    def _full_sketch(self, seq: np.array):
        Ts = self._empty_tensor()

        Ts[0][0] = 1

        # sketch
        for c in seq:
            for i in range(self.t - 1, -1, -1):
                for j in range(len(Ts[i])):
                    
                    Ts[i + 1][self.A * j + c] += Ts[i][j]
        return Ts

    # def sketch_one(self, seq: Sequence) -> SketchedSequence:
    #     full_sketch = self._full_sketch(seq.seq)
    #     self._normalize(seq.len(), full_sketch[self.t])
    #     sketch = np.array([x for x in full_sketch[self.t]], dtype=nb.float32)
    #     return SketchedSequence(seq, sketch)

    # Returns the sketch for the given t as frequencies.
    # def sketch(self, seqs: List[Sequence]) -> List[SketchedSequence]:
    #     return [self.sketch_one(seq) for seq in seqs]

    ## functions added by Sayan ##

    # @final
    # def _full_sketch_str(self, seq: str):
    #     return self._full_sketch(Sequence.remap(seq))

    @final
    def _normalize(self, seq_len, full_sketch):
        if self.normalize:
            # Normalization factor.
            n = seq_len
            nct = nb.float64(1)
            for i in range(self.t):
                nct = nct * (n - i) / (i + 1)
            full_sketch /= nct

    @final
    def sketch(self, seq: str) -> np.ndarray:
        full_sketch = self._full_sketch(encode_seq(seq))

        self._normalize(len(seq), full_sketch[self.t])
        
        sketch = np.array([x for x in full_sketch[self.t]], dtype=nb.float32)
        return sketch
    
    '''@final
    def won_sketch(self, seq: str) -> np.ndarray:
        full_sketch = self._full_sketch(encode_seq(seq))
        
        sketch = np.array([x for x in full_sketch[self.t]], dtype=nb.float32)
        return sketch'''
