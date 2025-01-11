# TENSOR EMBEDDING

from lib.base import *
from numba import jit, njit, prange
import mmh3

#@jitclass(sketchparams_spec +[('char_list', nb.int8[:])])
class MH(Sketcher):#params = SketchParams(A=4, t=subseq_len, D=tensor_sketch_dim)
    __init__Sketcher = Sketcher.__init__
    def __init__(self, params):
        self.__init__Sketcher(params)
        random.seed(31415)
        self.hashers = [lambda x, i=i: mmh3.hash(x,i,False) for i in range(self.D)]
        
    @final
    def sketch(self, seq): 
        assert len(seq)>0    
        vector = []
        kmers = [seq[i:i+self.t] for i in range(len(seq) - self.t + 1)]
        assert len(kmers)>0
        for i in range(self.D):
            hash_values = [self.hashers[i](kmer) for kmer in kmers]
            vector.append(min(hash_values))
    
        
        return vector
    
    @final
    def mh_sketch(self,seq):
        assert len(seq)>=self.t  
        vector = []
        kmers = [seq[i:i+self.t] for i in range(len(seq) - self.t + 1)]
        hash_values = [hash(kmer) for kmer in kmers]
        hash_values.sort()
        vector = hash_values[0:self.D]
           
        
        return vector
    
@jitclass(sketchparams_spec + [('hashes', nb.int32[:, :]), ('signs', nb.float32[:, :]), ('char_list', nb.int8[:])])
class TS(Sketcher):
    __init__Sketcher = Sketcher.__init__

    def __init__(self, params):
        self.__init__Sketcher(params)

        random.seed(31415)
        
        
        # An A*t array of random integers in [0, D)
        self.hashes = np.empty((self.A, self.t), dtype=np.int32)
        # An A*t array of random +-1
        self.signs = np.empty((self.A, self.t), dtype=np.float32)
        for c in range(self.A):
            for k in range(self.t):
                self.hashes[c][k] = random.randrange(0, self.D)
                self.signs[c][k] = random.randrange(-1, 2, 2)


    def _full_sketch(self, seq: np.array):
        # NOTE: The sketch is stored as float64 here so counting won't overflow.
        T = np.zeros((self.t + 1, self.D), dtype=np.float64)
        T[0][0] = 1

        for c in seq:
            for k in range(self.t - 1, -1, -1):
                h = self.hashes[c][k]
                s = self.signs[c][k]
                for l in range(self.D):
                    r = l + h if l + h < self.D else l + h - self.D
                    T[k + 1][l] += s * T[k][r]

        return T

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
    
class KS(Sketcher):
    __init__Sketcher = Sketcher.__init__
    def __init__(self, params):
        self.__init__Sketcher(params)
        self.RADIX = 4

    @staticmethod
    @njit
    def encode_seq(seq: str) -> np.ndarray:
        """
        Encode a sequence of DNA characters into an array of integers (A->0, C->1, T->2, G->3)
        :param seq: DNA sequence compsed of A,C,G,T only (no Ns)
        :return: The encoded sequence
        """
        v = np.empty(len(seq), dtype=np.int8)
        for i, x in enumerate(seq):
            v[i] = ((ord(x) >> 1) & 3)  # maps DNA bases to {0, 1, 2, 3}. Defined only on {A,C,G.T}
        return v
    
    @staticmethod
    @njit
    def unique(x: np.ndarray):
        """
        Find the unique values and their indices (first occurrence) in the original array (which is unchanged)
        :param x: an 1-dimensional array
        :return: a 2-tuple of the sorted unique values and their indices in the original array
        """
        assert x.ndim == 1, 'x must be a 1D array'
        assert x.shape[0] > 0, 'x must not be empty'
        sorted_indices = np.argsort(x, kind='mergesort')
        uniq_values = np.empty(len(x), dtype=x.dtype)
        uniq_indices = np.empty(len(x), dtype=np.int16)
        uniq_indices[0] = sorted_indices[0]
        uniq_values[0] = x[uniq_indices[0]]
        c = 1
        for i in range(1, len(sorted_indices)):
            if x[sorted_indices[i]] != x[sorted_indices[i - 1]]:
                uniq_indices[c] = sorted_indices[i]
                uniq_values[c] = x[sorted_indices[i]]
                c += 1
        return uniq_values[:c], uniq_indices[:c]

    @staticmethod
    @njit
    def seq2kmers(encseq: np.ndarray, K, RADIX) -> np.ndarray:
        """
        A vector of k-mers from a sequence
        :param K: kmer length
        :param encseq: encoded sequence
        :return: vector of kmers (encoded into integers)
        """
        if encseq.shape[0] < K:
            return None
        D = RADIX ** K
        M1 = D // RADIX
        v = np.empty((encseq.shape[0] - K + 1,), dtype=np.int16)  # we will not have k-mers longer than 8
        v[0] = 0
        for i in range(K):
            v[0] = ((v[0] * RADIX) + encseq[i])
        for i in prange(1, encseq.shape[0] - K + 1):
            outgoing = encseq[i - 1]
            incoming = encseq[i + K - 1]
            v[i] = ((v[i - 1] - M1 * outgoing) * RADIX) + incoming
        return v

    @final
    def vector_of_distances(self, seq: str) -> np.ndarray:
        """
        Compute the distances between k-mers consecutive (k_i to k_{i+1}) in the ordered universe of all k-mers
        and use the distances to form a vector. If a k-mer is not present in the sequence, its distance from its previous
        and next k-mers is 0.
        :param seq: input sequence
        :return: a (D-1)-dimensional vector v, such that v[i] = the distance in the sequence between the i-th and (i+1)-th k-mer
        in the ordered universe of all k-mers, or 0 if either k-mer is not present in the sequence.
        """
        kmers = KS.seq2kmers(KS.encode_seq(seq), self.t, self.RADIX)
        assert kmers.size < 128  # the distances must be in [-128, 127] so the sequence cannot be longer than 127
        unique_kmers, indices = KS.unique(kmers)
        v = np.zeros(self.RADIX ** self.t - 1, dtype=np.int8)  # we cannot handle sequences larger than 127
        for i in range(unique_kmers.size - 1):
            if unique_kmers[i + 1] == unique_kmers[i] + 1:
                v[i] = indices[i + 1] - indices[i]
        return v
    
    @final
    def vector_of_positions(self,seq: str) -> np.ndarray:
        """
        Encode the positions of k-mers into a D-dimensional vector
        :param K: kmer length
        :param seq: input sequence
        :return: a D-dimensional vector v, such that v[i] = the position (1-based) in the sequence of the i-th k-mer
        in the ordered universe of all k-mers, or 0 is the k-mer is not present in the sequence
        """
        kmers = KS.seq2kmers(KS.encode_seq(seq), self.t, self.RADIX)
        assert kmers.size < 255
        unique_kmers, indices = KS.unique(kmers)
        v = np.zeros(self.RADIX ** self.t, dtype=np.uint8)  # we cannot handle sequences larger than 255
        for i in range(unique_kmers.size):
            v[unique_kmers[i]] = indices[i] + 1
        return v


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
