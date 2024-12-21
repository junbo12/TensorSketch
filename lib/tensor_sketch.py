# TENSOR SKETCH
from lib.base import *
import gc
import mmh3
class MH(Sketcher):#params = SketchParams(A=4, t=subseq_len, D=tensor_sketch_dim)
    __init__Sketcher = Sketcher.__init__
    def __init__(self, params):
        self.__init__Sketcher(params)
        self.hashers = [lambda x, i=i: mmh3.hash(x,i,False) for i in range(self.D)]

    def sketch(self, seq):       
        vector = []
        kmers = [seq[i:i+self.t] for i in range(len(seq) - self.t + 1)]
        
        for i in range(self.D):
            hash_values = [mmh3.hash(kmer, i) for kmer in kmers]
            vector.append(min(hash_values))
            
        
        return vector
    
    def mh_sketch(self,seq):
        vector = []
        kmers = [seq[i:i+self.t] for i in range(len(seq) - self.t + 1)]
        
        
        hash_values = [self.hashers[0](kmer) for kmer in kmers]
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
    
    

      
    '''def min_hash_sketch(seq: str, k: int=15, sketch_dim:int = 128):
        hashers = [hasher(seed=i) for i in range(128)]
        kmers = seq2kmers(seq, k)
        vector = []
        for i in range(128):
            hash_values = [hashers[i].hash(kmer) for kmer in kmers]
            vector.append(min(hash_vales))
	return vector
    '''
    # def sketch_one(self, seq: Sequence) -> SketchedSequence:
    #     full_sketch = self._full_sketch(seq.seq)
    #
    #     self._normalize(seq.len(), full_sketch[self.t])
    #
    #     sketch = np.array([x for x in full_sketch[self.t]], dtype=nb.float32)
    #     return SketchedSequence(seq, sketch)
    #
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
