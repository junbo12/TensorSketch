import os, sys

import annoy
import math

import numpy as np
import usearch.index
from numba import jit, njit, prange
import pyfastx
from sklearn.random_projection import SparseRandomProjection
import faiss
from datetime import datetime as dt
from annoy import AnnoyIndex
from usearch.index import Index
from tqdm import tqdm
import pickle
from threading import Thread
from queue import Queue as TQueue
from multiprocessing import Process
from multiprocessing import Queue as PQueue
import random
from collections import namedtuple

# sys.path.append(os.path.abspath('.'))

from lib.tensor_sketch import SketchParams, TS
from lib.tensor_embedding import TE

#debug
#from memory_profiler import profile

K = 3  # must not be larger than 8
RADIX = 4
D = RADIX ** K
M1 = D // RADIX
Vectorizer = namedtuple('Vectorizer', ['func', 'ndim'])

def info(msg: str) -> None:
    """
    Utility function to print message with timestamp
    :param msg: message to print
    """
    print("[{}] {}".format(dt.now().strftime('%H:%M:%S'), msg))


def get_num_lines(filename: str) -> int:
    """
    Get number of lines in a file
    :param filename: name of the file
    :return: the number of lines in the file
    """
    with open(filename, "rb") as f:
        num_lines = sum(1 for _ in f)
    return num_lines


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


@njit
def seq2kmers(encseq: np.ndarray) -> np.ndarray:
    """
    A vector of k-mers from a sequence
    :param encseq: encoded sequence
    :return: vector of kmers (encoded into integers)
    """
    if encseq.shape[0] < K:
        return None
    v = np.empty((encseq.shape[0] - K + 1,), dtype=np.int16)  # we will not have k-mers longer than 8
    v[0] = 0
    for i in range(K):
        v[0] = ((v[0] * RADIX) + encseq[i])
    for i in prange(1, encseq.shape[0] - K + 1):
        outgoing = encseq[i - 1]
        incoming = encseq[i + K - 1]
        v[i] = ((v[i - 1] - M1 * outgoing) * RADIX) + incoming
    return v


@njit
def unique(x: np.ndarray) -> (np.ndarray, np.ndarray):
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


@njit
def vector_of_distances(seq: str) -> np.ndarray:
    """
    Compute the distances between k-mers consecutive (k_i to k_{i+1}) in the ordered universe of all k-mers
    and use the distances to form a vector. If a k-mer is not present in the sequence, its distance from its previous
    and next k-mers is 0.
    :param seq: input sequence
    :return: a (D-1)-dimensional vector v, such that v[i] = the distance in the sequence between the i-th and (i+1)-th k-mer
    in the ordered universe of all k-mers, or 0 if either k-mer is not present in the sequence.
    """
    kmers = seq2kmers(encode_seq(seq))
    assert kmers.size < 128  # the distances must be in [-128, 127] so the sequence cannot be longer than 127
    unique_kmers, indices = unique(kmers)
    v = np.zeros(D - 1, dtype=np.int8)  # we cannot handle sequences larger than 127
    for i in range(unique_kmers.size - 1):
        if unique_kmers[i + 1] == unique_kmers[i] + 1:
            v[i] = indices[i + 1] - indices[i]
    return v


@njit
def vector_of_positions(seq: str) -> np.ndarray:
    """
    Encode the positions of k-mers into a D-dimensional vector
    :param seq: input sequence
    :return: a D-dimensional vector v, such that v[i] = the position (1-based) in the sequence of the i-th k-mer
    in the ordered universe of all k-mers, or 0 is the k-mer is not present in the sequence
    """
    kmers = seq2kmers(encode_seq(seq))
    assert kmers.size < 255
    unique_kmers, indices = unique(kmers)
    v = np.zeros(D, dtype=np.uint8)  # we cannot handle sequences larger than 255
    for i in range(unique_kmers.size):
        v[unique_kmers[i]] = indices[i] + 1
    return v


'''

distance_vector = Vectorizer(func=vector_of_distances, ndim=D - 1)
position_vector = Vectorizer(func=vector_of_positions, ndim=D)

tensor_sketch_dim = 16
te_subseq_len = 4
tensor_embedding_dim = 4 ** te_subseq_len

params = SketchParams(A=4, t=te_subseq_len, D=tensor_sketch_dim)
ts = TS(params)
te = TE(params)

tensor_sketch = Vectorizer(func=ts.sketch, ndim=tensor_sketch_dim)
tensor_embedding = Vectorizer(func=te.sketch, ndim=tensor_embedding_dim)'''

def seq2vec(seq: str, vectorizer: Vectorizer) -> np.ndarray:
    """
    A vector representation of the sequence.
    :param seq: sequence
    :param vectorizer: the function that is used to vectorize the sequence
    :return: a vector representation of the sequence
    """
    if len(seq) > K:
        if vectorizer is None:
            return vector_of_distances(seq)
        return vectorizer.func(seq)
    return None


class RefIdx:
    def __init__(self, filename: str, index='usearch', vectorizer: Vectorizer = None,
                 w: int = 100, o: int = 80, rebuild: bool = False, sketchdim: int = 0, wf: bool = True):
        """
        Creates an index of the reference sequences
        :param filename: reference file
        :param vectorizer: the function that is used to vectorize the sequences
        :param w: sequence window to turn into a vector
        :param o: overlap between windows
        :param rebuild: if true, rebuild the index even though an index was found in the current directory
        :param sketchdim: if greater than 0, sketch the vector into this dimension
        """
        self.w = w
        self.o = o
        self.vectorizer = vectorizer if vectorizer is not None else position_vector
        self.refs = []
        self.index = index
        self.d = sketchdim if sketchdim > 0 else self.vectorizer.ndim
        self.wf = wf
        if sketchdim <= 0:
            self.sketcher = None
        else:
            rng = np.random.RandomState(42)
            self.sketcher = SparseRandomProjection(random_state=rng, n_components=self.d)
        if index == 'annoy':
            self.t = AnnoyIndex(self.d, "euclidean")
            self.build_annoy_index(filename, rebuild, wf)
        elif index == 'usearch':
            dt = np.float32 if sketchdim > 0 else np.int8
            self.t = Index(ndim=self.d, dtype=dt, metric='ip', connectivity=16)
            self.build_usearch_index(filename, rebuild, wf)
        elif index == 'faiss':
            quantizer = faiss.IndexFlatL2(self.d)
            self.t = faiss.IndexIVFFlat(quantizer, self.d, 100)
            self.build_faiss_index(filename, rebuild, wf)
        else:
            assert False, "Not implemented"


    def vectorize_references(self, filename):
        nlines = get_num_lines(filename) // 2
        vectors = []
        info("Reading file {}".format(filename))
        with tqdm(total=nlines, unit='lines', mininterval=60) as pbar:
            for name, seq in pyfastx.Fasta(filename, build_index=False):
                if len(seq) >= self.w:
                    offset = 0
                    while offset + self.w < len(seq):
                        vectors.append(seq2vec(seq[offset:offset + self.w], self.vectorizer))
                        self.refs.append((name, offset))
                        offset += (self.w - self.o)
                    last_pos = len(seq)   
                    if 2*(last_pos-offset) >= self.w: #if the remaining part is larger than half the window size, then add it to the vectors
                        offset = last_pos-self.w
                        vectors.append(seq2vec(seq[offset:last_pos], self.vectorizer)) 
                        self.refs.append((name, offset))                     
                pbar.update(1)
        info("Done.")
        return vectors
    
    
    def build_annoy_index(self, filename: str, rebuild: bool = False, wf:bool = True):
        assert self.index == 'annoy'
        assert isinstance(self.t, annoy.AnnoyIndex)

        if not rebuild and os.path.exists(filename + '.annoy') and os.path.exists(filename + '.refmap'):
            info("Loading annoy index from disk..")
            self.t.load(filename + '.annoy')
            with open(filename + '.refmap', 'rb') as fp:
                self.refs = pickle.load(fp)
            info('done.')
        else:
            
            vectors = self.vectorize_references(filename)
            info("Adding items to annoy index..")
            for i in tqdm(range(len(vectors)), mininterval=60):
                self.t.add_item(i, vectors[i])
            info("Building index..")
            self.t.build(n_trees=64, n_jobs=32)

            if wf:
                info('Writing index to file..')
                self.t.save(filename + '.annoy')
                with open(filename + '.refmap', 'wb') as fp:
                    pickle.dump(self.refs, fp, protocol=pickle.HIGHEST_PROTOCOL)
            info("Done.")
    @staticmethod
    def summarize_scores( matches: list) -> tuple:
        """
        Summarize matches for windows into a single tuple containing (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query.
        :param matches: a list of 3-tuples - (ref-name, offset, distance)
        :return: a 3-tuple - (ref-name, distance, unique)
        """
        def all_same(items):
            return all(x == items[0] for x in items)
        
        ref_names = [el[0] for el in matches]
        dists = [el[2] for el in matches]
        if len(matches) > 0:
            # check if the matches are all with the same reference

            if all_same(ref_names):
                # return the average distance
                return ref_names[0], sum(dists) / len(matches), True
            
            else:
                # return the smallest distance
                matches = sorted(matches, key=lambda x: x[2])
                return matches[0][0], matches[0][2], False
        return '', math.inf, False


    def query_window_check(self, q: str, target_refname, read_name, search_k = 1) -> tuple:
            assert (self.w - self.o) <= len(q) <= (self.w + self.o)
            v = seq2vec(q, vectorizer=self.vectorizer)
            ids, distances = self.t.get_nns_by_vector(v, search_k, include_distances=True)
            name_off_list = [self.refs[x] for x in ids]
            concat_list = list(map(lambda x, y:(x[0],y,x[1]), name_off_list, distances))
            #sorted_list = sorted(concat_list, key=lambda x: x[1])
            #print(sorted_list[0][0], concat_list[0][0])
            name, distance, offset = concat_list[0]
            true_distance = self.t.get_distance(int(target_refname),int(read_name))
            is_close = (target_refname in [x[0] for x in name_off_list])
            return name, offset, distance, is_close, true_distance


    
    def query_found(self, read_name:str, q: str, target_refname: str):
        """
        Return the distance between the query string and the closest segment in the reference
        :param q: the query string
        :return: a summary of scores consisting of (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query
        """
        matches = []
        offset = 0
        neighboring = 0
        while offset + self.w <= len(q):
            name, ref_offset, dist, is_close, true_distance = self.query_window_check(q[offset:offset + self.w], target_refname, read_name, search_k=4)
            
            matches.append((name,offset, dist,ref_offset))
            if (is_close):
                neighboring += 1

            offset += 1
        ref_names = [el[0] for el in matches]

        ref_name,dist,uniq = self.summarize_scores(matches)
        return ref_name, dist, uniq, ref_names.count(target_refname)/len(matches), neighboring/len(matches), true_distance
    
    def query_file(self, infile, outfile, frac=1.0):
        n_lines = get_num_lines(infile) // 2
        write_list = []
        if frac > 1.0:
            frac = 1.0
            info("reading query file {}".format(infile))
        for name, seq in pyfastx.Fasta(infile, build_index=False):
            if random.random() < frac:
                
                target_refname = name
                refname, distance, uniq, occ, neighboring_score, true_distance = self.query_found(name, seq, target_refname)
                
                write_list.append("{},{},{},{},{},{}\n".format(
                    distance, true_distance, int(uniq), int(refname == target_refname), occ, neighboring_score))
           

        with open(outfile, 'w') as f:

            f.write('distance, true_distance, unique,correct,occurence_matching, occurence_querying\n')
            with tqdm(total=n_lines, mininterval=60) as pbar:
                for stat in write_list:
                    f.write(stat)
                    pbar.update(1)
        info('Done.')





