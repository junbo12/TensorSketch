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

K = 3  # must not be larger than 8
RADIX = 4
D = RADIX ** K
M1 = D // RADIX


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


Vectorizer = namedtuple('Vectorizer', ['func', 'ndim'])

distance_vector = Vectorizer(func=vector_of_distances, ndim=D - 1)
position_vector = Vectorizer(func=vector_of_positions, ndim=D)

tensor_sketch_dim = 16
te_subseq_len = 4
tensor_embedding_dim = 4 ** te_subseq_len

params = SketchParams(A=4, t=te_subseq_len, D=tensor_sketch_dim)
ts = TS(params)
te = TE(params)

tensor_sketch = Vectorizer(func=ts.sketch, ndim=tensor_sketch_dim)
tensor_embedding = Vectorizer(func=te.sketch, ndim=tensor_embedding_dim)

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
                 w: int = 80, o: int = 79, rebuild: bool = False, sketchdim: int = 0):
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
        if sketchdim <= 0:
            self.sketcher = None
        else:
            rng = np.random.RandomState(42)
            self.sketcher = SparseRandomProjection(random_state=rng, n_components=self.d)
        if index == 'annoy':
            self.t = AnnoyIndex(self.d, "euclidean")
            self.build_annoy_index(filename, rebuild)
        elif index == 'usearch':
            dt = np.float32 if sketchdim > 0 else np.int8
            self.t = Index(ndim=self.d, dtype=dt, metric='ip', connectivity=16)
            self.build_usearch_index(filename, rebuild)
        elif index == 'faiss':
            quantizer = faiss.IndexFlatL2(self.d)
            self.t = faiss.IndexIVFFlat(quantizer, self.d, 100)
            self.build_faiss_index(filename, rebuild)
        else:
            assert False, "Not implemented"

    @staticmethod
    def vectorize_sequence_par(read_queue: PQueue, write_queue: PQueue, w: int, o: int, vectorizer: Vectorizer):
        keys = []
        vectors = []
        while True:
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                write_queue.put((keys, vectors))
                break
            else:
                if len(seq) > w:
                    offset = 0
                    while offset + w < len(seq):
                        keys.append((name, offset))
                        vectors.append(seq2vec(seq[offset:offset + w], vectorizer))
                        offset += (w - o)
        info('Worker exiting.')

    def vectorize_references_par(self, filename):
        nthreads = 8
        readQueue = PQueue(maxsize=2 * nthreads)
        writeQueue = PQueue(maxsize=nthreads)
        n_lines = get_num_lines(filename) // 2
        worker_threads = []
        for i in range(nthreads):
            worker_thread = Process(target=RefIdx.vectorize_sequence_par,
                                    args=(readQueue, writeQueue, self.w, self.o, self.vectorizer))
            worker_threads.append(worker_thread)
            worker_thread.start()
        reader_thread = Process(target=read_reference_file, args=(filename, readQueue))
        reader_thread.start()
        reader_thread.join()

        vectors = []
        for i in range(nthreads):
            t_keys, t_vectors = writeQueue.get()
            self.refs.extend(t_keys)
            vectors.extend(t_vectors)

        for worker in worker_threads:
            worker.join()

        return vectors

    def vectorize_references(self, filename):
        nlines = get_num_lines(filename) // 2
        vectors = []
        info("Reading file {}".format(filename))
        with tqdm(total=nlines, unit='lines', mininterval=60) as pbar:
            for name, seq in pyfastx.Fasta(filename, build_index=False):
                if len(seq) > self.w:
                    offset = 0
                    while offset + self.w < len(seq):
                        vectors.append(seq2vec(seq[offset:offset + self.w], self.vectorizer))
                        self.refs.append((name, offset))
                        offset += (self.w - self.o)
                pbar.update(1)
        info("Done.")
        return vectors

    def build_usearch_index(self, filename: str, rebuild: bool = False):
        assert self.index == 'usearch'
        assert isinstance(self.t, usearch.index.Index)

        if not rebuild and os.path.exists(filename + '.usearch') and os.path.exists(filename + '.refmap'):
            info("Loading usearch index from disk..")
            self.t.load(filename + '.usearch')
            with open(filename + '.refmap', 'rb') as fp:
                self.refs = pickle.load(fp)
            info('done.')
        else:
            vectors = self.vectorize_references_par(filename)
            info("Building usearch index..")
            matrix = np.vstack(vectors)
            keys = np.arange(len(vectors))
            self.t.add(keys, matrix, copy=True, threads=32, log=True)
            info("Done")

            info('Writing index to file..')
            self.t.save(filename + '.usearch')
            with open(filename + '.refmap', 'wb') as fp:
                pickle.dump(self.refs, fp, protocol=pickle.HIGHEST_PROTOCOL)
            info("Done.")

    def build_annoy_index(self, filename: str, rebuild: bool = False):
        assert self.index == 'annoy'
        assert isinstance(self.t, annoy.AnnoyIndex)

        if not rebuild and os.path.exists(filename + '.annoy') and os.path.exists(filename + '.refmap'):
            info("Loading annoy index from disk..")
            self.t.load(filename + '.annoy')
            with open(filename + '.refmap', 'rb') as fp:
                self.refs = pickle.load(fp)
            info('done.')
        else:
            vectors = self.vectorize_references_par(filename)
            info("Adding items to annoy index..")
            for i in tqdm(range(len(vectors)), mininterval=60):
                self.t.add_item(i, vectors[i])
            info("Building index..")
            self.t.build(n_trees=64, n_jobs=32)

            info('Writing index to file..')
            self.t.save(filename + '.annoy')
            with open(filename + '.refmap', 'wb') as fp:
                pickle.dump(self.refs, fp, protocol=pickle.HIGHEST_PROTOCOL)
            info("Done.")

    def build_faiss_index(self, filename: str, rebuild: bool = False):
        assert self.index == 'faiss'
        assert isinstance(self.t, faiss.IndexIVFFlat)
        if not rebuild and os.path.exists(filename + '.faiss') and os.path.exists(filename + '.refmap'):
            info("Loading faiss index from disk..")
            self.t = faiss.read_index(filename + '.faiss')
            with open(filename + '.refmap', 'rb') as fp:
                self.refs = pickle.load(fp)
            info('done.')
        else:
            vectors = self.vectorize_references_par(filename)
            info("Building faiss index..")
            matrix = np.vstack(vectors, dtype=np.float32)
            info("Created a matrix of shape {} and type {}".format(matrix.shape, matrix.dtype))

            if self.sketcher:
                info("Transforming data..")
                ref_matrix_new = self.sketcher.fit_transform(matrix)
                matrix = np.array(ref_matrix_new, dtype=np.float32)
                info("Obtained matrix of shape {} and type {}".format(ref_matrix_new.shape, ref_matrix_new.dtype))

            info('Training index..')
            self.t.train(matrix)
            info("Adding vectors to index..")
            self.t.add(matrix)
            info('Writing index to file..')
            faiss.write_index(self.t, filename + '.faiss')
            with open(filename + '.refmap', 'wb') as fp:
                pickle.dump(self.refs, fp, protocol=pickle.HIGHEST_PROTOCOL)
            info("Done.")
        self.t.nprobe = 1

    @staticmethod
    def summarize_scores(matches: list) -> tuple:
        """
        Summarize matches for windows into a single tuple containing (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query.
        :param matches: a list of 3-tuples - (ref-name, offset, distance)
        :return: a 3-tuple - (ref-name, distance, unique)
        """
        if len(matches) > 0:
            # check if the matches are all with the same reference
            all_same = True
            ref_name = matches[0][0]
            for i in range(1, len(matches)):
                if matches[i][0] != matches[i - 1][0]:
                    all_same = False
                    break

            if all_same:
                # return the average distance
                return ref_name, sum(z for x, y, z in matches) / len(matches), True
            else:
                # return the smallest distance
                matches = sorted(matches, key=lambda x: x[2])
                return matches[0][0], matches[0][2], False
        return '', math.inf, False

    def query_window(self, q: str) -> tuple:
        """
        Vectorise a string and return the closest reference window
        :param q: query string (must be in [w-o, w+o], where w = window length and o = overlap length
        :return: a 3-tuple of the reference name, the offset of the matching window, and its distance from the query
        """
        assert (self.w - self.o) <= len(q) <= (self.w + self.o)
        v = seq2vec(q, vectorizer=self.vectorizer)
        if self.index == 'usearch':
            matches = self.t.search(v, 1)
            name, offset = self.refs[matches[0].key]
            return name, offset, matches[0].distance
        elif self.index == 'faiss':
            v = np.array([v])
            if self.sketcher:
                sv = self.sketcher.transform(v)
                v = np.array(sv, dtype=np.float32)
            distances, ids = self.t.search(v, 1)
            name, offset = self.refs[ids[0][0]]
            return name, offset, distances[0][0]
        elif self.index == 'annoy':
            ids, distances = self.t.get_nns_by_vector(v, 1, include_distances=True)
            name, offset = self.refs[ids[0]]
            return name, offset, distances[0]
        else:
            assert False, 'Not implemented'

    def query(self, q: str):
        """
        Return the distance between the query string and the closest segment in the reference
        :param q: the query string
        :return: a summary of scores consisting of (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query
        """
        matches = []
        offset = 0
        while offset + self.w < len(q):
            matches.append(self.query_window(q[offset:offset + self.w]))
            offset += (self.w - self.o)
        return self.summarize_scores(matches)

    def query_thread(self, read_queue: PQueue, write_queue: TQueue, check_correct: bool = False):
        while True:
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                write_queue.put("X")
                break
            else:
                refname, distance, uniq = self.query(seq)
                if check_correct:
                    target_refname, offset = self.split_query_header(name)
                    write_queue.put("{},{},{}\n".format(
                        distance, int(uniq), int(refname == target_refname)
                    ))
                else:
                    write_queue.put("{},{}\n".format(distance, int(uniq)))
        info('Worker exiting.')

    @staticmethod
    def split_query_header(header: str):
        substr1 = header.split('!')[1]
        tokens = substr1.split(':')
        ref_name = tokens[0]
        start = int(tokens[1].split('-')[0])
        return ref_name, start

    def query_file(self, infile, outfile, frac=1.0, check_correct=True):
        nthreads = 32
        info("Querying")
        readQueue = PQueue(maxsize=4 * nthreads)
        writeQueue = TQueue(maxsize=4 * nthreads)
        n_lines = get_num_lines(infile) // 2
        worker_threads = []

        for i in range(nthreads):
            worker_thread = Thread(target=self.query_thread, args=(readQueue, writeQueue, check_correct))
            worker_threads.append(worker_thread)
            worker_thread.start()
        reader_thread = Process(target=read_query_file, args=(infile, frac, readQueue))
        reader_thread.start()

        with open(outfile, 'w') as f:
            if check_correct:
                f.write('distance,unique,correct\n')
            else:
                f.write('distance,unique\n')

            n_threads_remaining = nthreads
            with tqdm(total=n_lines, mininterval=60) as pbar:
                while n_threads_remaining:
                    stat = writeQueue.get()
                    if stat[0] == 'X':
                        n_threads_remaining -= 1
                    else:
                        f.write(stat)
                        pbar.update(1)
        reader_thread.join()
        for i in range(nthreads):
            worker_threads[i].join()
        info('Done.')


def read_query_file(infile: str, frac: float, queue: PQueue):
    if frac > 1.0:
        frac = 1.0
    info("reading query file {}".format(infile))
    for name, seq in pyfastx.Fasta(infile, build_index=False):
        if random.random() < frac:
            if len(seq) > 200:
                queue.put((name, seq[:200]))
            else:
                queue.put((name, seq))
    queue.put(('END', ''))
    info('Reader exiting.')


def read_reference_file(infile: str, queue: PQueue):
    info("reading reference file {}".format(infile))
    for name, seq in pyfastx.Fasta(infile, build_index=False):
        queue.put((name, seq))
    queue.put(('END', ''))
    info('Reader exiting.')
