import os, sys
import pathlib

import annoy
import math
import numpy as np
from numba import jit, njit, prange
import pyfastx
from sklearn.random_projection import SparseRandomProjection
from datetime import datetime as dt
from annoy import AnnoyIndex
from tqdm import tqdm
import pickle
from threading import Thread
from queue import Queue as TQueue
from multiprocessing import Process
from multiprocessing import Queue as PQueue
import random
from collections import namedtuple
import argparse
from functools import partial

from typing import List

from lib.tensor_sketch import SketchParams, TS
from lib.tensor_sketch import TE

from lib.vector_sketch import info, encode_seq, unique, get_num_lines, read_query_file, read_reference_file

RADIX = 4

@njit
def seq2kmers(encseq: np.ndarray, K: int) -> np.ndarray:
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


@njit
def vector_of_distances(seq: str, K: int) -> np.ndarray:
    """
    Compute the distances between k-mers consecutive (k_i to k_{i+1}) in the ordered universe of all k-mers
    and use the distances to form a vector. If a k-mer is not present in the sequence, its distance from its previous
    and next k-mers is 0.
    :param seq: input sequence
    :return: a (D-1)-dimensional vector v, such that v[i] = the distance in the sequence between the i-th and (i+1)-th k-mer
    in the ordered universe of all k-mers, or 0 if either k-mer is not present in the sequence.
    """
    kmers = seq2kmers(encode_seq(seq), K)
    assert kmers.size < 128  # the distances must be in [-128, 127] so the sequence cannot be longer than 127
    unique_kmers, indices = unique(kmers)
    v = np.zeros(RADIX ** K - 1, dtype=np.int8)  # we cannot handle sequences larger than 127
    for i in range(unique_kmers.size - 1):
        if unique_kmers[i + 1] == unique_kmers[i] + 1:
            v[i] = indices[i + 1] - indices[i]
    return v


@njit
def vector_of_positions(seq: str, K: int) -> np.ndarray:
    """
    Encode the positions of k-mers into a D-dimensional vector
    :param K: kmer length
    :param seq: input sequence
    :return: a D-dimensional vector v, such that v[i] = the position (1-based) in the sequence of the i-th k-mer
    in the ordered universe of all k-mers, or 0 is the k-mer is not present in the sequence
    """
    kmers = seq2kmers(encode_seq(seq), K)
    assert kmers.size < 255
    unique_kmers, indices = unique(kmers)
    v = np.zeros(RADIX ** K, dtype=np.uint8)  # we cannot handle sequences larger than 255
    for i in range(unique_kmers.size):
        v[unique_kmers[i]] = indices[i] + 1
    return v


class TestAnnoyIndex:
    def __init__(self):
        self.args = self.parse_args()
        self.config_str = '### ' + ', '.join('{}={}'.format(k, v) for k, v in vars(self.args).items() if k != 'query')
        print(self.config_str)
        self.K = self.args.kmer_length
        self.normalize = self.args.normalize
        self.sketch_dim = self.args.sketch_dim
        assert self.sketch_dim < RADIX ** self.K - 1, ""
        self.distance = self.args.distance
        self.W = self.args.window_size
        self.O = int(self.args.overlap_frac * self.W)

        if self.args.vectorizer == 'kmer-dist':
            self.vectorizer = lambda seq: vector_of_distances(seq, self.K)
            self.D = self.sketch_dim if self.sketch_dim > 0 else RADIX ** self.K - 1
        elif self.args.vectorizer == 'kmer-pos':
            self.vectorizer = lambda seq: vector_of_positions(seq, self.K)
            self.D = self.sketch_dim if self.sketch_dim > 0 else RADIX ** self.K
        elif self.args.vectorizer == 'subseq-count':
            params = SketchParams(A=4, t=self.K, D=self.sketch_dim, normalize=self.args.normalize)
            if self.sketch_dim > 0:
                self.tensor_sketch = TS(params)
                self.vectorizer = lambda seq: self.tensor_sketch.sketch(seq)
                self.D = self.sketch_dim
                self.sketch_dim = 0
            else:
                self.tensor_embedding = TE(params)
                self.vectorizer = lambda seq: self.tensor_embedding.sketch(seq)
                self.D = RADIX ** self.K
        else:
            raise Exception('Unrecognized vectorizer', self.args.vectorizer)

        self.index = AnnoyIndex(self.D, self.distance)
        self.refs = []
        rng = np.random.RandomState(42)
        self.sketcher = SparseRandomProjection(random_state=rng, n_components=self.D)

    def parse_args(self):
        class FloatRange(object):
            def __init__(self, start: float, end: float):
                self.start = start
                self.end = end

            def __eq__(self, other: float) -> bool:
                return self.start <= other <= self.end

            def __repr__(self):
                return '[{0},{1}]'.format(self.start, self.end)

        parser = argparse.ArgumentParser()
        parser.add_argument('reference_file')
        parser.add_argument('-V', '--vectorizer', choices=['kmer-dist', 'kmer-pos', 'subseq-count'], default='kmer-pos')
        parser.add_argument('-N', '--normalize', action='store_true')
        parser.add_argument('-S', '--sketch_dim', type=int, default=0)
        parser.add_argument('-K', '--kmer_length', type=int, choices=range(1, 8), default=3)
        parser.add_argument('-D', '--distance',
                            choices=('euclidean', 'angular', 'manhattan', 'hamming', 'dot'), default='euclidean')
        parser.add_argument('-W', '--window_size', type=int, choices=range(10, 180), default=100)
        parser.add_argument('-O', '--overlap_frac', type=float, choices=[FloatRange(0.01, 0.3)], default=.1)
        parser.add_argument('-Q', '--query', action='append')
        parser.add_argument('-F', '--query_frac', type=float, choices=[FloatRange(0.01, 1.0)], default=0.1)
        parser.add_argument('-P', '--out_prefix', default='./')
        parser.add_argument('-B', '--rebuild_index', action='store_true')

        args = parser.parse_args()
        return args

    def build_index(self):
        rebuild = self.args.rebuild_index
        filename = self.args.reference_file
        if not rebuild and os.path.exists(filename + '.annoy') and os.path.exists(filename + '.refmap'):
            info("Loading annoy index from disk..")
            self.index.load(filename + '.annoy')
            with open(filename + '.refmap', 'rb') as fp:
                self.refs = pickle.load(fp)
            info('done.')
        else:
            vectors = self.vectorize_references_par(filename)
            if self.sketch_dim > 0:
                info("Sketching..")
                matrix = np.vstack(vectors)
                ref_matrix_new = self.sketcher.fit_transform(matrix)
                matrix = np.array(ref_matrix_new, dtype=np.float32)

            info("Adding items to annoy index..")
            for i in tqdm(range(len(vectors)), mininterval=60):
                self.index.add_item(i, vectors[i])
            info("Building index..")
            self.index.build(n_trees=64, n_jobs=32)

            info('Writing index to file..')
            self.index.save(filename + '.annoy')
            with open(filename + '.refmap', 'wb') as fp:
                pickle.dump(self.refs, fp, protocol=pickle.HIGHEST_PROTOCOL)
            info("Done.")

    def vectorize_references_par(self, filename):
        nthreads = 8
        readQueue = PQueue(maxsize=2 * nthreads)
        writeQueue = PQueue(maxsize=nthreads)
        n_lines = get_num_lines(filename) // 2
        worker_threads = []
        for i in range(nthreads):
            worker_thread = Process(target=self.vectorize_sequence_par, args=(readQueue, writeQueue))
            worker_threads.append(worker_thread)
            worker_thread.start()
        reader_thread = Process(target=read_reference_file, args=(filename, readQueue))
        reader_thread.start()
        reader_thread.join()

        vectors = []
        for i in range(nthreads):
            t_keys, t_vectors = writeQueue.get()
            assert len(t_keys) == len(t_vectors), "Keys and values must be the same length"
            self.refs.extend(t_keys)
            vectors.extend(t_vectors)

        for worker in worker_threads:
            worker.join()

        return vectors

    def vectorize_sequence_par(self, read_queue: PQueue, write_queue: PQueue):
        keys = []
        vectors = []
        while True:
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                write_queue.put((keys, vectors))
                break
            else:
                if len(seq) > self.W:
                    offset = 0
                    while offset + self.W < len(seq):
                        keys.append((name, offset))
                        vectors.append(self.vectorizer(seq[offset:offset + self.W]))
                        offset += (self.W - self.O)
        # info('Worker exiting.')

    def test(self):
        print(self.args)
        seq = 'TGGAGCCGGAGACCGGCGTCGACGCGGTGAACGGGTCCCGCCGAACCCCTCGAAGCAGCCGCCGCGAAGGTGGCCGCCACGATCCAGGACAGCAGGCGGGCG'
        v = self.vectorizer(seq)
        print(v.shape)
        print(v)
        info("Building index..")
        self.index.add_item(0, v)
        self.index.build(1, 1)

        ids, distances = self.index.get_nns_by_vector(v, 1, include_distances=True)
        print(ids, distances)

        print(self.args)

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
        assert (self.W - self.O) <= len(q) <= (self.W + self.O)
        v = self.vectorizer(q)
        ids, distances = self.index.get_nns_by_vector(v, 1, include_distances=True)
        name, offset = self.refs[ids[0]]
        return name, offset, distances[0]

    def query(self, q: str):
        """
        Return the distance between the query string and the closest segment in the reference
        :param q: the query string
        :return: a summary of scores consisting of (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query
        """
        matches = []
        offset = 0
        while offset + self.W < len(q):
            matches.append(self.query_window(q[offset:offset + self.W]))
            offset += (self.W - self.O)
        return self.summarize_scores(matches)

    def query_thread(self, read_queue: PQueue, write_queue: TQueue):
        while True:
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                write_queue.put("X")
                break
            else:
                refname, distance, uniq = self.query(seq)
                write_queue.put("{},{}\n".format(distance, int(uniq)))
        # info('Worker exiting.')

    def query_file(self, infile: str, outfile: str):
        nthreads = 32
        info("Querying")
        readQueue = PQueue(maxsize=4 * nthreads)
        writeQueue = TQueue(maxsize=4 * nthreads)
        n_lines = get_num_lines(infile) // 2
        worker_threads = []

        for i in range(nthreads):
            worker_thread = Thread(target=self.query_thread, args=(readQueue, writeQueue))
            worker_threads.append(worker_thread)
            worker_thread.start()
        reader_thread = Process(target=read_query_file, args=(infile, self.args.query_frac, readQueue))
        reader_thread.start()

        with open(outfile, 'a') as f:
            f.write(self.config_str + '\n')
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

    def query_files(self):
        for infile in self.args.query:
            if not os.path.isfile(infile):
                raise Exception("File not found -", infile)
            fname = os.path.splitext(os.path.basename(infile))[0]
            outfile = self.args.out_prefix + fname + '.csv'
            outfile = os.path.abspath(outfile)
            outdir = os.path.dirname(outfile)
            pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
            self.query_file(infile, outfile)

