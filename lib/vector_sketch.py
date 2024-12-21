import os

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
import time

# sys.path.append(os.path.abspath('.'))

#debug
from memory_profiler import profile


Vectorizer = namedtuple('Vectorizer', ['func', 'name'])

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
def seq2kmers(encseq: np.ndarray, K: int = 3) -> np.ndarray:
    """
    A vector of k-mers from a sequence
    :param encseq: encoded sequence
    :return: vector of kmers (encoded into integers)
    """
    
    RADIX = 4
    D = RADIX ** K
    M1 = D // RADIX
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


class RefIdx:
    def __init__(self, filename: str, index='annoy', vectorizer: Vectorizer = None,
                 w: int = 100, o: int = 80, sketchdim: int = 0, rebuild: bool = False, wf: bool = True, n_trees=16):
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
        self.vectorizer = vectorizer if vectorizer is not None else None
        self.refs = []
        self.ref_dict = {}
        self.index = index
        self.d = sketchdim 
        self.wf = wf
        self.vectorizing_time = -1
        self.build_time = -1
        self.n_trees = n_trees
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
    def vectorize_sequence_par(read_queue: PQueue, write_queue: PQueue, w: int, o: int, vectorizer: Vectorizer, seq_queue:PQueue):
        keys = []
        
        vectors = []
        dict_list= {}
        while True:
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                seq_queue.put(dict_list)
                write_queue.put((keys,vectors))
                break
            else:
                
                length = len(seq)
                dict_list[name]= (length,seq)
                if length >= w:
                    offset = 0
                    while offset + w <= length:
                        keys.append((name, offset))
                        vectors.append(vectorizer.func(seq[offset:offset + w]))
                        offset += (w - o)
                     
                    if 2*(length-offset) >= w: #if the remaining part is larger than half the window size, then add it to the vectors
                        offset = length-w
                        vectors.append(vectorizer.func(seq[offset:length])) 
                        keys.append((name, offset))  
            print('vectorized', name, length)
            
        info('Worker exiting.')

    
    #vectorizes references in parallel

    def vectorize_references_par(self, filename):
        nthreads = 8
        readQueue = PQueue(maxsize=2 * nthreads)
        writeQueue = PQueue(maxsize=nthreads)
        seqQueue = PQueue(maxsize=2 * nthreads)
        
        worker_threads = []
        for _ in range(nthreads):
            worker_thread = Process(target=RefIdx.vectorize_sequence_par,
                                    args=(readQueue, writeQueue, self.w, self.o, self.vectorizer, seqQueue))
            worker_threads.append(worker_thread)
            worker_thread.start()
        reader_thread = Process(target=read_reference_file, args=(filename, readQueue))
        reader_thread.start()
        
        vectors = []
        for i in range(nthreads):
            t_keys, t_vectors = writeQueue.get()
            self.refs.extend(t_keys)
            vectors.extend(t_vectors)
            new_dict = seqQueue.get()
            self.ref_dict.update(new_dict)
        reader_thread.join()

        for worker in worker_threads:
            worker.join()
            
        
        return vectors
    
    def build_annoy_index(self, filename: str, rebuild: bool = False):
        assert self.index == 'annoy'
        assert isinstance(self.t, annoy.AnnoyIndex)

        if not rebuild and os.path.exists(filename + '.annoy') and os.path.exists(filename + '.refmap'):
            info("Loading annoy index from disk..")
            self.t.load(filename + '.annoy')
            with open(filename + '.refmap', 'rb') as fp:
                self.refs = pickle.load(fp)
            with open(filename + '.refdict', 'rb') as fp:
                self.ref_dict = pickle.load(fp)
            info('done.')
        else:
            starttime = time.time()
            vectors = self.vectorize_references_par(filename)
            self.vectorizing_time = time.time() - starttime
            starttime = time.time()
            info("Adding items to annoy index..")
            for i in tqdm(range(len(vectors)), mininterval=60):
                self.t.add_item(i, vectors[i])
            del vectors
            info("Building index..")
            self.t.build(n_trees=self.n_trees, n_jobs=16)
            self.build_time = time.time() - starttime
            if self.wf:
                info('Writing index to file..')
                filename = filename + '_({},{},{},{})'.format(self.vectorizer.name, self.sketchdim, self.w, self.stride)
                self.t.save(filename + '.annoy')
                with open(filename + '.refmap', 'wb') as fp:
                    pickle.dump(self.refs, fp, protocol=pickle.HIGHEST_PROTOCOL)
                with open(filename + '.refdict', 'wb') as fp:
                    pickle.dump(self.ref_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
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
            if self.wf:
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
        :param matches: a list of 4-tuples - (ref_name, distance, read_offset, ref_offset)
        :return: a 5-tuple - (ref-name, distance, unique, read_offset, ref_offset)
        """
        
        
        ref_names = [el[0] for el in matches]
        dists = [el[1] for el in matches]
        if len(matches) > 0:
            # check if the matches are all with the same reference

            if all(x == ref_names[0] for x in ref_names):
                # return the average distance
                return ref_names[0], sum(dists) / len(matches), True, matches[0][2], matches[0][3]
            
            else:
                # return the smallest distance
                matches = sorted(matches, key=lambda x: x[1])
                return matches[0][0], matches[0][1], False, matches[0][2], matches[0][3]
        return '', math.inf, False, -1, -1
    @staticmethod
    def summarize_scores_annoy(matches: list) -> tuple:
        
        
        ref_names = [el[0] for el in matches]
        dists = [el[1] for el in matches]
        if len(matches) > 0:
            # check if the matches are all with the same reference

            if all(x == ref_names[0] for x in ref_names):
                # return the average distance
                return ref_names[0], sum(dists) / len(matches)
            
            else:
                # return the smallest distance
                matches = sorted(matches, key=lambda x: x[1])
                return matches[0][0], matches[0][1]
        return '', math.inf
    
    
    def query_window_check(self, q: str, target_refname = '', n_nearest = 1,fac=2) -> tuple:
            assert (self.w - self.o) <= len(q) <= (self.w + self.o)
            v = self.vectorizer.func(q)
            if self.index == 'annoy':
                
                ids, distances = self.t.get_nns_by_vector(v, n_nearest, search_k = fac*self.t.get_n_trees(), include_distances=True)
                #search_k parameter
                name_off_list = [self.refs[x] for x in ids]
                concat_list = list(map(lambda x, y:(x[0],y,x[1]), name_off_list, distances))
                name, distance, offset = concat_list[0]

                is_close = (target_refname in [x[0] for x in name_off_list])
                return name, offset, distance, is_close
            elif self.index == 'faiss':
                v = np.array([v])
                if self.sketcher:
                    sv = self.sketcher.transform(v)
                    v = np.array(sv, dtype=np.float32)
                distances, ids = self.t.search(v, n_nearest)

                name_off_list = [self.refs[x] for x in ids[0]]
                is_close = (target_refname in [x[0] for x in name_off_list])
                name, offset = name_off_list[0]
                return name, offset, distances[0][0], is_close
            else:
                assert False, ('Not implemented')
    
    def query_found(self, q: str, target_refname: str, ws: bool =False, fac=2):
        """
        Return the distance between the query string and the closest segment in the reference
        :param q: the query string
        :return: a summary of scores consisting of (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query
        """
        matches = []
        read_offset = 0
        neighboring = 0
        assert self.w <= len(q)
        while read_offset + self.w <= len(q):
            name, ref_offset, dist, is_close = self.query_window_check(q[read_offset:read_offset + self.w], target_refname,n_nearest=1,fac=fac)
            
            matches.append((name,dist,read_offset, ref_offset))
            if (is_close):
                neighboring += 1

            read_offset += 1
        
        ref_name,dist,_, read_offset, ref_offset = self.summarize_scores(matches)
        
        read_seq = []
        if(ws):
            read_seq = q[read_offset:read_offset + self.w]
        
        return ref_name, dist, read_offset, ref_offset,read_seq
    
    def query_thread(self, read_queue: PQueue, write_queue: TQueue, check_correct: bool = True, seq_queue = None, fac=2):
        while True:
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                write_queue.put("X")
                if(seq_queue is not None):
                    seq_queue.put('X')
                break
            else:
                if check_correct:
                    ws=seq_queue is not None
                    
                    target_refname, true_offset = self.split_query_header(name)
                   
                    #refname, distance, uniq, occ, neighboring_score, read_offset, ref_offset, read_seq = self.query_found(seq, target_refname,ws)
                    refname, distance, read_offset, ref_offset, read_seq = self.query_found(seq, target_refname,ws,fac=fac)
                    is_in_pool = int(target_refname in self.ref_dict.keys())
                    if (not is_in_pool):
                        print('not in pool query found')
                    true_distance = -1
                    
                    if is_in_pool:
                        th=10
                        nearest_distance = read_offset%(self.w-self.o)
                        true_distance = min([math.dist(self.vectorizer.func(self.ref_dict[target_refname][1][true_offset+read_offset+i:true_offset+read_offset+self.w+i*nearest_distance]),
                                                    self.vectorizer.func(seq[read_offset+i:read_offset+self.w+i*nearest_distance])) for i in range(-th,th+1) if i*nearest_distance < th])
                    
                    '''
                    write_queue.put("{},{},{},{},{},{},{},{},{}\n".format(
                        distance, int(refname == target_refname), is_in_pool, int(uniq), occ, neighboring_score, read_offset, ref_offset, true_offset+read_offset
                    ))
                    '''
                    write_queue.put("{},{},{},{},{},{},{}\n".format(
                        distance, true_distance, int(refname == target_refname), is_in_pool, read_offset, ref_offset, true_offset+read_offset
                    ))
                    if(ws):
                        desc =  "{},{},{},{},{},{}".format(target_refname, refname,  refname == target_refname, true_offset+read_offset, ref_offset, read_offset)
                        true_ref = self.ref_dict[target_refname][1][true_offset+read_offset:true_offset+read_offset+self.w]
                        match_ref =  self.ref_dict[refname][1][ref_offset:ref_offset+self.w]
                        
                       
                        seq_queue.put((desc,true_ref,match_ref,read_seq))
                else:
                    refname, distance, _, _, _ = self.query(seq)
                    write_queue.put("{},{}\n".format(distance, refname))
        info('Worker exiting.')

    @staticmethod
    def split_query_header(header: str):
        substr1 = header.split('!')[1]
        tokens = substr1.split(':')
        ref_name = tokens[0]
        start = int(tokens[1].split('-')[0])
        return ref_name, start
    

    def query_file(self, infile, outfile, frac=1.0, check_correct=True, ws = False, fac = 2):
        nthreads =8 
        info("Querying")
        readQueue = PQueue(maxsize=4 * nthreads)
        writeQueue = TQueue(maxsize=4 * nthreads)
        seqQueue = TQueue(maxsize=4 * nthreads)
        n_lines = get_num_lines(infile) // 2
        worker_threads = []
        reader_thread = Process(target=read_query_file, args=(infile, frac, readQueue, self.w))
        reader_thread.start()
        starttime = time.time()
        for i in range(nthreads):
            if (ws):
                worker_thread = Thread(target=self.query_thread, args=(readQueue, writeQueue, check_correct, seqQueue, fac))
            else:    
                worker_thread = Thread(target=self.query_thread, args=(readQueue, writeQueue, check_correct,None, fac))
            worker_threads.append(worker_thread)
            worker_thread.start()

        if(ws):
            tmp = outfile.split('.')[0]
            
            filename = tmp+'_seq.csv'
            seq_thread = Thread(target=write_file, args=(filename, seqQueue,nthreads))
            seq_thread.start()
        with open(outfile, 'w') as f:
            if check_correct:
                #f.write('distance, correct, is_in_pool, unique, occurence_matching, occurence_querying, read_offset, matched_offset, true_offset\n')
                f.write('distance, true_distance, correct, is_in_pool, read_offset, matched_offset, true_offset\n')
            else:
                f.write('distance,refname\n')

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
        if(ws):
            seq_thread.join()
        for i in range(nthreads):
            worker_threads[i].join()
        info('Done.')
        return time.time()-starttime

def read_reference_file(infile: str, queue: PQueue):
    info("reading reference file {}".format(infile))
    ref_dict = {}
    for name, seq in pyfastx.Fasta(infile, build_index=False):
        ref_dict[name]=(len(seq), seq)
        
        queue.put((name, seq))
        

    queue.put(('END', ''))
    info('Reader exiting.')

def write_file(filename, seq_queue: TQueue, nthreads):
        with open(filename, 'w') as f:
            f.write('true_ref,match_ref,correct,true_ref_off,match_ref_off,read_off\n')
            while nthreads:
                stat = seq_queue.get()
                
                if stat[0][0] == 'X':
                        nthreads -= 1
                       
                else:
                    
                    f.write(stat[0]+'\n')
                    f.write(stat[1]+'\n')
                    f.write(stat[2]+'\n')
                    f.write(stat[3]+'\n')   

def read_query_file(infile: str, frac: float, queue: PQueue, w):
    if frac > 1.0:
        frac = 1.0
    info("reading query file {}".format(infile))
    for name, seq in pyfastx.Fasta(infile, build_index=False):
        if random.random() < frac:
            if len(seq) > 200:
                queue.put((name, seq[:200]))
            elif (len(seq)>=w):
                queue.put((name, seq))
    queue.put(('END', ''))
    info('Reader exiting.')