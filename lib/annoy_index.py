import os

from math import dist
import sys
import pyfastx
from datetime import datetime as dt
from annoy import AnnoyIndex
import random
from tqdm import tqdm
import pickle
from multiprocessing import Array as PArray
from multiprocessing import Value as PValue
from multiprocessing import Process
from multiprocessing import Queue as PQueue
from collections import namedtuple
import time
from threading import Thread
import psutil
import numpy as np
from fast_edit_distance import edit_distance
# sys.path.append(os.path.abspath('.'))

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


class RefIdx:
    def __init__(self, filename: str, tmp_directory,index, vectorizer: Vectorizer, sketchdim: int,
                 w: int, s: int, n_trees, rebuild: bool, on_disk:bool, dict_flag:bool,write_flag:bool, build_threads, prefault):
        """
        Creates an index of the reference sequences
        :param filename: reference file
        :param vectorizer: the function that is used to vectorize the sequences
        :param w: sequence window to turn into a vector
        :param o: overlap between windows
        :param rebuild: if true, rebuild the index even though an index was found in the current directory
        :param sketchdim: if greater than 0, sketch the vector into this dimension
        """
        random.seed(42)
        self.id = time.time()*1000
        self.dict_flag = dict_flag
        self.prefault = prefault
        self.on_disk = on_disk
        self.w = w
        self.s = s
        self.write_flag = write_flag
        self.vectorizer = vectorizer 
   
        self.index = index # should only be annoy
        self.d = sketchdim 

        self.t = None
        self.arr_hash = None
        self.arr_offset = None
        self.ref_dict = {}
        self.hash_dict = None
        self.n_trees = n_trees
        self.n_items = 0

        self.tmp_directory = tmp_directory
        self.filename = filename
        self.savefilename = ''
        
        self.ref_num = 0
        self.avg_ref_size =0
        self.vectorizing_time = -1
        self.build_time = -1
        self.total_pss = 0 #in MB

        if index == 'annoy':
            self.t = AnnoyIndex(self.d,'euclidean')
            self.t.set_seed(42)
            vectorizing_time, build_time = self.build_annoy_index(filename, rebuild, build_threads)
            self.vectorizing_time = vectorizing_time
            self.build_time = build_time 
        else:
            assert False, "Not implemented"

    def build_annoy_index(self, filename: str, rebuild: bool = False, build_threads=8):
        

        if (not rebuild and os.path.exists(self.filename + '_{}.annoy'.format(self.vectorizer.name)) 
                and os.path.exists(self.filename+ '_{}.ref_hash'.format(self.vectorizer.name)) 
                and os.path.exists (self.filename + '_{}.hash_dict'.format(self.vectorizer.name))
                and os.path.exists (self.filename+ '_{}.ref_offset'.format(self.vectorizer.name))):
                load_time = time.time()
                info("Loading annoy index from disk.." + (self.filename + '_{}.annoy'.format(self.vectorizer.name)))
                self.t.load(self.filename + '_{}.annoy'.format(self.vectorizer.name), prefault = self.prefault)
                if os.path.exists(self.tmp_directory): #save to tmp directory if possible 
                    self.savefilename = self.tmp_directory +'/{}.annoy'.format(self.vectorizer.name)
                    self.t.save(self.savefilename,prefault=False)
                else:
                    self.savefilename = self.filename +'_{}.annoy'.format(self.vectorizer.name)
                self.n_items = self.t.get_n_items()
                self.n_trees = self.t.get_n_trees()
                with open(self.filename + '_{}.ref_hash'.format(self.vectorizer.name), 'rb') as fp:
                    self.arr_hash = pickle.load(fp)
                with open(self.filename + '_{}.ref_offset'.format(self.vectorizer.name), 'rb') as fp:
                    self.arr_offset = pickle.load(fp)
                with open(self.filename + '_{}.hash_dict'.format(self.vectorizer.name), 'rb') as fp:
                    self.hash_dict = pickle.load(fp)
                if self.dict_flag:
                    if os.path.exists (self.filename + '_{}.ref_dict'.format(self.vectorizer.name)):

                        with open(self.filename + '_{}.ref_dict'.format(self.vectorizer.name), 'rb') as fp:
                            self.ref_dict = pickle.load(fp)
                load_time = time.time()-load_time
                info('loaded in {}'.format(load_time))

                info('done.')
                return -1,load_time 
        else:
            proc = psutil.Process()
            info("Adding items to annoy index..")

            if os.path.exists(self.tmp_directory):
                print('tmp_directory exists!')
                self.filepath = self.tmp_directory
                self.savefilename = self.filepath +'/{}_{}.annoy'.format(self.vectorizer.name,self.id)
            else:
                self.savefilename = filename +'_{}.annoy'.format(self.vectorizer.name)
            
            if self.on_disk:
                self.t.on_disk_build(self.savefilename)  
            
            num_items =0
            gen = self.vectorize_references_par(filename, build_threads)
            
            hash_dict = {}
            arr_offset = []
            arr_hash = []
            vectorizing_time = time.time()
            for offsets,vectors,name,hash_name in gen:
                
                hash_time = time.time()
                arr_hash.extend(np.full(len(offsets),hash_name))
                arr_offset.extend(offsets)
                hash_dict[hash_name]=name
                info('hash_time {}'.format(time.time()-hash_time))
                add_time = time.time()
                for vector in vectors:
                    
                    self.t.add_item(num_items,vector)

                    num_items +=1
                info('add_time {}'.format(time.time()-add_time))   
                sys.stdout.flush() 
            self.n_items = num_items
            self.hash_dict = hash_dict
            self.arr_hash = np.array(arr_hash,dtype=np.int32)
            self.arr_offset = np.array(arr_offset,dtype=np.int32)
            self.num_items=num_items
            vectorizing_time = time.time() - vectorizing_time
            build_start= time.time()
            

            self.total_pss= max(proc.memory_full_info().pss, self.total_pss)

            info("Building index..{}".format(self.n_trees))
            self.t.build(n_trees=self.n_trees, n_jobs=2*build_threads)
            
            self.total_pss = max(proc.memory_full_info().pss, self.total_pss)
            info('Writing index to file.. ' + self.savefilename)
            
            
            if (self.on_disk):
                self.t.unload()
                self.t.load(self.savefilename,prefault=False)
            else:
                self.t.save(self.savefilename,prefault=False)
            if self.write_flag:
                if not os.path.exists(self.filename+'.annoy',):
                    self.t.save(self.filename+'.annoy',prefault=False)
                
                with open(self.filename + '_{}.ref_hash'.format(self.vectorizer.name), 'wb') as fp:
                    pickle.dump(self.arr_hash, fp, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.filename + '_{}.hash_dict'.format(self.vectorizer.name), 'wb') as fp:
                    pickle.dump(self.hash_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                with open(self.filename + '_{}.ref_offset'.format(self.vectorizer.name), 'wb') as fp:
                    pickle.dump(self.arr_offset, fp, protocol=pickle.HIGHEST_PROTOCOL)
                if self.dict_flag:
                    with open(self.filename + '_{}.ref_dict'.format(self.vectorizer.name), 'wb') as fp:
                        pickle.dump(self.ref_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                
            
            build_time = time.time() - build_start
            
            
            self.total_pss = max(proc.memory_full_info().pss, self.total_pss)/1000000

            info("Done.")
            
            return vectorizing_time, build_time

    @staticmethod
    def vectorize_sequence_par(read_queue: PQueue, write_queue: PQueue, w: int, s: int, vectorizer: Vectorizer):
        while True:
            
            offsets = [] 
            vectors = []
            name, seq = read_queue.get()
            hash_name = np.int32(hash(name)& 0xFFFFFFFF)
            if name == 'END':

                read_queue.put((name, seq))

                write_queue.put((0,0,0,'END'))
                break
            else:
                
                length = len(seq)
                if length >= w:
                    offset = np.int32(0)
                    while offset + w <= length:
                        offsets.append(offset)
                        vectors.append(vectorizer.func(seq[offset:offset + w]))
                        
                        offset += s
                    
                    if 2*(length-offset) >= w: #if the remaining part is larger than half the window size, then add it to the vectors
                        offset = length-w
                        vectors.append(vectorizer.func(seq[offset:offset + w]))
                        offsets.append(offset)  

            write_queue.put((np.array(offsets),np.array(vectors),name,hash_name))
            
        info('Worker exiting.')

    
    #vectorizes references in parallel

    def vectorize_references_par(self, filename, build_threads):

        info(f'vectorizing with {build_threads}')

        nthreads = build_threads
        readQueue = PQueue(maxsize=4 * nthreads)
        writeQueue = PQueue(maxsize=2 * nthreads)
        reader_thread =Thread(target=self.read_reference_file, args=(filename, readQueue,self.dict_flag))
        reader_thread.start()
        
        worker_threads = []
        for _ in range(nthreads):
            worker_thread = Process(target=RefIdx.vectorize_sequence_par,
                                    args=(readQueue, writeQueue, self.w, self.s, self.vectorizer))
            worker_threads.append(worker_thread)
            worker_thread.start()
        finished = 0
        while True:
            offsets, t_vectors,name,hash_name = writeQueue.get()    
            
            if hash_name == 'END': 
                finished+=1
                if finished == nthreads:
                    readQueue.get()
                    break
            else:      
                yield (offsets,t_vectors,name,hash_name)
        reader_thread.join()
        for worker in worker_threads:
            worker.join()
    
    def read_reference_file(self, infile: str, queue: PQueue,dict_flag):
        info("reading reference file {}".format(infile))
        ref_num = 0
        avg_ref_size =0
        for name, seq in pyfastx.Fasta(infile, build_index=False):
            len_seq = len(seq)
            if len_seq>=self.w:
                if(dict_flag):
                    self.ref_dict[name]=(len_seq, seq)
                
                queue.put((name, seq))
                avg_ref_size += len_seq
            
            ref_num += 1

        self.ref_num = ref_num
        self.avg_ref_size = int(avg_ref_size/ref_num)
        queue.put(('END', 0))
        info('Reader exiting.')
    
    @staticmethod
    def summarize_scores(matches: list) -> tuple:
        """
        Summarize matches for windows into a single tuple containing (1) the name of the closest reference,
        (2) distance from this reference, and (3) whether this reference was the closest in all windows over the query.
        :param matches: a list of 4-tuples - (ref_name, distance, read_offset, ref_offset)
        :return: a 4-tuple - (ref-name, distance, read_offset, ref_offset)
        """
        if len(matches) > 0:
            # return the smallest distance
            matches = sorted(matches, key=lambda x: x[1])
            return matches[0][0], matches[0][1], matches[0][2], matches[0][3]
        return '', -1, -1, -1

    
    @staticmethod
    def query_window_check(q: str,t, w,s, vec_func, arr_hash, arr_offset, search_fac) -> tuple:
            assert s <= len(q) <= (2*w - s)

            vec_time = time.time()
            v = vec_func(q)
            vec_time = time.time()-vec_time

            index_time = time.time()
            
            ids, distances = t.get_nns_by_vector(v, 1, search_k = max(int(search_fac*t.get_n_trees()),1), include_distances=True)
            index_time = time.time()-index_time
            
            name_hash = arr_hash[ids[0]]
            offset = arr_offset[ids[0]]
            distance = distances[0]
            
            return name_hash, offset, distance, vec_time, index_time
   
    @staticmethod
    def query_found(q: str,t, vec_func, w,s, arr_hash,arr_offset, ws, read_stride, search_fac,hash_dict):
        
        
        
        read_offset =0
        matches = []
        len_q = len(q)
        total_vec_time =0 
        total_index_time=0
        
        #query at least once 
        first = min(w, len(q))
        name_hash, ref_offset, dist,vec_time,index_time = RefIdx.query_window_check(q[read_offset:read_offset + first],t,w,s,vec_func,arr_hash,arr_offset,search_fac)
        total_vec_time += vec_time
        total_index_time += index_time          
        matches.append((name_hash,dist,read_offset, ref_offset))
        read_offset += read_stride
        
        while read_offset + w <= len_q:
            name_hash, ref_offset, dist,vec_time,index_time = RefIdx.query_window_check(q[read_offset:read_offset + first],t,w,s,vec_func,arr_hash,arr_offset,search_fac)
            total_vec_time += vec_time
            total_index_time += index_time
            matches.append((name_hash,dist,read_offset, ref_offset))

            read_offset += read_stride
        
        
        refname_hash,dist, read_offset, ref_offset = RefIdx.summarize_scores(matches)
        read_seq = None
        if(ws):
            read_seq = q[read_offset:read_offset + w]
        return hash_dict[refname_hash], dist, read_offset, ref_offset,read_seq, total_vec_time, total_index_time
    
    @staticmethod
    def query_thread(idx_path,read_queue: PQueue, write_queue: PQueue,sketch_dim, vec_func, arr_hash,arr_offset,ref_dict,hash_dict,check_correct,w,s, eer=0.1,seq_queue = None, vec_t=None, index_t=None,mem_t=None,prefault=False,read_stride=1, search_fac=2):
        t = AnnoyIndex(sketch_dim,'euclidean')     
        
        t.load(idx_path,prefault=prefault) #use annoys mmap to share the index
        ws = (seq_queue is not None)
        while True:
            
            name, seq = read_queue.get()
            if name == 'END':
                read_queue.put((name, seq))
                write_queue.put("X")
                if ws:
                    seq_queue.put('X')
                break
            else:

                target_refname, read_name,true_offset = split_query_header(name)

                if check_correct:
                    assert ref_dict, 'check_correct not allowed when ref_dict not built'

                    refname, distance, read_offset, ref_offset, read_seq,total_vec_time, total_index_time = RefIdx.query_found(seq,t,vec_func,w,s,arr_hash,arr_offset,ws,read_stride,search_fac,hash_dict)

                    
                    is_in_pool = int(target_refname in ref_dict.keys())
                    true_distance = -1
                    edit_dist = -1
                    match_edit_dist = -1
                    th=int(eer*100)
                    if is_in_pool: #soft check
                        '''
                        bruteforce_check = False #setting this to true is very slow
                        match_edit_dist = edit_distance(seq[read_offset:read_offset+w],ref_dict[refname][1][ref_offset:ref_offset+w],max_ed=2*(int(th))) 
                        if bruteforce_check:#bruteforce check the true vector sketch distance and edit distance
                            len_seq = len(seq)
                            true_distance = min([dist(vec_func(ref_dict[target_refname][1][i:i+w]),vec_func(seq[j:j+w])) for j in range(0,len_seq-w+1,read_stride) for i in range(max(0,true_offset-th*s),min(ref_dict[target_refname][0],true_offset+len_seq-w+1+th*s),s)])
                            edit_dist = min([edit_distance(ref_dict[target_refname][1][i:i+w],seq[j:j+w],max_ed=2*(int(th))) for j in range(0,len_seq-w+1,read_stride) for i in range(max(0,true_offset-th*s),min(ref_dict[target_refname][0],true_offset+len_seq-w+1+th*s),s)])
                        '''
                        
                        stride = s
                        nearest_distance = (true_offset + read_offset)%stride
                        nearest_offset = true_offset+read_offset-nearest_distance
                        true_distance = min([dist(vec_func(ref_cdt),vec_func(read_cdt)) for i in range(-th,th+1)  if abs(i*stride) <= th 
                                            and len(ref_cdt:=ref_dict[target_refname][1][nearest_offset+i*stride:max((nearest_offset+i*stride)+w,0)])==w  
                                                and len(read_cdt:=seq[read_offset:read_offset+w])==w])
                        edit_dist = min([edit_distance(ref_cdt,read_cdt,max_ed=2*(int(th))) for i in range(-th,th+1)  if abs(i*stride) <= th 
                                            and len(ref_cdt:=ref_dict[target_refname][1][nearest_offset+i*stride:max((nearest_offset+i*stride)+w,0)])==w  
                                                and len(read_cdt:=seq[read_offset:read_offset+w])==w])
                    write_queue.put("{},{},{},{},{},{},{},{},{}\n".format(
                        distance, true_distance, int(refname == target_refname), is_in_pool, read_offset, ref_offset, true_offset+read_offset,match_edit_dist,edit_dist
                    ))
                   
                    if ws:
                        desc =  "{},{},{},{},{},{}".format(target_refname, ref_dict[refname], refname == target_refname, true_offset+read_offset, ref_offset, read_offset)
                        true_ref = ref_dict[target_refname][1][true_offset+read_offset:true_offset+read_offset+w]
                        match_ref = ref_dict[refname][1][ref_offset:ref_offset+w]
                        
                       
                        seq_queue.put((desc,true_ref,match_ref,read_seq))
                else:
                    target_refname,true_offset, read_name = split_query_header(name)
                    refname, distance, read_offset, ref_offset, read_seq,total_vec_time, total_index_time = RefIdx.query_found(seq,t,vec_func,w,s,arr_hash,arr_offset,ws,read_stride,search_fac,hash_dict)
                    write_queue.put("{},{},{},{},{},{}\n".format(refname, true_offset,read_name,read_offset,distance, int(refname == target_refname)))
            
            vec_t.value += total_vec_time
            index_t.value += total_index_time
        mem_t.value=psutil.Process().memory_full_info().pss
        info('Worker exiting.')

    def query_file(self, infile, outfile, check_correct=True, query_frac=1.0,search_fac = 2, read_stride=1,eer=0.01, ws = False, prefault=False, query_threads = 8):
        pid = os.getpid()
        proc = psutil.Process(pid)
        cpu_num = proc.cpu_num()

        if self.dict_flag:          
            ref_dict  = self.ref_dict           
        else:
            ref_dict = None
            
        hash_dict = self.hash_dict 
        arr_hash = PArray('i',self.arr_hash,lock=False)
        arr_offset = PArray('i',self.arr_offset,lock=False)
        
        val_query_num = PValue('i')
        val_avg_query_size = PValue('i')
        nthreads = query_threads
        vec_t  = [PValue('f',0.0,lock=False) for _ in range (nthreads)]
        index_t  = [PValue('f',0.0,lock=False) for _ in range (nthreads)]   
        mem_t  = [PValue('f',0.0,lock=False) for _ in range (nthreads)]   
        readQueue = PQueue(maxsize= 10 * nthreads)
        writeQueue = PQueue(maxsize= 8 * nthreads)
        if ws:
            seqQueue = PQueue(maxsize= 8 * nthreads)
        else:
            seqQueue = None
        
        worker_threads = []
        query_num = 0
        info("Querying with {} threads".format(query_threads))
        
        
        start_time = time.time()
        
        reader_thread = Process(target=read_query_file, args=(infile, query_frac, readQueue, self.w,val_query_num,val_avg_query_size))
        reader_thread.start()
        for i in range(nthreads):
            
            worker_thread = Process(target=RefIdx.query_thread, args=(self.savefilename,readQueue, writeQueue,self.d, self.vectorizer.func, arr_hash,arr_offset, ref_dict,hash_dict, check_correct,self.w,self.s, eer, seqQueue,vec_t[i],index_t[i],mem_t[i],prefault,read_stride,search_fac))
            worker_threads.append(worker_thread)
            worker_thread.start()
        
        if ws:
            tmp = outfile.split('.')[0]
            filename = tmp+'_seq.csv'
            seq_thread = Process(target=RefIdx.write_file, args=(filename, seqQueue,nthreads))
            seq_thread.start()

        with open(outfile, 'w') as f:
            if check_correct:
                #f.write('distance, correct, is_in_pool, unique, occurence_matching, occurence_querying, read_offset, matched_offset, true_offset\n')
                f.write('distance, true_distance, correct, is_in_pool, read_offset, matched_offset, true_offset,ed,true_ed\n')
            else:
                f.write('name,refname,distance,correct\n')
            avg_write_time = 0
            n_threads_remaining = nthreads
            n_lines = get_num_lines(infile) // 2
            with tqdm(total=n_lines, mininterval=60) as pbar:
                while n_threads_remaining:
                    write_time = time.time()
                    stat = writeQueue.get()
                    if stat[0] == 'X':
                        n_threads_remaining -= 1
                    else:
                        f.write(stat)
                        pbar.update(1)

                    avg_write_time += time.time() - write_time
               
        
        reader_thread.join()
        for i in range(nthreads):
            worker_threads[i].join()
        if ws :
            seq_thread.join()
        query_num = val_query_num.value

        avg_query_size = val_avg_query_size.value
        total_pss = proc.memory_full_info().pss  
        total_pss += sum([val.value for val in mem_t])
        total_vec_time = sum([val.value for val in vec_t])/query_threads
        total_index_time = sum([val.value for val in index_t])/query_threads

        info('Done.')
        return query_num, avg_query_size, time.time()-start_time, total_vec_time, total_index_time, total_pss/1000000

    @staticmethod
    def write_file(filename, seq_queue: PQueue, nthreads):
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
#S1_45!NC_004719.1:74975-75374!0!400!+@45[194]
def split_query_header(header: str):
    substr1 = header.split('!')[1]
    tokens = substr1.split(':')
    ref_name = tokens[0]
    start = int(tokens[1].split('-')[0])
    return ref_name,  header.split('!')[0],start


def read_query_file(infile: str, frac: float, queue: PQueue, w, val_query_num, val_avg_query_size):
    query_num = 0
    avg_query_size = 0
    if frac > 1.0:
        frac = 1.0
    info("reading query file {}".format(infile))
    for name, seq in pyfastx.Fasta(infile, build_index=False):
        
        if random.random() < frac:
            if len(seq) > 200:
                avg_query_size += 200
                queue.put((name, seq[:200]))
            elif (len(seq)>=w):
                queue.put((name, seq))
                avg_query_size += len(seq)
            query_num += 1

    queue.put(('END', ''))
    val_query_num.value = query_num
    val_avg_query_size.value = int(avg_query_size/query_num)
    info('Reader exiting.')
    return query_num, avg_query_size