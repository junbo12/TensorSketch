import csv
import datetime
import argparse
from collections import namedtuple
import math
from lib.vectorizations import TS, MH, TE, KS
from lib.annoy_index import RefIdx
from lib.base import SketchParams
import os
from multiprocessing import cpu_count
# sys.path.append(os.path.abspath('.'))
Vectorizer = namedtuple('Vectorizer', ['func', 'name'])

#distance, true_distance, correct, is_in_pool, read_offset, matched_offset, true_offset   
def classify(filepath, th=math.inf):
    with open(filepath,'r') as f:

        reader = csv.reader(f)
        l = [row for row in reader]
        data = [[float(el) for el in row] for row in l[1:]]
        
        
        tp = 0
        fp = 0 
        tn = 0
        fn = 0
        missr = 0
        edr=0
        miss_num = 0
        for row in data:
            
            dist = row[0]
            ed=row[7]
            true_ed = row[8]
            correct = int(row[2])
            is_in_pool = bool(int(row[3]))

            if is_in_pool and not correct:
                true_dist = row[1]
                miss_num += 1
                if true_ed<ed:
                    edr+=1
                if true_dist < dist:
                    missr += 1

            if (dist<=th):
                if is_in_pool:
                    tp += correct
                    fp += 1-correct
                else:
                    fn += 1

            else:
                if(is_in_pool):
                    fp +=1
                else:
                    tn += 1

    fpr = -1 if tp+fp==0 else fp/(tp+fp)
    fnr = -1 if tn+fn==0 else fn/(tn+fn)
    missr = 0 if miss_num == 0 else missr/miss_num
    edr = 0 if miss_num == 0 else edr/miss_num
    return fpr, fnr, missr,edr


def build(filename, tmp_directory, rebuild, vectorizer, kmer_len, sketch_dim, index_type, window=20,stride=2,n_trees=32, dict_flag=True, on_disk=False, write_flag=True,prefault=True, build_threads = cpu_count()):
    assert kmer_len <= sketch_dim
    params = SketchParams(A=4, t=kmer_len, D=sketch_dim)
    sketcher = None 
    match vectorizer:
        case 'kmer_pos':
            assert sketch_dim == 4**kmer_len
            ks = KS(params)
            sketcher = Vectorizer(func=ks.vector_of_positions, name='KP')
        case 'kmer_dist':
            assert sketch_dim == 4**kmer_len
            kd = KS(params)
            sketcher = Vectorizer(func=kd.vector_of_distances, name='KD')
        case 'tensor_sketch':
            ts = TS(params)
            sketcher = Vectorizer(func=ts.sketch, name='TS')
        case 'tensor_embedding':
            assert sketch_dim == 4**kmer_len
            te = TE(params)
            sketcher = Vectorizer(func=te.sketch, name='TE')
        case 'min_hash_simple':
            assert window-kmer_len+1 >= sketch_dim
            mhs = MH(params)
            sketcher = Vectorizer(func=mhs.mh_sketch, name='MHS')
        case 'min_hash':
            mh = MH(params)
            sketcher = Vectorizer(func=mh.sketch, name='MHS')
    #self, filename: str, index, vectorizer: Vectorizer,w: int, s: int, sketchdim: int, n_trees, rebuild: bool, build_dict, build_threads, prefault
    idx = RefIdx(filename,tmp_directory, index_type, sketcher, sketch_dim, window, stride, n_trees,rebuild,on_disk,dict_flag,write_flag,build_threads,prefault)

    return idx


def query(idx:RefIdx, query_file, out_file,check_correct=True,query_frac=1.0,search_fac=2,read_stride=1,eer=0.1,ws=False, prefault=False, query_threads=8):
    
    query_num, avg_query_size, query_time,total_vec_time,total_index_time,total_pss = idx.query_file(query_file, out_file,check_correct, query_frac, search_fac,read_stride,eer,ws,prefault,query_threads)
    if check_correct:
        fpr, fnr,missr,edr = classify(out_file)  
        return query_num, avg_query_size, fpr, fnr, missr, edr,query_time, total_vec_time, total_index_time,total_pss 
    else:
        return query_num, avg_query_size, -1, -1,-1, -1,query_time, total_vec_time, total_index_time,total_pss 
    


def write_to_csv(row, file_path, file_name):
    filepath = '{}/{}.csv'.format(file_path, file_name)
    file_size = os.path.getsize(file_path)
    flag = (file_size==0 or not os.path.exists(filepath))
    with open(filepath,'a+') as f:
        if flag:
            
            f.write('n_trees,search_fac,sketch_dim,kmer_len,window,stride,fpr,missr,edr,build_time,ref_max_mem,query_per_sec,query_time,vec_time,index_time,query_max_mem\n')
        f.write(row)
    return 


class FloatRange(object):
            def __init__(self, start: float, end: float):
                self.start = start
                self.end = end

            def __eq__(self, other: float) -> bool:
                return self.start <= other <= self.end

            def __repr__(self):
                return '[{0},{1}]'.format(self.start, self.end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #build specific parameters
    parser.add_argument('-PR', '--file_path_ref', default = './scratch/data')
    parser.add_argument('-FR', '--reference_file', default = 'filter_random_ref.fasta')
    parser.add_argument('-O', '--on_disk', action='store_false')
    parser.add_argument('-R', '--rebuild_index', action='store_false')
    parser.add_argument('-BT', '--build_threads', type=int, default=8)  
    parser.add_argument('-Df', '--dict_flag', action='store_false')   
    parser.add_argument('-Nt', '--n_trees', type=int, default=32)
    parser.add_argument('-Wf', '--write_flag', action='store_true')

    #used by both build and query
    parser.add_argument('-V', '--vectorizer', choices=['kmer_pos','kmer_dist','tensor_sketch','tensor_embedding','min_hash_simple','min_hash'], default='tensor_sketch')
    parser.add_argument('-I', '--index_type', choices=['annoy'] ,default = 'annoy')
    parser.add_argument('-K', '--kmer_length', type=int, default=6)  
    parser.add_argument('-D', '--sketch_dim', type=int, default=64)
    parser.add_argument('-W', '--window_size', type=int, choices=range(10, 300), default=100)
    parser.add_argument('-S', '--stride', type=int, default=1)
    parser.add_argument('-Pf', '--pre_fault', action='store_false')
    parser.add_argument('-TMP', '--tmp_directory', default = './tmp')

    #query specific arguments
    parser.add_argument('-PQ', '--file_path_query', default = './scratch/data')
    parser.add_argument('-FQ', '--query_file', default = 'filter_random_read.fasta')
    parser.add_argument('-F', '--query_frac', type=float, choices=[FloatRange(0.01, 1.0)], default=1.0)
    parser.add_argument('-P', '--output_prefix', default='./scratch/out')
    parser.add_argument('-QT', '--query_threads', type=int, default=8) 
    parser.add_argument('-C', '--check_correct', action='store_false')
    parser.add_argument('-Fn', '--fac_nearest', type=float, default=2.0)
    parser.add_argument('-Ws', '--write_sequence', action='store_true')
    args = parser.parse_args()
    reference_name = args.reference_file
    reference_file = args.file_path_ref + '/' + reference_name
    query_name = args.query_file
    query_file = args.file_path_query + '/' + query_name
    vectorizer = args.vectorizer
    index_type = args.index_type
    sketch_dim = args.sketch_dim
    kmer_len = args.kmer_length
    window = args.window_size
    stride = args.stride
    output_prefix = args.output_prefix
    query_frac = args.query_frac
    rebuild = args.rebuild_index
    build_threads = args.build_threads
    query_threads = args.query_threads
    prefault = args.pre_fault
    write_sequence = args.write_sequence
    check_correct = args.check_correct
    n_trees = args.n_trees
    search_fac = args.fac_nearest
    dict_flag = args.dict_flag
    tmp_directory = args.tmp_directory
    write_flag = args.write_flag
    on_disk = args.on_disk
    ws =args.write_sequence
    
    def format_time(num):
        print(num)
        tmp = str(datetime.timedelta(seconds=round(num,2))).split('.')
        return tmp[0]+ '.' + tmp[1][:2]


    idx = build(reference_file,tmp_directory, rebuild, vectorizer, kmer_len, sketch_dim, index_type, window, stride,n_trees, dict_flag,on_disk, write_flag, prefault, build_threads)
    
    output_file  =  '{}/{}_{}.csv'.format(output_prefix,reference_name,idx.id)

    #ref_max_mem, idx = memory_usage(proc=(build, () ,{'filename': reference_file, 'build_threads': build_threads, 'vectorizer':vectorizer, 'index_type':index_type, 
    #                            'kmer_len': kmer_len, 'sketch_dim': sketch_dim, 'rebuild':rebuild, 'prefault':prefault, 'window':window,'stride':stride}),interval=.5, include_children=True, multiprocess=True, max_usage=True,retval=True)
    vectorizing_time = idx.vectorizing_time
    build_time = idx.build_time
    ref_max_mem = idx.total_pss
    print('ref num: {}, avg ref size: {}, num elements: {}'.format(idx.ref_num,idx.avg_ref_size, idx.n_items))
    print(f'vectorizing_time: {vectorizing_time:.2f}, build_time : {build_time:.2f}, max_mem_usage: {ref_max_mem:.2f}MB')

    #query_max_mem, ret_val = memory_usage(proc=(query,(),{'idx':idx,'query_file': query_file, 'query_threads': query_threads, 'out_file':out_file, 'check_correct':check_correct, 
    #                           'frac': 1.0, 'ws': False}),interval=.5, include_children=True, multiprocess=True, max_usage=True,retval=True)

    query_num, avg_query_size, fpr, fnr, missr,edr,query_time,total_vec_time,total_index_time,total_pss = query(idx,query_file,output_file, check_correct, query_frac, search_fac,1,0.1,ws,prefault, query_threads)
    query_max_mem = total_pss 
    print(f'query_num: {query_num}, avg query size:{avg_query_size}, false positive rate: {fpr}, query_per_sec{query_num/query_time},query_time: {query_time:.2f},index_time: {total_index_time:.3f},vec_time: {total_vec_time:.3f},query_mem_usage: {query_max_mem:.2f}MB')
    str_row = f'{n_trees},{search_fac},{sketch_dim},{kmer_len},{window},{stride},{fpr},{missr},{edr},{build_time + vectorizing_time},{ref_max_mem},{query_num/query_time},{query_time},{total_vec_time},{total_index_time},{query_max_mem}\n'
    
    #write statistics into a specified csv file
    csv_name = '{}_{}_test'.format(index_type, vectorizer)
    out_prefix = './scratch/eval/'
    write_to_csv(str_row, out_prefix, csv_name)