from lib.vector_sketch import *
import csv
import numpy as np
import pandas as pd
import os.path
import openpyxl
import gc
import time
import datetime
import seaborn as sns
Vectorizer = namedtuple('Vectorizer', ['func', 'ndim'])



def get_score_csv(filepath):

    with open(filepath,'r') as f:

        reader = csv.reader(f)
        l = [row for row in reader]
        data = [[float(el) for el in row] for row in l[1:]]
        
        query_num = len(data)

        assert query_num > 0
        uniq_num = 0
        occ_r = 0
        neigh_r = 0
        tp = 0
        fp = 0
        for row in data:
            correct = row[1]
            uniq = row[3]
            occ = row[4]
            neigh = row[5]
            tp += correct
            fp += 1-correct
            uniq_num += uniq
            occ_r += occ
            neigh_r += neigh

        return query_num, uniq_num/query_num, fp/(tp+fp), occ_r/query_num, neigh_r/query_num
    
#distance, true_distance, correct, is_in_pool, read_offset, matched_offset, true_offset   
def classify_csv(filepath, th=math.inf):
    with open(filepath,'r') as f:

        reader = csv.reader(f)
        l = [row[:4] for row in reader]
        data = [[float(el) for el in row] for row in l[1:]]
        
        query_num = len(data)
        tp = 0
        fp = 0 
        tn = 0
        fn = 0
        for row in data:
            
            dist = row[0]
            correct = row[2]
            is_in_pool = bool(int(row[3]))
            if (dist<=th):
                if(is_in_pool):
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
    return query_num, fpr, fnr


def build_TE(filename, subseq_len: int = 3, indextype = 'annoy', w: int = 120, o:int = 100,tensor_sketch_dim: int = 16, rebuild=True, wf=False):
    params = SketchParams(A=4, t=subseq_len, D=tensor_sketch_dim)
    te = TE(params)
    
    tensor_embedding = Vectorizer(func=te.sketch, ndim=4**subseq_len)
    start_time = time.time()
    idx = RefIdx(filename = filename, vectorizer = tensor_embedding, w=w, o=o, index=indextype, rebuild=rebuild,  wf=wf)
    total_time = time.time() - start_time
    return idx, total_time

def build_TS(filename, subseq_len: int = 3, indextype = 'annoy', w: int = 120, o:int = 100,tensor_sketch_dim: int = 16, rebuild=True, wf=False):
    params = SketchParams(A=4, t=subseq_len, D=tensor_sketch_dim)
    ts = TS(params)
    
    tensor_sketching = Vectorizer(func=ts.sketch, ndim=4**subseq_len)
    start_time = time.time()
    idx = RefIdx(filename = filename, vectorizer = tensor_sketching,sketchdim = tensor_sketch_dim, w=w, o=o, index=indextype, rebuild=rebuild,  wf=wf)
    total_time = time.time() - start_time
    return idx, total_time

def build_MH(filename, subseq_len: int = 3, indextype = 'annoy', w: int = 120, o:int = 100,tensor_sketch_dim: int = 16, rebuild=True, wf=False):
    params = SketchParams(A=4, t=subseq_len, D=tensor_sketch_dim)
    mh = MH(params)
    
    tensor_sketching = Vectorizer(func=mh.mh_sketch, ndim=4**subseq_len)
    start_time = time.time()
    idx = RefIdx(filename = filename, vectorizer = tensor_sketching,sketchdim = tensor_sketch_dim, w=w, o=o, index=indextype, rebuild=rebuild,  wf=wf)
    total_time = time.time() - start_time
    return idx, total_time

def eval(filename, fread_file, subseq_len, indextype, w_list, s_list, rebuild= True, wf=False, sketch_dim = 16, th=math.inf):

    outfile = 'scratch/out/ts_{}_{}.csv'.format(indextype, subseq_len)
    res = np.zeros((len(w_list),len(s_list)))
    res_fnr = np.zeros((len(w_list),len(s_list)))
    build_table = [ ['']*len(s_list)for _ in range(len(w_list))]
    query_table = [ ['']*len(s_list)for _ in range(len(w_list))]
    for i,w in enumerate(w_list):
        for j,s in enumerate(s_list):
            idx, build_time = build_TS(filename, subseq_len = subseq_len, indextype = indextype, w = w, o = w-s, rebuild=rebuild, wf=wf, tensor_sketch_dim = sketch_dim)
            start_time = time.time()
            idx.query_file(fread_file, outfile, check_correct=True, frac=1.0, ws=False)
            query_time = time.time() - start_time
            del idx
            build_table[i][j] = str(datetime.timedelta(seconds=round(build_time)))
            query_table[i][j] = str(datetime.timedelta(seconds=round(query_time)))
            gc.collect()
            #query_num, uniq_r, fpr, match_r , neigh_r= get_score_csv(outfile)
            _, fpr, fnr = classify_csv(outfile,th)
            res[i][j] = fpr
            res_fnr[i][j] = fnr
            print('finished', subseq_len, w, s)
    
    return res, build_table, query_table, res_fnr

def table_to_csv(table, file_path,file_name, windows, strides):
    filepath = '{}/{}.csv'.format(file_path, file_name)
    with open(filepath,'w') as f:
        
        csvWriter = csv.writer(f,delimiter=',')
        csvWriter.writerows([windows])
        csvWriter.writerows([strides])
        csvWriter.writerows(table)

    return 

def table_to_excel(res, file_path, file_name, sheet_name, subseq_len, row = 0, col = 0, index = [], columns = []):
        filepath = '{}/{}.xlsx'.format(file_path, file_name)
        first_time = False
        
        if not os.path.isfile(filepath):
            df_empty = pd.DataFrame()
            df_empty.to_excel(filepath)
            first_time = True
            
        workbook = openpyxl.load_workbook(filepath)
        
        if first_time and not (sheet_name in workbook.sheetnames):
            ex_sheet = workbook['Sheet1']
            ex_sheet.title = sheet_name
            workbook.save(filepath)
        workbook.close()
        
        with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer: 
            df = pd.DataFrame(res, index = index, columns = columns)
            df.to_excel(writer, sheet_name=sheet_name, startrow = row, startcol = col)
        workbook = openpyxl.load_workbook(filepath)
        worksheet = workbook[sheet_name]
        worksheet.cell(row=row+1, column = col+1).value = 't = {}'.format(subseq_len)
        workbook.save(filepath)
        workbook.close()



if __name__ == '__main__':
    indextype = 'annoy'     # can be 'faiss', 'annoy', 'usearch'

    
    query_files = ['scratch/data/{}_reads.fa'.format(x) for x in [
        'bacteria_1g_a',
        'bacteria_1g_b',
        'human'
    ]]
    r9_basecalled_query_files = ['scratch/data/{}_basecalled_r9.fasta'.format(x) for x in [
        'bacteria_1g_a',
        'bacteria_1g_b',
        'human'
    ]]
    
    outfiles = ['scratch/out/{}_{}.csv'.format(indextype, x) for x in 'abh']
    
    if indextype == 'faiss':
        #idx = RefIdx(reference_file, d=D - 1, index='faiss', rebuild=True, vectorizer=distance_vector)
        #idx = RefIdx(reference_file, index='faiss', rebuild=True, vectorizer=position_vector, sketchdim=16)
        pass
    elif indextype == 'annoy':
        #idx = RefIdx(reference_file, index='annoy', rebuild=True, vectorizer=distance_vector)
        #idx = RefIdx(reference_file, index='annoy', rebuild=True, vectorizer=position_vector)
        #idx = build_TE(reference_file, subseq_len = 5, indextype = 'annoy', w = 80, o = 79,tensor_sketch_dim = 16, rebuild=True)
        pass
    elif indextype == 'usearch':
        #idx = RefIdx(reference_file, index='usearch', rebuild=True, vectorizer=distance_vector)
        #idx = RefIdx(reference_file, index='usearch', rebuild=True, vectorizer=position_vector)
        pass
    else:
        raise Exception('Not implemented')

    #idx.query_file(fread_file, outfiles[0], check_correct=True, frac=1.0)
    #idx.query_file(r9_basecalled_query_files[1], outfiles[1], check_correct=False, frac=.05)
    #idx.query_file(r9_basecalled_query_files[2], outfiles[2], check_correct=False, frac=.05)

    #subseq_len, indextype, w, o, tensor_sketch_dim
    reference_file = 'scratch/data/filter_random_ref.fasta'
    fread_file= 'scratch/data/filter_random_read.fasta'
    w_list = [100,120]
    s_list = [5,10]
    sub_l = [8,9]
    sketch_dim = 128
    indextype = 'annoy' 
    filepath = 'scratch/eval'
    excel_name = 'annoy_Random_TS'
    
    sheetname = '128, 2n_trees, 16'
    offset = 0
    '''
    for sl in sub_l:
        
        res, build_time, query_time,res_fnr = eval(reference_file, fread_file, sl,  indextype, w_list, s_list, rebuild=True, wf = False, sketch_dim=sketch_dim)
        table_to_excel(res,filepath, filename, excel_name , sl, row = offset*len(w_list)+2*offset, index = w_list, columns = s_list)
        table_to_excel(res_fnr, filepath, excel_name , sheetname, sl, row = offset*len(w_list)+2*offset,col = len(s_list) + 2, index = w_list, columns = s_list)
        table_to_excel(build_time, filepath, fexcel_name , sheetname, sl, row = offset*len(w_list)+2*offset, col = 2*len(s_list) + 4, index = w_list, columns = s_list)
        table_to_excel(query_time, filepath, excel_name , sheetname, sl, row = offset*len(w_list)+2*offset, col = 3*len(s_list) + 6, index = w_list, columns = s_list)
        offset+=1
        print('added to excel', filename, sheetname, sl)
    '''
    for sl in sub_l:
        csv_name = 'annoy_{}'.format(sl)
        res, build_time, query_time,res_fnr = eval(reference_file, fread_file, sl,  indextype, w_list, s_list, rebuild=True, wf = False, sketch_dim=sketch_dim)
        
        table_to_csv(res, filepath, csv_name, w_list, s_list)
    
    