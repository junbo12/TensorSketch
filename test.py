from lib.test_emb import *
import csv
import numpy as np
import pandas as pd
import os.path
import openpyxl
import gc
from filter_same import *

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
            uniq = row[2]
            correct = row[3]
            occ = row[4]
            neigh = row[5]
            tp += correct
            fp += 1-correct
            uniq_num += uniq
            occ_r += occ
            neigh_r += neigh
        
        return query_num, uniq_num/query_num, fp/(tp+fp), occ_r/query_num, neigh_r/query_num     

def build_TE(subseq_len: int = 3, indextype = 'annoy', w: int = 120, o:int = 100,tensor_sketch_dim: int = 16):
    params = SketchParams(A=4, t=subseq_len, D=tensor_sketch_dim)
    te = TE(params)
    Vectorizer = namedtuple('Vectorizer', ['func', 'ndim'])
    tensor_embedding = Vectorizer(func=te.sketch, ndim=4**subseq_len)
    idx = RefIdx(filename = reference_file, vectorizer = tensor_embedding, w=w, o=o, index=indextype, rebuild=True,  wf=False)
    return idx

def eval_te(subseq_len, indextype, w_list, s_list):

    outfile = 'scratch/out/te_{}_{}_same.csv'.format(indextype, subseq_len)
    res = np.zeros((len(w_list),len(s_list)))

    for i,w in enumerate(w_list):
        for j,s in enumerate(s_list):
            idx = build_TE(subseq_len = subseq_len, indextype = 'annoy', w = w, o = w-s)
            idx.query_file(fread_file, outfile, frac=1.0)
            del idx
            gc.collect()
            query_num, uniq_r, fpr, match_r , neigh_r= get_score_csv(outfile)
            res[i][j] = fpr

            print('finished', subseq_len, w, s)
    
    return res

def table_to_excel(res, filename, sheet_name, row = 0, col = 0, index = [], columns = []):
        filepath = 'scratch/eval/{}.xlsx'.format(filename)
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
        
        with pd.ExcelWriter(filepath, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer: 
            df = pd.DataFrame(res, index = index, columns = columns)
            df.to_excel(writer, sheet_name=sheet_name, startrow = row, startcol = col)

if __name__ == '__main__':
    indextype = 'annoy'     # can be 'faiss', 'annoy', 'usearch'

    reference_file = 'scratch/data/filter_random_ref.fa'

    fread_file= 'scratch/data/filter_random_read.fa'
    '''outfiles = ['scratch/out/{}_{}.csv'.format(indextype, x) for x in 'abh']
    idx = build_TE(subseq_len = 5, indextype = 'annoy', w = 80, o = 79,tensor_sketch_dim = 16)
    idx.query_file(fread_file, outfiles[0],  frac=1.0)
    print(get_score_csv(outfiles[0], check_correct = True))
    '''
    length_ref = 180
    filename = 'annoy_random'
    sheetname = '1e6, 10, 1000, serial'.format(length_ref)
    min_ref_size = 1e6
    max_ref_size=2e6
    #filter(max_ref_size,length_ref, min_ref_size)
    w_list = [80,100,120]
    s_list = [1,5,10]
    sub_l = [5]
    for sl in sub_l:
        res = eval_te(sl, indextype, w_list, s_list)
        table_to_excel(res, filename, sheetname, row = (sl-3)*len(w_list)+2*(sl-3), index = w_list, columns = s_list)
        print('added to excel', filename, sheetname, sl)
