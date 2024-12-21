from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import random

def filter( max_ref_size, window_size, min_ref_size = 0):

    input = "scratch/data/bacteria_1g_a.fa"

    output = "scratch/data/filter_same.fa"

    #ref_fa = random.sample(list(r for r in SeqIO.parse(input_ref, "fasta") if len(r.seq) < max_ref_size), ref_count)
    ref_fa = [r for r in SeqIO.parse(input, "fasta") if min_ref_size <= len(r.seq) < max_ref_size]
    ref_fa = random.sample(ref_fa, 1)[0]

    ref_id = ref_fa.id
    ref_seq = ref_fa.seq
    print(len(ref_seq))
    output_list = []
    i = 0
    id = 0
    while i < len(ref_seq)-window_size:

        output_list.append(SeqRecord(Seq(ref_seq[i:i+window_size]),id = str(id), description=''))
        i += window_size
        id +=1

    count_ref = SeqIO.write(output_list, output, 'fasta-2line')
    print("Saved %i random records from %s to %s" % (count_ref, input, output))


