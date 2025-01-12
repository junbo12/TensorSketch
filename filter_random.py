from Bio import SeqIO
from Bio.Seq import Seq
import random

#format S1_45!NC_004719.1:74975-75374!0!400!+@45[194]
#>S1_349!tig00000306!118409!118719!+
def get_ref(record):
    tmp1 = record.id.split('!')
    tmp2 = tmp1[1].split(':')
    return tmp2[0]


max_ref = 2
max_read = 10
max_ref_size = 100000
flag = False

input_ref = "./scratch/data/bacteria_1g_a.fasta"
input_read= "./scratch/data/bacteria_1g_a_basecalled_r9.fasta"
input_read2 = "./scratch/data/bacteria_1g_b_basecalled_r9.fasta"
output_ref = "./scratch/data/filter_random_ref.fasta"
output_read = "./scratch/data/filter_random_read.fasta"


#ref_fa = random.sample(list(r for r in SeqIO.parse(input_ref, "fasta") if len(r.seq) < max_ref_size), ref_count)
ref_fa = [r for r in SeqIO.parse(input_ref, "fasta") if len(r.seq) < max_ref_size]
if(len(ref_fa)>max_ref):
    ref_fa = random.sample(ref_fa, max_ref)

avg_len = 0
ref_count = 0
for r in ref_fa:
    len_seq = len(r.seq)
    print(len_seq)
    avg_len += len_seq
    ref_count += 1
print('avg len: ', int(avg_len/ref_count))
count_ref = SeqIO.write(ref_fa, output_ref, 'fasta-2line')

print("Saved %i random records from %s to %s" % (count_ref, input_ref, output_ref))



ref_id = [r.id for r in ref_fa]

read_fa =  [r for r in SeqIO.parse(input_read, "fasta") if get_ref(r) in ref_id]

if len(read_fa)> max_read:
    read_fa = random.sample(read_fa, max_read)
if(flag):
    read_fa = read_fa + (random.sample([r for r in SeqIO.parse(input_read2, "fasta")], max_read))

print(len(read_fa))
avg_read = 0
read_count = 0
for r in read_fa:
    len_seq = len(r.seq)
    avg_read += len_seq
    read_count += 1
print(int(avg_read/read_count))
count_read = SeqIO.write(read_fa, output_read, 'fasta-2line')

print("Saved %i random records from %s to %s" % (count_read, input_read, output_read))

