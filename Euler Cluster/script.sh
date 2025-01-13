#!/bin/bash

#SBATCH --job-name=array
#SBATCH --array=1-1692
#SBATCH --output=extend/%A_%a.out
#SBATCH --error=extend/%A_%a.out
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G              
#SBATCH --job-name=tensorsketch
#SBATCH --tmp=24G
activate_venv() {
    local env_name=${1:-".venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create one."
        return 1
    fi

    source "./$env_name/bin/activate"
    
}

/cluster/scratch/junbzhang

activate_venv
# default values:
F_REF="filter_random_ref.fasta"
F_QUE="filter_random_read.fasta"
#F_REF="Refs_positive.fasta "
#F_QUE="Sigs_positive.fasta"
#F_REF="test_ref.fasta"
#F_QUE="test_read.fasta"
config=tree_TS.txt


# Print the task id.
sketch_dim=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
kmer_len=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
window=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
stride=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
n_trees=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
fac=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)
echo ''ArrayTask $SLURM_ARRAY_TASK_ID computing kmer: $kmer_len window: $window stride: $stride sketch_dim: $sketch_dim n_trees: $n_trees fac $fac''
srun python main.py -FR $F_REF -FQ $F_QUE -K $kmer_len -D $sketch_dim -W $window -S $stride -Nt $n_trees -Fn $fac -BT 16 -QT 16 -TMP $TMPDIR -O 
#srun python tree_main2.py -R $F_REF -Q $F_QUE -K 3 -D 64 -W $window -S $stride -Nt $n_trees -Fn $fac -BT 16 -QT 16 -TMP $TMPDIR -V 'tensor_embedding'
#mprof run -C -M tree_main.py -R $F_REF -Q $F_QUE -K $kmer_len -D $sketch_dim -W $window -S $stride -Nt $n_trees -Fn $fac -BT 16 -QT 16 -TMP $TMPDIR
#mprof plot --output=./mem_info/