## About The Project

This is the code for the thesis paper "Tensor Sketch for Fast similary Search Using Annoy". It is based on the paper "Fast Alignment-Free Similarity Estimation By Tensor Sketching" written by Joudaki et al. (2020) and Python's Approximate Nearest Neighbors Oh Yeah (https://github.com/spotify/annoy). The code is written in Python 3.10. In lib/annoy_sketch.py you will find the backend of the code that build and queries the AnnoyIndex. In lib/vectorizations.py you will find different vectorization methods, MinHash, Tensor Embedding, Tensor Sketch and additionally, Min Hash Simple, kmer-pos and kmer-distance, where last 3 are not introduced in the paper. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/github_username/repo_name.git
   ```
2. Create new virutal Python environment and install the dependencies requirements.txt
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### How to run the code
You could directly run main.py with different arguments then it automatically builds and subsequently queries with the given arguments.

Arguments
---------------
#build specific parameters:
* ``-PR, --file_path_ref`` specifies the file path of the reference data to build AnnoyIndex
* ``-FR, --reference-file`` specifies the reference file name
* ``-O, --on_disk`` uses Annoy's on_disk_build to build the AnnoyIndex on hard disk. The default is True
* ``-R, --rebuild_index`` rebuilds the AnnoyIndex. If False it tries to find stored AnnoyIndex specified by file_path_ref/reference_file_name + _vectorizer.annoy, where reference_file_name is just reference_file without .fasta at the end. Default is True
* ``-BT, --build_threads`` number of processes to build
* ``-Df, --dict_flag`` builds a dictionary containing reference name -> length(sequence), sequence. Default is True
* ``-Nt, --number of trees``this is the same as n_trees specified in the paper.
* ``-Wf, --write_flag`` after building the AnnoyIndex it will save the AnnoyIndex and other necessary datastrucutres in file_path_ref. AnnoyIndex will be saved as file_path_ref/reference_file_name + _vectorizer.annoy. Default is False.

#used by both build and query
* ``-V, --vectorizer`` vectorization method to build and query AnnoyIndex. Choices are ['kmer_pos','kmer_dist','tensor_sketch','tensor_embedding','min_hash_simple','min_hash']. Default tensor_sketch.
* ``-K, --kmer_length`` kmer length to vectorize. This is the same as kmer length introduced in the paper.
* ``-D, --sketch_dim`` sketch dimension of vectorization. This is the same as sketch dimension introduced in the paper.
* ``-I, --index_type`` only Annoy is available.
* ``-W, --window_size`` window size to vectorize. This is the same as window size introduced in the paper.
* ``-S, --stride`` stride size to shift the windows. This is the same as stride size introduced in the paper.
* ``-Pf, --pre_fault`` pre_fault determines how Annoy loads the AnnoyIndex from disk. If True it uses MAP_POPULATE to preload the AnnoyIndex into memory for faster querying. MMAP will be used regardless to share the same AnnoyIndex. Default is True.
* ``-TMP, --tmp_directory`` tmp_directory to store the AnnoyIndex temporarily for faster IO-operations and automatic deletion after program execution. This is the same as TMPDIR introduced in the paper where I used it in the settings of Euler Cluster. The code tries to save to tmp_directory first, if it does not exist it write_flag should be set when you want to store it in hard disk.

#query specific arguments
* ``-PQ, --file_path_query`` specifies the file path of the read data to query into AnnoyIndex
*  ``-PQ, --file_path_query`` specifies the file name of the read data to query into AnnoyIndex
*  ``-F, --query_frac`` specifies fraction of queries it should use to query. This is useful if you have too many queries in the file and want to query only a fraction of it
*  ``-P, --out_prefix`` specifies the filepath where the results of each query should be stored
*  ``-QT, --query_threads`` specifies the number of processes to query
*  ``-C, --check_correct`` if True write more information like correctly labeled, memory usage etc. to out_prefix
*  ``-Fn, --fac_nearest`` specifies search factor. This is the same as search factor introduced in the paper.
*  ``-Ws, --write_sequence`` if True write true sequence, matched sequence and read sequence to another file in out_prefix for wrongly labeled queries.
*  
Notes:
query assumes to be of the form S1_45!NC_004719.1:74975-75374!0!400!+@45[194], where S1_45 is the query name, NC_004719.1 being the true reference name and 74975-75374 being the offset start and offset end.
Zymo has the header S1_349!tig00000306!118409!118719!+ but I've rewritten the header using filter_newdataset.ipynb

there are also build and query functions in the main.py that you can call. It has similar paramteres but query has 2 additional parameters eer and read_stride. eer stands for expected error of the queries which has default value 0.1 and read_stride is the stride you use to slide through the windows for the queries. Default is 1.




