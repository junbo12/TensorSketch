## About The Project

This is the code for the thesis paper "Tensor Sketch for Fast similary Search Using Annoy". It is based on the paper "Fast Alignment-Free Similarity Estimation By Tensor Sketching" written by Joudaki et al. (2020) and Python's Approximate Nearest Neighbors Oh Yeah (https://github.com/spotify/annoy). The code is written in Python 3.10. In lib/annoy_sketch.py you will find the backend of the code that build and queries the AnnoyIndex. In lib/vectorizations.py you will find different vectorization methods, MinHash, Tensor Embedding, Tensor Sketch and additionally, Min Hash Simple, kmer-pos and kmer-distance, where last 3 are not introduced in the paper. 
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/junbo12/tensorsketch.git
   ```
2. Create new virutal Python environment and install the dependencies requirements.txt
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### How to run the code
You could directly run main.py with different arguments then it automatically builds and subsequently queries with the given arguments.

Arguments
---------------
#build specific parameters:
* ``-PR, --file_path_ref`` specifies the file path of the reference data to build AnnoyIndex
* ``-FR, --reference-file`` specifies the reference file name, usually a fasta file
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
*  ``-F, --query_frac`` specifies fraction of queries it should use to query. This is useful if you have too many queries in the file and want to query only a fraction of it. Default is 1.0.
*  ``-P, --out_prefix`` specifies the filepath where the results of each query should be stored
*  ``-QT, --query_threads`` specifies the number of processes to query
*  ``-C, --check_correct`` if True write more information like correctly labeled, memory usage etc. to out_prefix
*  ``-Fn, --fac_nearest`` specifies search factor. This is the same as search factor introduced in the paper.
*  ``-Ws, --write_sequence`` if True write true sequence, matched sequence and read sequence to another file in out_prefix for wrongly labeled queries.
  
Notes
---------------
Query header assumes to be of the form S1_45!NC_004719.1:74975-75374!0!400!+@45[194], where S1_45 is the query name, NC_004719.1 being the true reference name and 74975-75374 being the offset start and offset end.
Zymo has the header S1_349!tig00000306!118409!118719!+ but I've rewritten the header using filter_newdataset.ipynb. If you have other headers you might need to change the split_query_header function found in lib/annoy_index.py

Calling main.py will just call build and query functions in the main.py, that has additional checks like if Tensor Embedding received the correct sketch dimension=4^kmer length and classifies the results. Query function has 2 additional parameters eer and read_stride. eer stands for expected error of the queries which has default value 0.1 and read_stride is the stride you use to slide through the windows for the queries. Default is 1.

Furthermore the query function requires a Vectorizer type argument which is just a namedtuple of the form (func: vectorizing function, name: arbitrary name). You can not pass out_prefix to the function argument outfile , as outfile requires filepath + csv name, an example is provided in main.py.

You might find filter_random.py useful for filtering a large dataset of refernce file and query file for testing.

Return values of build() and query()
---------------
returns an instance of RefIdx containing class attributes:
build():
* ``id`` id to identify the specific RefIdx. Used in main.py combined with out_prefix so there is no clashing.
* ``dict_flag`` same as in Argument above
*  ``prefault `` same as in Argument above. This will used for building, loading and querying.
*  ``on_disk `` same as in Argument above. If True the code unloads the AnnoyIndex and load it with prefault=False after building the AnnoyIndex. This is due to the observation that I made, that if you you on_disk_build flag in Annoy, the memory of building the AnnoyIndex stays as opposed to unloading it first and then loading it with prefault = False.
*  ``w `` window size
*  ``s `` stride size
*  ``d `` sketch dimension
*  ``write_flag `` same as in Argument above
*  ``vectorizer `` a Namedtuple of the form (func: vectorizing function, name: arbitrary name)
*  ``index`` indextype. Only Annoy is implemented.
*  ``t`` instance of AnnoyIndex
*  ``n_trees`` number of trees to build t
*  ``n_items`` number of items in t
#used for query
*  ``arr_hash `` list of hashed reference names
*  ``arr_offset `` list of offsets for each item in AnnoyIndex. This is as large as number of elements in the AnnoyIndex
*  ``hash_dict`` dictionary translating hashed reference names and reference names
*  ``ref_dict`` dictionary containing length of reference sequence and the actual reference sequence
*  ``tmp_directory`` temporary directory to save t. This is powerful in Euler Cluster in terms of speedup.
* ``filename`` filename of reference sequence data
* ``savefilename`` filename to save t, this is either tmp_directory/reference_file_vectorizer.annoy or filename
#others
*  ``ref_num`` number of references. Only available if you rebuild and read the reference files.
*  ``avg_ref_size`` average size of references. Only available if you rebuild and read the reference files.
*  ``vectorizing_time`` time to vectorize the reference data and adding it to AnnoyIndex. Only available if you rebuild and read the reference files.
*  ``build_time`` time to build AnnoyIndex. Or time to load if rebuild=False.
*  ``total_pss`` max of psutils.memory_full_info().pss after vectorizing and building. This does sum up across all processes, since I did not want to double count the vectorized reference sequences that are being sent between MultiProcessing Queues.


query():
* ``query_num`` number of queries. This number is after filtering with query_frac.
* ``avg_query_size`` average size of queries
* ``fpr`` false positive rate calculated by #mismatches/(#mismatches+matches). This is the same metric used in the paper.
* ``fnr`` false negative rate. This is not yet implemented. The idea is to set a threshhold value to determine if a query matches to the AnnoyIndex or not by setting th in classify() function in main.py. If a query matches with Euclidean distance < th, then it is a match, else it doesn't match any of the references in AnnoyIndex. fnr is not introduced in the paper.
* ``missr`` this should be a metric to calculate how many queries out of the incorrectly labeled ones were due to Annoy's search. At the moment it is calculated as following: given a wrongly matched query, calculate the minimal Euclidean Distance between the neighborhood of the true offset and matched query window. If there exists a smaller Euclidean Distance, this means that Annoy is theoretically able to find it.
* ``edr`` this stands for edit distance rate and should be a metric to calculate how many misses were due to edit distance of the neighborhood of the true offset being larger than the edit distance of the matched refence sequence. At the moment it calculates in the same way as missr but using edit distance
* ``query_time`` the total execution time of the query function. Includes Process Calling, result aggregation and calculating missr, edr.
* ``total_vec_time`` the total time spent vectorizing the queries. This is calculated average over the total_vec_time across all processes
* ``total_index_time`` the total time spent searching (calling the function t.get_nns_by_vector()). This is calculated average over the total_index_time across all processes
* ``total_pss`` sum of pss=proportional set size across all processes in the querying using psutil.memory_full_info().pss. This memory cost metric is used in the paper.
