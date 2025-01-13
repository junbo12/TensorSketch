w_list = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
s_list=[1,2,4]
k_list = [3,4,5,6,7,8,9,10,11,12]
sd_list=[32,64,96,128,160,192,224,256]
tree_list = [8]
fac_list = [2]
with open('job_index.txt', 'w') as f:
    f.write('ArrayTaskID sketch kmer window stride n_trees fac\n') 
    id = 1
    for sd in sd_list:
        for k in k_list:
            for w in w_list:
                for s in s_list:
                    for t in tree_list:
                        for fr in fac_list:
                            if(w>=s and w>=k):                                
                                f.write('{} {} {} {} {} {} {} \n'.format(id,sd,k,w,s,t,fr))
                                id +=1
print(id-1)