from optparse import OptionParser
import numpy as np
import pandas as pd
# import time
import os
import scipy.sparse
import sys
import subprocess
import math
# import configparser
import yaml

##======================
## Sourya
## adapted from the script utils/hic_to_graph.py from the GraphReg package 
## https://github.com/karbalayghareh/GraphReg

## Input:
## FitHiChIP loops along with their significance (all loops)
## Output:
## compressed numpy matrix - graph represenation of the contacts
## Optional: FDR threshold - contacts with FDR < specified threshold are only considered 
##======================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)    
    parser.add_option('-C', dest='configfile', default=None, type='str', help='Configuration file. Mandatory parameter.')
    
    (options, args) = parser.parse_args()
    return options, args

##==============
## main routine
##==============
def main():
    options, args = parse_options()

    ## read the input configuration file
    # config = configparser.ConfigParser()
    # config.read(options.configfile)

    config_fp = open(options.configfile, "r")
    config = yaml.load(config_fp, Loader=yaml.FullLoader)    

    refgenome = config['General']['Genome']
    BaseOutDir = config['General']['OutDir']
    chrsizefile = config['General']['ChrSize']
    FDRThr = float(config['Loop']['FDRThr'])
    LoopFile = config['Loop']['loopfile']
    Resolution = int(config['Loop']['resolution'])
    SampleLabel = config['Loop']['SampleLabel']

    CurrOutDir = BaseOutDir + "/" + SampleLabel + "/FitHiChIP_to_mat/contact_mat_FDR_" + str(FDRThr)
    if not os.path.exists(CurrOutDir):
        os.makedirs(CurrOutDir)

    ## read the chromosome size file
    chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])        

    ## check individual chromosomes   
    for rowidx in range(chrsize_df.shape[0]):      
        currchr = str(chrsize_df.iloc[rowidx, 0])
        print("\n\n rowidx : ", rowidx, "  processing chromosome : ", currchr)      

        OutMatFile = CurrOutDir + "/" + str(currchr) + ".npz"
        if (os.path.exists(OutMatFile) == True):
            continue

        ## discard any chromosomes other than chr1 to chr22
        if (currchr == "chrX" or currchr == "chrY" or currchr == "chrM" or "un" in currchr or "Un" in currchr or "random" in currchr or "Random" in currchr or "_" in currchr):
            continue

        print("\n\n ==>> Extracting loop information for the chromosome : " + str(currchr))

        ## extract chromosome, start of first interacting bin, start of second interacting bin, FDR and contact count
        tempfile = CurrOutDir + "/temp_contact_" + str(currchr) + ".txt"
        sys_cmd = "awk \'{if ((NR>1) && ($1 == \"" + str(currchr) + "\") && ($NF < " + str(FDRThr) + ")) {print $1\"\t\"$2\"\t\"$5\"\t\"$NF\"\t\"$7}}\' " + str(LoopFile) + " > " + str(tempfile)
        os.system(sys_cmd)

        hic_dataframe = pd.read_csv(tempfile, header=None, delimiter='\t')
        hic_dataframe.columns = ["chr", "start_i", "start_j", "qval", "count"]
        # hic_dataframe = hic_dataframe.astype({"start_i":"int","start_j":"int","qval":"float","count":"int"})
        if 0:
            print(hic_dataframe)

        ## sequence file for the current chromosome
        ## generated from the previous step
        filename_seqs = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution) + '/sequences_' + currchr + '.bed'
        if (os.path.exists(filename_seqs) == False):
            continue

        seq_dataframe = pd.read_csv(filename_seqs, header=None, delimiter='\t')
        seq_dataframe.columns = ["chr", "start", "end"]
        if 0:
            print(seq_dataframe)

        ##### write the whole hic matrix for a chromosome as a sparse matrix #####

        ## number of intervals for this chromosome
        nodes_list_all = sorted(list(set(seq_dataframe['start'].values)))
        n_nodes_all = len(nodes_list_all)
        print('\n ==>> number of all nodes (bins): ', n_nodes_all, nodes_list_all)
        
        ## data structure to store the contact matrix
        hic = np.zeros([n_nodes_all, n_nodes_all])

        # ## old code
        # if 0:
        #     nodes_dict_all = {}
        #     for i in range(n_nodes_all):
        #         nodes_dict_all[nodes_list_all[i]] = i

        #     for i in range(n_nodes_all):
        #         ## get the interactions whose first interacting bin start coordinate = nodes_list_all[i]
        #         cut = hic_dataframe.loc[hic_dataframe['start_i'] == nodes_list_all[i]]
        #         ## get the set of other interacting bins
        #         idx = cut['start_j'].values
        #         print("\n ==>> i : ", str(i), " first bin: (", nodes_list_all[i], ",", (nodes_list_all[i]+Resolution), " number of interactions associated - ", len(idx))
        #         ## there exists at least one interaction involving the current first interacting bin
        #         if (len(idx) > 0):
        #             idx_col = np.zeros(len(idx), dtype=int)
        #             for j in range(len(idx)):
        #                 ## store the other interacting bin indices
        #                 idx_col[j] = nodes_dict_all[idx[j]]
        #             ## store the contact counts
        #             hic[i, idx_col] = cut['count'].values
        #             if 0:
        #                 print(cut['count'].values)

        ## new code - much simpler
        ## compatible to FitHiChIP format
        ## (start_bin, end_bin, contact count)
        if 1:
            ## the astype(int) conversion is very important - other numpy stores as floating point numbers
            bin1_idx = (hic_dataframe['start_i'].values / Resolution).astype(int)
            bin2_idx = (hic_dataframe['start_j'].values / Resolution).astype(int)
            count_values = (hic_dataframe['count'].values).astype(int)
            hic[bin1_idx, bin2_idx] = count_values

        ## create symmetric matrix
        ## store as numpy sparse matrix
        hic_t = hic.transpose()
        hic_sym = hic + hic_t
        #row_max = np.max(hic_sym, axis=1)
        #print('hic_max: ', row_max)
        #hic_sym = hic_sym + np.diag(row_max)
        hic_sym = hic_sym.astype(np.float32)
        # print(hic_sym[2000:2020,2000:2020])
        sparse_matrix = scipy.sparse.csr_matrix(hic_sym)

        ## write the HiC matrix        
        scipy.sparse.save_npz(OutMatFile, sparse_matrix)

        ## delete temporary files
        sys_cmd = "rm " + tempfile
        os.system(sys_cmd)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
                   
