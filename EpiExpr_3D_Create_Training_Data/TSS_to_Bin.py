from optparse import OptionParser
import numpy as np
import pandas as pd
# import time
import os

##======================
## adapted from the script utils/find_tss.py from the GraphReg package 
## https://github.com/karbalayghareh/GraphReg

## Objective: TSS Annotations - to get the number of TSS in individual bins.
## Writes for each bin, the TSS sites (and corresponding genes) falling in that bin.

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##======================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-s', 
                      dest='seqfile', 
                      default=None, 
                      type='str', 
                      help='File containing the bin intervals for the current chromosome. Mandatory parameter.')
    
    parser.add_option('-O', 
                      dest='OutDir', 
                      default=None, 
                      type='str', 
                      help='Output directory. Mandatory parameter.')
    
    parser.add_option('-c', 
                      dest='currchr', 
                      default=None, 
                      type='str', 
                      help='Current chromosome. Mandatory parameter.')
    
    parser.add_option('-T', 
                      dest='TSSFile', 
                      default=None, 
                      type='str', 
                      help='Reference Genome specific TSS file. Mandatory parameter.')

    (options, args) = parser.parse_args()
    return options, args

##==============
## main routine
##==============
def main():
    options, args = parse_options()

    filename_seqs = options.seqfile
    OutDir = options.OutDir
    os.makedirs(OutDir, exist_ok = True)
    currchr = options.currchr
    filename_tss = options.TSSFile

    print("\n\n ===>> Function TSS_to_Bin ---- Input parameters <<==== \n\n")
    print("\n filename_seqs : ", filename_seqs)
    print("\n BaseOutDir : ", OutDir)    
    print("\n Target Chromosome : ", currchr)    
    print("\n GTF file (for reference genome) : ", filename_tss)

    ## read the reference GTF file
    ## this GTF file contains the following subset of fields extracted from the original reference genome specific GTF file
    tss_dataframe = pd.read_csv(filename_tss, header=None, delimiter='\t')
    tss_dataframe.columns = ["chr", "tss", "ens", "gene", "strand", "type"]

    ##===========
    ## list of output files to be created
    ##===========
    ## read the sequence file
    seq_dataframe = pd.read_csv(filename_seqs, delimiter='\t')
    seq_dataframe.columns = ["chr", "start", "end"]

    seq_dataframe['start'] = seq_dataframe['start'].astype(np.int64)
    seq_dataframe['end'] = seq_dataframe['end'].astype(np.int64)

    ##============
    ## extract protein coding genes within current chromosome
    ## these genes will be only used for training and testing the model
    ##============
    protein_coding_tss = tss_dataframe.loc[(tss_dataframe['type'] == 'protein_coding') 
                                           & (tss_dataframe['chr'] == currchr)]
    protein_coding_tss = protein_coding_tss.reset_index(drop=True)
    # print(protein_coding_tss)
    n_tss = len(protein_coding_tss) ## number of TSS / genes (protein coding)
    print("\n Number of protein coding genes for this chromosome : ", n_tss)

    ## dump the protein coding genes and corresponding GTF information for the current chromosome
    if 0:
        outfilename = OutDir + '/Protein_Coding_Genes_' + str(currchr) + '.csv'
        protein_coding_tss.to_csv(outfilename, index=False)

    ##================
    ## Define a data frame for individual fixed sized bins
    ## columns: chr, bin_start, bin_end, numTSS, TSS_index_vec
    ## where numTSS = number of TSS falling in that bin (we'll use the protein coding genes for the current chromosome file)
    ## TSS_index_vec = vector of indices (rows) w.r.t. protein coding genes for the current chromosome file such that corresponding TSS falls within that bin
    ## for example, numTSS = 2 and TSS_index_vec = 1042,1055 means that the 1042th and 1055th entry of the protein coding gene TSS file falls within the particular bin
    ##================
    n_bin = len(seq_dataframe)
    TSSDF = pd.DataFrame(
        {
        'chr': seq_dataframe['chr'].values, 
        'bin_start': seq_dataframe['start'].values,
        'bin_end': seq_dataframe['end'].values,
        'numTSS': np.zeros(n_bin).astype(np.int64),
        'TSS_index_vec': ['-' for _ in range(n_bin)]
        }
    )

    ## fill the TSSDF data structure
    for i in range(n_bin):
        ## identify the indices of "protein_coding_tss" structure
        ## such that the TSS falls within the particular bin
        ## we consider the exact overlap
        curr_bin_start = seq_dataframe['start'][i]
        curr_bin_end = seq_dataframe['end'][i]

        idx_tss = protein_coding_tss.loc[(protein_coding_tss['tss'] >= curr_bin_start) 
                                         & (protein_coding_tss['tss'] <= curr_bin_end)].index
        TSSDF['numTSS'][i] = len(idx_tss)
        
        if (len(idx_tss) > 0):
            ## comma separated indices - final output is a string
            TSSDF['TSS_index_vec'][i] = ','.join(map(str, idx_tss))
        
        if 1:
            print('==>>> Checking TSS and bin overlap - processing chromsome : ', str(currchr), 
                  ' processing bin index : ', i, 
                  ' number of TSS falling in the bin : ', len(idx_tss), 
                  '  TSS index vector : ', TSSDF['TSS_index_vec'][i])
        
    ## dump the TSSDF data structure
    ## this will be useful later to fetch the gene / TSS specific information
    ## from the CAGE resolution
    outfilename1 = OutDir + '/TSS_Info_' + str(currchr) + '.csv'
    TSSDF.to_csv(outfilename1, index=False)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()


