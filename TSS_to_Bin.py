from optparse import OptionParser
import numpy as np
import pandas as pd
import time
import os

##======================
## Sourya
## adapted from the script utils/find_tss.py from the GraphReg package 
## https://github.com/karbalayghareh/GraphReg

## Objective: TSS Annotations - to get the number of TSS in individual bins.
## Writes for each bin, the TSS sites (annd corresponding genes) falling in that bin.
##======================

##======================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-g', dest='refgenome', default=None, type='str', help='Reference Genome. Mandatory parameter.')
    parser.add_option('-c', dest='chrsizefile', default=None, type='str', help='Reference Genome specific chromosome size file. Mandatory parameter.')
    parser.add_option('-O', dest='BaseOutDir', default=None, type='str', help='Base output directory. Mandatory parameter.')
    parser.add_option('-r', dest='Resolution', default=5000, type='int', help='Loop resolution. Default 5000 (5 Kb)')
    parser.add_option('-T', dest='GTF', default=None, type='str', help='Reference Genome specific GTF file. Mandatory parameter.')

    (options, args) = parser.parse_args()
    return options, args

##==============
## main routine
##==============
def main():
    options, args = parse_options()

    refgenome = options.refgenome
    chrsizefile = options.chrsizefile
    BaseOutDir = options.BaseOutDir
    Resolution = int(options.Resolution)
    filename_tss = options.GTF

    OutDir = BaseOutDir + '/TSS/' + refgenome + '/' + str(Resolution)
    if not os.path.exists(OutDir):
        os.makedirs(OutDir)    

    ## printing / debug variables
    n_tss_bins_all = 0
    n_tss_bins_all_1 = 0
    n_tss_bins_all_2 = 0
    n_tss_bins_all_3 = 0
    n_tss_bins_all_4 = 0
    n_tss_bins_all_5 = 0
    n_tss_bins_all_6 = 0

    # only keep the tss bins whose distance from bin borders are more than "thr" 
    # (only applicable when want to consider the bins with 1 tss, otherwise thr = 0)
    thr = 0

    ## read the chromosome size file
    chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])

    ## read the reference GTF file
    tss_dataframe = pd.read_csv(filename_tss, header=None, delimiter='\t')
    tss_dataframe.columns = ["chr", "tss_1", "tss_2", "ens", "gene", "strand", "type"]

    ## check individual chromosomes   
    for rowidx in range(chrsize_df.shape[0]):      
        currchr = str(chrsize_df.iloc[rowidx, 0])
        print("\n\n rowidx : ", rowidx, "  processing chromosome : ", currchr)      

        ## discard any chromosomes other than chr1 to chr22
        if (currchr == "chrX" or currchr == "chrY" or currchr == "chrM" or "un" in currchr or "Un" in currchr or "random" in currchr or "Random" in currchr or "_" in currchr):
            continue

        ## check if files exist
        bin_start_file = OutDir + '/bin_start_' + currchr + '.npy'
        tss_bins_file = OutDir + '/tss_bins_' + currchr + '.npy'
        tss_gene_file = OutDir + '/tss_gene_' + currchr + '.npy'
        tss_pos_file = OutDir + '/tss_pos_' + currchr + '.npy'

        if (os.path.exists(bin_start_file) == True) and (os.path.exists(tss_bins_file) == True) and (os.path.exists(tss_gene_file) == True) and (os.path.exists(tss_pos_file) == True):
            continue

        ## sequence file for the current chromosome
        ## generated from the previous step
        filename_seqs = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution) + '/sequences_' + currchr + '.bed'
        if (os.path.exists(filename_seqs) == False):
            continue

        ## read the sequence file
        seq_dataframe = pd.read_csv(filename_seqs, header=None, delimiter='\t')
        seq_dataframe.columns = ["chr", "start", "end"]
        # print(seq_dataframe)

        ## extract protein coding genes within current chromosome
        protein_coding_tss = tss_dataframe.loc[(tss_dataframe['type'] == 'protein_coding') & (tss_dataframe['chr'] == currchr)]
        protein_coding_tss = protein_coding_tss.reset_index(drop=True)
        # print(protein_coding_tss)
        n_tss = len(protein_coding_tss)
        print("\n Number of protein coding genes for this chromosome : ", n_tss)
    
        bin_start = seq_dataframe['start'].values
        bin_end = seq_dataframe['end'].values
        n_bin = len(bin_start)
        print("\n Sequence file : ", filename_seqs, " number of bins in the sequence file : ", n_bin)

        ## store the bin start coordinates
        np.save(OutDir + '/bin_start_' + currchr, bin_start)
        
        ### write tss
        tss = np.zeros(n_bin)
        gene_name = []
        for i in range(n_tss):
            idx_tss = seq_dataframe.loc[(seq_dataframe['start'] <= protein_coding_tss['tss_1'][i]) & (seq_dataframe['end'] > protein_coding_tss['tss_1'][i])].index
            if len(idx_tss)==0:
                print(protein_coding_tss['tss_1'][i])
            tss[idx_tss] = tss[idx_tss] + 1
        
        print('number of all tss bins:', np.sum(tss>0))
        print('number of bins with 1 tss:', np.sum(tss==1))
        print('number of bins with 2 tss:', np.sum(tss==2))
        print('number of bins with 3 tss:', np.sum(tss==3))
        print('number of bins with 4 tss:', np.sum(tss==4))
        print('number of bins with 5 tss:', np.sum(tss==5))

        n_tss_bins_all = n_tss_bins_all + np.sum(tss>0)
        n_tss_bins_all_1 = n_tss_bins_all_1 + np.sum(tss==1)
        n_tss_bins_all_2 = n_tss_bins_all_2 + np.sum(tss==2)
        n_tss_bins_all_3 = n_tss_bins_all_3 + np.sum(tss==3)
        n_tss_bins_all_4 = n_tss_bins_all_4 + np.sum(tss==4)
        n_tss_bins_all_5 = n_tss_bins_all_5 + np.sum(tss==5)
        n_tss_bins_all_6 = n_tss_bins_all_6 + np.sum(tss==6)
        
        print('number of tss: ', np.sum(tss).astype(np.int64))
        np.save(OutDir + '/tss_bins_' + currchr, tss)
        
        ### find gene names and their tss positions in the bins 
        pos_tss = np.zeros(n_bin).astype(np.int64)
        gene_name = np.array([""]*n_bin, dtype='|U16')
        for i in range(n_bin):
            if tss[i] >= 1:     # if want to choose only bins with one tss: tss[i] == 1
               pos_tss1 = protein_coding_tss.loc[(seq_dataframe['start'][i] <= protein_coding_tss['tss_1'] - thr) & (seq_dataframe['end'][i] > protein_coding_tss['tss_1'] + thr)]['tss_1'].values
               if len(pos_tss1)>0:
                  pos_tss1 = pos_tss1[0]
                  gene_name1 = protein_coding_tss.loc[(seq_dataframe['start'][i] <= protein_coding_tss['tss_1'] - thr) & (seq_dataframe['end'][i] > protein_coding_tss['tss_1'] + thr)]['gene'].values
                  gene_names = gene_name1[0]
                  for j in range(1,len(gene_name1)):
                      gene_names = gene_names + '+' + gene_name1[j]
                  print('gene_names: ', gene_names)
        
                  pos_tss[i] = pos_tss1
                  gene_name[i] = gene_names

        if 0:        
            print(len(pos_tss), pos_tss[0:800])
            print(len(gene_name), gene_name[:800])
        
        np.save(OutDir + '/tss_pos_' + currchr, pos_tss)
        np.save(OutDir + '/tss_gene_' + currchr, gene_name)

    ## debug
    if 0:
        print('number of all tss bins:', n_tss_bins_all)
        print('number of bins with 1 tss:', n_tss_bins_all_1)
        print('number of bins with 2 tss:', n_tss_bins_all_2)
        print('number of bins with 3 tss:', n_tss_bins_all_3)
        print('number of bins with 4 tss:', n_tss_bins_all_4)
        print('number of bins with 5 tss:', n_tss_bins_all_5)
        print('number of bins with 6 tss:', n_tss_bins_all_6)

        protein_coding_tss_all = tss_dataframe.loc[tss_dataframe['type'] == 'protein_coding']
        print('number of all tss: ', len(protein_coding_tss_all))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()


