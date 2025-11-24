#!/usr/bin/env python

## partly adapted from 
## https://github.com/calico/basenji and modified.

## modified 
## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037

# =========================================================================
# Write chunk level data in Pytorch data loader compatible format

## Example Resolutions:
## Default: Writes 6 Mb chunks at a time, and then does sliding window on 2 Mb.
## Middle 2 Mb regions are used to predict CAGE expression values. 
## Hi-C / HiChIP and CAGE tracks: 5 Kb - thus, 1200 entries for 6 Mb span
## Epigenomic tracks: 100 bp - thus, 60000 entries for 6 Mb span

## For each batch of 6Mb, the dimensions of data would be: 
## 60,000 for each epigenomic track (since 100 bp resolution was employed), 
## 1200 for CAGE (since 5 Kb resolution was employed), 
## and 1200 x 1200 (5 Kb X 5 Kb) for adjacency matrices. 

## The predicted CAGE values in the middle 400 bins would appear in the loss function 
## so that all the genes could see their distal enhancers up to 2Mb up- and downstream of their TSS.
# =========================================================================

from optparse import OptionParser
import collections
import os
import h5py
import numpy as np
import pandas as pd
# from dna_io import dna_1hot
import scipy.sparse
from scipy.sparse import csr_matrix

import torch
import pickle
from sklearn.preprocessing import normalize

import psutil

## import the local utility file within current directory
from UtilFunc import *

#from basenji_data import ModelSeq
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

"""
Input files - batches of model sequences:
(1) epigenomic coverage files (saved in h5 format), 
(2) The epigenomic signals for Epi undergo an extra log-normalization, via function log2(x+1), to reduce their dynamic ranges, as they are inputs in Epi models.
(3) Chromatin contact map
"""

##======================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-x', 
                      dest='CAGETrackLabel', 
                      default=None, 
                      type='str', 
                      help='CAGE track label. Mandatory parameter.')
    
    parser.add_option('-c', 
                      dest='chr', 
                      default=None, 
                      type='str', 
                      help='Current chromosome. Mandatory parameter.')
    
    parser.add_option('-O', 
                      dest='BaseOutDir', 
                      default=None, 
                      type='str', 
                      help='Base output directory. Mandatory parameter.')

    parser.add_option('--SeqDir', 
                      dest='SeqDir', 
                      default=None, 
                      type='str', 
                      help='Directory containing reference genome specific chromosome-wise bin files. Default None')

    parser.add_option('-L', 
                      dest='loopcelltype', 
                      default=None, 
                      type='str', 
                      help='Cell type corresponding to FitHiChIP contact information. Default None')

    parser.add_option('--EpiTrackBaseDir', 
                      dest='EpiTrackBaseDir', 
                      default=None, 
                      type='str', 
                      help='Directory containing Epigenomic track information. Default None')
    
    parser.add_option('--CAGETrackBaseDir', 
                      dest='CAGETrackBaseDir', 
                      default=None, 
                      type='str', 
                      help='Directory containing CAGE track information. Default None')

    parser.add_option('--Offset', 
                      dest='Offset', 
                      default=2000000, 
                      type='int', 
                      help='Sliding window offset. Default = 2000000 (2 Mb), as suggested in EpiGraphReg paper.'),

    parser.add_option('-n', 
                      dest='LoopLabel', 
                      default=None, 
                      type='str', 
                      help='Loop label. Mandatory parameter.')

    parser.add_option('--loopfile', 
                      dest='LoopFile', 
                      default=None, 
                      type='str', 
                      help='Loop file. Mandatory parameter.')

    parser.add_option('--TrackTable', 
                      dest='TrackTable', 
                      default=None, 
                      type='str', 
                      help='Table containing the CAGE and epigenomic track information. Mandatory parameter.')
    
    parser.add_option('-C', 
				   dest='CAGEBinSize', 
				   default=4096, 
				   type='int', 
				   help='CAGE bin size. Default 4096 (4 Kb)')
    
    parser.add_option('-E', 
				   dest='EpiBinSize', 
				   default=128, 
				   type='int', 
				   help='Epigenomic track bin size. Default 128 bp')

    parser.add_option('--EdgeNorm', 
				   dest='EdgeNorm', 
				   default=1, 
				   type='int', 
				   help='Edge feature normalization technique. 1 (default): scikit learn normalization, 2: doubly stochastic normalization')    
    
    (options, args) = parser.parse_args()
    return options, args


################################################################################
# main
################################################################################
def main():

    options, args = parse_options()

    CurrCAGETrackLabel = options.CAGETrackLabel
    currchr = options.chr
    BaseOutDir = options.BaseOutDir
    SeqDir = options.SeqDir
    EpiTrackBaseDir = options.EpiTrackBaseDir
    CAGETrackBaseDir = options.CAGETrackBaseDir    
    Offset = int(options.Offset)    ## Sliding window - default = 2 Mb
    TrackTable = options.TrackTable
    LoopFile = options.LoopFile
    CAGEBinSize = int(options.CAGEBinSize)
    EpiBinSize = int(options.EpiBinSize)

    Span = (3 * Offset)     #int(options.Span)  ## default = 6 Mb - 3 times the sliding window
    model = "epi"   #options.Model    ## sequence or epigenomic model
    
    print("\n\n *** Processing CAGE track label (and writing records for this data) : ", CurrCAGETrackLabel)
    print("\n\n *** Input loop file : ", LoopFile)
    
    ## output directory to store Pytorch compatible data
    ## specific to this CAGE sample, loop file, and also for the chromosome
    TorchDataDir = BaseOutDir + '/OUT_MODEL_EPI_' + str(EpiBinSize) + 'bp_CAGE_' + str(CAGEBinSize) + 'bp_3D_GAT/TrainingData/Offset_' + str(Offset) + '/EdgeNorm_' + str(options.EdgeNorm) + "/" + str(options.loopcelltype) + '/' + options.LoopLabel + "/" + CurrCAGETrackLabel + "/" + currchr      
    os.makedirs(TorchDataDir, exist_ok = True)
    
    ## initialize seeds
    np.random.seed(0)
    
    ## sequence file for the current chromosome (generated from previous codes)
    seqs_bed_file = SeqDir + '/sequences_' + currchr + '.bed'
    print("\n seqs_bed_file : ", seqs_bed_file)

    ################################################################
        
    ## adapts for header information
    seq_dataframe = pd.read_csv(seqs_bed_file, delimiter='\t')

    ## number of intervals for this chromosome    
    num_seqs = seq_dataframe.shape[0]    #len(seq_dataframe)
    print('==>> number of all nodes (bins) for the current chromosome: ', num_seqs)

    ## start coordinates of individual bins
    bin_start = seq_dataframe["start"].values.astype(np.int64)

    ## T: number of bins (having size = "Resolution") within the sliding window interval ("Offset")
    # T = 400
    T = Offset // options.CAGEBinSize
    
    ## TT: ( number of bins (having size = "Resolution") within the specified "Span" ) / 2
    # TT = T+T//2
    TT = (Span // options.CAGEBinSize) // 2

    ################################################################
    ## determine sequence coverage files for the epigenomic track list, and CAGE track, for current chromosome
    seqs_cov_files = []

    ## add the CAGE track
    currfile = CAGETrackBaseDir + "/" + CurrCAGETrackLabel + "/seq_cov_" + currchr + ".h5"
    if 1:
        print("==>>> Adding CAGE file to the list of tracks : ", currfile)
    seqs_cov_files.append(currfile)

    ## add the other epigenomic tracks
    ## check the input track table file, to get the folder names and get the epigenomic track labels
    TrackTableData = pd.read_csv(TrackTable, delimiter='\t')
    for index, curr_row in TrackTableData.iterrows():
        ## skip the entries with CAGE and Loop
        if curr_row.iloc[1] != "CAGE" and curr_row.iloc[1] != "Loop":
            ## folder containing the input epigenomic tracks - bigwig files
            inpdir = curr_row.iloc[2]
            if 0:
                print("===>> Epigenomic data - Input directory : ", inpdir)
            ## list of bigwig files
            bw_files = [f for f in os.listdir(inpdir) if f.endswith(('.bw', '.bigwig', '.BigWig', 'bigWig'))]
            for f in bw_files:                
                samplelabel = os.path.basename(f).split('.')[0]
                ## epigenomic coverage file for the current bigwig track and for the current chromosome
                currfile = EpiTrackBaseDir + "/" + samplelabel + "/seq_cov_" + currchr + ".h5"
                if 1:
                    print("==>> Adding Other epigenomic file to the list of tracks : ", currfile)
                seqs_cov_files.append(currfile)

    if 0:
        print("\n ==>>> seqs_cov_files : ", seqs_cov_files)
    seq_pool_len = h5py.File(seqs_cov_files[1], 'r')['seqs_cov'].shape[1]
    num_targets = len(seqs_cov_files)
    if 1:
        print("==>>> Number of epigenomic coverage files / targets : ", str(num_targets), 
            " seq_pool_len : ", str(seq_pool_len))

    ################################################################
    ## extend targets
    num_targets_tfr = num_targets

    ## initialize targets - a numpy array for all the epigenomic tracks
    targets = np.zeros((num_seqs, seq_pool_len, num_targets_tfr), dtype='float32')
    if 1:
        print("** targets shape : " + str(targets.shape))

    # read each target (epigenomic track)
    for ti in range(num_targets):
        if 0:
            print("=>> reading file index (from seqs_cov_files) : " + str(ti))
        seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
        tmp = seqs_cov_open['seqs_cov'][0:num_seqs, :]

        if (ti == 0):         
            ## the first track (CAGE) is used as the testing data
            ## store it in "targets_y" variable        
            ## Note: no log transform is performed
            targets_y = tmp
        else:
            ## this track is used as the training data      
            if model == 'epi':
                ## for epigenomic tracks and for "epi" model specification
                ## Note: bigwig epigenomic tracks are log normalized
                tmp = np.log2(tmp+1)
            ## store it in the "targets" variable (training data)
            ## note that starting index is 1 (ti = 1)
            targets[:, :, ti] = tmp

        ## close the stream
        seqs_cov_open.close()
        ## delete temporary variables
        del tmp

    ##================================
    ## extract chromatin interactions for the current chromosome
    ## first 6 fields: chr1, start1, end1, chr2, start2, end2
    ## additional features can be put after that, which would be used as the edge features
    ##================================
    temp_interaction_file = TorchDataDir + "/temp_interactions_" + currchr + ".txt"
    sys_cmd = "awk \'(($1 == \"" + currchr + "\") && ($2~/^[0-9]+$/) && ($5~/^[0-9]+$/))\' " + LoopFile + " > " + temp_interaction_file
    os.system(sys_cmd)

    df = pd.read_csv(temp_interaction_file, header = None, delimiter='\t')  # Adjust delimiter if needed
    ncol = df.shape[1]
    print("\n\n -->> number of interactions : ", str(df.shape[0]), 
          "  number of columns in interaction file : ", ncol, 
          "  number of edge features : ", (ncol - 6))

    ##=========================
    ## declare an array of CSR matrices
    ## each matrix would store one of the edge features    
    ##=========================
    csr_matrices = []

    ## indices
    rows = (df.iloc[:, 1].values / CAGEBinSize).astype(np.int64) 
    cols = (df.iloc[:, 3].values / CAGEBinSize).astype(np.int64)

    ## navigate through individual edge features
    ## and construct csr matrices + normalize
    ## features start from colmumn 6 (i.e. col_idx = 5)
    for col_idx in range(5, ncol):
        currdata = df.iloc[:, col_idx].values
        ## construct matrix
        curr_csr_mat = csr_matrix((currdata, (rows, cols)), shape=(num_seqs, num_seqs), dtype = np.float32)
        ## replace any NAN entries with 0
        curr_csr_mat.data[np.isnan(curr_csr_mat.data)] = 0
        ## normalize if EdgeNorm > 0
        if options.EdgeNorm == 1:
            ## scikit-learn normalize
            curr_csr_mat = normalize(curr_csr_mat, axis=0)
        elif options.EdgeNorm == 2:
            ## double stochastic normalize
            curr_csr_mat = sinkhorn_knopp(curr_csr_mat)
        ## replace any NAN entries with 0
        curr_csr_mat.data[np.isnan(curr_csr_mat.data)] = 0
        print("-->> col_idx : ", col_idx, 
              " After normalization - curr_csr_mat shape : ", str(curr_csr_mat.shape), 
              " number of nonzero indices : ", len(curr_csr_mat.nonzero()[0]))
        csr_matrices.append(curr_csr_mat)

    ## number of batches (iterations) considered
    n_batch = 0
    print("\n *** Initialization before loop - T (num bins within sliding window) : ", str(T), 
          "TT (num bins within span) : ", str(TT), 
          "num_seqs (number of bins / intervals for this chromosome) : ", str(num_seqs), 
          "n_batch : ", str(n_batch), " *** ")

    ##================
    ## in each iteration, read the current span (default 6 Mb)
    ## and then advance by the sliding window (default 2 Mb)
    ##================
    for si in range(TT, num_seqs-TT, T):
        
        n_batch = n_batch + 1        
                        
        print("\n ==>> Writing Pytorch data file - epigenomic tracks - n_batch : ", str(n_batch), 
                " si: ", str(si), 
                " Slice range : ", str(si-TT), " - ", str(si+TT), 
                " dimension : ", (2 * TT), " X ", (2 * TT))
        
        bin_idx = torch.from_numpy(bin_start[si-TT:si+TT]).to(dtype=torch.int64)
        if 0:
            print("==>> bin_idx shape : ", str(bin_idx.shape))

        ##========== output label - gene expression
        Y = torch.from_numpy(targets_y[si-TT:si+TT]).to(dtype=torch.float16)
        if 0:
            print("==>> Y shape : ", str(Y.shape))

        ##========== training data - epigenomic track 
        ## note the last index - starts from 1
        X_1d = torch.from_numpy(targets[si-TT:si+TT,:,1:]).to(dtype=torch.float16)
        if 0:
            print("==>> X_1d shape : ", str(X_1d.shape))

        ##========== insert the edge information (having nonzero contact)
        currmat = csr_matrices[0][si-TT:si+TT, si-TT:si+TT]
        a, b = currmat.nonzero()
        if 1:
            print("==>> length of nonzero indices : ", len(a))
            if (len(a) > 0):
                print("  min a : ", min(a), " max a : ", max(a), 
                      "  min b : ", min(b), " max b : ", max(b))

        ## edge indices
        a_t = torch.from_numpy(a).to(dtype=torch.int64)
        b_t = torch.from_numpy(b).to(dtype=torch.int64)
        edge_idx = torch.stack([a_t, b_t],dim=0).to(dtype=torch.int64)
        if 1:
            print("==>> edge_idx shape : ", str(edge_idx.shape))
        
        ## edge features
        edge_feat_list = []
        for feat_idx in range(len(csr_matrices)):
            if 1:
                print("==>> processing feature index : ", str(feat_idx))
            currmat = csr_matrices[feat_idx][si-TT:si+TT, si-TT:si+TT]
            currmat_t = torch.from_numpy(np.asmatrix(currmat.toarray(), dtype=np.float16)).to(dtype=torch.float16)
            if 1:
                print("==>> currmat_t shape : ", str(currmat_t.shape), 
                      "  currmat_t[a, b] shape : ", str(currmat_t[a, b].shape))
            ## make sure to have at least 2 dimensions for indivdual entries
            ## that's why torch.unsqueeze operation is done
            edge_feat_list.append(torch.unsqueeze(currmat_t[a, b], 1))  
        if 1:
            print("==>> length edge_feat_list : ", len(edge_feat_list))
        edge_feat = torch.cat(edge_feat_list, dim=1)
        if 1:
            print("==>> edge_feat shape (final) : ", str(edge_feat.shape))
        
        ## Write file - one chunk in one file
        currObj = TorchDataClass_Epigenome_Contact(X_1d, bin_idx, Y, edge_idx, edge_feat)

        # Write to a .pkl file
        targetoutfile = TorchDataDir + "/" + str(n_batch) + ".pkl"
        with open(targetoutfile, "wb") as f:
            pickle.dump(currObj, f)

        ## delete temporary values - per iteration
        del X_1d 
        del bin_idx
        del Y
        del currmat
        del currmat_t
        del edge_idx
        del edge_feat
        del edge_feat_list

    ## delete temporary structures - free memory
    del targets
    del targets_y
    del bin_start
    del seq_dataframe
    del csr_matrices
    del curr_csr_mat

    if os.path.exists(temp_interaction_file):
        sys_cmd = "rm " + temp_interaction_file
        os.system(sys_cmd)
    

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

