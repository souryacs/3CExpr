#!/usr/bin/env python

## partly adapted from 
## https://github.com/calico/basenji and modified.

# =========================================================================
## Write chunk level data in Pytorch data loader (1D epigenomes)

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

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
# =========================================================================

from optparse import OptionParser
import collections
import os
import h5py
import numpy as np
import pandas as pd
import torch
import pickle
# from sklearn.preprocessing import normalize
from pathlib import Path

## import the local utility file within current directory
from UtilFunc import *

#from basenji_data import ModelSeq
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

"""
Input files - batches of model sequences:
(1) epigenomic coverage files (saved in h5 format), 
(2) The epigenomic signals for EPI undergo an extra log-normalization, via function log2(x+1), to reduce their dynamic ranges, as they are inputs in Epi models.
(3) Chromatin contact map
"""

##============
## function to get the bigwig files
##============
def get_bigwig_files(inpdir):
	extensions = ['.bw', '.bigwig', '.BigWig', '.bigWig']
	return [str(file) for file in Path(inpdir).rglob("*") if file.suffix in extensions]

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
    
    parser.add_option('--Label', 
                      dest='Label', 
                      default=None, 
                      type='str', 
                      help='Label of the particular experiment. Default None')    

    parser.add_option('--Offset', 
                      dest='Offset', 
                      default=2000000, 
                      type='int', 
                      help='Sliding window offset. Default = 2000000 (2 Mb), as suggested in EpiGraphReg paper.'),

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
    
    (options, args) = parser.parse_args()
    return options, args


################################################################################
# main
################################################################################
def main():

    options, args = parse_options()

    Resolution = int(options.CAGEBinSize)
    CurrCAGETrackLabel = options.CAGETrackLabel
    currchr = options.chr
    BaseOutDir = options.BaseOutDir
    SeqDir = options.SeqDir
    EpiTrackBaseDir = options.EpiTrackBaseDir
    CAGETrackBaseDir = options.CAGETrackBaseDir    
    Offset = int(options.Offset)    ## Sliding window - default = 2 Mb
    TrackTable = options.TrackTable
            
    Span = (3 * Offset)     #int(options.Span)  ## default = 6 Mb - 3 times the sliding window
    model = "epi"           #options.Model    ## sequence or epigenomic model
    
    print("\n\n *** Processing CAGE track label (and writing records for this data) : ", CurrCAGETrackLabel)

    ## output directory to store Pytorch compatible data
    ## specific to this CAGE sample and also for the chromosome
    TorchDataDir = BaseOutDir + '/OUT_MODEL_EPI_' + str(options.EpiBinSize) + 'bp_CAGE_' + str(options.CAGEBinSize) + 'bp_1D_CNN/TrainingData/Offset_' + str(Offset) + "/" + str(options.Label) + "/" + CurrCAGETrackLabel + "/" + currchr
    os.makedirs(TorchDataDir, exist_ok = True)

    ##=================
    ## process input files and create the training data

    ## initialize seeds
    np.random.seed(0)

    ## sequence file for the current chromosome (generated from previous codes)
    seqs_bed_file = SeqDir + '/sequences_' + currchr + '.bed'
    print("\n seqs_bed_file : ", seqs_bed_file)

    ################################################################
    # read model sequences
    model_seqs = []
        
    ## adapts for header information
    seq_dataframe = pd.read_csv(seqs_bed_file, delimiter='\t')
    for i in range(seq_dataframe.shape[0]):
        model_seqs.append(ModelSeq(seq_dataframe.iloc[i,0], 
                                    int(seq_dataframe.iloc[i,1]), 
                                    int(seq_dataframe.iloc[i,2]), 
                                    None))

    start_i = 0
    end_i = len(model_seqs)            
    num_seqs = end_i - start_i   
    if 1: 
        print("start_i : ", str(start_i), " end_i : ", str(end_i), " num_seqs : ", str(num_seqs))

    ################################################################
    ## determine sequence coverage files for the epigenomic track list, and CAGE track, for current chromosome
    seqs_cov_files = []

    ## add the CAGE track
    currfile = CAGETrackBaseDir + "/" + CurrCAGETrackLabel + "/seq_cov_" + currchr + ".h5"
    print("==>>> Adding CAGE file to the list of tracks : ", currfile)
    seqs_cov_files.append(currfile)

    ## read the input track table file
    ## cell type, track type, and the folders containing these tracks
    TrackTableData = pd.read_csv(TrackTable, delimiter='\t')

    ## for this particular input CAGE track
    ## find the specific cell type, and the corresponding epigenomic tracks
    ## for the current CAGE track, the torch record file will only incorporate 
    ## information for the specific cell type
    for index, curr_row in TrackTableData.iterrows():
        if curr_row[1] == "CAGE":
            inpdir = curr_row[2]
            ## list of bigwig files
            bw_files = get_bigwig_files(inpdir)
            for f in bw_files:
                samplelabel = os.path.basename(f).split('.')[0]
                if CurrCAGETrackLabel == samplelabel:
                    currCAGETrack_CellType = curr_row[0]

    print("\n ==>>> current CAGE file : ", currfile, 
          "  CurrCAGETrackLabel : ", CurrCAGETrackLabel, 
          " Cell type with respect to the current CAGE track : ", currCAGETrack_CellType)

    ## add the other epigenomic tracks
    ## check the input track table file, to get the folder names and get the epigenomic track labels
    TrackTableData = pd.read_csv(TrackTable, delimiter='\t')
    for index, curr_row in TrackTableData.iterrows():
        ## skip the entries with CAGE and Loop
        ## consider non-CAGE tracks for the specific cell type
        if curr_row[1] != "CAGE" and curr_row[1] != "Loop" and curr_row[0] == currCAGETrack_CellType:
            ## folder containing the input epigenomic tracks - bigwig files
            inpdir = curr_row[2]
            if 1:
                print("===>> Epigenomic data - Input directory : ", inpdir)            
            ## list of bigwig files
            bw_files = get_bigwig_files(inpdir)
            for f in bw_files:                
                samplelabel = os.path.basename(f).split('.')[0]
                ## epigenomic coverage file for the current bigwig track and for the current chromosome
                currfile = EpiTrackBaseDir + "/" + samplelabel + "/seq_cov_" + currchr + ".h5"
                if os.path.exists(currfile):
                    print("==>> Adding Other epigenomic file to the list of tracks : ", currfile)
                seqs_cov_files.append(currfile)

    if 1:
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
        if 1:
            print("=>> reading file index (from seqs_cov_files) : " + str(ti))
        seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
        tmp = seqs_cov_open['seqs_cov'][start_i:end_i, :]
        if 1:
            print("** tmp shape : " + str(tmp.shape))

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

    ################################################################
    ## write Pytorch compatible data (to be loaded by the DataLoader)
    ################################################################
    
    ##=======================
    bin_start = seq_dataframe["start"].values
    
    ## T: number of bins (having size = "Resolution") within the sliding window interval ("Offset")
    # T = 400
    T = Offset // Resolution
    
    ## TT: ( number of bins (having size = "Resolution") within the specified "Span" ) / 2
    # TT = T+T//2
    TT = (Span // Resolution) // 2
    
    ## number of batches (iterations) considered
    n_batch = 0
    print("*** Initialization before loop - T (num bins within sliding window) : ", str(T), 
          "TT (num bins within span) : ", str(TT), 
          "num_seqs (number of bins / intervals for this chromosome) : ", str(num_seqs), 
          "n_batch : ", str(n_batch), " *** ")

    ##============== write the epigenome data (and optionally chromatin contact data) 
    ## in Pytorch compatible .pkl format
    ##============== one chunk at a time

    ##================
    ## in each iteration, read the current span (default 6 Mb)
    ## and then advance by the sliding window (default 2 Mb)
    ##================
    for si in range(TT, num_seqs-TT, T):
        
        n_batch = n_batch + 1        
                        
        ## determine if the current iteration indicates the last batch
        if np.abs(num_seqs-TT - si < T):
            last_batch = 1
        else:
            last_batch = 0
        print("==>> Writing Pytorch data file - epigenomic tracks - n_batch : ", str(n_batch), 
                "  num_seqs : ", str(num_seqs), 
                " TT : ", str(TT), 
                " si: ", str(si), 
                " num_seqs-TT-si : ", str(num_seqs-TT-si), 
                " T : ", str(T), 
                " last_batch : ", str(last_batch), 
                " Slice range : ", str(si-TT), " - ", str(si+TT))
        
        bin_idx = torch.tensor(bin_start[si-TT:si+TT], dtype=torch.int64)

        ##========== output label - gene expression
        Y = torch.tensor(targets_y[si-TT:si+TT], dtype=torch.float16)

        ##========== training data - epigenomic track 
        ## note the last index - starts from 1
        X_1d = torch.tensor(targets[si-TT:si+TT,:,1:], dtype=torch.float16)

        ## Write file - one chunk in one file
        currObj = TorchDataClass_Epigenome(X_1d, bin_idx, Y)

        # Write to a .pkl file
        targetoutfile = TorchDataDir + "/" + str(n_batch) + ".pkl"
        with open(targetoutfile, "wb") as f:
            pickle.dump(currObj, f)

        ## delete temporary values - per iteration
        del X_1d 
        del bin_idx
        del Y

    ## delete temporary structures - free memory
    del targets
    del targets_y
    del bin_start
    del seq_dataframe

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()


