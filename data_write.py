#!/usr/bin/env python

# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

# This code is copied from https://github.com/calico/basenji and modified.


## Writing the epigenomic and HiC tracks into tensorflow compatible TFRecord format
## Writes 6 Mb chunks at a time, and then does sliding window on 2 Mb.
## For HiC and CAGE tracks, it writes 1200 entries (at 5 Kb resolution).
## For other epigenomic tracks, it writes 60000 entries (at 100 bp resolution).
## For TSS bins (5 Kb bin size), 1200 entries, including the TSS and gene name information.
## All of these data is encapsulated in TFRecord format.

# =========================================================================

from optparse import OptionParser
import collections
import os
import sys
import h5py
import numpy as np
import pdb
import pysam
import pandas as pd
import scipy.sparse
import re
import tensorflow as tf
from dna_io import dna_1hot
# import configparser
import yaml

#from basenji_data import ModelSeq
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])


"""
Write TF Records for batches of model sequences. (TFRecord implemented in TensorFlow)

(1) epigenomic coverage files (saved in h5 format), 
(2) sparse adjacency matrices (saved in npz format), and 
(3) TSS files (saved in np format) and save them sequentially in TFRecords in each chromosome. 

Write epigenomic and CAGE coverages, adjacency matrices, and TSS annotations for the regions of 6Mb. 
Then we sweep the entire chromosome by steps of 2Mb. 
This way, there is no overlap for the middle 2Mb regions where we predict gene expression values. 

For each batch of 6Mb, the dimensions of data would be: 
60,000 for each epigenomic track (since 100 bp resolution was employed), 
1200 for CAGE (since 5 Kb resolution was employed), 
and 1200 x 1200 (5 Kb X 5 Kb) for adjacency matrices. 

The predicted CAGE values in the middle 400 bins would appear in the loss function 
so that all the genes could see their distal enhancers up to 2Mb up- and downstream of their TSS.

*** 
The TFRecord files are slightly different for models Epi and Seq models: 
(1) TFRecords for Seq also contain one-hot-coded DNA sequences of the size 6,000,000 x 4, as the DNA sequence is an input for these models, 
(2) The epigenomic signals for Epi undergo an extra log-normalization, via function log2(x+1), to reduce their dynamic ranges, as they are inputs in Epi models.
"""

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def check_symmetric(a, tol=1e-4):
    return np.all(np.abs(a-a.transpose()) < tol)

## global parameters
## max HiC / HiChIP contact counts (trimming)
MAX_HiC_HiChIP_CC = 1000

##======================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
    parser.add_option('-C', dest='configfile', default=None, type='str', help='Configuration file. Mandatory parameter.'),  
    parser.add_option('-M', dest='Model', default="epi", type='str', help='Prediction model. Either seq or epi. Default = epi'),

    ## a few default options - currently we do not edit them
    # parser.add_option('-s', dest='start_i', default=0, type='int', help='Sequence start index [Default: %default]'),
    # parser.add_option('-e', dest='end_i', default=None, type='int', help='Sequence end index [Default: %default]'),
    parser.add_option('--te', dest='target_extend', default=None, type='int', help='Extend targets vector [Default: %default]'),
    parser.add_option('--ts', dest='target_start', default=0, type='int', help='Write targets into vector starting at index [Default: %default'),
    parser.add_option('-u', dest='umap_npy', default=None, help='Unmappable array numpy file'),
    parser.add_option('--umap_set', dest='umap_set', default=None, type='float', help='Sequence distribution value to set unmappable positions to, eg 0.25.')

    (options, args) = parser.parse_args()
    return options, args


################################################################################
# main
################################################################################
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
    Resolution = int(config['Loop']['resolution'])
    fasta_file = config['General']['Fasta']
    SampleLabel = config['Loop']['SampleLabel']
    FDRThr = float(config['Loop']['FDRThr'])
    ## total span of a genomic region considered at a time
    ## default = 6 Mb
    Span = int(config['Model']['Span'])
    ## Sliding window - default = 2 Mb
    Offset = int(config['Model']['Offset'])

    CAGEBinSize = int(config['Epigenome']['CAGEBinSize'])
    EpiBinSize = int(config['Epigenome']['EpiBinSize'])
    ## track list using comma or colon as delimiter
    CAGETrackList = re.split(r':|,', config['Epigenome']['CAGETrack']) 
    ## track list using comma or colon as delimiter
    EpiTrackList = re.split(r':|,', config['Epigenome']['EpiTrack']) 
    ## label list using comma or colon as delimiter
    CAGELabelList = re.split(r':|,', config['Epigenome']['CAGELabel']) 
    ## label list using comma or colon as delimiter
    EpiLabelList = re.split(r':|,', config['Epigenome']['EpiLabel']) 

    ## other parameters
    model = options.Model

    ## read the chromosome size file
    chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])

    # open FASTA file for sequence model
    if model == 'seq':
        fasta_open = pysam.Fastafile(fasta_file)

    ## process individual CAGE files
    ## and create a TFRecord set
    for cagefileidx in range(len(CAGETrackList)):
        CurrCAGETrackFile = CAGETrackList[cagefileidx]
        CurrCAGETrackLabel = CAGELabelList[cagefileidx]
        print("\n\n *** Processing CAGE file : ", CurrCAGETrackFile, "  label : ", CurrCAGETrackLabel)

        ## output directory to store TFrecord
        ## specific to this CAGE sample
        TFRecordDir = BaseOutDir + "/" + SampleLabel + "/TFRecord/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel
        if not os.path.exists(TFRecordDir):
            os.makedirs(TFRecordDir)

        ## initialize seeds
        np.random.seed(0)
        q = np.zeros(3)

        for rowidx in range(chrsize_df.shape[0]):      
            currchr = str(chrsize_df.iloc[rowidx, 0])
            print("\n\n rowidx : ", rowidx, "  processing chromosome : ", currchr)

            ## discard any chromosomes other than chr1 to chr22
            if (currchr == "chrX" or currchr == "chrY" or currchr == "chrM" or "un" in currchr or "Un" in currchr or "random" in currchr or "Random" in currchr or "_" in currchr):
                continue

            ## sequence file for the current chromosome (generated from previous codes)
            seqs_bed_file = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution) + '/sequences_' + currchr + '.bed'
            print("\n seqs_bed_file : ", seqs_bed_file)
            if (os.path.exists(seqs_bed_file) == False):
                continue

            ## output TF record file
            tfr_file = TFRecordDir + "/" + currchr + '.tfr'
            print("\n output tfr_file : " + str(tfr_file))
            if (os.path.exists(tfr_file) == True):
                continue

            ################################################################
            # read model sequences
            model_seqs = []
            for line in open(seqs_bed_file):
                a = line.split()
                model_seqs.append(ModelSeq(a[0],int(a[1]),int(a[2]),None))

            start_i = 0
            end_i = len(model_seqs)            
            num_seqs = end_i - start_i    
            print("\n start_i : " + str(start_i) + " end_i : " + str(end_i) + " num_seqs : " + str(num_seqs))

            ################################################################
            ## determine sequence coverage files for the epigenomic track list
            ## and also the CAGE track 
            ## with respect to the current chromosome
            seqs_cov_files = []

            ## add the CAGE track
            pattern_to_replace = "_seq_cov_" + currchr + ".h5"
            currfile = BaseOutDir + "/Epigenome_Tracks/" + refgenome + "/" + CurrCAGETrackLabel + "/" + os.path.basename(CurrCAGETrackFile).replace(".bw", pattern_to_replace)
            seqs_cov_files.append(currfile)

            ## add the other epigenomic tracks
            for epitrackidx in range(len(EpiTrackList)):
                pattern_to_replace = "_seq_cov_" + currchr + ".h5"
                currfile = BaseOutDir + "/Epigenome_Tracks/" + refgenome + "/" + EpiLabelList[epitrackidx] + "/" + os.path.basename(EpiTrackList[epitrackidx]).replace(".bw", pattern_to_replace)
                seqs_cov_files.append(currfile)

            print("\n ==>>> seqs_cov_files : ", seqs_cov_files)
            seq_pool_len = h5py.File(seqs_cov_files[1], 'r')['seqs_cov'].shape[1]
            num_targets = len(seqs_cov_files)
            print("\n ==>>> Number of epigenomic coverage files / targets : " + str(len(seqs_cov_files)) + " seq_pool_len : " + str(seq_pool_len))

            ################################################################
            ## extend targets
            num_targets_tfr = num_targets
            if options.target_extend is not None:
                assert(options.target_extend >= num_targets_tfr)
                num_targets_tfr = options.target_extend

            ## initialize targets - a numpy array for all the epigenomic tracks
            targets = np.zeros((num_seqs, seq_pool_len, num_targets_tfr), dtype='float32')
            print("\n ** targets shape : " + str(targets.shape))

            # read each target (epigenomic track)
            for ti in range(num_targets):
                print("\n =>> reading file index (from seqs_cov_files) : " + str(ti))
                seqs_cov_open = h5py.File(seqs_cov_files[ti], 'r')
                tii = options.target_start + ti
                tmp = seqs_cov_open['seqs_cov'][start_i:end_i, :]
                print("\n =>> tmp shape " + str(tmp.shape))

                if (ti == 0):         
                    ## the first track (CAGE) is used as the testing data
                    ## store it in "targets_y" variable        
                    targets_y = tmp
                else:
                    ## this track is used as the training data      
                    if model == 'epi':
                        ## for epigenomic tracks and for "epi" model specification
                        ## log normalize the data
                        tmp = np.log2(tmp+1)
                    ## store it in the "targets" variable (training data)
                    ## note that starting index is 1 (ti = 1)
                    targets[:, :, ti] = tmp

                if 0:
                    print(ti, np.sort(tmp.ravel())[-200:])

                ## close the stream
                seqs_cov_open.close()
    
            ################################################################
            # modify unmappable
            if options.umap_npy is not None and options.umap_set is not None:
                unmap_mask = np.load(options.umap_npy)
                for si in range(num_seqs):
                    msi = start_i + si
                    # determine unmappable null value
                    seq_target_null = np.percentile(targets[si], q=[100*options.umap_set], axis=0)[0]
                    # set unmappable positions to null
                    targets[si,unmap_mask[msi,:],:] = np.minimum(targets[si,unmap_mask[msi,:],:], seq_target_null)

            ################################################################
            # write TFRecords

            ## Graph from HiChIP interactions
            ## load sparse matrix and convert to dense format
            hic_matrix_file = BaseOutDir + "/" + SampleLabel + "/FitHiChIP_to_mat/contact_mat_FDR_" + str(FDRThr) + "/" + str(currchr) + ".npz"            
            sparse_matrix = scipy.sparse.load_npz(hic_matrix_file)
            hic_matrix = sparse_matrix.todense()
            if 1:
                print('hic_matrix shape: ', hic_matrix.shape)

            ## TSS files (TSS positions)
            tss_bin_file = BaseOutDir + "/TSS/" + refgenome + "/" + str(Resolution) + "/tss_pos_" + currchr + ".npy"            
            tss_bin = np.load(tss_bin_file, allow_pickle=True)
            if 0:
                print('num tss:', np.sum(tss_bin))

            ## TSS bin start files
            bin_start_file = BaseOutDir + "/TSS/" + refgenome + "/" + str(Resolution) + "/bin_start_" + currchr + ".npy"
            bin_start = np.load(bin_start_file, allow_pickle=True)
            if 0:
                print('bin start:', bin_start)

            ## T: number of bins (having size = "Resolution")
            ## within the sliding window interval ("Offset")
            # T = 400
            T = Offset // Resolution
            ## TT: ( number of bins (having size = "Resolution") within the specified "Span" ) / 2
            # TT = T+T//2
            TT = (Span // Resolution) // 2
            ## number of bathes (iterations) considered
            n_batch = 0
            print("\n\n *** Initialization before loop - T (num bins within sliding window) : " + str(T) + "  TT (num bins within span) : " + str(TT) + " num_seqs (number of bins / intervals for this chromosome) : " + str(num_seqs) + " n_batch : " + str(n_batch) + " *** \n\n")

            ## write the epigenome data in TFRecords format
            # define options
            tf_opts = tf.io.TFRecordOptions(compression_type = 'ZLIB')

            with tf.io.TFRecordWriter(tfr_file, tf_opts) as writer:
                ## in each iteration, read the current span (default 6 Mb)
                ## and then advance by the sliding window (default 2 Mb)
                for si in range(TT,num_seqs-TT,T):
                    n_batch = n_batch + 1
                    ## dimension of hic_slice : (2 * TT) X (2 * TT)
                    hic_slice = hic_matrix[si-TT:si+TT,si-TT:si+TT]
                    print("\n ==>> processing n_batch : " + str(n_batch) + " si : " + str(si) + " HiC Slice range : " + str(si-TT) + " - " + str(si+TT))
                    
                    ## copy the HiC contacts  
                    adj_real = np.copy(hic_slice)
                    ## trim the HiC / HiChIP data above a certain contact count
                    adj_real[adj_real>=MAX_HiC_HiChIP_CC] = MAX_HiC_HiChIP_CC
                    ## log2 transform
                    adj_real = np.log2(adj_real+1)
                    ## make diagonal elements as 0
                    adj_real = adj_real * (np.ones([2*TT, 2*TT]) - np.eye(2*TT))
                    if 0:
                        print('real_adj: ', adj_real)

                    ## also store the adjacency information 
                    ## if any pair of bins have HiC/HiChIP contacts 
                    ## store 1 as a mark of adjacency - otherwise store 0
                    adj = np.copy(adj_real)
                    adj[adj>0] = 1

                    ## determine if the current iteration indicates the last batch
                    if np.abs(num_seqs-TT - si < T):
                        last_batch = 1
                    else:
                        last_batch = 0
                    print("\n ==>> num_seqs : ", str(num_seqs), " TT : ", str(TT), " si: ", str(si), " num_seqs-TT-si : " + str(num_seqs-TT-si) + " T : " + str(T) + " last_batch : " + str(last_batch))

                    ## obtain the TSS positions and bin information for the current slice / span
                    tss_idx = tss_bin[si-TT:si+TT]
                    bin_idx = bin_start[si-TT:si+TT]

                    ## output label - gene expression
                    Y = targets_y[si-TT:si+TT]

                    ## training data label 
                    ## note the last index - starts from 1
                    X_1d = targets[si-TT:si+TT,:,1:]

                    ## data type conversion
                    X_1d = X_1d.astype(np.float16)
                    adj = adj.astype(np.float16)
                    adj_real = adj_real.astype(np.float16)
                    Y = Y.astype(np.float16)
                    bin_idx = bin_idx.astype(np.int64)
                    tss_idx = tss_idx.astype(np.float16)

                    # read FASTA
                    if model == 'seq':
                        seq_1hot = np.zeros([1,4])
                        for msi in range(si-TT,si+TT):
                            mseq = model_seqs[msi]
                            seq_dna = fasta_open.fetch(mseq.chr, mseq.start, mseq.end)
                            # one hot code
                            seq_1hot = np.append(seq_1hot, dna_1hot(seq_dna), axis=0)

                        seq_1hot = np.delete(seq_1hot, 0, axis=0)
                        print('seq: ', np.shape(seq_1hot), seq_1hot)

                    if model == 'seq':
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'last_batch': _int_feature(last_batch),
                            'sequence': _bytes_feature(seq_1hot.flatten().tostring()),
                            'adj': _bytes_feature(adj.flatten().tostring()),
                            #'adj_real': _bytes_feature(adj_real.flatten().tostring()),
                            'X_1d': _bytes_feature(X_1d.flatten().tostring()),
                            'tss_idx': _bytes_feature(tss_idx.flatten().tostring()),
                            'bin_idx': _bytes_feature(bin_idx.flatten().tostring()),
                            'Y': _bytes_feature(Y.flatten().tostring())}))
                    elif model == 'epi':
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'last_batch': _int_feature(last_batch),
                            'adj': _bytes_feature(adj.flatten().tostring()),
                            #'adj_real': _bytes_feature(adj_real.flatten().tostring()),
                            'X_1d': _bytes_feature(X_1d.flatten().tostring()),
                            'tss_idx': _bytes_feature(tss_idx.flatten().tostring()),
                            'bin_idx': _bytes_feature(bin_idx.flatten().tostring()),
                            'Y': _bytes_feature(Y.flatten().tostring())}))

                    writer.write(example.SerializeToString())

                print('check symetric: ', check_symmetric(adj))
                print('number of batches: ', n_batch)
                print('q: ', q)

    ## close the fasta file, if opened
    if model == 'seq':
        fasta_open.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

