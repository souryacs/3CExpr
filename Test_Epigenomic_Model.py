##===================
## adapted from Epi-GraphReg_test_multiple_runs.py script of the GraphReg package
## https://github.com/karbalayghareh/GraphReg

## modified 
## Sourya Bhattacharyya
##===================

from __future__ import division
from optparse import OptionParser
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from gat_layer import GraphAttention

import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr
from scipy.stats import wilcoxon
from scipy.stats import ranksums

import pyBigWig

# from adjustText import adjust_text
import matplotlib.patches as mpatches
# import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
# from statannot import add_stat_annotation

## import the local utility file within current directory
from UtilFunc_Epigenomic_Model import *

import re

## debug variable 
debug_text = True

## number of models
## model 1: validation chr: 1, 11 test chr: 2, 12
## model 2: validation chr: 2, 12 test chr: 3, 13
## model 3: validation chr: 3, 13 test chr: 4, 14
## ....
## model 9: validation chr: 9, 19 test chr: 10, 20
## model 10: validation chr: 10, 20 test chr: 11, 21
# NUMMODEL = 10

## use sequence specific NN 
## currently we do not use any sequence specific information
USE_SEQNN_MODEL = False

##=========
## correlation of true and predicted CAGE expression (log2 scale)
##=========
def Plot_Corr_log2(plotfile, true_cage, pred_cage, n_contacts, plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(np.log2(true_cage+1),np.log2(pred_cage+1), c=np.log2(n_contacts+1), s=100, cmap=cm, alpha=.7)  #, edgecolors='')
    # plt.xlim((-.5,15))
    # plt.ylim((-.5,15))
    plt.title(plotlabel, fontsize=20)
    plt.xlabel("log2 (true + 1)", fontsize=20)
    plt.ylabel("log2 (pred + 1)", fontsize=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.grid(alpha=.5)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0,14.5, 'R= ' + "{:5.3f}".format(rho_gat) + '\nSP= ' + "{:5.3f}".format(sp_gat) + '\nNLL= '+str(np.float16(NLL_gat)), horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
    cbar = plt.colorbar(sc)
    cbar.set_label(label='log2 (n + 1)', size=20)
    cbar.ax.tick_params(labelsize=15)
    #plt.show()
    plt.tight_layout()
    plt.savefig(plotfile)

def Plot_Corr(plotfile, true_cage, pred_cage, n_contacts, plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(true_cage, pred_cage, c=n_contacts, s=100, cmap=cm, alpha=.7)  #, edgecolors='')
    # plt.xlim((-.5,15))
    # plt.ylim((-.5,15))
    plt.title(plotlabel, fontsize=20)
    plt.xlabel("true", fontsize=20)
    plt.ylabel("pred", fontsize=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)
    plt.grid(alpha=.5)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0,14.5, 'R= ' + "{:5.3f}".format(rho_gat) + '\nSP= ' + "{:5.3f}".format(sp_gat) + '\nNLL= '+str(np.float16(NLL_gat)), horizontalalignment='left', verticalalignment='top', bbox=props, fontsize=20)
    cbar = plt.colorbar(sc)
    cbar.set_label(label='n', size=20)
    cbar.ax.tick_params(labelsize=15)
    #plt.show()
    plt.tight_layout()
    plt.savefig(plotfile)


##===================
## parse options
##===================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-M', dest='Model', default="epi", type='str', help='Prediction model. Either seq or epi. Default = epi')
    
    parser.add_option('--Span', dest='Span', default=6000000, type='int', help='Span (complete window). Default = 6000000 (6 Mb), as suggested in the paper.'),
    parser.add_option('--Offset', dest='Offset', default=2000000, type='int', help='Offset (middle portion / sliding window). Default = 2000000 (2 Mb), as suggested in the paper.'),

    parser.add_option('-g', dest='refgenome', default=None, type='str', help='Reference Genome. Mandatory parameter.')
    parser.add_option('-O', dest='BaseOutDir', default=None, type='str', help='Base output directory. Mandatory parameter.')
    parser.add_option('-c', dest='chrsizefile', default=None, type='str', help='Reference Genome specific chromosome size file. Mandatory parameter.')
    parser.add_option('-r', dest='Resolution', default=5000, type='int', help='Loop resolution. Default 5000 (5 Kb)')
    parser.add_option('-n', dest='SampleLabel', default=None, type='str', help='Sample label. Mandatory parameter.')
    parser.add_option('-f', dest='FDRThr', default=0.01, type='float', help='FDR threshold of Loops. Default 0.01')
    parser.add_option('-C', dest='CAGEBinSize', default=5000, type='int', help='CAGE bin size. Default 5000 (5 Kb)')
    parser.add_option('-E', dest='EpiBinSize', default=5000, type='int', help='Epigenomic track bin size. Default 100 bp')
    parser.add_option('-X', dest='CAGETrackList', default=None, type='str', help='Comma or colon separated CAGE track list. Mandatory parameter.')
    parser.add_option('-Y', dest='EpiTrackList', default=None, type='str', help='Comma or colon separated Epigenome track list. Mandatory parameter.')
    parser.add_option('-x', dest='CAGELabelList', default=None, type='str', help='Comma or colon separated CAGE track label list. Mandatory parameter.')
    parser.add_option('-y', dest='EpiLabelList', default=None, type='str', help='Comma or colon separated Epigenome track label list. Mandatory parameter.')

    parser.add_option('-v', dest='valid_chr', default='1,11', type='str', help="Comma separated list (numbers) of the validation chromosomes. Default 1,11 means that chr1 and chr11 would be used as the validation chromosomes.")
    parser.add_option('-t', dest='test_chr', default='2,12', type='str', help="Comma separated list (numbers) of the test chromosomes. Default 2,12 means that chr2 and chr12 would be used as the test chromosomes.")

    (options, args) = parser.parse_args()
    return options, args

##===================
## main code
##===================
def main():
    options, args = parse_options()

    ##=============
    ## configuration file parameters
    ##=============

    refgenome = options.refgenome
    BaseOutDir = options.BaseOutDir
    Resolution = int(options.Resolution)
    SampleLabel = options.SampleLabel 
    chrsizefile = options.chrsizefile
    FDRThr = float(options.FDRThr)

    CAGEBinSize = int(options.CAGEBinSize)
    EpiBinSize = int(options.EpiBinSize)

    ## track list using comma or colon as delimiter
    CAGETrackList = re.split(r':|,', options.CAGETrackList) 
    ## track list using comma or colon as delimiter
    EpiTrackList = re.split(r':|,', options.EpiTrackList) 

    ## label list using comma or colon as delimiter
    CAGELabelList = re.split(r':|,', options.CAGELabelList) 
    ## label list using comma or colon as delimiter
    EpiLabelList = re.split(r':|,', options.EpiLabelList) 
    ## default = 6 Mb
    Span = int(options.Span)
    ## Sliding window - default = 2 Mb
    Offset = int(options.Offset)

    ##=================
    ## derived parameters
    ##=================
    # T = Offset // Resolution    #400 
    num_slide_window_loop_bin = Offset // Resolution

    num_slide_window_epi_bin = Offset // EpiBinSize

    # N = Span // Resolution    #3*T
    num_span_loop_bin = Span // Resolution

    num_span_epi_bin = Span // EpiBinSize

    # b = Resolution // EpiBinSize   ## 50
    ratio_loopbin_epibin = Resolution // EpiBinSize

    # feature dimension - number of Epigenomic tracks used in model
    # F = len(EpiTrackList)   ## 6   #3
    num_epitrack = len(EpiTrackList)

    ##=============
    ## other parameters
    ##=============
    batch_size = 1
    
    # write the predicted CAGE to bigwig files
    write_bw = False

    prediction = True
    
    load_np = False     #True
    
    # logfold = False
    # plot_violin = False
    # plot_box = False
    # plot_scatter = False
    # save_R_NLL_to_csv = True

    ##=================
    ## read the chromosome size file
    ## and exclude unnecessary chromosomes
    ##=================
    chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])
    print("\n\n ** Original chromosome size file - number of rows : ", chrsize_df.shape[0])
    idx = [i for i in range(chrsize_df.shape[0]) if chrsize_df.iloc[i, 0] != "chrX" and chrsize_df.iloc[i, 0] != "chrY" and chrsize_df.iloc[i, 0] != "chrM" and "un" not in chrsize_df.iloc[i, 0] and "random" not in chrsize_df.iloc[i, 0] and "Random" not in chrsize_df.iloc[i, 0] and "_" not in chrsize_df.iloc[i, 0]]
    chrsize_df = chrsize_df.iloc[idx, :]
    print("\n\n ** Filtered chromosome size file - number of rows : ", chrsize_df.shape[0])

    ##==========
    ## function
    ## reads one window at a time
    ## outputs:
    ## X_epi: epigenome track
    ## X_avg: average of epigenome track according to loop resolution
    ## Y: output gene expression - CAGE - according to loop resolution
    ## adj: adjacency matrix - loop information
    ## idx: middle region (sliding window)
    ## tss_idx: indices corresponding to TSS (gene) information
    ## pos: current window (6 Mb), in epigenome track width (EpiBinSize, usually 100 bp) spaced interval
    ##==========
    def read_tf_record_1shot(iterator, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack, CAGEBinSize, EpiBinSize):

        if debug_text == True:
            print("\n\n\n ******** Within function read_tf_record_1shot ********** \n\n\n")

        try:
            ## compatible with tensorflow 1
            if 0:
                next_datum = iterator.get_next()
            ## compatible with tensorflow 2
            if 1:
                next_datum = next(iterator)
            data_exist = True
        
        # sourya - modify the error handler
        # except tf.errors.OutOfRangeError:
        except Exception as e:
            print('\n\n !!!!! Data read TF record - error occured ')
            data_exist = False

        if data_exist:

            ## now we have provided these values as parameters
            # T = 400 #num_slide_window_loop_bin       # number of 5kb bins inside middle 2Mb region 
            # b = 50  #ratio_loopbin_epibin       # number of 100bp bins inside 5Kb region
            # F = 6  #num_epitrack #3      # number of Epigenomic tracks used in model

            X_epi = next_datum['X_epi']
            if debug_text == True:
                print("\n X_epi - shape : ", X_epi.shape)

            batch_size = tf.shape(X_epi)[0]
            if debug_text == True:
                print("\n batch_size : ", batch_size)
            
            ## content of epigenomic tracks
            # X_epi = tf.reshape(X_epi, [batch_size, 3*T*b, F])
            X_epi = tf.reshape(X_epi, [batch_size, num_span_epi_bin, num_epitrack])
            if debug_text == True:
                print("\n after reshape - X_epi - shape : ", X_epi.shape)
            
            ## average of epigenomic tracks, considering the loop resolution
            ## reshape() operation is used
            # X_avg = tf.reshape(X_epi, [3*T, b, F])
            X_avg = tf.reshape(X_epi, [num_span_loop_bin, ratio_loopbin_epibin, num_epitrack])
            if debug_text == True:
                print("\n X_avg - shape : ", X_avg.shape)

            X_avg = tf.reduce_mean(X_avg, axis=1)
            if debug_text == True:
                print("\n after mean - X_avg - shape : ", X_avg.shape)
            
            ## adjacency matrix - loop resolution
            adj = next_datum['adj']
            # adj = tf.reshape(adj, [batch_size, 3*T, 3*T])
            adj = tf.reshape(adj, [batch_size, num_span_loop_bin, num_span_loop_bin])
            if debug_text == True:
                print("\n adj - shape : ", adj.shape)
            
            tss_idx = next_datum['tss_idx']
            # tss_idx = tf.reshape(tss_idx, [3*T])
            tss_idx = tf.reshape(tss_idx, [num_span_loop_bin])
            if debug_text == True:
                print("\n tss_idx - shape : ", tss_idx.shape)
            
            bin_idx = next_datum['bin_idx']
            # bin_idx = tf.reshape(bin_idx, [3*T])
            bin_idx = tf.reshape(bin_idx, [num_span_loop_bin])
            if debug_text == True:
                print("\n bin_idx - shape : ", bin_idx.shape)

            ## idx: middle region (sliding window)
            idx = tf.range(num_slide_window_loop_bin, 2*num_slide_window_loop_bin)
            
            ##======
            Y = next_datum['Y']
            if debug_text == True:
                print("\n Y - shape : ", Y.shape)
            
            if 0:
                # Y = tf.reshape(Y, [batch_size, 3*T, b])
                Y = tf.reshape(Y, [batch_size, num_span_loop_bin, ratio_loopbin_epibin])
                Y = tf.reduce_sum(Y, axis=2)

            # Y = tf.reshape(Y, [batch_size, 3*T])
            Y = tf.reshape(Y, [batch_size, num_span_loop_bin])
            if debug_text == True:
                print("\n after reshape -- Y - shape : ", Y.shape)

            ## current window (6 Mb), in epigenome track width (EpiBinSize, usually 100 bp) spaced interval
            start = bin_idx[0].numpy()
            end = bin_idx[-1].numpy()
            pos = np.arange(start, end, EpiBinSize).astype(int)
            if debug_text == True:
                print("\n start : ", start, '  end : ', end, '  pos[1:10] : ', pos[1:10])

        else:
            X_epi = 0
            X_avg = 0
            Y = 0
            adj = 0
            tss_idx = 0
            idx = 0
            pos = 0
        
        if debug_text == True:
            print("\n\n\n ******** Going out of the function read_tf_record_1shot ********** \n\n\n")        
        
        return data_exist, X_epi, X_avg, Y, adj, idx, tss_idx, pos

    ##==========
    ## loss function
    ## added with parameters mentioning the loop and epigenome track resolutions
    ## outputs:
    ## 
    ##==========

    def calculate_loss(TSSDir, TFRecordDir, TestOutDir, model_gat, chrDF, valid_chr, test_chr, batch_size, write_bw, num_slide_window_epi_bin, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack, CAGEBinSize, EpiBinSize):
        
        if debug_text == True:
            print("\n\n\n ******** Within function calculate_loss ********** \n\n\n")

        loss_gat_all = np.array([])        
        Y_hat_gat_all = np.array([])

        Y_all = np.array([])
        y_gene = np.array([])
        y_hat_gene_gat = np.array([])

        if USE_SEQNN_MODEL == True:
            loss_cnn_all = np.array([])
            Y_hat_cnn_all = np.array([])
            y_hat_gene_cnn = np.array([])

        gene_names = np.array([])
        gene_tss = np.array([])
        gene_chr = np.array([])
        n_contacts = np.array([])
        n_tss_in_bin = np.array([])

        y_bw = np.array([])
        y_pred_gat_bw = np.array([])
        
        if USE_SEQNN_MODEL == True:
            y_pred_cnn_bw = np.array([])
        
        chroms = np.array([])
        starts = np.array([])
        ends = np.array([])
               
        test_chr_str = [str(i) for i in test_chr]
        test_chr_str = ','.join(test_chr_str)
        valid_chr_str = [str(i) for i in valid_chr]
        valid_chr_str = ','.join(valid_chr_str)

        if write_bw == True:
            ## output bigwig file - Y - CAGE gene expression (true)
            bw_y_true_filename = TestOutDir + "/CAGE_True.bw"
            bw_y_true = pyBigWig.open(bw_y_true_filename, "w")
            bw_y_true.addHeader(chrDF)
            if debug_text == True:
                print("\n write_bw option is True - bw_y_true_filename : ", bw_y_true_filename)

            ## output bigwig file - Y - CAGE gene expression (predicted from GAT)
            bw_y_pred_gat_filename = TestOutDir + "/Epi-GraphReg_CAGE_pred.bw"
            bw_y_pred_gat = pyBigWig.open(bw_y_pred_gat_filename, "w")
            bw_y_pred_gat.addHeader(chrDF)
            if debug_text == True:
                print("\n write_bw option is True - bw_y_pred_gat_filename : ", bw_y_pred_gat_filename)

        ## process individual chromosomes
        chr_list = test_chr.copy()
        for i in chr_list:        
            if debug_text == True:
                print('\n\n ==>> Loop - processing chromosome : ', i)
            
            TFRecordFileName = TFRecordDir + '/chr' + str(i) + '.tfr'
            if debug_text == True:
                print('\n  TFRecord file : ', TFRecordFileName)
            
            iterator = dataset_iterator(TFRecordFileName, batch_size)

            ## load TSS position
            tss_pos_filename = TSSDir + '/tss_pos_chr' + str(i) + '.npy'
            tss_pos = np.load(tss_pos_filename, allow_pickle=True)            

            ## load TSS and gene information
            TSS_gene_names_filename = TSSDir + '/tss_gene_chr' + str(i) + '.npy'
            gene_names_all = np.load(TSS_gene_names_filename, allow_pickle=True)
            
            ## load number of TSS information
            n_tss_filename = TSSDir + '/tss_bins_chr' + str(i) + '.npy'
            n_tss = np.load(n_tss_filename, allow_pickle=True)

            ## filter these structures to keep only TSS information
            ## i.e. filter the zeros (or 0 related entries)
            if debug_text == True:
                print("\n ==>> before filtering tss_pos - number of entries : ", len(tss_pos))
            tss_pos = tss_pos[tss_pos > 0]
            if debug_text == True:
                print("\n ==>> after filtering tss_pos - number of entries : ", len(tss_pos))
            if debug_text == True:
                print('tss_pos - first 10 entries: ', tss_pos[0:10])

            if debug_text == True:
                print("\n ==>> before filtering gene_names_all - number of entries : ", len(gene_names_all))
            gene_names_all = gene_names_all[gene_names_all != ""]
            if debug_text == True:
                print("\n ==>> after filtering gene_names_all - number of entries : ", len(gene_names_all))
            if debug_text == True:
                print('gene_names_all - first 10 entries: ', gene_names_all[0:10])
            
            if debug_text == True:
                print("\n ==>> before filtering n_tss - number of entries : ", len(n_tss))
            n_tss = n_tss[n_tss >= 1]
            if debug_text == True:
                print("\n ==>> after filtering n_tss - number of entries : ", len(n_tss))
            if debug_text == True:
                print('n_tss - first 10 entries: ', n_tss[0:10])

            pos_bw = np.array([])
            y_bw_ = np.array([])
            y_pred_gat_bw_ = np.array([])
            y_pred_cnn_bw_ = np.array([])
            
            while True:
                
                ## read one window (record)
                data_exist, X_epi, X_avg, Y, adj, idx, tss_idx, pos = read_tf_record_1shot(iterator, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack, CAGEBinSize, EpiBinSize)

                if data_exist:

                    ### Creating BigWig files for true and predicted CAGE tracks ###
                    if write_bw == True:
                        
                        ## middle portion (sliding window)
                        # pos_mid = pos[20000 : 40000]
                        pos_mid = pos[num_slide_window_epi_bin : (2 * num_slide_window_epi_bin)]
                        if debug_text == True:
                            print("\n ==>>> Within condition write_bw True")
                            print("\n len pos_mid : ", len(pos_mid))

                        if (pos_mid[-1] < 10**15):
                            pos_bw = np.append(pos_bw, pos_mid)
                            
                            ## we do not use the sequence specific prediction
                            if USE_SEQNN_MODEL == True:
                                Y_hat_cnn = model_cnn(X_epi)

                            ## apply the current "X_epi" and "adj" to the pre-trained "model_gat"
                            Y_hat_gat, att = model_gat([X_epi, adj])
                            if debug_text == True:
                                print("\n ==>> Y_hat_gat shape : ", Y_hat_gat.shape)
                            
                            ##=========== indices of original CAGE track
                            # Y_idx = tf.gather(Y, tf.range(T, 2*T), axis=1)
                            Y_idx = tf.gather(Y, tf.range(num_slide_window_loop_bin, 2*num_slide_window_loop_bin), axis=1)
                            if debug_text == True:
                                print("\n ==>> Y_idx len : ", len(Y_idx))
                                print("\n Y_idx[1:10] : ", Y_idx[1:10])
                            y1 = np.repeat(Y_idx.numpy().ravel(), ratio_loopbin_epibin)
                            if debug_text == True:
                                print("\n ==>> y1 len : ", len(y1))
                            y_bw_ = np.append(y_bw_, y1)
                            if debug_text == True:
                                print("\n ==>> y_bw_ len : ", len(y_bw_))
                            
                            ## we do not use the sequence specific prediction
                            if USE_SEQNN_MODEL == True:
                                Y_hat_cnn_idx = tf.gather(Y_hat_cnn, tf.range(T, 2*T), axis=1)

                            ##=========== indices of predicted CAGE track
                            # Y_hat_gat_idx = tf.gather(Y_hat_gat, tf.range(T, 2*T), axis=1)
                            Y_hat_gat_idx = tf.gather(Y_hat_gat, tf.range(num_slide_window_loop_bin, 2*num_slide_window_loop_bin), axis=1)
                            if debug_text == True:
                                print("\n ==>> Y_hat_gat_idx len : ", len(Y_hat_gat_idx))
                                print("\n Y_hat_gat_idx[1:10] : ", Y_hat_gat_idx[1:10])
                            y2 = np.repeat(Y_hat_gat_idx.numpy().ravel(), ratio_loopbin_epibin)
                            if debug_text == True:
                                print("\n ==>> y2 len : ", len(y2))
                            y_pred_gat_bw_ = np.append(y_pred_gat_bw_, y2)
                            if debug_text == True:
                                print("\n ==>> y_pred_gat_bw_ len : ", len(y_pred_gat_bw_))

                            ## we do not use the sequence specific prediction
                            if USE_SEQNN_MODEL == True:
                                y3 = np.repeat(Y_hat_cnn_idx.numpy().ravel(), ratio_loopbin_epibin)
                                y_pred_cnn_bw_ = np.append(y_pred_cnn_bw_, y3)
                
                    ## loss calculation using the original and predicted CAGE gene expression
                    if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                        
                        ## we do not use the sequence specific prediction
                        if USE_SEQNN_MODEL == True:
                            Y_hat_cnn = model_cnn(X_epi)
                        
                        ## apply the current "X_epi" and "adj" to the pre-trained "model_gat"
                        Y_hat_gat, att = model_gat([X_epi, adj])
                        if debug_text == True:
                            print("\n ==>> Y_hat_gat shape : ", Y_hat_gat.shape)
                        
                        ## indices of original CAGE track
                        Y_idx = tf.gather(Y, idx, axis=1)
                        Y_all = np.append(Y_all, Y_idx.numpy().ravel())

                        ## indices of predicted CAGE track
                        Y_hat_gat_idx = tf.gather(Y_hat_gat, idx, axis=1)
                        Y_hat_gat_all = np.append(Y_hat_gat_all, Y_hat_gat_idx.numpy().ravel())

                        ## loss between original and predicted CAGE tracks
                        loss_gat = poisson_loss(Y_idx, Y_hat_gat_idx)
                        loss_gat_all = np.append(loss_gat_all, loss_gat.numpy())
                        if debug_text == True:
                            print("\n ==>> loss_gat : ", loss_gat.numpy())

                        if USE_SEQNN_MODEL == True:
                            Y_hat_cnn_idx = tf.gather(Y_hat_cnn, idx, axis=1)
                            Y_hat_cnn_all = np.append(Y_hat_cnn_all, Y_hat_cnn_idx.numpy().ravel())
                            loss_cnn = poisson_loss(Y_idx,Y_hat_cnn_idx)
                            loss_cnn_all = np.append(loss_cnn_all, loss_cnn.numpy())
                                            
                        ## number of contacts from individual bins
                        ## from input loop data and adjacency matrix
                        ## originally contacts are in loop resolution (like 5 Kb)
                        ## np.repeat operation expands the contacts to epigenomic track resolution (100 Kb)
                        row_sum = tf.squeeze(tf.reduce_sum(adj, axis=-1))
                        num_contacts = np.repeat(row_sum.numpy().ravel(), ratio_loopbin_epibin)
                        if debug_text == True:
                            print("\n ==>> num_contacts len : ", len(num_contacts))
                            print("\n num_contacts[1:10] : ", num_contacts[1:10])

                        ## extract gene tss's, with respect to the middle portion 
                        tss_pos_1 = tss_pos[np.logical_and(tss_pos >= pos[num_slide_window_epi_bin], tss_pos < pos[(2*num_slide_window_epi_bin)])]
                        if debug_text == True:
                            print("\n ==>> tss_pos len : ", len(tss_pos), "   tss_pos_1 len : ", len(tss_pos_1))

                        ## converting the loop resolution specific CAGE gene expression
                        ## to the epigenomic track resolution
                        y_true_ = np.repeat(Y.numpy().ravel(), ratio_loopbin_epibin)
                        y_hat_gat_ = np.repeat(Y_hat_gat.numpy().ravel(), ratio_loopbin_epibin)
                        if debug_text == True:
                            print("\n ==>>> len(y_true_) : ", len(y_true_))
                            print("\n ==>>> len(y_hat_gat_) : ", len(y_hat_gat_))                        

                        for j in range(len(tss_pos_1)):
                            idx_tss = np.where(pos == int(np.floor(tss_pos_1[j]/100)*100))[0][0]
                            idx_gene = np.where(tss_pos == tss_pos_1[j])[0][0]  # sourya - added last [0]
                            if debug_text == True:
                                print("\n ->> within iteration - j : ", j, " tss_pos_1[j] : ", tss_pos_1[j], " idx_tss : ", idx_tss, " idx_gene : ", idx_gene)                            
                            
                            if USE_SEQNN_MODEL == True:
                                y_hat_cnn_ = np.repeat(Y_hat_cnn.numpy().ravel(), ratio_loopbin_epibin)

                            y_gene = np.append(y_gene, y_true_[idx_tss])
                            y_hat_gene_gat = np.append(y_hat_gene_gat, y_hat_gat_[idx_tss])
                            if debug_text == True:
                                print(" y_true_[idx_tss] : ", y_true_[idx_tss], " y_hat_gat_[idx_tss] : ", y_hat_gat_[idx_tss])
                            
                            if USE_SEQNN_MODEL == True:
                                y_hat_gene_cnn = np.append(y_hat_gene_cnn, y_hat_cnn_[idx_tss])
                            
                            gene_tss = np.append(gene_tss, tss_pos_1[j])
                            gene_chr = np.append(gene_chr, 'chr' + str(i))
                            gene_names = np.append(gene_names, gene_names_all[idx_gene]) 
                            n_tss_in_bin = np.append(n_tss_in_bin, n_tss[idx_gene])
                            n_contacts = np.append(n_contacts, num_contacts[idx_tss])
                            if debug_text == True:
                                print(" gene_tss : ", tss_pos_1[j], " gene_chr : chr", str(i), " gene_names : ", gene_names_all[idx_gene], "  n_tss_in_bin : ", n_tss[idx_gene], "  n_contacts : ", num_contacts[idx_tss])
                        
                else:
                    if write_bw == True:
                        assert len(pos_bw) == len(y_bw_) == len(y_pred_gat_bw_)
                        chroms_ = np.array(["chr" + str(i)] * len(pos_bw))
                        if debug_text == True:
                            print('chr' + str(i), len(pos_bw))
                        starts_ = pos_bw.astype(np.int64)
                        ends_ = starts_ + 100
                        ends_ = ends_.astype(np.int64)

                        chroms = np.append(chroms, chroms_)
                        starts = np.append(starts, starts_)
                        ends = np.append(ends, ends_)
                        y_bw = np.append(y_bw, y_bw_)
                        y_pred_gat_bw = np.append(y_pred_gat_bw, y_pred_gat_bw_)

                        if USE_SEQNN_MODEL == True:
                            y_pred_cnn_bw = np.append(y_pred_cnn_bw, y_pred_cnn_bw_)

                    break

        if write_bw == True:
            starts = starts.astype(np.int64)
            idx_pos = np.where(starts>0)[0]
            starts = starts[idx_pos]
            ends = ends.astype(np.int64)[idx_pos]
            y_bw = y_bw.astype(np.float64)[idx_pos]
            y_pred_gat_bw = y_pred_gat_bw.astype(np.float64)[idx_pos]

            if USE_SEQNN_MODEL == True:
                y_pred_cnn_bw = y_pred_cnn_bw.astype(np.float64)[idx_pos]

            chroms = chroms[idx_pos]
            print(len(chroms), len(starts), len(ends), len(y_bw))
            print(chroms)
            print(starts)
            print(ends)
            print(y_bw)
            bw_y_true.addEntries(chroms, starts, ends=ends, values=y_bw)
            bw_y_pred_gat.addEntries(chroms, starts, ends=ends, values=y_pred_gat_bw)
            
            if USE_SEQNN_MODEL == True:
                bw_y_pred_cnn.addEntries(chroms, starts, ends=ends, values=y_pred_cnn_bw)

            bw_y_true.close()
            bw_y_pred_gat.close()
            
            if USE_SEQNN_MODEL == True:
                bw_y_pred_cnn.close()

        if debug_text == True:            
            print('\n\n len of test/valid Y: ', len(y_gene))

        if debug_text == True:
            print("\n\n\n ******** Going out of the function calculate_loss ********** \n\n\n")        
        
        return y_gene, y_hat_gene_gat, gene_names, gene_tss, gene_chr, n_contacts, n_tss_in_bin


    ############################################################# 
    ## main code
    #############################################################

    if prediction == True:
        
        ##======================
        ## process individual CAGE files
        ## and generate test statistics
        ##======================
        for cagefileidx in range(len(CAGETrackList)):

            ##==============
            ## initialize parameters
            ##==============
            valid_loss_gat = np.zeros([4])
            valid_rho_gat = np.zeros([4])
            valid_sp_gat = np.zeros([4])
            n_gene = np.zeros([4])

            # df_all_predictions = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'average_dnase', 'average_h3k27ac', 'average_h3k4me3', 'true_cage', 'pred_cage_EpiGAT', 'pred_cage_epi_cnn', 'nll_EpiGAT', 'nll_epi_cnn', 'delta_nll'])
            df_all_predictions = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'true_cage', 'pred_cage_EpiGAT', 'nll_EpiGAT'])

            ##==============
            ## process current CAGE file and corresponding training model
            ##==============
            CurrCAGETrackFile = CAGETrackList[cagefileidx]
            CurrCAGETrackLabel = CAGELabelList[cagefileidx]

            ## TFrecord directory - specific to this CAGE sample
            TFRecordDir = BaseOutDir + "/" + SampleLabel + "/TFRecord/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel

            ## training model directory - specific to this CAGE sample
            TrainDir = BaseOutDir + "/" + SampleLabel + "/TrainingModel/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel

            ## directory storing TSS information
            TSSDir = BaseOutDir + "/TSS/" + refgenome + '/' + str(Resolution)

            valid_chr_list = re.split(r':|,', options.valid_chr)
            test_chr_list = re.split(r':|,', options.test_chr)

            valid_chr_str = '_'.join(valid_chr_list)
            test_chr_str = '_'.join(test_chr_list)

            ## training model file name
            model_name_gat = TrainDir + '/Model_valid_chr_' + valid_chr_str + '_test_chr_' + test_chr_str + '.h5'            

            ## output directory (to store the testing output)
            TestOutDir = BaseOutDir + "/" + SampleLabel + "/TestModel/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel + "/valid_chr_" + valid_chr_str + '_test_chr_' + test_chr_str
            if not os.path.exists(TestOutDir):
                os.makedirs(TestOutDir)

            ## output file which will store the performance summary in a data frame
            ## for individual combinations of test and validation chromosomes
            ## stores the loss and correlation statistics
            FinalSummaryFile = BaseOutDir + "/" + SampleLabel + "/TestModel/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel + "/Final_Summary_Metrics.txt"

            ## open output log file and dump all the prints
            old_stdout = sys.stdout
            logfilename = TestOutDir + "/out.log"
            log_file = open(logfilename, "w")
            sys.stdout = log_file
            
            print('\n\n *** TestOutDir : ', str(TestOutDir))
            print("\n\n *** Processing CAGE file : ", CurrCAGETrackFile, "  label : ", CurrCAGETrackLabel)
            print("\n\n *** Valid chromosome list : ", str(options.valid_chr), "  test chromosome list : ", str(options.test_chr))
            print('\n\n *** training model file - model_name_gat : ', str(model_name_gat))

            ## load the training model
            ## sourya: we needed to add initializers and regularizers
            ## compile=False option prevents the model from further compiling / training - use it as it is
            model_gat = tf.keras.models.load_model(model_name_gat, custom_objects={'GraphAttention': GraphAttention, "GlorotUniform": tf.keras.initializers.glorot_uniform, "Zeros": tf.keras.initializers.Zeros, "L2": tf.keras.regularizers.L2}, compile=False)

            model_gat.trainable = False
            model_gat._name = 'Epigenome_3C_GAT'
            model_gat.summary()

            ## execute test model
            y_gene, y_hat_gene_gat, gene_names, gene_tss, gene_chr, n_contacts, n_tss_in_bin = calculate_loss(TSSDir, TFRecordDir, TestOutDir, model_gat, chrsize_df, valid_chr_list, test_chr_list, batch_size, write_bw, num_slide_window_epi_bin, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack, CAGEBinSize, EpiBinSize)

            ## save the predictions in numpy objects
            if 0:
                np.save(TestOutDir +'/true_cage.npy', y_gene)
                np.save(TestOutDir + '/Epigenome_GAT_predicted_cage.npy', y_hat_gene_gat)        
                if USE_SEQNN_MODEL == True:
                    np.save(TestOutDir + '/Epi-CNN_predicted_cage.npy', y_hat_gene_cnn)
                np.save(TestOutDir + '/n_contacts.npy', n_contacts)
                np.save(TestOutDir + '/gene_names.npy', gene_names)
                np.save(TestOutDir + '/gene_tss.npy', gene_tss)
                np.save(TestOutDir + '/gene_chr.npy', gene_chr)

            # df_tmp = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'average_dnase', 'average_h3k27ac', 'average_h3k4me3', 'true_cage', 'pred_cage_EpiGAT', 'pred_cage_epi_cnn', 'nll_EpiGAT', 'nll_epi_cnn', 'delta_nll'])
            df_tmp = pd.DataFrame(columns=['chr', 'genes', 'n_tss', 'tss', 'tss_distance_from_center', 'n_contact', 'true_cage', 'pred_cage_EpiGAT', 'nll_EpiGAT'])

            df_tmp['chr'] = gene_chr
            df_tmp['genes'] = gene_names
            df_tmp['n_tss'] = n_tss_in_bin.astype(np.int64)
            df_tmp['tss'] = gene_tss.astype(np.int64)
            df_tmp['tss_distance_from_center'] = np.abs(np.mod(gene_tss, 5000) - 2500).astype(np.int64)
            df_tmp['n_contact'] = n_contacts.astype(np.int64)
            # df_tmp['average_dnase'] = x_dnase
            # df_tmp['average_h3k27ac'] = x_h3k27ac
            # df_tmp['average_h3k4me3'] = x_h3k4me3
            df_tmp['true_cage'] = y_gene
            df_tmp['pred_cage_EpiGAT'] = y_hat_gene_gat
            
            if USE_SEQNN_MODEL == True:
                df_tmp['pred_cage_epi_cnn'] = y_hat_gene_cnn
            
            df_tmp['nll_EpiGAT'] = poisson_loss_individual(y_gene, y_hat_gene_gat).numpy()
            
            if USE_SEQNN_MODEL == True:
                df_tmp['nll_epi_cnn'] = poisson_loss_individual(y_gene, y_hat_gene_cnn).numpy()
                # if delta_nll > 0 then GraphReg prediction is better than CNN
                df_tmp['delta_nll'] = poisson_loss_individual(y_gene, y_hat_gene_cnn).numpy() - poisson_loss_individual(y_gene, y_hat_gene_gat).numpy()    

            df_all_predictions = df_all_predictions.append(df_tmp).reset_index(drop=True)

            ##==================
            ## define 4 gene sets for validation
            ## j = 0 - set 1: all genes - no expression and contact condition
            ## j = 1 - set 2: expressed genes (expression >= 5) - no contact condition 
            ## j = 2 - set 3: expressed and interacting genes (expression >= 5, number of contacts >= 1)
            ## j = 3 - set 4: expressed and highly interacting genes (expression >= 5, number of contacts >= 5)
            ## In the "df_all_predictions.csv" file, two columns "n_contact" and "true_cage" show these values

            for j in range(4):
                if j==0:
                    min_expression = 0 
                    min_contact = 0
                elif j==1:
                    min_expression = 5 
                    min_contact = 0
                elif j==2:
                    min_expression = 5
                    min_contact = 1
                else:
                    min_expression = 5 
                    min_contact = 5

                if debug_text == True:
                    print("\n\n *** j : ", j, "  min_expression : ", min_expression, "  min_contact : ", min_contact)

                idx = np.where(np.logical_and(n_contacts >= min_contact, y_gene >= min_expression))[0]
                y_gene_idx = y_gene[idx]
                y_hat_gene_gat_idx = y_hat_gene_gat[idx]                
                if USE_SEQNN_MODEL == True:
                    y_hat_gene_cnn_idx = y_hat_gene_cnn[idx]
                if debug_text == True:
                    print("\n\n *** len idx : ", len(idx))

                ## metrics on validation chromosomes
                ## loss: poisson distribution
                ## rho and sp: (pearson) correlation between true and predicted gene expression
                valid_loss_gat[j] = poisson_loss(y_gene_idx, y_hat_gene_gat_idx).numpy()
                valid_rho_gat[j] = np.corrcoef(np.log2(y_gene_idx+1),np.log2(y_hat_gene_gat_idx+1))[0,1]
                valid_sp_gat[j] = spearmanr(np.log2(y_gene_idx+1),np.log2(y_hat_gene_gat_idx+1))[0]
                
                if USE_SEQNN_MODEL == True:
                    valid_loss_cnn[j] = poisson_loss(y_gene_idx, y_hat_gene_cnn_idx).numpy()
                    valid_rho_cnn[j] = np.corrcoef(np.log2(y_gene_idx+1),np.log2(y_hat_gene_cnn_idx+1))[0,1]
                    valid_sp_cnn[j] = spearmanr(np.log2(y_gene_idx+1), np.log2(y_hat_gene_cnn_idx+1))[0]

                n_gene[j] = len(y_gene_idx)

            ## Final results
            print('\n\n *** NLL GAT: ', valid_loss_gat, ' rho: ', valid_rho_gat, ' sp: ', valid_sp_gat)
            if USE_SEQNN_MODEL == True:
                print('\n\n *** NLL CNN: ', valid_loss_cnn, ' rho: ', valid_rho_cnn, ' sp: ', valid_sp_cnn)

            print('\n\n ****** Mean Loss GAT: ', np.mean(valid_loss_gat, axis=0), ' +/- ', np.std(valid_loss_gat, axis=0), ' std')
            if USE_SEQNN_MODEL == True:
                print('Mean Loss CNN: ', np.mean(valid_loss_cnn, axis=0), ' +/- ', np.std(valid_loss_cnn, axis=0), ' std \n')

            print('\n\n ****** Mean R GAT: ', np.mean(valid_rho_gat, axis=0), ' +/- ', np.std(valid_rho_gat, axis=0), ' std')
            if USE_SEQNN_MODEL == True:
                print('Mean R CNN: ', np.mean(valid_rho_cnn, axis=0), ' +/- ', np.std(valid_rho_cnn, axis=0), ' std \n')

            print('\n\n ****** Mean SP GAT: ', np.mean(valid_sp_gat, axis=0), ' +/- ', np.std(valid_sp_gat, axis=0), ' std')
            if USE_SEQNN_MODEL == True:
                print('Mean SP CNN: ', np.mean(valid_sp_cnn, axis=0), ' +/- ', np.std(valid_sp_cnn, axis=0), ' std')

            ##===================
            ## comparison between epigenome model and sequence based CNN models
            ##===================
            if USE_SEQNN_MODEL == True:
                w_loss = np.zeros(4)
                w_rho = np.zeros(4)
                w_sp = np.zeros(4)
                p_loss = np.zeros(4)
                p_rho = np.zeros(4)
                p_sp = np.zeros(4)
                for j in range(4):
                    w_loss[j], p_loss[j] = wilcoxon(valid_loss_gat[:,j], valid_loss_cnn[:,j], alternative='less')
                    w_rho[j], p_rho[j] = wilcoxon(valid_rho_gat[:,j], valid_rho_cnn[:,j], alternative='greater')
                    w_sp[j], p_sp[j] = wilcoxon(valid_sp_gat[:,j], valid_sp_cnn[:,j], alternative='greater')
                print('Wilcoxon Loss: ', w_loss, ' , p_values: ', p_loss)
                print('Wilcoxon R: ', w_rho, ' , p_values: ', p_rho)
                print('Wilcoxon SP: ', w_sp, ' , p_values: ', p_sp)

            ##===================
            ## summary
            ##===================

            ## close the output log file
            sys.stdout = old_stdout
            log_file.close()

            ## write the prediction to csv file
            df_all_predictions.to_csv(TestOutDir + '/df_all_predictions.csv', sep="\t", index=False)            

            ## define the summary metric data frame
            SummaryDF = pd.DataFrame(data={'Validation_chr': [options.valid_chr], 'Test_chr': [options.test_chr], 'NLL_GAT_MinExpr_0_MinContact_0': [valid_loss_gat[0]], 'NLL_GAT_MinExpr_5_MinContact_0': [valid_loss_gat[1]], 'NLL_GAT_MinExpr_5_MinContact_1': [valid_loss_gat[2]], 'NLL_GAT_MinExpr_5_MinContact_5': [valid_loss_gat[3]], 'NLL_GAT_Mean': [np.mean(valid_loss_gat, axis=0)], 'NLL_GAT_std': [np.std(valid_loss_gat, axis=0)], 'rho_MinExpr_0_MinContact_0': [valid_rho_gat[0]], 'rho_MinExpr_5_MinContact_0': [valid_rho_gat[1]], 'rho_MinExpr_5_MinContact_1': [valid_rho_gat[2]], 'rho_MinExpr_5_MinContact_5': [valid_rho_gat[3]], 'rho_Mean': [np.mean(valid_rho_gat, axis=0)], 'rho_std': [np.std(valid_rho_gat, axis=0)], 'sp_MinExpr_0_MinContact_0': [valid_sp_gat[0]], 'sp_MinExpr_5_MinContact_0': [valid_sp_gat[1]], 'sp_MinExpr_5_MinContact_1': [valid_sp_gat[2]], 'sp_MinExpr_5_MinContact_5': [valid_sp_gat[3]], 'sp_Mean': [np.mean(valid_sp_gat, axis=0)], 'sp_std': [np.std(valid_sp_gat, axis=0)]})
            SummaryDF = SummaryDF.reset_index(drop=True)

            ## write the summary metric data frame
            bool_file_exist = os.path.exists(FinalSummaryFile)
            if (bool_file_exist == False):
                SummaryDF.to_csv(FinalSummaryFile, header=True, mode="w", sep="\t", index=False)
            else:
                SummaryDF.to_csv(FinalSummaryFile, header=False, mode="a", sep="\t", index=False)

            ## plot correlations between true and predicted CAGE values            
            ## for different subsets of genes
            for j in range(4):
                if j==0:
                    ## considers all genes
                    min_expression_thr = 0 
                    min_contact_thr = 0
                elif j==1:
                    ## considers genes with expression >= 5
                    min_expression_thr = 5 
                    min_contact_thr = 0
                elif j==2:
                    ## considers genes with expression >= 5 and at least 1 contact associated
                    min_expression_thr = 5
                    min_contact_thr = 1
                else:
                    ## considers genes with expression >= 5 and at least 5 contacts associated
                    min_expression_thr = 5 
                    min_contact_thr = 5

                idx = np.where(np.logical_and(n_contacts >= min_contact_thr, y_gene >= min_expression_thr))[0]
                y_gene_idx = y_gene[idx]
                y_hat_gene_gat_idx = y_hat_gene_gat[idx]
                n_contacts_idx = n_contacts.astype(np.int64)[idx]

                plotlabel = "Expr_" + str(min_expression_thr) + "_Contact_" + str(min_contact_thr)
                plotfile_logscale = TestOutDir + "/Scatter_Expr_" + str(min_expression_thr) + "_Contact_" + str(min_contact_thr) + "_log_scale.png"
                Plot_Corr_log2(plotfile_logscale, y_gene_idx, y_hat_gene_gat_idx, n_contacts_idx, plotlabel, valid_loss_gat[j], valid_rho_gat[j], valid_sp_gat[j])
                plotfile = TestOutDir + "/Scatter_Expr_" + str(min_expression_thr) + "_Contact_" + str(min_contact_thr) + ".png"
                Plot_Corr(plotfile, y_gene_idx, y_hat_gene_gat_idx, n_contacts_idx, plotlabel, valid_loss_gat[j], valid_rho_gat[j], valid_sp_gat[j])


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

