##===================
## Training model using both epigenomic tracks and chromatin contact, to predict gene expression

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##===================

from __future__ import division
from optparse import OptionParser

import numpy as np
import random
import pandas as pd
import os
import psutil
import re
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

## import the local utility file within current directory
from Model_Epigenomic_Contact import *

## import the local utility file within current directory
from UtilFunc import *

## debug variable 
debug_text = True

## hyperparameters
OUT_CHANNEL_DIM = 128   ## output channel dimension
KERNEL_SIZE = 5     ## kernel size for convolutions
CNN_DROPOUT_RATE = 0.0          ## dropout rate for CNN model
# GAT_DROPOUT_RATE = 0.0  #0.1  #0.3            ## dropout rate for GAT model
LEARNING_RATE = 1e-3    #2e-3            ## learning rate    
l2_reg = 1e-2   ## 1e-5   #0.0            ## factor for l2 regularization - weight decay    

## early stopping criteria - max 10 epoch for performance improvement
max_early_stopping = 10

## max epoch count
n_epochs = 100

batch_size = 1

## define loss function
huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)

##===================
## parse options
##===================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-g', 
                    dest='refgenome', 
                    default=None, 
                    type='str', 
                    help='Reference Genome. Mandatory parameter.')

    parser.add_option('-v', 
                      dest='valid_chr', 
                      default='1_11', 
                      type='str', 
                      help="Underscore separated list (numbers) of the validation chromosomes. Default 1_11 means that chr1 and chr11 would be used as the validation chromosomes.")

    parser.add_option('-t', 
                      dest='test_chr', 
                      default='2_12', 
                      type='str', 
                      help="Underscore separated list (numbers) of the test chromosomes. Default 2_12 means that chr2 and chr12 would be used as the test chromosomes.")
    
    parser.add_option('-D', 
                      dest='TrainDataDirList', 
                      default=None, 
                      type='str', 
                      help='List of directories (comma separated) storing training data (for different chromosomes). Mandatory parameter.')

    parser.add_option('-O', 
                      dest='TrainModelDir', 
                      default=None, 
                      type='str', 
                      help='Output directory to store the training model. Mandatory parameter.')

    parser.add_option('--Offset', 
                    dest='Offset', 
                    default=2000000, 
                    type='int', 
                    help='Offset (middle portion / sliding window). Default = 2000000 (2 Mb), as suggested in the EpiGraphReg paper.')
    
    parser.add_option('-C', 
                      dest='CAGEBinSize', 
                      default=5000, 
                      type='int', 
                      help='CAGE (gene expression) bin size. Default 5000 (5 Kb), same as the default loop resolution')

    parser.add_option('-E', 
                      dest='EpiBinSize', 
                      default=100, 
                      type='int', 
                      help='Epigenomic track bin size. Default 100 bp')

    parser.add_option('-R', 
                      dest='ResidGAT', 
                      default=1, 
                      type='int', 
                      help='If 1 (default), residual information is used in GAT.')

    parser.add_option('--ActFun', 
                      dest='ActFun', 
                      default="gelu", 
                      type='str', 
                      help='Activation function. Default = gelu')
    
    parser.add_option('--ModelEPI', 
                      dest='ModelEPI', 
                      default=2, 
                      type='int', 
                      help='1D Model (epigenomic data) (1: CNN + Maxpool, 2: Residual Net). Default = 2')

    parser.add_option('--Model3D', 
                      dest='Model3D', 
                      default=1, 
                      type='int', 
                      help='3D Loop data model - 1: GAT, 2: GT. Default = 1')

    parser.add_option('--NumGATLayer', 
                      dest='NumGATLayer', 
                      default=2, 
                      type='int', 
                      help='Number of GAT layers. Default = 8')

    parser.add_option('--NumHeadGAT', 
                      dest='NUM_HEADS_GAT', 
                      default=8, 
                      type='int', 
                      help='Number of attention heads in GAT. Default = 4')

    parser.add_option('--EdgeFeatCols', 
                      dest='EdgeFeatCols', 
                      default="NA", 
                      type='str', 
                      help='Edge features. Options: 1. NA: means no edge features are used. 2. Colon or underscore separated list of columns storing the target edge features.')

    parser.add_option('-p', 
                      dest='ProjectedChannelDim', 
                      default=32, 
                      type='int', 
                      help='Projected input channel dimension, used to define the convolution. Default 32.')
    
    parser.add_option('--gatdropout', 
                      dest='gat_dropout_rate', 
                      default=0, 
                      type='float', 
                      help='GAT dropout rate. Default 0')
    
    parser.add_option('--fcn', 
                      dest='use_FCN', 
                      default=0, 
                      type='int', 
                      help='If 1, uses fully connected network (FCN) after GAT. Else uses convolution. Default 0')    
        
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
    TrainModelDir = options.TrainModelDir    ## output directory to store training model
    TrainDataDirList = options.TrainDataDirList.split(",")  ## parse input comma separated list
    CAGEBinSize = int(options.CAGEBinSize)
    EpiBinSize = int(options.EpiBinSize)
    Offset = int(options.Offset)        ## Sliding window - default = 2 Mb
    ProjectedChannelDim = int(options.ProjectedChannelDim)
    NUM_LAYERS_GAT = int(options.NumGATLayer)           ## number of GAT layers    
    NUM_HEADS_GAT = int(options.NUM_HEADS_GAT)          ## number of headers in GAT
    activation_fn_name = options.ActFun                 ## activation function name
    INITIAL_RESIDUAL_GAT = int(options.ResidGAT)        ## initial residual GAT

    GAT_DROPOUT_RATE = float(options.gat_dropout_rate)
    use_FCN = int(options.use_FCN)
    
    ## 1D and 3D models
    ## 1D: CNN or ResNet
    ## 3D: GAT or GT
    ModelEPI = int(options.ModelEPI)
    Model3D = int(options.Model3D)

    ## list of columns in the interaction file storing the edge features
    ## which will be used for cobstructing graph attention / graph transformer methods
    EdgeFeatCols = options.EdgeFeatCols.replace(":", "_")
    EdgeFeatCols = EdgeFeatCols.replace(",", "_")    
    if EdgeFeatCols != "NA":
        EdgeFeatColList = re.split(r'_', EdgeFeatCols)
        if len(EdgeFeatColList) > 0:
            EdgeFeatColList = [int(x) for x in EdgeFeatColList]     ## convert to integers
    else:
        EdgeFeatColList = []

    ## output directory
    os.makedirs(TrainModelDir, exist_ok = True)

    ##==========
    ## output folders and file names
    ##==========
    ## training model file name (to be created)    
    model_filename = TrainModelDir + '/Model_valid_chr_' + str(options.valid_chr) + '_test_chr_' + str(options.test_chr) + '.pt'
    print('*** output model_filename : ', str(model_filename))

    ## training model plot name
    model_plotname = model_filename.replace(".pt", "_plot.png")
    print('*** output model_plotname : ', str(model_plotname))

    ## set the current working directory as TrainDir
    os.chdir(TrainModelDir)

    ## open output log file and print input parameters, output file names
    old_stdout = sys.stdout
    logfilename = TrainModelDir + "/out_valid_chr_" + str(options.valid_chr) + "_test_chr_" + str(options.test_chr) + ".log"
    log_file = open(logfilename, "w")
    sys.stdout = log_file

    ##=================
    ## derived parameters
    ##=================
    ## default = 6 Mb - 3 times sliding window
    Span = (3 * Offset) #int(options.Span)
    num_slide_window_loop_bin = Offset // CAGEBinSize
    num_span_loop_bin = Span // CAGEBinSize
    num_span_epi_bin = Span // EpiBinSize

    ##===================
    ## list the parameters
    ##===================
    ## log file
    print("\n PyTorch version: {torch.__version__}")
    ## check if CUDA device is available
    print("\n torch cuda available : " + str(torch.cuda.is_available()))
    print("\n torch cuda device count : " + str(torch.cuda.device_count()))
    print("\n torch cuda current device : " + str(torch.cuda.current_device()))
    if torch.cuda.device_count() > 0:        
        for i in range(torch.cuda.device_count()):
            print("\n torch cuda device " + str(i) + " is " + str(torch.cuda.get_device_name(i)))

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('\n Using device:' + str(device))

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('\n Memory Usage:')
        print('\n Allocated:' + str(round(torch.cuda.memory_allocated(0)/1024**3,1)) + 'GB')
        print('\n Cached:   ' + str(round(torch.cuda.memory_reserved(0)/1024**3,1)) + 'GB')
    
    print('\n\n ==>> Input Parameters <<<==== ')
    print('\n TrainModelDir : ' + str(TrainModelDir))    
    print('\n refgenome : ' + str(refgenome))
    print('\n 1D Epigenomic data model (1: CNN, 2: ResNet): ' + str(ModelEPI))
    print('\n 3D Chromatin contact data model (1: GAT, 2: GT): ' + str(Model3D))
    print('\n CAGEBinSize : ' + str(CAGEBinSize))
    print('\n EpiBinSize : ' + str(EpiBinSize))

    print('\n activation_fn_name : ' + str(activation_fn_name))
    print('\n CNN_DROPOUT_RATE : ' + str(CNN_DROPOUT_RATE))
    print('\n NUM_LAYERS_GAT : ' + str(NUM_LAYERS_GAT))
    print('\n INITIAL_RESIDUAL_GAT : ' + str(INITIAL_RESIDUAL_GAT))
    print('\n GAT_DROPOUT_RATE : ' + str(GAT_DROPOUT_RATE))
    
    print('\n Offset : ' + str(Offset))
    print('\n Span : ' + str(Span))
    print('\n num_slide_window_loop_bin : ' + str(num_slide_window_loop_bin))
    print('\n num_span_loop_bin : ' + str(num_span_loop_bin))
    print('\n num_span_epi_bin : ' + str(num_span_epi_bin))

    print('\n Edge feature columns in interaction file : ' + str(EdgeFeatColList))

    ##==============
    ## define the training, validation and test chromosomes
    ##==============

    ## extract the validation and test chromosomes
    ## all other chromosomes are used for training
    valid_chr_list = re.split(r':|,|_', options.valid_chr)
    test_chr_list = re.split(r':|,|_', options.test_chr)    
    print('\n valid chromosomes : ' + str(options.valid_chr))
    print('\n test chromosomes : ' + str(options.test_chr))

    if refgenome == 'mm9' or refgenome == 'mm10':
        train_chr_list = [c for c in range(1,1+19)]
    elif refgenome == 'hg19' or refgenome == 'hg38':
        train_chr_list = [c for c in range(1,1+22)]
    for j in range(len(valid_chr_list)):
        train_chr_list.remove(int(valid_chr_list[j]))
    for j in range(len(test_chr_list)):
        train_chr_list.remove(int(test_chr_list[j]))
        
    print('\n ==>>> Training chromosomes : ' + str(train_chr_list))

    ##================================
    ## main code
    ##================================    

    ##==========
    ## training model parameters
    ##==========

    ##========= CNN related parameters            
    best_loss = 1e20
    best_rho = 0

    ## write model parameters
    print('\n l2_reg : ' + str(l2_reg))
    print('\n max_early_stopping : ' + str(max_early_stopping))
    print('\n n_epochs : ' + str(n_epochs))
    print('\n batch_size : ' + str(batch_size))

    ## set initilization seed for reproducibility
    ## check: https://pytorch.org/docs/stable/notes/randomness.html
    ## both CPU and GPU
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    ## all GPUs
    random.seed(seed)
    np.random.seed(seed)

    ## CUDA deterministic algorithm
    torch.backends.cudnn.deterministic = True

    ## empty Cache memory
    torch.cuda.empty_cache() 

    ##=======================
    ## model related functions
    ## implemented within the main code
    ## to avoid repeated parameter passing
    ##=======================

    ##=============================
    ## validation / test routine
    ##=============================
    def model_validate(device, out_model, chr_list, batch_size, 
                        num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, 
                        epoch, TrainDataDirList, EdgeFeatColList):
                
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])

        ## data frame containing individual epoch / chunk wise statistics
        curr_Detailed_Stat_df = pd.DataFrame(columns=['Epoch', 'Data_Type', 'chromosome', 
                                                      'chunk', 'interval', 'window', 
                                                      'loss', 'rho'])
                
        ## middle of the sliding window
        idx = torch.arange(num_slide_window_loop_bin, 2*num_slide_window_loop_bin).to(torch.int64)
        idx = idx.to(device)
        gather_idx = torch.unsqueeze(idx, 0).to(device)

        ##=== iterate through the training directories
        ##=== there can be more than one training directory
        ##=== each containing multiple chromosome information
        for TorchDataDir_Base in TrainDataDirList:
            if debug_text == True:
                print("\n ***** training directory : ", str(TorchDataDir_Base), " *** \n")
            
            ## iterate through individual chromosomes
            for currchr in chr_list:
                if debug_text == True:
                    print(f"\n ===>>> training chunks in chromosome : {currchr}", file=sys.stdout)
                sys.stdout.flush()

                ## directory containing Pytorch data (chunks) for the current chromosome
                CurrDataDir = TorchDataDir_Base + "/chr" + str(currchr)
                
                chunk_num = 0
                
                ## define data loader for this folder (inherits from class DataLoader)
                testdataset = ReadFolderData_CC(CurrDataDir)
                testdataloader = DataLoader(testdataset, 
                                            batch_size = batch_size, 
                                            shuffle = False)        
        
                for batch_idx, (data_exist, X_epi, Y, bin_idx, edge_idx, edge_feat) in enumerate(testdataloader):

                    if data_exist:
                        chunk_num = chunk_num + 1

                        current_batch_size = X_epi.size(0)  # This gives the actual batch size

                        X_epi = X_epi.to(device)
                        Y = Y.to(device)
                        edge_idx = edge_idx.to(device)
                        edge_feat = edge_feat.to(device)

                        ## remove the batch information in edge index and edge features
                        edge_idx = torch.squeeze(edge_idx, dim = 0)
                        edge_feat = torch.squeeze(edge_feat, dim = 0)

                        if debug_text == True:
                            print("\n Read chunk : ", chunk_num, 
                                  " ===>>> X_epi shape : ", str(X_epi.shape), 
                                  "  Y shape : ", str(Y.shape),
                                  "  edge_idx shape : ", str(edge_idx.shape),
                                  "  edge_feat shape : ", str(edge_feat.shape))

                        ## resize the input 1D data
                        X_epi, Y, bin_idx, edge_feat = \
                            ProcessInputData_CC(X_epi, Y, bin_idx, current_batch_size, 
                                                num_span_epi_bin, num_span_loop_bin, 
                                                edge_feat, EdgeFeatColList)
                        
                        if debug_text == True:
                            print("===>>> iterator - chunk : ", chunk_num, 
                                " - interval: [", bin_idx[0, 0].numpy(), " - ", (bin_idx[0, -1].numpy() + CAGEBinSize - 1), "]",
                                " - middle window: [", bin_idx[0, idx[0]].numpy(), " - ", (bin_idx[0, idx[-1]].numpy() + CAGEBinSize - 1), "]")
                        
                        ## during validation or testing, we need to make sure to stop gradient computation
                        with torch.no_grad():

                            ## Use both epigenetic track and the chromatin interactions as model inputs
                            ## Return: predicted gene expression and also the attention scores
                            ## attn_scores is a tuple: (edge_idx, score)
                            Y_hat, attn_scores = out_model(X_epi, edge_idx, edge_feat, device)
                            Y_hat = Y_hat.to(device)

                            ## plot the attention scores
                            if 0:
                                plotfile = "attn_heatmap_training_epoch_" + str(epoch) + "_chr" + str(currchr) + "_chunk_" + str(chunk_num) + ".pdf"
                                Plot_Att_Heatmap(plotfile, attn_scores)
                    
                            Y_hat = torch.squeeze(Y_hat, dim=1)
                            Y_hat_idx = torch.gather(Y_hat, 1, gather_idx)  ## predicted output  
                            Y_idx = torch.gather(Y, 1, gather_idx)  ## original gene expression
                            
                            ## loss estimation
                            if 0:
                                loss = poisson_loss(Y_idx, Y_hat_idx)
                            else:
                                ## employ Huber loss between the observed and predicted CAGE
                                loss = huber_loss(Y_idx, Y_hat_idx)

                            ## here no backpropagation is employed

                            ## store this loss value
                            curr_loss = loss.item()
                            loss_gat_all = np.append(loss_gat_all, curr_loss)

                            Y_idx_copy = Y_idx.detach().clone().cpu()
                            Y_hat_idx_copy = Y_hat_idx.detach().clone().cpu()

                            curr_rho_gat = compute_pearson_corr(Y_idx_copy, Y_hat_idx_copy)
                            # curr_rho_gat = batch_pearson_corrcoef(Y_hat_idx_copy, Y_idx_copy)
                            rho_gat_all = np.append(rho_gat_all, curr_rho_gat)
                            
                            if debug_text == True:
                                print(" -- curr_loss : ", curr_loss, 
                                    " curr_rho : ", curr_rho_gat)

                            ## append in the detailed statistics
                            new_row = {'Epoch': epoch, 
                                    'Data_Type': "Validation", 
                                    'chromosome': currchr, 
                                    'chunk': chunk_num, 
                                    'interval': str(bin_idx[0, 0].numpy()) + " - " + str(bin_idx[0, -1].numpy() + CAGEBinSize - 1), 
                                    'window': str(bin_idx[0, idx[0]].numpy()) + " - " + str(bin_idx[0, idx[-1]].numpy() + CAGEBinSize - 1), 
                                    'loss': curr_loss, 
                                    'rho': curr_rho_gat}
                                                
                            ## add the detailed statistics         
                            if curr_Detailed_Stat_df.empty:
                                curr_Detailed_Stat_df = pd.DataFrame([new_row])
                            else:
                                curr_Detailed_Stat_df = pd.concat([curr_Detailed_Stat_df, 
                                                            pd.DataFrame([new_row])], 
                                                            ignore_index=True)
                                
                        ## delete memory
                        del X_epi
                        del Y
                        del bin_idx
                        del edge_idx 
                        del edge_feat
                        del Y_hat
                        del attn_scores
                        del Y_hat_idx
                        del Y_idx
                        del Y_idx_copy
                        del Y_hat_idx_copy

                    else:
                        break
        
        ## ignore non-NaN values
        valid_loss = np.nanmean(loss_gat_all)
        valid_rho = np.nanmean(rho_gat_all)

        return valid_loss, valid_rho, curr_Detailed_Stat_df


    ##=============================
    ## model training routine
    ##=============================
    def model_train(device, out_model, train_chr_list, batch_size, 
                        num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, 
                        epoch, TrainDataDirList, EdgeFeatColList):

        ## initialize the loss and rho containing data structures
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])

        ## data frame containing individual epoch / chunk wise statistics
        curr_Detailed_Stat_df = pd.DataFrame(columns=['Epoch', 'Data_Type', 'chromosome', 
                                                      'chunk', 'interval', 'window', 
                                                      'loss', 'rho'])
        
        ## middle of the sliding window
        idx = torch.arange(num_slide_window_loop_bin, 2*num_slide_window_loop_bin).to(torch.int64)
        idx = idx.to(device)
        gather_idx = torch.unsqueeze(idx, 0).to(device)    

        ##=== iterate through the training directories
        ##=== there can be more than one training directory
        ##=== each containing multiple chromosome information
        for TorchDataDir_Base in TrainDataDirList:
            if debug_text == True:
                print("\n ***** training directory : ", str(TorchDataDir_Base), " *** \n")
            
            ## iterate through individual chromosomes
            for tc in train_chr_list:
                if debug_text == True:
                    print(f"\n ===>>> training chunks in chromosome : {tc}", file=sys.stdout)
                sys.stdout.flush()

                ## directory containing Pytorch data (chunks) for the current chromosome
                CurrDataDir = TorchDataDir_Base + "/chr" + str(tc)
                
                chunk_num = 0
                
                ## define data loader for this folder (inherits from class DataLoader)
                traindataset = ReadFolderData_CC(CurrDataDir)
                traindataloader = DataLoader(traindataset, batch_size = batch_size, shuffle = False)

                for batch_idx, (data_exist, X_epi, Y, bin_idx, edge_idx, edge_feat) in enumerate(traindataloader):

                    if data_exist:
                        chunk_num = chunk_num + 1

                        current_batch_size = X_epi.size(0)  # This gives the actual batch size

                        X_epi = X_epi.to(device)
                        Y = Y.to(device)
                        edge_idx = edge_idx.to(device)
                        edge_feat = edge_feat.to(device)

                        ## remove the batch information in edge index and edge features
                        edge_idx = torch.squeeze(edge_idx, dim = 0)
                        edge_feat = torch.squeeze(edge_feat, dim = 0)

                        if debug_text == True:
                            print("\n Read chunk : ", chunk_num, 
                                  " ===>>> X_epi shape : ", str(X_epi.shape), 
                                  "  Y shape : ", str(Y.shape),
                                  "  edge_idx shape : ", str(edge_idx.shape),
                                  "  edge_feat shape : ", str(edge_feat.shape))

                        ## resize the input 1D data
                        X_epi, Y, bin_idx, edge_feat = \
                            ProcessInputData_CC(X_epi, Y, bin_idx, current_batch_size, 
                                                num_span_epi_bin, num_span_loop_bin, 
                                                edge_feat, EdgeFeatColList)
                        
                        if debug_text == True:
                            print("===>>> iterator - chunk : ", chunk_num, 
                                " - interval: [", bin_idx[0, 0].numpy(), " - ", (bin_idx[0, -1].numpy() + CAGEBinSize - 1), "]",
                                " - middle window: [", bin_idx[0, idx[0]].numpy(), " - ", (bin_idx[0, idx[-1]].numpy() + CAGEBinSize - 1), "]")
                        
                        with torch.autograd.set_detect_anomaly(True):

                            ## reset optimizer gradient
                            opt.zero_grad()     

                            ## Use both epigenetic track and the chromatin interactions as model inputs
                            ## Return: predicted gene expression and also the attention scores
                            ## attn_scores is a tuple: (edge_idx, score)
                            Y_hat, attn_scores = out_model(X_epi, edge_idx, edge_feat, device)
                            Y_hat = Y_hat.to(device)

                            ## plot the attention scores
                            if 0:
                                plotfile = "attn_heatmap_training_epoch_" + str(epoch) + "_chr" + str(tc) + "_chunk_" + str(chunk_num) + ".pdf"
                                Plot_Att_Heatmap(plotfile, attn_scores)
                    
                            Y_hat = torch.squeeze(Y_hat, dim=1)
                            Y_hat_idx = torch.gather(Y_hat, 1, gather_idx)  ## predicted output  
                            Y_idx = torch.gather(Y, 1, gather_idx)  ## original gene expression
                            
                            ## loss estimation
                            if 0:
                                loss = poisson_loss(Y_idx, Y_hat_idx)
                            else:
                                ## employ Huber loss between the observed and predicted CAGE
                                loss = huber_loss(Y_idx, Y_hat_idx)

                            ## now apply backpropagation
                            loss.backward()
                            if 0:                     
                                ## sourya - stop parameters from exploding
                                torch.nn.utils.clip_grad_norm_(out_model.parameters(), 1.0)     
                            opt.step()

                            ## print parameter gradient values
                            if 0:
                                print("\n ===>>> Printing parameter gradients ")
                                for p in out_model.parameters():
                                    print(p.grad)

                            ## store this loss value
                            curr_loss = loss.item()
                            loss_gat_all = np.append(loss_gat_all, curr_loss)

                            Y_idx_copy = Y_idx.detach().clone().cpu()
                            Y_hat_idx_copy = Y_hat_idx.detach().clone().cpu()

                            curr_rho_gat = compute_pearson_corr(Y_idx_copy, Y_hat_idx_copy)
                            # curr_rho_gat = batch_pearson_corrcoef(Y_hat_idx_copy, Y_idx_copy)
                            rho_gat_all = np.append(rho_gat_all, curr_rho_gat)
                            
                            if debug_text == True:
                                print(" -- curr_loss : ", curr_loss, 
                                    " curr_rho : ", curr_rho_gat)

                            ## append in the detailed statistics
                            new_row = {'Epoch': epoch, 
                                    'Data_Type': "Training", 
                                    'chromosome': tc, 
                                    'chunk': chunk_num, 
                                    'interval': str(bin_idx[0, 0].numpy()) + " - " + str(bin_idx[0, -1].numpy() + CAGEBinSize - 1), 
                                    'window': str(bin_idx[0, idx[0]].numpy()) + " - " + str(bin_idx[0, idx[-1]].numpy() + CAGEBinSize - 1), 
                                    'loss': curr_loss, 
                                    'rho': curr_rho_gat}
                            
                            ## add the detailed statistics         
                            if curr_Detailed_Stat_df.empty:
                                curr_Detailed_Stat_df = pd.DataFrame([new_row])
                            else:
                                curr_Detailed_Stat_df = pd.concat([curr_Detailed_Stat_df, 
                                                            pd.DataFrame([new_row])], 
                                                            ignore_index=True)
                                                        
                        ## delete memory
                        del X_epi
                        del Y
                        del bin_idx
                        del edge_idx 
                        del edge_feat
                        del Y_hat
                        del attn_scores
                        del Y_hat_idx
                        del Y_idx
                        del Y_idx_copy
                        del Y_hat_idx_copy

                    else:
                        break
        
        ## current epoch - training data specific loss 
        ## ignore NaN values
        train_loss = np.nanmean(loss_gat_all)
        train_rho = np.nanmean(rho_gat_all)

        return train_loss, train_rho, curr_Detailed_Stat_df


    ##***************
    ## Define the deep learning model (using CNN)
    ## to process the epigenomic tracks and predict the gene expression            
    ##***************

    ##===========
    ##======== Step 1: define downsampling / residual layers (>= 3)
    ##===========
    if (CAGEBinSize // EpiBinSize) > 1:
        DownSample_Ratio_List = prime_factors_sequence(EpiBinSize, CAGEBinSize)        
        if (len(DownSample_Ratio_List) < 3):
            DownSample_Ratio_List.extend([1] * (3 - len(DownSample_Ratio_List)))
    else:
        DownSample_Ratio_List = [1] * 3
    if debug_text == True:
        print("==>>  DownSample_Ratio_List : ", str(DownSample_Ratio_List))    

    ##===========
    ##======= step 2: define instance of the full model
    ##===========
    out_model = FullModel(ModelEPI,
                          Model3D,                           
                          OUT_CHANNEL_DIM,
                          KERNEL_SIZE,
                          ProjectedChannelDim, 
                          NUM_LAYERS_GAT, 
                          NUM_HEADS_GAT, 
                          INITIAL_RESIDUAL_GAT,
                          len(EdgeFeatColList),     ## number of edge features in GAT 
                          DownSample_Ratio_List, 
                          CNN_DROPOUT_RATE, 
                          GAT_DROPOUT_RATE,
                          use_FCN,
                          activation_fn_name).to(device)

    #############################
    ########## step 3: training ##########
    #############################

    ## sourya - later implement variable weighted step size based learning rate
    ## define optimizer - AdamW
    opt = torch.optim.AdamW(out_model.parameters(), 
                            lr=LEARNING_RATE, 
                            weight_decay=l2_reg)

    ## This dataframe will store the statistics for loss for individual epochs   
    Stat_df = pd.DataFrame(columns=['Epoch', 'Train_Loss', 'Train_Rho', 
                                    'Valid_loss', 'Valid_rho', 'best_loss', 
                                    'early_stopping_counter', 'Time'])    

    t0 = time.time()        

    ###==================
    ## Model Training
    ###==================
    ##======== train for "n_epochs"
    for epoch in range(1, n_epochs+1):

        if debug_text == True:
            print("\n\n ***** Iteration --- epoch : ", epoch, " *** \n\n")

        ###==============
        ## Model training
        if debug_text == True:
            print('\n\n ***** Processing training Chromosomes ***** \n\n') 

        train_loss, train_rho, train_stat_df = \
            model_train(device, out_model, train_chr_list, batch_size, 
                        num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, 
                        epoch, TrainDataDirList, EdgeFeatColList)

        ## add the detailed statistics         
        if epoch == 1:
            Detailed_Stat_df = train_stat_df
        else:
            Detailed_Stat_df = pd.concat([Detailed_Stat_df, train_stat_df], 
                                         axis = 0, ignore_index=True)

        if debug_text == True:
            print('\n ==>>> ***** epoch: ', epoch, 
                  ', mean train loss: ', train_loss, 
                  ', mean train rho: ', train_rho, 
                  ', time taken: ', (time.time() - t0), ' sec')
        
        ###==============
        ## Validation
        
        ## now run the trained model on the validation data
        ## and compute the loss function
        if debug_text == True:
            print('\n\n ***** Processing validation Chromosomes ***** \n\n') 
        
        valid_loss, valid_rho, valid_stat_df = \
            model_validate(device, out_model, valid_chr_list, batch_size, 
                        num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, 
                        epoch, TrainDataDirList, EdgeFeatColList)
        
        ## add the detailed statistics         
        Detailed_Stat_df = pd.concat([Detailed_Stat_df, valid_stat_df], 
                                     axis = 0,
                                     ignore_index=True)
                
        if debug_text == True:
            print('\n ==>>> ***** epoch: ', epoch, 
                  ', mean valid_loss : ', valid_loss, 
                  ' mean valid_rho : ', valid_rho,
                  ' time passed: ', (time.time() - t0), ' sec') 
        
        ##=========== condition - better model selection - select by improved correlation
        ##=========== based on validation data correlation
        
        if valid_rho > best_rho:
            if debug_text == True:
                print('\n ***** Condition valid_rho > best_rho: *** current valid loss : ', valid_loss, 
                      ' current valid_rho : ', valid_rho,
                      '  existing best_loss : ', best_loss, 
                      '  existing best_rho : ', best_rho)

            ## reset early_stopping_counter
            early_stopping_counter = 1
            
            best_rho = valid_rho

            ## update only if we found a better metric
            if valid_loss < best_loss:
                best_loss = valid_loss
            
            ## save the complete model
            if 1:
                torch.save(out_model, model_filename)

            ## save the state_dict etc. separately
            if 0:
                torch.save({
                    'model_state_dict': out_model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'epoch': epoch,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state(),
                }, model_filename)
                
            ## we generate the data frame summarizing all these measures
            new_row = {'Epoch': epoch, 
                       'Train_Loss': train_loss, 
                       'Train_Rho': train_rho, 
                       'Valid_loss': valid_loss, 
                       'Valid_rho': valid_rho, 
                       'best_loss': best_loss, 
                       'early_stopping_counter': early_stopping_counter, 
                       'Time': (time.time() - t0)}
            Stat_df = pd.concat([Stat_df, pd.DataFrame([new_row])], ignore_index=True)

        else:
            early_stopping_counter += 1

            ## we generate the data frame summarizing all these measures
            new_row = {'Epoch': epoch, 
                       'Train_Loss': train_loss, 
                       'Train_Rho': train_rho, 
                       'Valid_loss': valid_loss, 
                       'Valid_rho': valid_rho, 
                       'best_loss': best_loss, 
                       'early_stopping_counter': early_stopping_counter, 
                       'Time': (time.time() - t0)}
            Stat_df = pd.concat([Stat_df, pd.DataFrame([new_row])], ignore_index=True)
            
            ## for the last "max_early_stopping" epochs, validation loss has not improved
            ## so there is no point in further training
            if early_stopping_counter == max_early_stopping:
                print("\n\n\n ******** \n The last update of validation loss (and corresponding training model) was done ", max_early_stopping, 
                      " iterations before \n So as the training model looks stable, in order to prevent overfitting, we are early terminating this model - current epoch: ", epoch, 
                      "  \n ******** \n\n\n")
                break
        
    ## finally write the epoch-specific mean statistics 
    Stat_df.to_csv(model_filename.replace(".pt", "_Complete_Stat.txt"), index=False)

    ## finally write the detailed statistics - all epochs and all chunks    
    Detailed_Stat_df.to_csv(model_filename.replace(".pt", "_Detailed_Epoch_Chunkwise_Stat.txt"), index=False)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

