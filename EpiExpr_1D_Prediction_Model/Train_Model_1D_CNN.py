##===================
## Training model using only epigenomic tracks, to predict gene expression
## model using CNN + residual Net / Max pooling

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##===================

from __future__ import division
from optparse import OptionParser

import random
import pandas as pd
import os
import psutil
import re
import time
import sys

import torch
import torch.nn as nn

## import the local utility file within current directory
from Model_Only_Epigenomic import *
from UtilFunc import *

## debug variable 
debug_text = True

## memory mangement
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

##*******
## usually fixed set of parameters
##*******

OUT_CHANNEL_DIM = 128   ## output channel dimension
KERNEL_SIZE = 5     ## kernel size for convolutions
DROPOUT_RATE = 0    #0.1  #0.5  # dropout rate
activation_fn_name = 'gelu' #'relu'
LEARNING_RATE = 1e-3    #2e-3    ## 5e-4

## early stopping criteria - max 10 epoch for performance improvement
max_early_stopping = 10
## max epoch count
n_epochs = 200  #100
## batch size
batch_size = 1  ## 16

## loss functions
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
    
    parser.add_option('--Method', 
                      dest='MethodType', 
                      default=2, 
                      type='int', 
                      help='Method type (1: CNN + Maxpool, 2: Residual Net, 3: Transformer. Default = 2')    

    parser.add_option('--Offset', 
                      dest='Offset', 
                      default=2000000, 
                      type='int', 
                      help='Offset (middle portion / sliding window). Default = 2000000 (2 Mb), as suggested in the EpiGraphReg paper.')

    parser.add_option('-p', 
                      dest='ProjectedChannelDim', 
                      default=32, 
                      type='int', 
                      help='Projected input channel dimension, used to define the convolution. Default 32.')
    
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
    CAGEBinSize = int(options.CAGEBinSize)
    EpiBinSize = int(options.EpiBinSize)
    Resolution = int(options.CAGEBinSize)
    TrainDataDirList = options.TrainDataDirList.split(",")  ## parse input comma separated list
    Offset = int(options.Offset)    ## Sliding window - default = 2 Mb
    TrainModelDir = options.TrainModelDir     ## output training model directory
    ProjectedChannelDim = int(options.ProjectedChannelDim)
    os.makedirs(TrainModelDir, exist_ok = True)

    ##==========
    ## output files
    ##==========
    ## training model file name (to be created)    
    model_filename = TrainModelDir + '/Model_valid_chr_' + str(options.valid_chr) + '_test_chr_' + str(options.test_chr) + '.pt'
    print('*** output model_filename : ', str(model_filename))

    ## training model plot name
    model_plotname = model_filename.replace(".pt", "_plot.png")
    print('*** output model_plotname : ', str(model_plotname))

    ##=================
    ## derived parameters
    ##=================
    ## default = 6 Mb - 3 times sliding window
    Span = (3 * Offset) #int(options.Span)
    # T = Offset // Resolution    #400 
    num_slide_window_loop_bin = Offset // Resolution
    # N = Span // Resolution    #3*T
    num_span_loop_bin = Span // Resolution
    num_span_epi_bin = Span // EpiBinSize
    # b = Resolution // EpiBinSize   ## 50
    # ratio_loopbin_epibin = Resolution // EpiBinSize

    ## open output log file and print input parameters, output file names
    old_stdout = sys.stdout
    logfilename = TrainModelDir + "/out_valid_" + str(options.valid_chr) + "_test_" + str(options.test_chr) + ".log"
    log_file = open(logfilename, "w")
    sys.stdout = log_file

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
    print('\n refgenome : ' + str(refgenome))
    print('\n CAGEBinSize : ' + str(CAGEBinSize))
    print('\n EpiBinSize : ' + str(EpiBinSize))
    print('\n Resolution : ' + str(Resolution))
    print('\n Offset : ' + str(Offset))
    print('\n Span : ' + str(Span))
    print('\n num_slide_window_loop_bin : ' + str(num_slide_window_loop_bin))
    print('\n num_span_loop_bin : ' + str(num_span_loop_bin))
    print('\n num_span_epi_bin : ' + str(num_span_epi_bin))

    ##==========
    ## main code
    ##==========
    ## extract the validation and test chromosomes
    ## all other chromosomes are used for training
    valid_chr_list = re.split(r':|,|_', options.valid_chr)
    test_chr_list = re.split(r':|,|_', options.test_chr)    
    print('\n valid chromosomes : ' + str(options.valid_chr))
    print('\n test chromosomes : ' + str(options.test_chr))
    
    ##==============
    ## define the training, validation and test chromosomes
    ##==============
    if refgenome == 'mm9' or refgenome == 'mm10':
        train_chr_list = [c for c in range(1,1+19)]
    elif refgenome == 'hg19' or refgenome == 'hg38':
        train_chr_list = [c for c in range(1,1+22)]
    
    for j in range(len(valid_chr_list)):
        train_chr_list.remove(int(valid_chr_list[j]))
    for j in range(len(test_chr_list)):
        train_chr_list.remove(int(test_chr_list[j]))
        
    print('\n ==>>> Training chromosomes : ' + str(train_chr_list))

    ##==========
    ## training model parameters
    ##==========

    ## CNN related parameters 
    best_loss = 1e20
    best_rho = 0

    ## factor for l2 regularization - weight decay  
    if options.MethodType == 3:
        ## transformer model
        l2_reg = 1e-2
    else:
        ## CNN models
        l2_reg = 1e-5

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

    ##================
    ##=========== validation
    ##================
    def model_validate(device, 
                       out_model, chr_list, batch_size, 
                       num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, 
                       inpEpochCount, TrainDataDirList):

        ## loss and correlation for all CAGE bins
        loss_gat_all = np.array([])
        rho_gat_all = np.array([])

        ## data frame containing individual epoch / chunk wise statistics
        curr_Detailed_Stat_df = pd.DataFrame(columns=['Epoch', 'Data_Type', 'chromosome', 'chunk', 
                                                      'interval', 'window', 'loss', 'rho'])
        
        ## middle of the sliding window (where prediction would be assessed)
        idx = torch.arange(num_slide_window_loop_bin, 
                            2*num_slide_window_loop_bin).to(torch.int64)
        gather_idx = torch.unsqueeze(idx, 0).to(device)
                
        ##=== iterate through the training directories
        ##=== there can be more than one training directory
        ##=== each containing multiple chromosome information
        for TorchDataDir_Base_Valid in TrainDataDirList:
            if debug_text == True:
                print("\n ***** training directory (validation) : ", str(TorchDataDir_Base_Valid), " *** \n")
        
            ## iterate through individual chromosomes
            for currchr in chr_list:                
                ## directory containing Pytorch data (chunks) for the current chromosome
                CurrDataDir = TorchDataDir_Base_Valid + "/chr" + str(currchr)
                
                chunk_num = 0
                
                if debug_text == True:
                    print(f"\n ===>>> Processing chunks in chromosome : {currchr}", file=sys.stdout)
                sys.stdout.flush()

                ## define data loader for this folder (inherits from class DataLoader)
                testdataset = ReadFolderData(CurrDataDir)
                testdataloader = DataLoader(testdataset, 
                                            batch_size = batch_size, 
                                            shuffle = False)

                ## iterate through individual chunks of validation data
                for batch_idx, (data_exist, X_epi, Y, bin_idx) in enumerate(testdataloader):
                # for batch_idx, (X_epi, Y, bin_idx) in enumerate(testdataloader):
                    if data_exist:
                    # if 1:
                        
                        chunk_num = chunk_num + 1

                        current_batch_size = X_epi.size(0)  # This gives the actual batch size
                        
                        ## data transfer to CUDA if available
                        X_epi = X_epi.to(device)
                        Y = Y.to(device)

                        ##===== resize the input data: define input, output dimension and the channels
                        X_epi, Y, bin_idx = ResizeDataset(X_epi, Y, bin_idx, current_batch_size,
                                                        num_span_epi_bin, num_span_loop_bin)
                        
                        if debug_text == True:
                            print("\n ===>>> iterator - chunk : ", chunk_num, 
                                " - interval: [", bin_idx[0, 0].numpy(), " - ", (bin_idx[0, -1].numpy() + Resolution - 1), "]",
                                " - middle window: [", bin_idx[0, idx[0]].numpy(), " - ", (bin_idx[0, idx[-1]].numpy() + Resolution - 1), "]")

                        # if debug_text == True:
                        #     print(f"\n ===>>> iterator - batch : {batch_idx}")

                        ## during validation / testing, stop gradient computation
                        with torch.no_grad():
                            
                            ## apply the trained model on the input epigenomic data
                            if options.MethodType == 3:
                                ## for transformer, use only one output
                                Y_hat = out_model(X_epi, return_intermediate=False)
                            else:
                                ## for CNN based models, return all intermediate outputs
                                outs = out_model(X_epi, return_intermediate=True)
                                Y_hat = outs["final"]
                            
                            Y_hat = Y_hat.to(device)
                            Y_hat = torch.squeeze(Y_hat, dim=1)

                            ## extract the middle portion which is going to be benchmarked and used
                            Y_hat_idx = torch.gather(Y_hat, 1, gather_idx)
                            Y_idx = torch.gather(Y, 1, gather_idx)

                            ## loss estimation
                            if 0:
                                loss = poisson_loss(Y_idx, Y_hat_idx)
                            if 1:
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

                            ## check memory consumption
                            # if debug_text == True:
                            if 0:
                                print(" **** Current memory usage (Mb): ", 
                                        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

                            ## append in the detailed statistics
                            new_row = {'Epoch': inpEpochCount, 
                                    'TrainingDir': TorchDataDir_Base_Valid,
                                    'chromosome': currchr, 
                                    'chunk': chunk_num, 
                                    'interval': str(bin_idx[0, 0].numpy()) + " - " + str(bin_idx[0, -1].numpy() + Resolution - 1), 
                                    'window': str(bin_idx[0, idx[0]].numpy()) + " - " + str(bin_idx[0, idx[-1]].numpy() + Resolution - 1), 
                                    'loss': curr_loss,                                     
                                    'rho': curr_rho_gat}

                            ## add the detailed statistics         
                            if curr_Detailed_Stat_df.empty:
                                curr_Detailed_Stat_df = pd.DataFrame([new_row])
                            else:
                                curr_Detailed_Stat_df = pd.concat([curr_Detailed_Stat_df, 
                                                            pd.DataFrame([new_row])], 
                                                            ignore_index=True)

                    else:
                        break
        
        ## current epoch - validation loss: mean across all chromosomes and all chunks
        ## mean, ignoring the NaN values
        valid_loss = np.nanmean(loss_gat_all)
        valid_rho = np.nanmean(rho_gat_all)

        return valid_loss, valid_rho, curr_Detailed_Stat_df


    ##================
    ##======== training function
    ##================
    def model_train(device, out_model, 
                    train_chr_list, batch_size, 
                    num_slide_window_loop_bin, 
                    num_span_loop_bin, num_span_epi_bin, 
                    inpEpochCount, TrainDataDirList):

        ## loss and correlation for all CAGE bins
        loss_gat_all = np.array([])                
        rho_gat_all = np.array([])

        ## data frame containing individual epoch / chunk wise statistics
        curr_Detailed_Stat_df = pd.DataFrame(columns=['Epoch', 'Data_Type', 'chromosome', 
                                                      'chunk', 'interval', 'window', 
                                                      'loss', 'rho'])
        
        ## middle of the sliding window (where prediction would be assessed)
        idx = torch.arange(num_slide_window_loop_bin, 
                            2*num_slide_window_loop_bin).to(torch.int64)
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
                traindataset = ReadFolderData(CurrDataDir)
                traindataloader = DataLoader(traindataset, batch_size = batch_size, shuffle = False)

                ## iterate through individual chunks of training data
                for batch_idx, (data_exist, X_epi, Y, bin_idx) in enumerate(traindataloader):
                # for batch_idx, (X_epi, Y, bin_idx) in enumerate(traindataloader):
                    if data_exist:
                    # if 1:
                        
                        chunk_num = chunk_num + 1

                        current_batch_size = X_epi.size(0)  # This gives the actual batch size
                        
                        ## data transfer to CUDA if available
                        X_epi = X_epi.to(device)
                        Y = Y.to(device)

                        ##===== resize the input data: define input, output dimension and the channels
                        X_epi, Y, bin_idx = ResizeDataset(X_epi, Y, bin_idx, current_batch_size, 
                                                        num_span_epi_bin, num_span_loop_bin)
                        
                    
                        if debug_text == True:
                            print("\n ===>>> iterator - chunk : ", chunk_num, 
                                " - interval: [", bin_idx[0, 0].numpy(), " - ", (bin_idx[0, -1].numpy() + Resolution - 1), "]",
                                " - middle window: [", bin_idx[0, idx[0]].numpy(), " - ", (bin_idx[0, idx[-1]].numpy() + Resolution - 1), "]")

                        # if debug_text == True:
                        #     print(f"\n ===>>> iterator - batch : {batch_idx}")
                    
                        with torch.autograd.set_detect_anomaly(True):
                            
                            ## reset optimizer gradient
                            opt.zero_grad()
                            
                            ## invokes the function "call" in the trained model
                            if options.MethodType == 3:
                                ## for transformer, use only one output
                                Y_hat = out_model(X_epi, return_intermediate=False)
                            else:
                                ## for CNN based models, return all intermediate outputs
                                outs = out_model(X_epi, return_intermediate=True)
                                Y_hat = outs["final"]

                            Y_hat = Y_hat.to(device)                        
                            Y_hat = torch.squeeze(Y_hat, dim=1)
                            
                            ## extract the middle portion which is going to be benchmarked and used
                            Y_hat_idx = torch.gather(Y_hat, 1, gather_idx)
                            Y_idx = torch.gather(Y, 1, gather_idx)
                            
                            ## loss between the observed and predicted values
                            if 0:
                                loss = poisson_loss(Y_idx, Y_hat_idx)
                            if 1:
                                loss = huber_loss(Y_idx, Y_hat_idx)
                            
                            ## now apply backpropagation                            
                            loss.backward()       
                            if 0:
                                ## sourya - stop parameters from exploding
                                torch.nn.utils.clip_grad_norm_(out_model.parameters(), 0.5)     
                            opt.step()

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
                            
                            ## check memory consumption
                            # if debug_text == True:
                            if 0:
                                print(" **** Current memory usage (Mb): ", 
                                        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

                            ## append in the detailed statistics
                            new_row = {'Epoch': inpEpochCount, 
                                    'TrainingDir': TorchDataDir_Base,
                                    'chromosome': tc, 
                                    'chunk': chunk_num, 
                                    'interval': str(bin_idx[0, 0].numpy()) + " - " + str(bin_idx[0, -1].numpy() + Resolution - 1), 
                                    'window': str(bin_idx[0, idx[0]].numpy()) + " - " + str(bin_idx[0, idx[-1]].numpy() + Resolution - 1), 
                                    'loss': curr_loss,                                     
                                    'rho': curr_rho_gat}
                                                
                            ## add the detailed statistics         
                            if curr_Detailed_Stat_df.empty:
                                curr_Detailed_Stat_df = pd.DataFrame([new_row])
                            else:
                                curr_Detailed_Stat_df = pd.concat([curr_Detailed_Stat_df, 
                                                            pd.DataFrame([new_row])], 
                                                            ignore_index=True)
                    
                    else:
                        break
        
        ## current epoch - training loss: mean across all chromosomes and all chunks
        ## mean ignoring the NaN values
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
    out_model = FullModel(options.MethodType, 
                          OUT_CHANNEL_DIM,
                          KERNEL_SIZE,
                          ProjectedChannelDim, 
                          num_span_epi_bin, 
                          num_span_loop_bin, 
                          DownSample_Ratio_List, 
                          DROPOUT_RATE, 
                          activation_fn_name).to(device)

    ## compile the model
    if 0:
        ## later
        out_model = torch.compile(out_model, mode="reduce-overhead")
    
    ## check memory consumption
    if debug_text == True:
        print(" **** Current memory usage (Mb): ", 
                psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

    #############################
    ########## step 3: training ##########
    #############################

    ## sourya - later implement variable weighted step size based learning rate
    ## define optimizer - AdamW
    opt = torch.optim.AdamW(out_model.parameters(), 
                            lr=LEARNING_RATE, weight_decay=l2_reg)

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

        train_loss, train_rho, train_stat_df = \
            model_train(device, out_model, train_chr_list, batch_size, 
                        num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, 
                        epoch, TrainDataDirList)

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
                           epoch, TrainDataDirList)    

        ## add the detailed statistics         
        Detailed_Stat_df = pd.concat([Detailed_Stat_df, valid_stat_df], 
                                     axis = 0,
                                     ignore_index=True)
                
        if debug_text == True:
            print('\n ==>>> ***** epoch: ', epoch, 
                  ', mean valid_loss : ', valid_loss, 
                  ' mean valid_rho : ', valid_rho,
                  ' time passed: ', (time.time() - t0), ' sec') 
            
        ## check memory consumption
        # if debug_text == True:
        if 0:
            print(" **** Current memory usage (Mb): ", 
                  psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)            

        ##=========== condition - better model selection
        ##=========== based on validation data correlation

        if valid_rho > best_rho:              
            best_rho = valid_rho  
            
            ## update only if we found a better metric
            if valid_loss < best_loss:
                best_loss = valid_loss
            
            if debug_text == True:
                print('\n ***** Condition valid_rho > best_rho: *** current valid loss : ', valid_loss, 
                    ' current valid_rho : ', valid_rho,
                    '  existing best_loss : ', best_loss, 
                    '  existing best_rho : ', best_rho)
                    
            ## current model is selected
            ## reset early_stopping_counter
            early_stopping_counter = 1
            if options.MethodType == 3:
                ## for transformer model,
                ## save the model state dict, optimizer state dict
                torch.save({
                    'model_state_dict': out_model.state_dict(), 
                    'optimizer_state_dict': opt.state_dict()
                }, model_filename)
            else:                
                ## for CNN type model, save the complete model
                torch.save(out_model, model_filename)
                
            ## we generate the data frame summarizing all these measures
            new_row = {'Epoch': epoch, 
                       'Train_Loss': train_loss, 
                       'Train_Rho': train_rho, 
                       'Valid_loss': valid_loss, 
                       'Valid_rho': valid_rho, 
                       'best_loss': best_loss, 
                       'early_stopping_counter': early_stopping_counter, 
                       'Time': (time.time() - t0)}
            
            Stat_df = pd.concat([Stat_df, pd.DataFrame([new_row])], 
                                ignore_index=True)

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
            
            Stat_df = pd.concat([Stat_df, pd.DataFrame([new_row])], 
                                ignore_index=True)
            
            ## for the last "max_early_stopping" epochs, validation loss has not improved
            ## so there is no point in further training
            if early_stopping_counter == max_early_stopping:
                print("\n\n\n ******** \n The last update of validation loss (and corresponding training model) was done ", max_early_stopping, 
                      " iterations before \n So as the training model looks stable, in order to prevent overfitting, we are early terminating this model - current epoch: ", epoch, 
                      "  \n ******** \n\n\n")
                break
    
    ## finally write the epoch-specific mean statistics 
    Stat_df.to_csv(model_filename.replace(".pt", "_Complete_Stat.txt"), 
                   index=False)

    ## finally write the detailed statistics - all epochs and all chunks    
    Detailed_Stat_df.to_csv(model_filename.replace(".pt", "_Detailed_Epoch_Chunkwise_Stat.txt"), 
                            index=False)

    ## close the output log file
    sys.stdout = old_stdout
    log_file.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

