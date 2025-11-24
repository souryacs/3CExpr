##===================
## Testing the model trained with only epigenomic tracks, to predict gene expression
## Model using CNN + Max pooling / Residual Net

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##===================

from __future__ import division
from optparse import OptionParser

# import math
import sys
import os
import re
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn

from scipy.stats import spearmanr

## include the training and testing model definition
from Model_Only_Epigenomic import *

## debug variable 
debug_text = True

OUT_CHANNEL_DIM = 128   ## output channel dimension
KERNEL_SIZE = 5     ## kernel size for convolutions
DROPOUT_RATE = 0    #0.1  #0.5  # dropout rate
activation_fn_name = 'gelu' #'relu'

batch_size = 1 #16

##===================
## parse options
##===================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)
        
    parser.add_option('--modelfile', 
                      dest='ModelFileName', 
                      default=None, 
                      type='str', 
                      help='Trained model file name. Mandatory parameter.')
    
    parser.add_option('--Method', 
                      dest='MethodType', 
                      default=2, 
                      type='int', 
                      help='Method type (1: CNN + Maxpool, 2: Residual Net, 3: Transformer. Default = 2') 

    parser.add_option('-p', 
                      dest='ProjectedChannelDim', 
                      default=32, 
                      type='int', 
                      help='Projected input channel dimension, used to define the convolution. Default 32.')
        
    parser.add_option('--Offset', 
                      dest='Offset', 
                      default=2000000, 
                      type='int', 
                      help='Offset (middle portion / sliding window). Default = 2000000 (2 Mb).')
    
    parser.add_option('--TSSDir', 
                      dest='TSSDir', 
                      default=None, 
                      type='str', 
                      help='Directory containing TSS information for the current resolution and reference genome. Mandatory parameter.')    
    
    parser.add_option('-D', 
                      dest='TestDataDir', 
                      default=None, 
                      type='str', 
                      help='Directory storing test data (for different chromosomes). Mandatory parameter.')
    
    parser.add_option('-t', 
                      dest='test_chr', 
                      default='2_12', 
                      type='str', 
                      help="Comma separated list (numbers) of the test chromosomes. Default 2_12 means that chr2 and chr12 would be used as the test chromosomes.")

    parser.add_option('-C', 
                      dest='CAGEBinSize', 
                      default=5000, 
                      type='int', 
                      help='CAGE bin size. Default 5000 (5 Kb)')
    
    parser.add_option('-E', 
                      dest='EpiBinSize', 
                      default=100, 
                      type='int', 
                      help='Epigenomic track bin size. Default 100 bp')

    parser.add_option('-O', 
                      dest='TestOutDir', 
                      default=None, 
                      type='str', 
                      help='Testing model output directory. Mandatory parameter.')    
        
    (options, args) = parser.parse_args()
    return options, args

##===================
## main code
##===================
def main():
    options, args = parse_options()

    print(f"PyTorch version: {torch.__version__}")
    ## check if CUDA device is available
    print("torch cuda available : ", torch.cuda.is_available())
    print("torch cuda device count : ", torch.cuda.device_count())
    print("torch cuda current device : ", torch.cuda.current_device())
    if torch.cuda.device_count() > 0:
        for i in range(torch.cuda.device_count()):
            print("torch cuda device ", i, " is ", torch.cuda.get_device_name(i))

    # setting device on GPU if available, else CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    ##=============
    ## configuration file parameters
    ##=============
    TestDataDir = options.TestDataDir
    TestOutDir = options.TestOutDir
    TSSDir = options.TSSDir ## directory storing TSS information
    CAGEBinSize = int(options.CAGEBinSize)
    EpiBinSize = int(options.EpiBinSize)    
    Offset = int(options.Offset)    ## Sliding window - default = 2 Mb    
    model_filename = options.ModelFileName
    MethodType = int(options.MethodType)
    ProjectedChannelDim = int(options.ProjectedChannelDim)

    os.makedirs(TestOutDir, exist_ok = True)
    ##=================
    ## derived parameters
    ##=================
    Span = (3 * Offset)     #int(options.Span)  ## default = 6 Mb - 3 times sliding window
    num_slide_window_loop_bin = Offset // CAGEBinSize
    num_span_loop_bin = Span // CAGEBinSize
    num_span_epi_bin = Span // EpiBinSize

    ##=============
    ## other parameters
    ##=============    
    
    ## set initilization seed
    ## both CPU and GPU
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)    ## all GPUs
    random.seed(seed)
    np.random.seed(seed)

    if 0:
        huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)

    ##=============
    ## extract the validation and test chromosomes
    ##=============
    test_chr_list = re.split(r':|,|_', options.test_chr)

    ##==========
    ## output folders and file names
    ##==========
    ## output file to store the performance summary (loss and correlation statistics) in a data frame
    ## for a specific test data
    FinalSummaryFile = TestOutDir + "/Summary_Metrics.txt"

    ## open output log file and print input parameters, output file names
    old_stdout = sys.stdout
    logfilename = TestOutDir + "/out.log"
    log_file = open(logfilename, "w")
    sys.stdout = log_file
    print('\n\n ==>> Input Parameters <<<==== ')
    print('\n CAGEBinSize : ', str(CAGEBinSize))
    print('\n test chromosomes : ', str(test_chr_list))
    print('\n\n *** training model file : ', str(model_filename))
    print('\n\n *** EpiBinSize : ', str(EpiBinSize))
    print('\n\n *** Offset : ', str(Offset))
    print('\n\n *** Span : ', str(Span))
    print('\n\n *** TSSDir : ', str(TSSDir))

    ##==========
    ## model testing function
    ##==========
    def model_test(device, TSSDir, TestDataDir, trained_model, test_chr_list, 
                   batch_size, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin):

        y_gene = np.array([])
        y_hat_gene_gat = np.array([])

        ## output contents
        out_chr = np.array([])
        out_bin_start = np.array([])
        out_bin_end = np.array([])
        out_bin_numTSS = np.array([])
        out_bin_TSSidxvec = np.array([])

        ## middle of the sliding window
        idx = torch.arange(num_slide_window_loop_bin, 
                            2*num_slide_window_loop_bin).to(torch.int64)
        gather_idx = torch.unsqueeze(idx, 0).to(device)
        
        ## process individual test chromosomes
        for testchr in test_chr_list:        
            if debug_text == True:
                print('\n\n ==>> Loop - Processing chunks in test chromosome : ', testchr)

            ## directory containing chunks for the current chromosome
            CurrDataDir = TestDataDir + "/chr" + str(testchr)

            ## TSS information file for the current chromosome
            TSS_info_filename = TSSDir + '/TSS_Info_chr' + str(testchr) + '.csv'
            TSSData = pd.read_csv(TSS_info_filename)

            chunk_num = 0

            ## define data loader for this folder (inherits from class DataLoader)
            testdataset = ReadFolderData(CurrDataDir)
            testdataloader = DataLoader(testdataset, batch_size = batch_size, shuffle = False)

            for batch_idx, (data_exist, X_epi, Y, bin_idx) in enumerate(testdataloader):

                if data_exist:
                    chunk_num = chunk_num + 1

                    X_epi = X_epi.to(device)
                    Y = Y.to(device)

                    ## resize the input data
                    X_epi, Y, bin_idx = ResizeDataset(X_epi, Y, bin_idx, batch_size, 
                                                      num_span_epi_bin, num_span_loop_bin)

                    ## extract the numTSS vector for this complete span
                    numTSSVec = torch.tensor(TSSData[(TSSData['bin_start'] >= bin_idx[0, 0].numpy()) & 
                                (TSSData['bin_start'] <= bin_idx[0, -1].numpy())]["numTSS"].values)
                    numTSSVec = torch.reshape(numTSSVec, (batch_size, num_span_loop_bin)).to(torch.int64)
                    numTSSVec = numTSSVec.to(device)

                    TSSData_mid = TSSData[(TSSData['bin_start'] >= bin_idx[0, idx[0]].numpy()) & 
                                (TSSData['bin_start'] <= bin_idx[0, idx[-1]].numpy())]
                    numTSS_total = np.sum(TSSData_mid["numTSS"].values)

                    if debug_text == True:
                        print("===>>> iterator - chunk : ", chunk_num, 
                            " - interval: [", bin_idx[0, 0].numpy(), " - ", (bin_idx[0, -1].numpy() + CAGEBinSize - 1), "]",
                            " - window: [", bin_idx[0, idx[0]].numpy(), " - ", (bin_idx[0, idx[-1]].numpy() + CAGEBinSize - 1), "]",
                            "  - num TSS : ", str(numTSS_total))

                    ## during validation or testing, we need to make sure to stop gradient computation
                    with torch.no_grad():

                        ## apply the trained model on the input epigenomic data
                        if MethodType == 3:
                            ## for transformer, use only one output
                            Y_hat = trained_model(X_epi, return_intermediate=False)
                        else:
                            ## for CNN based models, return all intermediate outputs
                            outs = trained_model(X_epi, return_intermediate=True)
                            Y_hat = outs["final"]

                        Y_hat = Y_hat.to(device)     ## , numTSSVec, device

                        Y_hat = torch.squeeze(Y_hat, dim=1)
                        Y_hat_idx = torch.gather(Y_hat, 1, gather_idx)
                        Y_idx = torch.gather(Y, 1, gather_idx)

                        ## loss estimation
                        ## here we use poisson loss (NLL) to compare with the epigraphReg paper
                        loss = poisson_loss(Y_idx, Y_hat_idx)
                        ## employ Huber loss between the observed and predicted CAGE
                        # loss = huber_loss(Y_idx, Y_hat_idx)

                        ## here no backpropagation is employed
                        curr_loss = loss.item()

                        Y_idx_copy = Y_idx.detach().clone().cpu()
                        Y_hat_idx_copy = Y_hat_idx.detach().clone().cpu()
                    
                        ## correlation between original and predicted expression
                        ## attenuated with random noise
                        e1 = np.random.normal(0,1e-6,size=len(Y_idx_copy.numpy().ravel()))
                        e2 = np.random.normal(0,1e-6,size=len(Y_idx_copy.numpy().ravel()))
                        curr_rho_gat = np.corrcoef(
                            np.log2(Y_idx_copy.numpy().ravel()+1)+e1, 
                            np.log2(Y_hat_idx_copy.numpy().ravel()+1)+e2
                            )[0,1]      
                        curr_rho_spearman = spearmanr(
                            np.log2(Y_idx_copy.numpy().ravel()+1)+e1, 
                            np.log2(Y_hat_idx_copy.numpy().ravel()+1)+e2)[0]
                                                                      
                        if debug_text == True:
                            print(" -- curr_loss : ", curr_loss,                                       
                                " curr_rho : ", curr_rho_gat,
                                " curr_rho_spearman : ", curr_rho_spearman)

                        y_gene = np.append(y_gene, 
                                           Y_idx_copy.numpy().ravel())  #Y_idx_copy)
                        y_hat_gene_gat = np.append(y_hat_gene_gat, 
                                                   Y_hat_idx_copy.numpy().ravel())  #Y_hat_idx_copy)
                        out_chr = np.append(out_chr, 
                                            TSSData_mid['chr'].values)
                        out_bin_start = np.append(out_bin_start, 
                                                  TSSData_mid['bin_start'].values)
                        out_bin_end = np.append(out_bin_end, 
                                                TSSData_mid['bin_end'].values)
                        out_bin_numTSS = np.append(out_bin_numTSS, 
                                                   TSSData_mid['numTSS'].values)
                        out_bin_TSSidxvec = np.append(out_bin_TSSidxvec, 
                                                      TSSData_mid['TSS_index_vec'].values)

                else:
                    break         
        
        return y_gene, y_hat_gene_gat, out_chr, out_bin_start, out_bin_end, out_bin_numTSS, out_bin_TSSidxvec

    ############################################################# 
    ## main code
    #############################################################

    ##==============
    ## initialize parameters
    ##==============
    valid_loss_gat = np.zeros([4])
    valid_rho_gat = np.zeros([4])
    valid_sp_gat = np.zeros([4])
   
    ## use only gene expression prediction output
    df_all_predictions = pd.DataFrame(columns=['chr', 'bin_start', 'bin_end', 
                                               'true_cage', 'pred_cage', 'numTSS', 
                                               'TSSIdxVec', 'nll'])

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

    ##==============
    ## load the training model
    ## we are only using the epigenomic tracks without the GAT and chromatin interactions
    ##==============
    if MethodType == 3:
        
        ## transformer based model
        trained_model = FullModel(MethodType, 
                                  OUT_CHANNEL_DIM, 
                                  KERNEL_SIZE, 
                                  ProjectedChannelDim,
                                  num_span_epi_bin, 
                                  num_span_loop_bin, 
                                  DownSample_Ratio_List, 
                                  DROPOUT_RATE, 
                                  activation_fn_name)
        
        ## also define optimizer
        opt = torch.optim.AdamW(trained_model.parameters())

        # Load the state_dict into the model
        checkpoint = torch.load(model_filename, weights_only=True)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
                
    else:
        ## applicable when the complete model is saved
        ## for CNN / residual net models
        trained_model = torch.load(model_filename)
        
    trained_model = trained_model.to(device)
    trained_model.eval()    ## evaluation mode

    ##==============
    ## execute the trained and loaded model on the test data
    ##==============
    y_gene, y_hat_gene_gat, out_chr, out_bin_start, out_bin_end, \
        out_bin_numTSS, out_bin_TSSidxvec = \
            model_test(device, 
                       TSSDir, 
                       TestDataDir,
                       trained_model, 
                       test_chr_list, 
                       batch_size, 
                       num_slide_window_loop_bin, 
                       num_span_loop_bin, 
                       num_span_epi_bin)

    ##=========
    ## Create data frame of the summary statistics
    df_tmp = pd.DataFrame(columns=['chr', 'bin_start', 'bin_end', 'true_cage', 'pred_cage', 
                                   'numTSS', 'TSSIdxVec', 'nll'])

    df_tmp['chr'] = out_chr
    df_tmp['bin_start'] = out_bin_start.astype(np.int64)
    df_tmp['bin_end'] = out_bin_end.astype(np.int64)
    df_tmp['true_cage'] = y_gene
    df_tmp['pred_cage'] = y_hat_gene_gat
    df_tmp['numTSS'] = out_bin_numTSS.astype(np.int64)
    df_tmp['TSSIdxVec'] = out_bin_TSSidxvec   
    df_tmp['nll'] = poisson_loss_individual(torch.tensor(y_gene), 
                                            torch.tensor(y_hat_gene_gat)).numpy()

    df_all_predictions = pd.concat([df_all_predictions, df_tmp], 
                                   ignore_index=True)    
    print("\n\n *** Complete number of rows in df_all_predictions : ", len(df_all_predictions))
    
    ## write the prediction to csv file
    df_all_predictions.to_csv(TestOutDir + '/df_all_predictions.txt', sep="\t", index=False)            
    
    ##==================
    ## define 3 gene sets for validation
    ## j = 0 - set 1: all genes - no expression and contact condition
    ## j = 1 - set 2: expressed genes (expression >= 1)
    ## j = 2 - set 3: expressed genes (expression >= 5)
    ## In the "df_all_predictions.csv" file, column "true_cage" show expression values

    for j in range(3):
        if j==0:
            min_expression = 0 
        elif j==1:
            min_expression = 1 
        elif j==2:
            min_expression = 5

        if debug_text == True:
            print("\n\n *** j : ", j, 
                  "  min_expression : ", min_expression)

        ## get the subset of data frame
        df_sub = df_all_predictions[(df_all_predictions['true_cage'] >= min_expression)]
        if debug_text == True:
            print("\n Complete data frame rows : ", len(df_all_predictions), 
                  " - subset data frame rows : ", len(df_sub))
                
        ## get the true and pred cage for this subset         
        true_cage_sub = df_sub['true_cage'].values
        pred_cage_sub = df_sub['pred_cage'].values

        ## metrics on validation chromosomes
        ## loss: poisson distribution
        ## rho and sp: (pearson) correlation between true and predicted gene expression
        valid_loss_gat[j] = poisson_loss(torch.tensor(true_cage_sub), 
                                         torch.tensor(pred_cage_sub)).numpy()
        valid_rho_gat[j] = np.corrcoef(np.log2(true_cage_sub+1),
                                       np.log2(pred_cage_sub+1))[0,1]
        valid_sp_gat[j] = spearmanr(np.log2(true_cage_sub+1),
                                    np.log2(pred_cage_sub+1))[0]
        # n_gene[j] = len(true_cage_sub)

        ## plot correlations between true and predicted CAGE values
        plotlabel = "Expr_" + str(min_expression)
        plotfile_logscale = TestOutDir + "/Scatter_Expr_" + str(min_expression) + "_log_scale.png"
        Plot_Corr_log2(plotfile_logscale, 
                        true_cage_sub, 
                        pred_cage_sub, 
                        plotlabel, 
                        valid_loss_gat[j], 
                        valid_rho_gat[j], 
                        valid_sp_gat[j])
        
        plotfile = TestOutDir + "/Scatter_Expr_" + str(min_expression) + ".png"
        Plot_Corr(plotfile, 
                  true_cage_sub, 
                  pred_cage_sub, 
                  plotlabel, 
                  valid_loss_gat[j], 
                  valid_rho_gat[j], 
                  valid_sp_gat[j])        


    ## Final results
    print('\n\n *** NLL GAT: ', valid_loss_gat, 
          ' rho: ', valid_rho_gat, 
          ' sp: ', valid_sp_gat)
    print('\n\n ****** Mean Loss GAT: ', np.mean(valid_loss_gat, axis=0), 
          ' +/- ', np.std(valid_loss_gat, axis=0), ' std')
    print('\n\n ****** Mean R GAT: ', np.mean(valid_rho_gat, axis=0), 
          ' +/- ', np.std(valid_rho_gat, axis=0), ' std')
    print('\n\n ****** Mean SP GAT: ', np.mean(valid_sp_gat, axis=0), 
          ' +/- ', np.std(valid_sp_gat, axis=0), ' std')

    ## close the output log file
    sys.stdout = old_stdout
    log_file.close()


    ##===================
    ## summary
    ##===================

    ## define the summary metric data frame
    SummaryDF = pd.DataFrame(
        data={'Test_chr': [options.test_chr], 
              'NLL_MinExpr_0': [valid_loss_gat[0]], 
              'NLL_MinExpr_1': [valid_loss_gat[1]], 
              'NLL_MinExpr_5': [valid_loss_gat[2]], 
              'rho_MinExpr_0': [valid_rho_gat[0]], 
              'rho_MinExpr_1': [valid_rho_gat[1]], 
              'rho_MinExpr_5': [valid_rho_gat[2]], 
              'sp_MinExpr_0': [valid_sp_gat[0]], 
              'sp_MinExpr_1': [valid_sp_gat[1]],
              'sp_MinExpr_5': [valid_sp_gat[2]]})
    SummaryDF = SummaryDF.reset_index(drop=True)

    ## write the summary metric data frame
    bool_file_exist = os.path.exists(FinalSummaryFile)
    if (bool_file_exist == False):
        SummaryDF.to_csv(FinalSummaryFile, header=True, mode="w", sep="\t", index=False)
    else:
        SummaryDF.to_csv(FinalSummaryFile, header=False, mode="a", sep="\t", index=False)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

