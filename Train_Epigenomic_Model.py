##===================
## adapted from Epi-GraphReg.py script of the GraphReg package
## https://github.com/karbalayghareh/GraphReg

## modified 
## Sourya Bhattacharyya
##===================

from __future__ import division
from optparse import OptionParser

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import spearmanr

from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras import datasets, layers, models  
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf

## code within the current directory
## implements graph attention network
from gat_layer import GraphAttention

## import the local utility file within current directory
from UtilFunc_Epigenomic_Model import *

import os
import re
# import configparser
import yaml

## debug variable 
debug_text = True

##=============
## Training of the GAT model to predict gene expression from epigenomic data.
## We have set of 2 chromosomes for validation and 2 chromosomes for testing the model.
## User can alter the set of validation and test chromosomes, by specifying comma or colon separated list of chromosomes.
##=============

##===================
## parse options
##===================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-C', dest='configfile', default=None, type='str', help='Configuration file. Mandatory parameter.'),
    parser.add_option('-M', dest='Model', default="epi", type='str', help='Prediction model. Either seq or epi. Default = epi'),    
    parser.add_option('-v', dest='valid_chr', default='1,11', type='str', help="Comma separated list (numbers) of the validation chromosomes. Default 1,11 means that chr1 and chr11 would be used as the validation chromosomes."),
    parser.add_option('-t', dest='test_chr', default='2,12', type='str', help="Comma separated list (numbers) of the test chromosomes. Default 2,12 means that chr2 and chr12 would be used as the test chromosomes."),    
    parser.add_option('-n', dest='n_gat_layers', default=2, type='int', help='Number of graph attention layers. Default = 2.')

    (options, args) = parser.parse_args()
    return options, args

##===================
## main code
##===================
def main():
    options, args = parse_options()

    ## read the input configuration file
    # config = configparser.ConfigParser()
    # config.read(options.configfile)

    config_fp = open(options.configfile, "r")
    config = yaml.load(config_fp, Loader=yaml.FullLoader)

    refgenome = config['General']['Genome']
    BaseOutDir = config['General']['OutDir']
    Resolution = int(config['Loop']['resolution'])
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

    ##=================
    ## derived parameters
    ##=================
    # T = Offset // Resolution    #400 
    num_slide_window_loop_bin = Offset // Resolution

    # N = Span // Resolution    #3*T
    num_span_loop_bin = Span // Resolution

    num_span_epi_bin = Span // EpiBinSize

    # b = Resolution // EpiBinSize   ## 50
    ratio_loopbin_epibin = Resolution // EpiBinSize

    # feature dimension - number of Epigenomic tracks used in model
    # F = len(EpiTrackList)   ## 6   #3
    num_epitrack = len(EpiTrackList)

    ##==========
    ## training model parameters
    ##==========
    # output size of GraphAttention layer    
    F_ = 32
    # number of attention heads in GAT layers
    n_attn_heads = 4
    # dropout rate
    dropout_rate = 0.5
    # factor for l2 regularization
    l2_reg = 0.0
    re_load = False
    
    best_loss = 1e20
    max_early_stopping = 10
    n_epochs = 200

    ##==========
    ## main code
    ##==========
    ## extract the validation and test chromosomes
    ## all other chromosomes are used for training
    valid_chr_str = options.valid_chr.split(',')
    valid_chr = [int(valid_chr_str[i]) for i in range(len(valid_chr_str))]
    test_chr_str = options.test_chr.split(',')
    test_chr = [int(test_chr_str[i]) for i in range(len(test_chr_str))]

    n_gat_layers = int(options.n_gat_layers)

    print('\n\n ==>> Input Parameters <<<==== ')
    print('\n valid chromosomes : ', str(valid_chr))
    print('\n test chromosomes : ', str(test_chr))
    print('\n number of GAT layers: ', str(options.n_gat_layers))

    ##==============
    ## define the training, validation and test chromosomes
    ##==============
    if refgenome == 'mm9' or refgenome == 'mm10':
        train_chr_list = [c for c in range(1,1+19)]
    elif refgenome == 'hg19' or refgenome == 'hg38':
        train_chr_list = [c for c in range(1,1+22)]
    valid_chr_list = valid_chr
    test_chr_list = test_chr        
    vt = valid_chr_list + test_chr_list
    for j in range(len(vt)):
        train_chr_list.remove(vt[j])
    
    if debug_text == True:
        print('\n Training chromosomes : ', str(train_chr_list))

    ##======================
    ## process individual CAGE files
    ## and create trainining models separately for individual CAGE files
    ##======================
    for cagefileidx in range(len(CAGETrackList)):
        CurrCAGETrackFile = CAGETrackList[cagefileidx]
        CurrCAGETrackLabel = CAGELabelList[cagefileidx]
        print("\n\n *** Processing CAGE file : ", CurrCAGETrackFile, "  label : ", CurrCAGETrackLabel)

        ## output directory to store TFrecord
        ## specific to this CAGE sample
        TFRecordDir = BaseOutDir + "/" + SampleLabel + "/TFRecord/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel

        ## output directory to store training model
        ## specific to this CAGE sample
        TrainDir = BaseOutDir + "/" + SampleLabel + "/TrainingModel/contact_mat_FDR_" + str(FDRThr) + "/" + CurrCAGETrackLabel
        if not os.path.exists(TrainDir):
            os.makedirs(TrainDir)

        ## training model file name (to be created)
        model_filename_gat = TrainDir + '/Model_valid_chr_' + "_".join(str(s) for s in valid_chr_str) + '_test_chr_' + "_".join(str(s) for s in test_chr_str) + '.h5'
        print('\n *** output model_filename_gat : ', str(model_filename_gat))
        if (os.path.exists(model_filename_gat) == True):
            continue

        ## training model plot name
        model_plotname_gat = model_filename_gat.replace(".h5", "_plot_GAT.png")
        print('\n *** output model_plotname_gat : ', str(model_plotname_gat))

        ##=======================
        ## model related functions
        ## implemented within the main code
        ## to avoid repeated parameter passing
        ##=======================
        
        ## loss function
        def calculate_loss(model_gat, chr_list, TFRecordDir, batch_size, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack):
            
            loss_gat_all = np.array([])
            rho_gat_all = np.array([])
            Y_hat_all = np.array([])
            Y_all = np.array([])
            
            for i in chr_list:
                file_name = TFRecordDir + "/chr" + str(i) + '.tfr'
                print("\n ===>>> current chromosome : ", i, "  TFRecord file name : ", file_name)

                iterator = dataset_iterator(file_name, batch_size)
                while True:
                    data_exist, X_epi, Y, adj, idx, tss_idx = read_tf_record_1shot_train(iterator, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack)
                    if data_exist:
                        if tf.math.reduce_sum(tf.gather(tss_idx, idx)) > 0:
                            Y_hat, _ = model_gat([X_epi, adj])
                            Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                            Y_idx = tf.gather(Y, idx, axis=1)

                            loss = poisson_loss(Y_idx, Y_hat_idx)
                            loss_gat_all = np.append(loss_gat_all, loss.numpy())
                            if debug_text == True:
                                print("\n !!! curr_loss (put in loss_gat_all structure) : ", loss.numpy())                            
                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            
                            curr_rho_gat = np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1]
                            rho_gat_all = np.append(rho_gat_all, curr_rho_gat)
                            if debug_text == True:
                                print("\n !!! curr_rho_gat (put in rho_gat_all structure) : ", curr_rho_gat)

                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())
                    else:
                        break

            print('\n\n ==>>> len of test/valid Y: ', len(Y_all))
            valid_loss = np.mean(loss_gat_all)
            rho = np.mean(rho_gat_all)
            print('\n ==>>> valid_loss / test loss: ', valid_loss)
            print('\n ==>>> rho: ', rho)

            return valid_loss, rho

        ##===================
        # Model definition
        ##===================
        if re_load:
            model = tf.keras.models.load_model(model_filename_gat, custom_objects={'GraphAttention': GraphAttention})
            model.summary()
        else:

            ## clear the existing TF variables
            tf.keras.backend.clear_session()

            ## input feature dimension (for epigenomic tracks)        
            # X_in = Input(shape=(3*T*b, F))
            X_in = Input(shape=(num_span_epi_bin, num_epitrack))

            ## input label dimension - for HiC tracks
            # A_in = Input(shape=(N, N))
            A_in = Input(shape=(num_span_loop_bin, num_span_loop_bin))

            if debug_text == True:
                print("\n\n ==>> Starting deep learning model \n X_in (for epigenomic tracks) shape : " + str(X_in.shape) + "  A_in (for HiC tracks) shape : " + str(A_in.shape))

            ##==============
            ## process the epigenomic tracks using CNN
            ## the first "Conv1D" function uses "X_in" as the input
            ##==============        

            ## number of input features: X_in - here 60000
            ## number of output features: 128
            ## filter size = 25
            ## Note: input is X_in
            x = layers.Conv1D(128, 25, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(X_in)

            ## normalizes the layer
            x = layers.BatchNormalization()(x)

            ## downsample by 2 - so, the dimension becomes 30000 X 128
            x = layers.MaxPool1D(2)(x)

            ## applies dropout nodes, according to the specified "dropout_rate"
            x = Dropout(dropout_rate)(x)

            ## another convolutional layer
            ## number of output features: 128
            ## filter size = 3
            ## Note: input is x, i.e. the output of the last step        
            x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)

            ## normalizes the layer
            x = layers.BatchNormalization()(x)

            ## downsample by 5 - so, the dimension becomes 6000 X 128
            x = layers.MaxPool1D(5)(x)

            ## applies dropout nodes, according to the specified "dropout_rate"
            x = Dropout(dropout_rate)(x)

            ## another convolutional layer
            ## number of output features: 128
            ## filter size = 3
            ## Note: input is x, i.e. the output of the last step        
            x = layers.Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)

            ## normalizes the layer
            x = layers.BatchNormalization()(x)

            ## downsample by 5 - so, the dimension becomes 1200 X 128
            x = layers.MaxPool1D(5)(x)  

            ##======================
            ## process the chromatin interactions (HiC) using GAT
            ## the "GraphAttention" function uses "A_in" as the input
            ##======================
            att=[]
            for i in range(n_gat_layers):

                ## Input: layer x - here x.shape is not changed by the graph-attention model
                ## only its information is stored, and the attention + activation is computed
                ## (see the call routine)
                ## the second input is the HiC track information (A_in)
                x, att_ = GraphAttention(F_,
                            attn_heads=n_attn_heads,
                            attn_heads_reduction='concat',
                            dropout_rate=dropout_rate,
                            activation='elu',   ## note the activation - exponential linear unit (elu)
                            kernel_regularizer=l2(l2_reg),
                            attn_kernel_regularizer=l2(l2_reg))([x, A_in])

                ## the contents of x (after storing activation + weighted combination by attention)
                ## perform a normalization
                x = layers.BatchNormalization()(x)

                ## store the attention output
                ## the list "att" has a size of "n_gat_layers"
                att.append(att_)

            ## apply dropout nodes
            x = Dropout(dropout_rate)(x)

            ## another convolutional layer
            ## number of output features: 64
            ## so, the dimension of x becomes 1200 X 64
            ## filter size = 1
            x = layers.Conv1D(64, 1, activation='relu', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)

            ## perform a normalization
            x = layers.BatchNormalization()(x)

            ## final output - output number of features = 1
            ## input features = 1200, since dimension of x is 1200 X 64
            ## predicted CAGE - output dimension 1200 X 1
            mu_cage = layers.Conv1D(1, 1, activation='exponential', padding='same', kernel_regularizer=l2(l2_reg), bias_regularizer=l2(l2_reg))(x)

            ## as the last layer is of dimension 1, we reshape "mu_cage"
            ## so the output becomes a vector of length [3*T] which is 1200 here
            # mu_cage = layers.Reshape([3*T])(mu_cage)
            mu_cage = layers.Reshape([num_span_loop_bin])(mu_cage)

            ##===============
            ## Build model
            ## using both epigenomic tracks and HiC interactions
            ## both "X_in" and "A_in" inputs are used
            ##===============
            model_gat = Model(inputs=[X_in, A_in], outputs=[mu_cage, att])
            model_gat._name = 'Epigenome_3C_GAT'
            model_gat.summary()        
            if debug_text == True:
                print("\n\n ** Number of trainable variables in this model : ", len(model_gat.trainable_variables))
        
            ## store the model schematic in a plot
            ## commented for the moment
            if 1:            
                keras.utils.plot_model(model_gat, model_plotname_gat, show_shapes = True)

        
        ########## training ##########

        ## define optimizer
        ## now deprecated - modified - sourya
        if 0:
            opt = tf.keras.optimizers.Adam(learning_rate=.0002, decay=1e-6)
        else:
            opt = tf.keras.optimizers.Adam(learning_rate=.0002)
       
        batch_size = 1
        t0 = time.time()

        ## train for "n_epochs"
        for epoch in range(1,n_epochs+1):
            loss_gat_all = np.array([])
            rho_gat_all = np.array([])
            Y_hat_all = np.array([])
            Y_all = np.array([])
            if debug_text == True:
                print("\n\n ***** epoch : ", epoch, " *** \n\n")

            ## read the TF records for individual training chromosomes
            for i in train_chr_list:            
                file_name_train = TFRecordDir + "/chr" + str(i) + '.tfr'
                if debug_text == True:
                    print("\n\n ***** training with chromosome : ", str(i), " -- reading TF record file : ", file_name_train, " *** \n\n")
                
                chunk_num = 0
                ## training with respect to current chromosome
                iterator_train = dataset_iterator(file_name_train, batch_size)
                
                while True:
                    ## get the current chunk of data
                    ## idx: points to the middle chunk (2 Mb region)
                    data_exist, X_epi, Y, adj, idx, tss_idx = read_tf_record_1shot_train(iterator_train, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack)
                    if data_exist:
                        chunk_num = chunk_num + 1
                        if debug_text == True:                            
                            print("\n ===>>> Data iterator - read chunk : ", chunk_num)
                        
                        if tf.reduce_sum(tf.gather(tss_idx, idx)) > 0:

                            ## tf.GradientTape() enables remembering the gradients for differentiation
                            with tf.GradientTape() as tape:
                                ## invokes the function "call" in the GAT
                                ## the underscore indicates that we do not care about this value
                                Y_hat, _ = model_gat([X_epi, adj])
                                ## specifically gather the middle part of the predicted expression
                                ## (idx contains the indices of this middle part)
                                ## predicted output
                                Y_hat_idx = tf.gather(Y_hat, idx, axis=1)
                                ## original gene expression
                                Y_idx = tf.gather(Y, idx, axis=1)
                                ## loss estimation
                                loss = poisson_loss(Y_idx, Y_hat_idx)

                            ## apply gradients (differentiation) - backpropagation
                            grads = tape.gradient(loss, model_gat.trainable_variables)
                            opt.apply_gradients(zip(grads, model_gat.trainable_variables))

                            ## append loss information
                            curr_loss = loss.numpy()
                            loss_gat_all = np.append(loss_gat_all, curr_loss)
                            if debug_text == True:
                                print("\n !!! curr_loss (put in loss_gat_all structure) : ", curr_loss)

                            e1 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))
                            e2 = np.random.normal(0,1e-6,size=len(Y_idx.numpy().ravel()))

                            ## correlation between original and predicted expression
                            ## attenuated with random noise
                            curr_rho_gat = np.corrcoef(np.log2(Y_idx.numpy().ravel()+1)+e1,np.log2(Y_hat_idx.numpy().ravel()+1)+e2)[0,1]
                            rho_gat_all = np.append(rho_gat_all, curr_rho_gat)
                            if debug_text == True:
                                print("\n !!! curr_rho_gat (put in rho_gat_all structure) : ", curr_rho_gat)

                            ## predicted expression with respect to the middle part (idx)
                            Y_hat_all = np.append(Y_hat_all, Y_hat_idx.numpy().ravel())
                            
                            ## original expression with respect to the middle part (idx)
                            Y_all = np.append(Y_all, Y_idx.numpy().ravel())

                    else:
                        break
            
            if epoch == 1:
                print('\n\n ==>>> Epoch : ', epoch, ' --- len of train Y: ', len(Y_all))

            train_loss = np.mean(loss_gat_all)
            rho = np.mean(rho_gat_all)
            print('\n\n ==>>> ***** epoch: ', epoch, ', train loss: ', train_loss, ', train rho: ', rho, ', time passed: ', (time.time() - t0), ' sec')

            if epoch%1 == 0:            
                ## now run the trained model on the validation data
                ## and compute the loss function
                print('\n\n ==>>> Within the function calculate_loss for validation data <<=== \n\n') 
                valid_loss, valid_rho = calculate_loss(model_gat, valid_chr_list, TFRecordDir, batch_size, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack)
                print('\n\n ---- calculate_loss - valid_loss : ', valid_loss, ' valid_rho : ', valid_rho) 

            if valid_loss < best_loss:
                print('\n\n *** Condition valid_loss < best_loss ***') 
                early_stopping_counter = 1
                best_loss = valid_loss
                model_gat.save(model_filename_gat)
                print('epoch: ', epoch, ', valid loss: ', valid_loss, ', valid rho: ', valid_rho, ', time passed: ', (time.time() - t0), ' sec')

                print('\n\n ==>>> Within the function calculate_loss for test data <<=== \n\n') 
                test_loss,  test_rho = calculate_loss(model_gat, test_chr_list, TFRecordDir, batch_size, num_slide_window_loop_bin, num_span_loop_bin, num_span_epi_bin, ratio_loopbin_epibin, num_epitrack)
                print('epoch: ', epoch, ', test loss: ', test_loss, ', test rho: ', test_rho, ', time passed: ', (time.time() - t0), ' sec')

            else:
                early_stopping_counter += 1
                if early_stopping_counter == max_early_stopping:
                    break

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

