##===================
## Pytorch model definition for predicting gene expression using only epigenomic tracks
## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##===================

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

## import the local utility file within current directory
from UtilFunc import *

##==========
## utility function
## defines the channels per block for encoder layer
def Define_Channels_per_block(Tot_Block):
    channel_count_list = []
   
    while Tot_Block > 0:
        if Tot_Block > 0:
            if 64 in channel_count_list:
                channel_count_list.insert(channel_count_list.index(64), 32)
            else:
                channel_count_list.append(32)
            Tot_Block = Tot_Block - 1
        if Tot_Block > 0:
            if 128 in channel_count_list:
                channel_count_list.insert(channel_count_list.index(128), 64)
            else:
                channel_count_list.append(64)
            Tot_Block = Tot_Block - 1
        if Tot_Block > 0:
            if 256 in channel_count_list:
                channel_count_list.insert(channel_count_list.index(256), 128)
            else:
                channel_count_list.append(128)
            Tot_Block = Tot_Block - 1
        if Tot_Block > 0:
            channel_count_list.append(256)
            Tot_Block = Tot_Block - 1
    
    return channel_count_list


##*********
## Reading Epigenomic data without chromatin contacts
##*********

##==============
## defining dataset iterator
## reading all files / chunks for a particular chromosome
##==============
class ReadFolderData(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        # Sort files - 1.pkl, 2.pkl, ..., 
        self.filenames = sorted(os.listdir(folder_path), key=sort_key)  
        # self.filenames = os.listdir(folder_path)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.folder_path, filename)
        if 0:
            print(" -->> data loader - reading file : ", filepath)        
        try:
            ## read the file 
            with open(filepath, "rb") as f:
                currdata = pickle.load(f)            
            data_exist = True            
            X_1d = currdata.X_1d.to(torch.float32)
            bin_idx = currdata.bin_idx.to(torch.int64)
            Y = currdata.Y.to(torch.float32)
        except Exception as e:
            data_exist = False
            X_1d = 0
            bin_idx = 0
            Y = 0

        ## return the data
        return data_exist, X_1d, Y, bin_idx 
        
##===============
## resize the data
##===============
def ResizeDataset(X_1d, Y, bin_idx, batch_size, num_span_epi_bin, num_span_loop_bin):

    ##===== Epigenetic data
    ##===== reshape in the format: 1 X N X C
    ##===== where 1: batch size, N = number of epigenomic bins, C = number of epigenomic tracks
    X_1d = torch.reshape(X_1d, (batch_size, num_span_epi_bin, -1)).to(torch.float32)
    
    ##========= Reshape X_1d in the format 1 X C X N (swap axes 1 and 2 )
    ## this is because, in Pytorch, number of channels is in the axis 1
    ## and here we are modeling the number of epigenetic tracks as the channels
    X_1d = torch.permute(X_1d, (0,2,1))

    ##======== both bin_idx and Y follow the CAGE resolution
    ##======== Dimension: 1 X N', where N' : number of CAGE bins
    bin_idx = torch.reshape(bin_idx, (batch_size, num_span_loop_bin)).to(torch.int64)
    Y = torch.reshape(Y, (batch_size, num_span_loop_bin)).to(torch.float32)
    
    return X_1d, Y, bin_idx

##================
## ResNet architecture - class definition of residual + scaling encoder block
## so far we have not put any dropout, employed stride = 2
## Lazy functions determine the input channel dimensions from the input data
##================
class ResNetBlock(nn.Module):
    def __init__(self, 
                hidden_in,     ## input channel dimension
                hidden_out,    ## output channel dimension
                kernel_size, 
                stride_val,
                activation_fn = "relu"):
                
        super(ResNetBlock, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation_fn = nn.GELU()

        self.pad_len = int(kernel_size / 2)

        ##========= scaling block - convolution + stride = stride_val to achieve downsampling
        if stride_val > 1:
            self.conv1 = nn.Conv1d(hidden_in,
                                hidden_out, 
                                kernel_size=kernel_size, 
                                padding=self.pad_len,
                                stride=stride_val)
        else:
            self.conv1 = nn.Conv1d(hidden_in,
                                hidden_out, 
                                kernel_size=kernel_size, 
                                padding=self.pad_len)
        
        ##========= convolutional block - no downsampling
        self.conv2 = nn.Conv1d(hidden_out,
                               hidden_out, 
                               kernel_size=kernel_size, 
                               padding=self.pad_len)
        
        ##======== downsampling, using stride = stride_val, kernel_size = 1
        if stride_val > 1:
            self.scale_no_act = nn.Conv1d(hidden_in,
                                    hidden_out, 
                                    kernel_size=1,
                                    stride=stride_val,
                                    padding="valid")
        else:
            self.scale_no_act = nn.Conv1d(hidden_in,
                                    hidden_out, 
                                    kernel_size=1,
                                    padding="valid")

        ##======= batch normalization 
        self.bn1 = nn.BatchNorm1d(hidden_out)
        self.bn2 = nn.BatchNorm1d(hidden_out)

        ##====== define the consolidated modules
        ##====== 1. scale + convlution block
        ##====== we should have one scaling layer (conv1) and one convolution layer (conv2)
        ##====== here we employed one scaling layer, three convolution layers - following ResNet18 architecture        

        ## ResNet18 model
        if 1:
            self.scale_conv_block_unit = nn.Sequential(self.conv1, self.bn1, self.activation_fn, 
                                                    self.conv2, self.bn2, self.activation_fn, 
                                                    self.conv2, self.bn2, self.activation_fn, 
                                                    self.conv2, self.bn2)
            
        ## decreasing number of convolutions     
        if 0:
            self.scale_conv_block_unit = nn.Sequential(self.conv1, self.bn1, self.activation_fn, 
                                                    self.conv2, self.bn2)
    
    def forward(self, X): 
        identity = self.scale_no_act(X)  # Skip connection
        out = self.scale_conv_block_unit(X)
        out += identity     # Adding the skip connection
        out = self.activation_fn(out)
        return out

#================
## class definition of Encoder module: using 1D CNN / ResNet models (multiple layers of CNN)
#================
class ConvEnc(nn.Module):
    def __init__(self,
                 ModelType,
                 in_channel_dim, 
                 out_channel_dim, 
                 kernel_size,
                 DownSample_Ratio_List,                 
                 activation_fn = "relu",
                 dropout_rate = 0.0):        

        super(ConvEnc, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation_fn = nn.GELU()

        self.DownSample_Ratio_List = DownSample_Ratio_List
        self.kernel_size = kernel_size

        if ModelType == 1:
            ##=============
            ## CNN + max pooling + downsampling
            ##=============
            self.CNNModel = nn.Sequential()            
            for i in range(len(DownSample_Ratio_List)):
                if i == 0:
                    ## first iteration: check the input and output channel dimension 
                    self.CNNModel = nn.Sequential(self.CNNModel, 
                                                  nn.Conv1d(in_channels = in_channel_dim, 
                                                            out_channels = out_channel_dim,
                                                            kernel_size = kernel_size, 
                                                            padding='same'))
                else:
                    ## subsequent iteration: keep the input and output channel dimensions identical
                    self.CNNModel = nn.Sequential(self.CNNModel, 
                                                  nn.Conv1d(in_channels = out_channel_dim, 
                                                            out_channels = out_channel_dim,
                                                            kernel_size = kernel_size, 
                                                            padding='same'))   
                    
                ## perform batch normalization + nonlinear activation
                self.CNNModel = nn.Sequential(self.CNNModel, 
                                              nn.BatchNorm1d(out_channel_dim),
                                              self.activation_fn)

                ## perform max pooling (if applicable)
                if self.DownSample_Ratio_List[i] > 1:
                    self.CNNModel = nn.Sequential(self.CNNModel,  
                                                  nn.MaxPool1d(self.DownSample_Ratio_List[i]))

                ## apply dropout, if dropout rate is nonzero
                if dropout_rate > 0:
                    self.CNNModel = nn.Sequential(self.CNNModel, 
                                                  nn.Dropout(p=dropout_rate))

        elif ModelType == 2:
            ##=============
            ## CNN + ResNet + downsampling
            ##=============

            ##===== first convolution layer
            ##===== And also define the residual blocks for downsampling
            if 1:
                ## dynamic number of output channels according to the iterations
                ## from 32 to 256            
                self.model_conv_start = nn.Sequential(
                    nn.Conv1d(in_channels = in_channel_dim, 
                            out_channels = 32,
                            kernel_size = kernel_size, 
                            padding='same'),
                    nn.BatchNorm1d(32),
                    self.activation_fn
                )
                channel_list = Define_Channels_per_block(len(self.DownSample_Ratio_List) - 1)
                hidden_ins = channel_list.copy()
                hidden_outs = channel_list.copy()
                hidden_ins.insert(0, 32)                ## first entry of input channel        
                hidden_outs.append(out_channel_dim)     ## last entry of output channel
            
            else:     
                ## output channels is constant - out_channel_dim (input parameter)
                self.model_conv_start = nn.Sequential(
                    nn.Conv1d(in_channels = in_channel_dim, 
                            out_channels = out_channel_dim, 
                            kernel_size = kernel_size,  
                            padding='same'),
                    nn.BatchNorm1d(out_channel_dim), 
                    self.activation_fn
                )
                hidden_ins = [out_channel_dim] * len(self.DownSample_Ratio_List)
                hidden_outs = [out_channel_dim] * len(self.DownSample_Ratio_List)
            
            ##========== define residual + scaling blocks 
            ##========== using the "DownSample_Ratio_List"
            self.ResidScaleBlock = self.ResidBlock(self.DownSample_Ratio_List, 
                                                   hidden_ins, 
                                                   hidden_outs, 
                                                   self.kernel_size, 
                                                   activation_fn)

            ##===== last convolution layer
            self.model_conv_end = nn.Sequential(
                nn.Conv1d(in_channels = out_channel_dim, 
                        out_channels = out_channel_dim,
                        kernel_size = 1,
                        padding='same'),
                nn.BatchNorm1d(out_channel_dim),
                self.activation_fn
            )
            
            if dropout_rate > 0:
                self.model_conv_end = nn.Sequential(self.model_conv_end, 
                                                    nn.Dropout(p=dropout_rate))        

            ## constructing the complete residual + scaling model
            self.CNNModel = nn.Sequential(self.model_conv_start, 
                                        self.ResidScaleBlock, 
                                        self.model_conv_end)
        
    ## function to assemble multiple residual blocks and return
    ## ResNet / ResNet with maxpooling
    def ResidBlock(self, Downsample_List, his, hos, kernel_size, activation_fn):
        blk = []
        for i, ho, hi in zip(range(len(Downsample_List)), hos, his):        
            ## ResNet architecture
            blk.append(ResNetBlock(hi, ho, kernel_size, 
                                    Downsample_List[i], activation_fn))
        res_blocks = nn.Sequential(*blk)
        return res_blocks

    def forward(self, x):
        return self.CNNModel(x)

#================
## class - Complete Model
#================
class FullModel(nn.Module):
    def __init__(self,
                 ModelType,
                 out_channel_dim,
                 kernel_size,                 
                 fixed_in_channel_dim, 
                 epi_seq_len, 
                 cage_seq_len,
                 DownSample_Ratio_List,
                 dropout_rate=0.0,
                 activation_fn = "relu"):        

        super(FullModel, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation_fn = nn.GELU()

        self.ModelType = ModelType
        self.fixed_in_channel_dim = fixed_in_channel_dim
        self.out_channel_dim = out_channel_dim  #256  #128  ## output channel dimension - fixed
        self.kernel_size = kernel_size  #5        ## kernel size for convolution - fixed
        self.DownSample_Ratio_List = DownSample_Ratio_List
        self.epi_seq_len = epi_seq_len
        self.cage_seq_len = cage_seq_len                

        ## To be defined dynamically
        ## Projection of input data onto fixed number of channels
        ## defined by LazyLinear since the input dimension is variable
        self.input_projection = nn.LazyLinear(self.fixed_in_channel_dim)  
        
        ##=============
        ## the model definition
        ##=============

        ##*************
        ## model type 1: CNN
        ## model type 2: residual net
        ##*************
        if self.ModelType == 1 or self.ModelType == 2:

            ##======= step 1: encoder
            self.Enc1_CAGE = ConvEnc(self.ModelType,
                                    self.fixed_in_channel_dim, 
                                    self.out_channel_dim, 
                                    self.kernel_size,
                                    self.DownSample_Ratio_List,                                 
                                    activation_fn,
                                    dropout_rate)
                
            ##=========
            ## final output formation        
            self.out_Decoder = nn.Sequential(
                nn.Conv1d(in_channels = self.out_channel_dim,
                        out_channels = 64, 
                        kernel_size = 1,  ## filter size: 1
                        padding='same'), 
                nn.BatchNorm1d(64),
                self.activation_fn
            )

            ## final CAGE expression layer - convolution
            self.CAGE_last = nn.Sequential(
                nn.Conv1d(in_channels = 64,
                        out_channels = 1, 
                        kernel_size = 1, 
                        padding='same'),
                nn.ELU()
            )


    def forward(self, x, return_intermediate=True):

        batch_size, m_in, N = x.shape   ## input data shape
        if 1:
            print(f"Model - Input X shape : {x.shape}")
        
        ##==========
        ## for CNN / ResNet models
        ##==========
        if self.ModelType == 1 or self.ModelType == 2:

            ## If input shape differs from projection channel dimension, 
            ## define the projection dynamically
            if m_in != self.fixed_in_channel_dim:                                    
                ## Project to fixed `fixed_in_channel_dim` channels
                x = x.permute(0, 2, 1)  # Change shape to [batch, N, m_in]
                x = self.input_projection(x)  # Project: [batch, N, m_in] → [batch, N, fixed_in_channel_dim]
                x = x.permute(0, 2, 1)  # Back to shape [batch, fixed_in_channel_dim, N]            
        
            ## out1 shape: (batch, out_channel_dim (constant = 256), Span/CAGEResolution)
            out1 = self.Enc1_CAGE(x)     

            ## use only if ConvEnc is used, and not ConvEnc2D
            ## final stage decoder        
            ## only if convolution + downsampling was performed
            out2 = self.out_Decoder(out1)             ## out shape: [batch, 64, Span/CAGEResolution]
            out = self.CAGE_last(out2)               ## out shape: [batch, 1, Span/CAGEResolution]

            ## clip the output - all negative entries are 0's
            torch.nn.functional.relu(out, inplace=True)

        ## for CNN based models, return intermediate outputs as well
        if return_intermediate and self.ModelType != 3:
            return {"encoder": out1, "decoder1": out2, "final": out}

        ## final output if no intermediate outputs are returned
        return out

##======================
## end model related class definitions
##======================

