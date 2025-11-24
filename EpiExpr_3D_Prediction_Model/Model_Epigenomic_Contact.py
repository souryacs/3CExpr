##===================
## Pytorch model definition for predicting gene expression
## using both epigenomic tracks and chromatin contact

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##===================

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import GATv2Conv, TransformerConv

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
## Reading Epigenomic data with chromatin contacts
##*********

##==============
## defining dataset iterator
## reading all files / chunks for a particular chromosome
##==============
class ReadFolderData_CC(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.filenames = sorted(os.listdir(folder_path), key=sort_key)  # Sort files - 1.pkl, 2.pkl, ..., 
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
            edge_idx = currdata.edge_idx.to(torch.int64)
            edge_feat = currdata.edge_feat.to(torch.float32)            
        except Exception as e:
            data_exist = False
            X_1d = 0
            bin_idx = 0
            Y = 0
            edge_idx = 0
            edge_feat = 0

        ## return the data
        return data_exist, X_1d, Y, bin_idx, edge_idx, edge_feat


##===============
## resize the data
##===============
def ProcessInputData_CC(X_1d, Y, bin_idx, batch_size, num_span_epi_bin, 
                     num_span_loop_bin, edge_feat, EdgeFeatColList):   

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

    ## extract only the edge features mentioned in the columns of "EdgeFeatColList"
    ## Note: we need to subtract 6 since features start from column 6 in the original interaction file
    ## if EdgeFeatColList is not provided, keep the "edge_feat" as it is
    if len(EdgeFeatColList) > 0:
        target_cols = [x - 6 for x in EdgeFeatColList]
        edge_feat = edge_feat[:, target_cols]

    return X_1d, Y, bin_idx, edge_feat

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
## class definition of Downsampling encoder module
## downsamples the input resolution to CAGE track resolution    
## Depending on the input parameter PoolType, implements different encoder module
## PoolType = 1 - CNN + max pooling 
## PoolType = 2 - ResNet (uses the above defined ResNetBlock class)
#================
class ConvEnc(nn.Module):
    def __init__(self,
                 PoolType,
                 in_channel_dim, 
                 out_channel_dim, 
                 kernel_size,
                 DownSample_Ratio_List,                 
                 activation_fn = "relu",
                 dropout_rate=0.0):        

        super(ConvEnc, self).__init__()

        self.activation_fn = activation_fn
        self.DownSample_Ratio_List = DownSample_Ratio_List
        self.kernel_size = kernel_size

        if PoolType == 1:
            ##=============
            ## CNN + max pooling
            ##=============
            self.CNNModel = nn.Sequential()            
            for i in range(len(DownSample_Ratio_List)):
                if i == 0:
                    ## for the first iteration, check the input and output channel dimension                    
                    self.CNNModel = nn.Sequential(self.CNNModel, 
                                                  nn.Conv1d(in_channels = in_channel_dim, 
                                                            out_channels = out_channel_dim,
                                                            kernel_size = kernel_size, 
                                                            padding='same'))
                else:
                    ## for subsequent iterations, keep the input and output channel dimension same
                    self.CNNModel = nn.Sequential(self.CNNModel, 
                                                  nn.Conv1d(in_channels = out_channel_dim, 
                                                            out_channels = out_channel_dim,
                                                            kernel_size = kernel_size, 
                                                            padding='same'))   

                ## now perform max pooling (if applicable)
                if self.DownSample_Ratio_List[i] > 1:
                    self.CNNModel = nn.Sequential(self.CNNModel,  
                                                  nn.MaxPool1d(self.DownSample_Ratio_List[i]))

                ## perform batch normalization
                self.CNNModel = nn.Sequential(self.CNNModel, nn.BatchNorm1d(out_channel_dim))

                ## nonlinear activation
                if (activation_fn == "relu"):
                    self.CNNModel = nn.Sequential(self.CNNModel, nn.ReLU())
                elif (activation_fn == "gelu"):
                    self.CNNModel = nn.Sequential(self.CNNModel, nn.GELU())

                ## apply dropout, if dropout rate is nonzero
                if dropout_rate > 0:
                    self.CNNModel = nn.Sequential(self.CNNModel, nn.Dropout(p=dropout_rate))

        elif PoolType == 2:
            ##=============
            ## ResNet
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
                    nn.BatchNorm1d(32)
                )
                if (activation_fn == "relu"):
                    self.model_conv_start = nn.Sequential(self.model_conv_start, nn.ReLU())
                elif (activation_fn == "gelu"):
                    self.model_conv_start = nn.Sequential(self.model_conv_start, nn.GELU())

                channel_list = Define_Channels_per_block(len(self.DownSample_Ratio_List) - 1)
                hidden_ins = channel_list.copy()
                hidden_outs = channel_list.copy()
                hidden_ins.insert(0, 32)                ## first entry of input channel        
                hidden_outs.append(out_channel_dim)     ## last entry of output channel
            
            else:     
                ## currently not used - sourya
                ## output channels is constant - out_channel_dim = 256
                self.model_conv_start = nn.Sequential(
                    nn.Conv1d(in_channels = in_channel_dim, 
                            out_channels = out_channel_dim, 
                            kernel_size = kernel_size,  
                            padding='same'),
                    nn.BatchNorm1d(out_channel_dim),   
                )
                if (activation_fn == "relu"):
                    self.model_conv_start = nn.Sequential(self.model_conv_start, nn.ReLU())
                elif (activation_fn == "gelu"):
                    self.model_conv_start = nn.Sequential(self.model_conv_start, nn.GELU())
                
                hidden_ins = [out_channel_dim] * len(self.DownSample_Ratio_List)
                hidden_outs = [out_channel_dim] * len(self.DownSample_Ratio_List)
            
            ##========== define residual + scaling blocks (stride = 2)
            ##========== kernel size is maintained as 3
            self.ResidScaleBlock = self.ResidBlock(PoolType,
                                                   self.DownSample_Ratio_List, 
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
                nn.BatchNorm1d(out_channel_dim)
            )
            if (activation_fn == "relu"):
                self.model_conv_end = nn.Sequential(self.model_conv_end, nn.ReLU())
            elif (activation_fn == "gelu"):
                self.model_conv_end = nn.Sequential(self.model_conv_end, nn.GELU())        
            
            ## dropout
            if dropout_rate > 0:
                self.model_conv_end = nn.Sequential(self.model_conv_end, 
                                                    nn.Dropout(p=dropout_rate))        

            ## constructing the complete residual + scaling model
            self.CNNModel = nn.Sequential(self.model_conv_start, 
                                        self.ResidScaleBlock, 
                                        self.model_conv_end)
                
    ## function to assemble multiple residual blocks and return
    ## different blocks according to the parameter "PoolType"
    ## ResNet / ResNet with maxpooling
    def ResidBlock(self, PoolType, Downsample_List, his, hos, kernel_size, activation_fn):
        blk = []
        for i, ho, hi in zip(range(len(Downsample_List)), hos, his):        
            ## ResNet architecture
            blk.append(ResNetBlock(hi, ho, kernel_size, Downsample_List[i], activation_fn))
        res_blocks = nn.Sequential(*blk)
        return res_blocks

    def forward(self, x):
        if 0:
            print(" -->> Before starting ConvEnc - x shape : ", str(x.shape))
        x = self.CNNModel(x)
        if 0:
            print(" -->> After ConvEnc - x shape : ", str(x.shape))
        return x


#================
## class definition of GAT model
## Either Pytorch GAT module is used, or custom designed GAT is used
#================
class GATModel(nn.Module):
    def __init__(self,
                 ModelType,     ## GAT or Graph Transformer specific model type
                 channel_dim,   ## channel dimension: input and output
                 num_heads,     ## number of attention heads
                 num_layers,    ## number of GAT layers
                 num_edge_dim,  ## number of edge dimension - 0: no edge feature
                 residual_GAT = True,   
                 dropout_rate=0.0,
                 record_attn = True):

        super(GATModel, self).__init__()

        ## 1: GATv2 (Brody et al. 2021), 
        ## 2: Graph Transformer (unified message passing - Shi et al. 2020)
        ## 3. DotGatConv: GAT with dot product specific attention
        self.ModelType = ModelType    
        self.num_heads = num_heads
        self.channel_dim = channel_dim          
        self.num_edge_dim = num_edge_dim        
        self.num_layers = num_layers
        self.record_attn = record_attn
        self.residGAT = residual_GAT
        self.dropout_rate = dropout_rate  

        ## set of GAT / graph transformer layers - just define an empty array
        self.GATLayers = []

        ##===============
        ## self.ModelType = 1 --> Use Pytorch routine GATv2Conv - dynamic attention
        ## self.ModelType = 2 --> Use graph transformer routine TransformerConv (message passing) - pytorch
        ## self.ModelType = 3 --> GAT using dot product attention - DGL
        ##===============  
        if not self.residGAT:

            ## Residual / skip connection is not employed
            ## employ concat = True, to allow more channels

            ##====== add GAT / graph transformer layers
            for i in range(self.num_layers):
                ## determine the input channel dimension
                if i == 0:
                    ## first layer
                    curr_inp_channel_dim = self.channel_dim
                else:
                    ## subsequent layers
                    curr_inp_channel_dim = (self.channel_dim * self.num_heads)

                ## define the model
                if self.ModelType == 1:                
                    ## GATv2conv routine
                    if self.num_edge_dim == 0:
                        ##===== without edge features
                        curr_layer = GATv2Conv(in_channels = curr_inp_channel_dim, 
                                                out_channels = self.channel_dim, 
                                                heads = self.num_heads, 
                                                dropout = self.dropout_rate)
                                                    
                    else:
                        ##===== with edge features
                        curr_layer = GATv2Conv(in_channels = curr_inp_channel_dim, 
                                               out_channels = self.channel_dim, 
                                               heads = self.num_heads, 
                                               edge_dim = self.num_edge_dim, 
                                               dropout = self.dropout_rate)
                            
                elif self.ModelType == 2:
                    ## Graph Transformer using message passing
                    if self.num_edge_dim == 0:
                        ## without edge features
                        curr_layer = TransformerConv(in_channels = curr_inp_channel_dim,  
                                                     out_channels = self.channel_dim, 
                                                     heads = self.num_heads,
                                                     beta = False, 
                                                     dropout = self.dropout_rate)
                    else:
                        ## with edge features
                        curr_layer = TransformerConv(in_channels = curr_inp_channel_dim, 
                                                     out_channels = self.channel_dim,
                                                     heads = self.num_heads,
                                                     edge_dim = self.num_edge_dim, 
                                                     beta = False, 
                                                     dropout = self.dropout_rate)
                
                # elif self.ModelType == 3:
                #     ## GAT with dot product attention - DGL based
                #     ## does not support edge features
                #     curr_layer = DotGatConv(in_feats = curr_inp_channel_dim, 
                #                             out_feats = self.out_channel_dim,
                #                             num_heads = self.num_heads,
                #                             allow_zero_in_degree = True)

                ## append the layer in the final set of GAT/GT layers
                self.GATLayers.append(curr_layer)
                
        else:

            ## Residual / skip connection is employed
            ## employ concat = False, to easily add the residual connection

            ##====== add GAT / graph transformer layers
            for i in range(self.num_layers):

                ##===== define the model
                ##===== here we set concat = False 
                ## simplification: to add residual connection
                if self.ModelType == 1:                
                    ## GATv2conv routine
                    if self.num_edge_dim == 0:
                        ##===== without edge features
                        curr_layer = GATv2Conv(in_channels = self.channel_dim, 
                                            out_channels = self.channel_dim, 
                                            heads = self.num_heads, 
                                            dropout = self.dropout_rate,
                                            concat = False)
                                                    
                    else:
                        ##===== with edge features
                        curr_layer = GATv2Conv(in_channels = self.channel_dim, 
                                            out_channels = self.channel_dim, 
                                            heads = self.num_heads, 
                                            edge_dim = self.num_edge_dim, 
                                            dropout = self.dropout_rate,
                                            concat = False)
                            
                elif self.ModelType == 2:
                    ## Graph Transformer using message passing
                    if self.num_edge_dim == 0:
                        ## without edge features
                        curr_layer = TransformerConv(in_channels = self.channel_dim,    
                                                    out_channels = self.channel_dim,
                                                    heads = self.num_heads,
                                                    beta = False, 
                                                    dropout = self.dropout_rate,
                                                    concat = False)
                    else:
                        ## with edge features
                        curr_layer = TransformerConv(in_channels = self.channel_dim,    
                                                    out_channels = self.channel_dim,
                                                    heads = self.num_heads,
                                                    edge_dim = self.num_edge_dim, 
                                                    beta = False,
                                                    dropout = self.dropout_rate,
                                                    concat = False)

                ## append the layer in the final set of GAT/GT layers
                self.GATLayers.append(curr_layer)

        # ## define batch normalization - not employed
        # if not self.residGAT:
        #     self.bn1 = nn.BatchNorm1d(self.channel_dim * self.num_heads)
        # else:
        #     self.bn1 = nn.BatchNorm1d(self.channel_dim)

        ## define layer normalization 
        if not self.residGAT:
            self.ln1 = nn.LayerNorm(self.channel_dim * self.num_heads)
        else:
            self.ln1 = nn.LayerNorm(self.channel_dim)

        ##=========
        ## end add - sourya
        ##=========
    
    def forward(self, GAT_inp, edge_idx, device, edge_feat=None):        
        if 0:
            print("\n ==>> Within GATmodel - GAT_inp shape : ", str(GAT_inp.shape),
                  "  edge_idx shape : ", str(edge_idx.shape))
            
        ## GAT_inp shape: (Span/CAGEResolution, in_channel_dim)
        if edge_feat is not None:
            if 0:
                print(" ==>> edge_feat shape : ", str(edge_feat.shape))

        if self.ModelType == 1 or self.ModelType == 2:
            X_in = GAT_inp  ## initialize the input
            for i, layer in enumerate(self.GATLayers):                
                layer = layer.to(device)
                if self.num_edge_dim == 0:
                    out_, att_ = layer(X_in, 
                                       edge_index = edge_idx, 
                                       return_attention_weights = self.record_attn)
                else:
                    out_, att_ = layer(X_in, 
                                       edge_index = edge_idx, 
                                       edge_attr = edge_feat, 
                                       return_attention_weights = self.record_attn)                

                ## add layer normalization and activation - only for GATv2 model
                ## for GT model, no such transformation is required
                if self.ModelType == 1:
                    out_ = F.gelu(self.ln1(out_))
                
                if i < (self.num_layers - 1):
                    ## add the residual / skip connection if supported by input settings
                    if self.residGAT:                        
                        out_ = out_ + X_in
                    ## reset input for the next iteration
                    X_in = out_
                
        if 0:
            print("\n ==>> Exiting GATmodel - out_ shape : ", str(out_.shape))
        
        return out_, att_

#================
## class - Complete Model
## Convolution + GAT / GT
#================
class FullModel(nn.Module):
    def __init__(self,
                 ModelEPI,
                 Model3D,
                 out_channel_dim,
                 kernel_size,
                 fixed_in_channel_dim, 
                 num_layer_GAT,
                 num_head_GAT,
                 InitResidGAT,
                 num_edge_dim_GAT,
                 DownSample_Ratio_List,
                 cnn_dropout_rate=0.0,
                 gat_dropout_rate=0.0,
                 use_FCN=0,
                 activation_fn = "gelu", 
                 record_attn = True):
        
        super(FullModel, self).__init__()

        if activation_fn == "relu":
            self.activation_fn = nn.ReLU()
        elif activation_fn == "gelu":
            self.activation_fn = nn.GELU()

        self.ModelEPI = ModelEPI        ## 1: CNN 2: ResNet
        self.Model3D = Model3D          ## 1: GAT, 2: TransformerConv
        self.fixed_in_channel_dim = fixed_in_channel_dim
        self.out_channel_dim = out_channel_dim  
        self.kernel_size = kernel_size  
        self.DownSample_Ratio_List = DownSample_Ratio_List
        self.record_attn = record_attn
        self.num_layer_GAT = num_layer_GAT
        self.num_head_GAT = num_head_GAT
        self.num_edge_dim_GAT = num_edge_dim_GAT
        self.cnn_dropout_rate = cnn_dropout_rate
        self.gat_dropout_rate = gat_dropout_rate        
        self.resid_GAT = bool(InitResidGAT)     ## residual GAT parameter
        self.use_FCN = use_FCN

        ## To be defined dynamically
        ## Projection of input data onto fixed number of channels
        ## defined by LazyLinear since the input dimension is variable
        self.input_projection = nn.LazyLinear(self.fixed_in_channel_dim)  
        
        ##=============
        ## the 1D Epigenomic data specific model definition
        ##=============

        ##*************
        ## model type 1: CNN
        ## model type 2: residual net
        ##*************

        ##======= step 1: encoder
        self.Enc1_CAGE = ConvEnc(self.ModelEPI,
                                self.fixed_in_channel_dim, 
                                self.out_channel_dim, 
                                self.kernel_size,
                                self.DownSample_Ratio_List,                                 
                                activation_fn,
                                self.cnn_dropout_rate)
        
        ##========= step 2: 
        ##========= define the instance of GAT                         
        self.GATModel = GATModel(self.Model3D,
                                self.out_channel_dim,   ## employing same channel dimension in input and output
                                self.num_head_GAT,
                                self.num_layer_GAT,
                                self.num_edge_dim_GAT,
                                self.resid_GAT,                            
                                self.gat_dropout_rate,
                                self.record_attn)
                
        ##=========
        ## final output formation        
        ##=========

        ##=============
        ## decoder for CNN / ResNet
        ##=============
        self.cnn_Decoder = nn.Sequential(
            nn.Conv1d(in_channels =self.out_channel_dim,
                    out_channels = 64, 
                    kernel_size = 1,  ## filter size: 1
                    padding='same'), 
            nn.BatchNorm1d(64),
            self.activation_fn
        )

        ##=============
        ## decoder for GATv2
        ##=============

        if self.resid_GAT:
            ## residual / skip connection is employed - channel concat is False
            self.out_Decoder = nn.Sequential(
                nn.Conv1d(in_channels =self.out_channel_dim,
                        out_channels = 64, 
                        kernel_size = 1,  ## filter size: 1
                        padding='same'), 
                nn.BatchNorm1d(64),
                self.activation_fn
            )
        else:            
            ## residual / skip connection is not employed - channel concat is True
            self.out_Decoder = nn.Sequential(
                nn.Conv1d(in_channels =(self.out_channel_dim * self.num_head_GAT),
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

        ##=============
        ## decoder for GT - FFN - works better than Conv
        ##=============
        if self.resid_GAT:
            ## residual / skip connection is employed - channel concat is False
            self.out_FFN = nn.Sequential(
                nn.Linear(self.out_channel_dim, 1),
                nn.GELU()
            )
        else:
            ## residual / skip connection is not employed - channel concat is True
            self.out_FFN = nn.Sequential(
                nn.Linear((self.out_channel_dim * self.num_head_GAT), 1),
                nn.GELU()
            )

    def forward(self, x, edge_idx, edge_feat, device):
        
        x = x.to(device)
        edge_idx = edge_idx.to(device)
        edge_feat = edge_feat.to(device)

        batch_size, m_in, N = x.shape   ## input data shape
        if 0:
            print(f"Model - Input X shape : {x.shape}")
        
        if 0: 
            print(" Starting CNN + GAT model - input x shape : ", str(x.shape), 
                "  edge_idx shape : ", str(edge_idx.shape),
                "  edge_feat shape : ", str(edge_feat.shape))

        ##======== Step 0
        ##======== If input shape differs from projection channel dimension, 
        ##======== define the projection dynamically
        if m_in != self.fixed_in_channel_dim:                                    
            ## Project to fixed `fixed_in_channel_dim` channels
            x = x.permute(0, 2, 1)  # -> [batch, N, m_in]
            x = self.input_projection(x)  # batch, N, m_in] -> [batch, N, fixed_in_channel_dim]
            x = x.permute(0, 2, 1)  # [batch, fixed_in_channel_dim, N]

        ##**************
        ##======== Step 1: CNN / ResNet
        ## downsample the epigenomic data to CAGE resolution
        ##**************
        
        cnn_out = self.Enc1_CAGE(x)     ## cnn_out shape: (1, out_channel_dim, N2)
        cnn_out = cnn_out.to(device)
        if 0:
            print(" After Enc1_CAGE - cnn_out shape : ", str(cnn_out.shape))

        ##**************
        ##========= Step 2 - GAT / GT: node features: |V| * F_in, 
        ##========= where |V| = number of nodes, F_in = number of node features
        ##**************
        gat_inp = cnn_out.squeeze()
        gat_inp = torch.permute(gat_inp, (1, 0))        ## (N2, out_channel_dim)
        if 0:
            print(" Before starting GAT layers - gat_inp shape : ", str(gat_inp.shape))

        ##======= apply the GAT/GT operation
        if self.num_edge_dim_GAT == 0:
            gat_out, att_ = self.GATModel(gat_inp, edge_idx, device)
        else:
            gat_out, att_ = self.GATModel(gat_inp, edge_idx, device, edge_feat)

        ##======= generate the output data for subsequent processing
        gat_out = torch.permute(gat_out, (1, 0))
        gat_out = gat_out.unsqueeze(0)      ## [1, (out_channel_dim * num_heads), N2]

        if self.use_FCN:
            ## use fully connected network
            gat_out = gat_out.permute(0, 2, 1)    ## [1, N2, M]
            gat_out = self.out_FFN(gat_out)       ## [1, N2, 1]
            gat_out = gat_out.permute(0, 2, 1)    ## [1, 1, N2]
        else:
            gat_out = self.out_Decoder(gat_out)       ## [1, 64, N2]
            gat_out = self.CAGE_last(gat_out)         ## [1, 1, N2]

        ## return both output and the final attention scores
        return gat_out , att_


