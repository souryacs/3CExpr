#!/bin/bash

##==============
## Reference genome
RefGenome="hg19"

## EPI resolution
EPISIZE=100 ## bp

## CAGE resolution
CAGESIZE=5000  ## bp

## Sliding window (offset) value
Offset=2000000 

## Label of the experiment
Label="GM12878_ENCFF915EIJ"

##==============
## Validation chromosomes
validchrlist="1_11"

## Test chromosomes
testchrlist="2_12"

## Projected channels (will be used as an input channel dimension of convolution)
## For EpigraphReg data, use 8 (not 6)
## For Basenji / Creator data, use 256
ProjChannel=8   #6

## GAT/GT layers
NumGatLayer=2

## GAT/GT heads
NumGATHeader=8

## FitHiChIP loop label: follow the configuration parameters
LoopLabel="FitHiChIP_P2A_b5000_L20000_U2000000_FDR0.1"

## Base output directory
BaseOutDir=${ProjDir}'/Results/OUT_MODEL_EPI_'${EPISIZE}'bp_CAGE_'${CAGESIZE}'bp_EpiExpr_3D'

## Graph edge normalization 
## 1: Scikit learn
## 2: Double stochastic
EdgeNorm=1

## Activation function
ActFn="gelu"

## Folder containing training data
currTrainingDataFolder=${BaseOutDir}/TrainingData/Offset_${Offset}/EdgeNorm_${EdgeNorm}/${LoopLabel}/${Label}

## 3D model type: 1: GAT, 2: GT
Modeltype_3D=2

## Using initial residual connection (1) or not (0)
ResidGAT=1

## output directory to store the training model
TrainModelDir=${BaseOutDir}"/TrainingModel"
mkdir -p $TrainModelDir

## output training model file
trainmodelfile=${TrainModelDir}/Model_valid_chr_${validchrlist}_test_chr_${testchrlist}.pt

## Invoke the script
python3 ${code_path}/Train_Model_Epi_plus_3D.py \
    -g ${RefGenome} \
    -v ${validchrlist} \
    -t ${testchrlist} \
    -D ${currTrainingDataFolder} \
    -O ${TrainModelDir} \
    -C ${CAGESIZE} \
    -E ${EPISIZE} \
    -R ${ResidGAT} \
    --Model3D ${Modeltype_3D} \
    --NumGATLayer ${NumGatLayer} \
    --NumHeadGAT ${NumGATHeader} \
    --Offset ${Offset} \
    -p ${ProjChannel} \
    --ActFun ${ActFn}


