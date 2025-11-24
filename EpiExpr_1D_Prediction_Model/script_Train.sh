#!/bin/bash

##==============
## Reference genome
RefGenome="hg19"

## EPI resolution
EPISIZE=100 #128     ## bp

## CAGE resolution
CAGESIZE=5000   ##4096  ##128   ## bp

## Sliding window (offset) value
Offset=2000000

## Label of the experiment
Label="1D_GM12878_ENCFF915EIJ"

##==============
## Validation chromosomes
validchrlist="1_11"

## Test chromosomes
testchrlist="2_12"

## Base output directory
BaseOutDir="./Results/OUT_MODEL_EPI_"${EPISIZE}"bp_CAGE_"${CAGESIZE}"bp_1D_CNN"

## Training model employed - 1: CNN, 2: residual net (recommended)
Trainmodel=2

## Projected channels (will be used as an input channel dimension of convolution)
## For EpigraphReg data, use 8 (not 6)
## For Basenji / Creator data, use 256
ProjChannel=8   #6

## Directory containing training data
TrainingDataDir=${BaseOutDir}/TrainingData/Offset_${Offset}/${Label}

## Directory containing output training model
TrainingModelDir=${BaseOutDir}/TrainingModel/Offset_${Offset}/Model_${Trainmodel}/${Label}

python3 ./Train_Model_1D_CNN.py \
    -g ${RefGenome} \
    -v ${validchrlist} \
    -t ${testchrlist} \
    -D ${TrainingDataDir} \
    -O ${TrainingModelDir} \
    -C ${CAGESIZE} \
    -E ${EPISIZE} \
    --Method ${Trainmodel} \
    --Offset ${Offset} \
    -p ${ProjChannel} \



