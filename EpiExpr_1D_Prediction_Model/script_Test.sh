#!/bin/bash

##==============
## Reference genome
RefGenome="hg19"

## EPI resolution
EPISIZE=100      ## bp

## CAGE resolution
CAGESIZE=5000    ## bp

## Blacklist file corresponding to the reference genome
BlackListFile="./Data/Blacklist_Genome/Blacklist/lists/hg19-blacklist.v2.bed"

## Sliding window (offset) value
Offset=2000000 

## Label of the experiment
Label="1D_GM12878_ENCFF915EIJ"

##===================
## Validation chromosomes
validchrlist="1_11"

## Test chromosomes
testchrlist="2_12"

## Base output directory
BaseOutDir="./Results/OUT_MODEL_EPI_"${EPISIZE}"bp_CAGE_"${CAGESIZE}"bp_1D_CNN"

## Directory containing TSS information
TSSDir="./Results/TSS/"${RefGenome}"/"${CAGESIZE}

## Projected channels (will be used as an input channel dimension of convolution)
## For EpigraphReg data, use 8 (not 6)
## For Basenji / Creator data, use 256
ProjChannel=8   #6

## Training model employed - 1: CNN, 2: residual net (recommended)
Trainmodel=2

## Directory containing training data
TrainingDataDir=${BaseOutDir}/TrainingData/Offset_${Offset}/${Label}

## Training model file name
ModelFileName=${BaseOutDir}"/TrainingModel/Offset_"${Offset}"/Model_"${Trainmodel}"/"${Label}"/Model_valid_chr_"${validchrlist}"_test_chr_"${testchrlist}".pt"

## Directory to contain output testing results
TestOutDir=${BaseOutDir}/TestModel/Offset_${Offset}/Model_${Trainmodel}/${Label}/valid_chr_${validchrlist}_test_chr_${testchrlist}

python3 ${code_path}/Test_Model_1D_CNN.py \
    --modelfile ${ModelFileName} \
    --Method ${Trainmodel} \
    -p ${ProjChannel} \
    -t ${testchrlist} \
    -C ${CAGESIZE} \
    -E ${EPISIZE} \
    --Offset ${Offset} \
    --TSSDir ${TSSDir} \
    -O ${TestOutDir} \
    -D ${TrainingDataDir}
