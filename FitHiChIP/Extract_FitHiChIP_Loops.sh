#!/bin/bash

## Code to extract FitHiChIP interactions
CodeExec="./Extract_FitHiChIP_Loops.py"

## FitHiChIP loop resolution
BinSize=5000

## FitHiChIP lower distance threshold
LowDist=20000

## FitHiChIP higher distance threshold
UppDist=2000000

## FDR threshold to be applied on FitHiChIP loops
FDRThr=0.1

## FitHiChIP loop calls
## This file contains all the input chromatin interactions along with their FitHiChIP significance (FDR)
## without any filtering based on the FDR threshold
InpFile="./Results/HiChIP_Loops_hg19_FitHiChIP/GM12878_H3K27ac/FitHiChIP.interactions_FitHiC.bed"

## This directory would store the FitHiChIP loops in the format compatible for EpiExpr-3D
OutDir="./Loops_Input_EpiExpr/HiChIP/hg19"
mkdir -p $OutDir

## FitHiChIP loops compatible to EpiExpr-3D
LoopFile=${OutDir}/FitHiChIP_Input_EpiExpr.txt

## extract loops according to the specified FDR threshold
## and the associated features
python3 $CodeExec -I $InpFile -O $LoopFile -f $FDRThr

