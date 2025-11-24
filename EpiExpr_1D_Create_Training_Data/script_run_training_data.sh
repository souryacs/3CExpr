#!/bin/bash

##==============
## Curate training data for EpiExpr-1D using Snakemake pipeline
##==============

## Resolution of epigenetic track
EPI_Res=100     ## bp

## Resolution of CAGE track
CAGE_Res=5000

## configuration file
configfile=./config/configfile.yaml

## pass the configuration file as a command line argument
snakemake \
    --rerun-triggers mtime \
    --rerun-incomplete \
    --cores \
    --jobs 150 \
    --resources mem_mb=100000 \
    --configfile $configfile \
    --latency-wait 120


