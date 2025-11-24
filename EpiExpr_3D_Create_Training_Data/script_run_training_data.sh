#!/bin/bash

##==============
## EpiExpr-3D: using both 1D epigenomic information and 3D contact data
## Create training dataset
##==============

## Epigenomic track resolution
EPI_RES=100

## CAGE track resolution
CAGE_RES=5000

## configuration file
configfile="./config/configfile.yaml"

## pass the configuration file as a command line argument
## --rerun-incomplete means that the files marked as incomplete would be re-run
## --config run_name=${run_look} \
snakemake \
    --rerun-triggers mtime \
    --rerun-incomplete \
    --cores \
    --jobs 150 \
    --resources mem_mb=80000 \
    --configfile $configfile \
    --latency-wait 120


