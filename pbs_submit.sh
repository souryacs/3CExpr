#!/bin/bash -ex
#PBS -l nodes=1:ppn=1
#PBS -l mem=10GB
#PBS -l walltime=20:00:00
#PBS -m ae
#PBS -j eo
#PBS -V
source ~/.bashrc
#source ~/.bash_profile
hostname
TMPDIR=/scratch
cd $PBS_O_WORKDIR

conda activate DeepLearning

##==============
## job submission
##==============

## path to the main script
code_path='/home/sourya/Code'
cd ${code_path}

## runID
run_look=GM12878_H3K27ac_HiChIP

## path to store the logs
log_path=${code_path}/logs/${run_look}
mkdir -p ${log_path}

## json file storing sequence of jobs for this run
jsonfile=${code_path}/cluster.json

snakemake --jobs 100 --latency-wait 120 --cluster-config $jsonfile --cluster "qsub -l {cluster.walltime} -l {cluster.cores} -l {cluster.memory} -m n -q default -e ${log_path}/{cluster.error} -o ${log_path}/{cluster.output}" --jobname 's.{rulename}.{jobid}' --stats ${log_path}/snakemake.stats >& ${log_path}/snakemake.log --config run_name=${run_look}
