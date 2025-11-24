#!/bin/bash

##==================
## script to download hg19 reference genome specific data 
## mentioned in the EpiGraphReg paper

## Follow the routines and adjust for downloading other datasets 
## such as hg38 data from Basenji paper
##==================

## Directory to store the downloaded data
Datadir='./Data'

## effective genome size
EGS_hg19=2864785220
EGS_hg38=2913022398
EGS_mm9=2620345972
EGS_mm10=2652783500

## blacklisted region
## Downloaded 
BlackListFile_hg19=${Datadir}"/Blacklist_Genome/hg19/ENCFF001TDO.bed"

## bin size for CAGE-seq data
## similar to the HiChIP loop resolution
Binsize_CAGE=5000	#4096

## bin size for other epigenomic tracks
Binsize_Other=100	#128

##==============================
## function to convert input BAM file to bigwig file
## using Deeptools "bamCoverage" routine
## and using the RPGC normalization
##==============================
BamToBigWigDeepToolsRPGC() {
	InpBamFile=${1}
	OutBWFile=${2}
	EGSVal=${3}
	binsize=${4}

	bamCoverageExec=`which bamCoverage`
	$bamCoverageExec -b ${InpBamFile} -o ${OutBWFile} -of bigwig -bs ${binsize} \
	--normalizeUsing RPGC \
	--effectiveGenomeSize ${EGSVal} -p max/2 --ignoreDuplicates
}

##========
## get ensemble GTF for reference genome
## and extract the TSS information
## output fields: "chr", "tss_1", "tss_2", "ens", "gene", "strand", "type"
##========

## hg19 reference genome - GTF annotation
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_42/GRCh37_mapping/gencode.v42lift37.basic.annotation.gtf.gz
gunzip gencode.v42lift37.basic.annotation.gtf.gz
RefGTFFile='gencode.v42lift37.basic.annotation.gtf'

OutDir=$Datadir'/GTF/hg19'
mkdir -p $OutDir
OutFile=$OutDir'/hg19_TSS.gtf'
if [[ ! -f $OutFile ]]; then
	awk -F'[\t| ]' '{if (substr($1,1,1)!="#") {for (i=0;i<=NF;i++) {if ($i=="gene_name") {N=i+1}; if ($i=="gene_id") {E=i+1}; if ($i=="gene_type") {T=i+1}}; print $1"\t"$4"\t"$5"\t"$E"\t"$N"\t"$7"\t"$T}}' $RefGTFFile | sort -k1,1 -k2,2n | uniq > $OutFile
	sed -i 's/;//g' $OutFile
	sed -i 's/\"//g' $OutFile
	tempfile=$OutDir'/temp.txt'
	awk -F'\t' '{if ($6=="+") {print $1"\t"$2"\t"$4"\t"$5"\t"$6"\t"$7} else {print $1"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7}}' $OutFile | sort -k1,1 -k2,2n | uniq > $tempfile
	mv $tempfile $OutFile
fi

##========
## download blacklist genome - hg19
##========
OutDir=${Datadir}'/Blacklist_Genome/hg19'
mkdir -p $OutDir
cd $OutDir
targetfile=$OutDir'/ENCFF001TDO.bed'
if [[ ! -f $targetfile ]]; then
	wget https://www.encodeproject.org/files/ENCFF001TDO/@@download/ENCFF001TDO.bed.gz
	gunzip ENCFF001TDO.bed.gz
fi

##========
## Tracks for GM12878 - hg19 reference genome
##========

##======= CAGE tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/GM12878/hg19/CAGE'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF915EIJ/@@download/ENCFF915EIJ.bam
samtools index ENCFF915EIJ.bam
wget https://www.encodeproject.org/files/ENCFF990KLZ/@@download/ENCFF990KLZ.bam
samtools index ENCFF990KLZ.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_CAGE}bp
mkdir -p $targetoutdir
for prefixstr in 'ENCFF915EIJ' 'ENCFF990KLZ'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_CAGE}
done 

fi 	## end hg19 reference genome specific files download

##======= DNase tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/GM12878/hg19/DNase'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF775ZJX/@@download/ENCFF775ZJX.bam
samtools index ENCFF775ZJX.bam
wget https://www.encodeproject.org/files/ENCFF783ZLL/@@download/ENCFF783ZLL.bam
samtools index ENCFF783ZLL.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_Other}bp
mkdir -p $targetoutdir
for prefixstr in 'ENCFF775ZJX' 'ENCFF783ZLL'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_Other}
done

fi 	## end hg19 reference genome specific files download

##======= H3K4me3 tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/GM12878/hg19/H3K4me3'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF818GNV/@@download/ENCFF818GNV.bigWig
wget https://www.encodeproject.org/files/ENCFF342CXS/@@download/ENCFF342CXS.bam
samtools index ENCFF342CXS.bam
wget https://www.encodeproject.org/files/ENCFF818UQV/@@download/ENCFF818UQV.bam
samtools index ENCFF818UQV.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_Other}bp
mkdir -p $targetoutdir

for prefixstr in 'ENCFF342CXS' 'ENCFF818UQV'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_Other}
done

fi 	## end hg19 reference genome specific files download


##======= H3K27ac tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/GM12878/hg19/H3K27ac'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF180LKW/@@download/ENCFF180LKW.bigWig
wget https://www.encodeproject.org/files/ENCFF197QHX/@@download/ENCFF197QHX.bam
samtools index ENCFF197QHX.bam
wget https://www.encodeproject.org/files/ENCFF882PRP/@@download/ENCFF882PRP.bam
samtools index ENCFF882PRP.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_Other}bp
mkdir -p $targetoutdir

for prefixstr in 'ENCFF197QHX' 'ENCFF882PRP'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_Other}
done

fi 	## end hg19 reference genome specific files download



##========
## Tracks for K562 - hg19 reference genome
##========

##======= CAGE tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/K562/hg19/EpiGraphReg_Paper/CAGE'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF623BZZ/@@download/ENCFF623BZZ.bam
samtools index ENCFF623BZZ.bam
wget https://www.encodeproject.org/files/ENCFF902UHF/@@download/ENCFF902UHF.bam
samtools index ENCFF902UHF.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_CAGE}bp
mkdir -p $targetoutdir
for prefixstr in 'ENCFF623BZZ' 'ENCFF902UHF'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_CAGE}
done 

fi 	## end hg19 reference genome specific files download

##======= DNase tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/K562/hg19/EpiGraphReg_Paper/DNase'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF826DJP/@@download/ENCFF826DJP.bam
samtools index ENCFF826DJP.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_Other}bp
mkdir -p $targetoutdir
for prefixstr in 'ENCFF826DJP'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_Other}
done

fi 	## end hg19 reference genome specific files download

##======= H3K4me3 tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/K562/hg19/EpiGraphReg_Paper/H3K4me3'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF689TMV/@@download/ENCFF689TMV.bigWig
wget https://www.encodeproject.org/files/ENCFF915MJO/@@download/ENCFF915MJO.bam
samtools index ENCFF915MJO.bam
wget https://www.encodeproject.org/files/ENCFF367FNL/@@download/ENCFF367FNL.bam
samtools index ENCFF367FNL.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_Other}bp
mkdir -p $targetoutdir

for prefixstr in 'ENCFF915MJO' 'ENCFF367FNL'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_Other}
done

fi 	## end hg19 reference genome specific files download


##======= H3K27ac tracks

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/K562/hg19/EpiGraphReg_Paper/H3K27ac'
mkdir -p $OutDir
cd $OutDir

wget https://www.encodeproject.org/files/ENCFF010PHG/@@download/ENCFF010PHG.bigWig
wget https://www.encodeproject.org/files/ENCFF384ZZM/@@download/ENCFF384ZZM.bam
samtools index ENCFF384ZZM.bam
wget https://www.encodeproject.org/files/ENCFF070PWH/@@download/ENCFF070PWH.bam
samtools index ENCFF070PWH.bam

targetoutdir=$OutDir'/Deeptools_RPGC_binsize_'${Binsize_Other}bp
mkdir -p $targetoutdir

for prefixstr in 'ENCFF384ZZM' 'ENCFF070PWH'; do
	BamToBigWigDeepToolsRPGC ${OutDir}/${prefixstr}.bam \
							${targetoutdir}/${prefixstr}.bw \
							${EGS_hg19} \
							${Binsize_Other}
done

fi 	## end hg19 reference genome specific files download

##======= JUND TF ChIP-seq

if [[ 1 == 1 ]]; then

OutDir=$Datadir'/K562/hg19/EpiGraphReg_Paper/JUND'
mkdir -p $OutDir
cd $OutDir

## fold change over control
wget https://www.encodeproject.org/files/ENCFF709JGL/@@download/ENCFF709JGL.bigWig

fi 	## end hg19 reference genome specific files download


