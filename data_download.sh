#!/bin/bash 

Datadir='/home/sourya/Data'

## effective genome size
EGS_hg19=2864785220
EGS_hg38=2913022398
EGS_mm9=2620345972
EGS_mm10=2652783500

## blacklisted region
#BlackListFile_hg19=$Datadir"/Blacklist_Genome/hg19/ENCFF001TDO.bed"

## bin size for CAGE-seq data
## similar to the HiChIP loop resolution
Binsize_CAGE=5000

## bin size for other epigenomic tracks
Binsize_Other=100

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
## download this file if not available
RefGTFFile=$Datadir'/hg19/gencode.v42lift37.basic.annotation.gtf'
OutDir=$Datadir'/GTF/hg19'
mkdir -p $OutDir
OutFile=$OutDir'/hg19_TSS.gtf'
awk -F'[\t| ]' '{if (substr($1,1,1)!="#") {for (i=0;i<=NF;i++) {if ($i=="gene_name") {N=i+1}; if ($i=="gene_id") {E=i+1}; if ($i=="gene_type") {T=i+1}}; print $1"\t"$4"\t"$5"\t"$E"\t"$N"\t"$7"\t"$T}}' $RefGTFFile | sort -k1,1 -k2,2n | uniq > $OutFile
sed -i 's/;//g' $OutFile
sed -i 's/\"//g' $OutFile

## hg38 reference genome - GTF annotation
RefGTFFile=$Datadir'/hg38/gencode.v42.basic.annotation.gtf'
OutDir=$Datadir'/GTF/hg38'
mkdir -p $OutDir
OutFile=$OutDir'/hg38_TSS.gtf'
awk -F'[\t| ]' '{if (substr($1,1,1)!="#") {for (i=0;i<=NF;i++) {if ($i=="gene_name") {N=i+1}; if ($i=="gene_id") {E=i+1}; if ($i=="gene_type") {T=i+1}}; print $1"\t"$4"\t"$5"\t"$E"\t"$N"\t"$7"\t"$T}}' $RefGTFFile | sort -k1,1 -k2,2n -k3,3n | uniq > $OutFile
sed -i 's/;//g' $OutFile
sed -i 's/\"//g' $OutFile

##========
## download blacklist genome - hg19
##========
OutDir=$Datadir'/Blacklist_Genome/hg19'
mkdir -p $OutDir
cd $OutDir
wget https://www.encodeproject.org/files/ENCFF001TDO/@@download/ENCFF001TDO.bed.gz
gunzip ENCFF001TDO.bed.gz

##========
## GM12878
##========
OutDir=$Datadir'/GM12878/CAGE'
mkdir -p $OutDir
cd $OutDir
wget https://www.encodeproject.org/files/ENCFF915EIJ/@@download/ENCFF915EIJ.bam
samtools index ENCFF915EIJ.bam
wget https://www.encodeproject.org/files/ENCFF990KLZ/@@download/ENCFF990KLZ.bam
samtools index ENCFF990KLZ.bam
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF915EIJ.bam ${Inpdir}/ENCFF915EIJ_Deeptools_RPGC_binsize_${Binsize_CAGE}bp.bw ${EGS_hg19} ${Binsize_CAGE}
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF990KLZ.bam ${Inpdir}/ENCFF990KLZ_Deeptools_RPGC_binsize_${Binsize_CAGE}bp.bw ${EGS_hg19} ${Binsize_CAGE}


OutDir=$Datadir'/GM12878/DNase'
mkdir -p $OutDir
cd $OutDir
wget https://www.encodeproject.org/files/ENCFF775ZJX/@@download/ENCFF775ZJX.bam
samtools index ENCFF775ZJX.bam
wget https://www.encodeproject.org/files/ENCFF783ZLL/@@download/ENCFF783ZLL.bam
samtools index ENCFF783ZLL.bam
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF775ZJX.bam ${Inpdir}/ENCFF775ZJX_Deeptools_RPGC_binsize_${Binsize_Other}bp.bw ${EGS_hg19} ${Binsize_Other} 
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF783ZLL.bam ${Inpdir}/ENCFF783ZLL_Deeptools_RPGC_binsize_${Binsize_Other}bp.bw ${EGS_hg19} ${Binsize_Other} 


OutDir=$Datadir'/GM12878/H3K4me3'
mkdir -p $OutDir
cd $OutDir
wget https://www.encodeproject.org/files/ENCFF818GNV/@@download/ENCFF818GNV.bigWig
wget https://www.encodeproject.org/files/ENCFF342CXS/@@download/ENCFF342CXS.bam
samtools index ENCFF342CXS.bam
wget https://www.encodeproject.org/files/ENCFF818UQV/@@download/ENCFF818UQV.bam
samtools index ENCFF818UQV.bam
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF342CXS.bam ${Inpdir}/ENCFF342CXS_Deeptools_RPGC_binsize_${Binsize_Other}bp.bw ${EGS_hg19} ${Binsize_Other} 
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF818UQV.bam ${Inpdir}/ENCFF818UQV_Deeptools_RPGC_binsize_${Binsize_Other}bp.bw ${EGS_hg19} ${Binsize_Other} 


OutDir=$Datadir'/GM12878/H3K27ac'
mkdir -p $OutDir
cd $OutDir
wget https://www.encodeproject.org/files/ENCFF180LKW/@@download/ENCFF180LKW.bigWig
wget https://www.encodeproject.org/files/ENCFF197QHX/@@download/ENCFF197QHX.bam
samtools index ENCFF197QHX.bam
wget https://www.encodeproject.org/files/ENCFF882PRP/@@download/ENCFF882PRP.bam
samtools index ENCFF882PRP.bam
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF197QHX.bam ${Inpdir}/ENCFF197QHX_Deeptools_RPGC_binsize_${Binsize_Other}bp.bw ${EGS_hg19} ${Binsize_Other}
BamToBigWigDeepToolsRPGC ${Inpdir}/ENCFF882PRP.bam ${Inpdir}/ENCFF882PRP_Deeptools_RPGC_binsize_${Binsize_Other}bp.bw ${EGS_hg19} ${Binsize_Other} 



