
3CExpr
=========

Predicting gene expression using chromatin conformation capture (3C) data


Developed by: Sourya Bhattacharyya

La Jolla Institute for Immunology, La Jolla, CA 92037, USA


** Note ** This implementation is an adaptation of the package **GraphReg** (https://github.com/karbalayghareh/GraphReg) specifically the module **Epi-GraphReg** corresponding to the paper "Chromatin interaction–aware gene regulatory modeling with graph attention networks" by Karbalayghareh et al. Genome Research 2022


Improvements
==============

The current implementation, is however, highly automated and free from any hard-coding or static assumptions used in the original Genome Research paper. Below are the list of improvements done by the current implementation:

	1. The current implementation is highly automated using a SnakeMake pipeline, and is free from any hard-coding or static assumptions (or parameters) associated with the original project.

		1a. The original work uses hard-coding and manual parameter setting for different reference genome and chromososome sizes. Here, user only needs to provide the reference genome and associated parameters in a configuration file, without any need to edit the codes.

		1b. The original work requires sequential execution of individual modules, and editing individual codes according to the parameters. Here, an automated Snakemake pipeline is executed.

	2. The Genome Reseach paper uses HiCDC+ (https://www.nature.com/articles/s41467-021-23749-x) as the loop caller to call HiChIP loops and use these loops for predicting the gene expression. The current implementation, on the other hand, uses FitHiChIP (https://www.nature.com/articles/s41467-019-11950-y) as a HiChIP loop caller since it is the most widely used and best performing HiChIP loop calling method.

	3. The results of the prediction model are better documented and streamlined for better evaluation.


On-going implementations
=========================

We are working on a better prediction model generalized and applicable to a wide range of 3C data for predicting gene expression.

	1. Support for HiC, HiChIP, Micro-C, ChIA-PET and other 3C datasets, and loop calls from any loop callers.

	2. Improveming the deep learning based predition model by accounting for additional feaures from 3C datasets and loop calls.


Execution of 3CExpr
=====================

User needs to perform the following steps to execute the predictive model.

	A. Download data

		1. User needs to download the reference GTF files from GENCODE depending on the reference genome.

			a. For hg19, download *gencode.v42lift37.basic.annotation.gtf*

			b. For hg38, download *gencode.v42.basic.annotation.gtf*

		2. Follow the script *data_download.sh* and edit necessary file paths to download the required datasets.

	B. Main code / pipeline:

		1. Apply FitHiChIP on your HiChIP data of interest (for details, please check FitHiChIP documentation https://ay-lab.github.io/FitHiChIP/html/index.html)

		2. Check the configuration file *configfile_GM12878.yaml* (sample configuration file provided for GM12878 H3K27ac HiChIP data) to include the HiChIP data details and other parameters.

		3. Check *Snakefile* and edit the path / configuration parameters.

		4. Edit the file *pbs_submit.sh* according to the file paths and execute (*qsub pbs_submit.sh*)



Configuration parameters
==========================

The sample configuration file provided (*configfile_GM12878.yaml*) lists the following parameters which users need to review / edit:

	1. *Genome*, *ChrSize*, *GTF*, and *Fasta*: Parameters related to the reference genome. Chromosome size and reference fasta file information can be obtained from UCSC while the reference GTF file is obtained by executing the script *data_download.sh*.

	2. *BlackList*: Blacklist genome. Please check the script *data_download.sh*.

	3. *OutDir*: Base output directory under which all the results will be stored.

	4. *loopfile*, *resolution*, *FDRThr* and *SampleLabel*: FitHiChIP loop file, resolution (bin size), FDR threshold, and sample label.

	5. *CAGETrack*, *CAGELabel* and *CAGEBinSize*: CAGE track lists, their labels and CAGE bin size. CAGE bin sizes should be equal to the FitHiChIP loop resolution.

		*Note*: To edit the CAGE track bin sizes, check the script *data_download.sh* where the parameter *Binsize_CAGE* is mentioned (and the CAGE bigwig tracks are created according to the bin size). We used 5 Kb for CAGE bin size and FitHiChIP loops, as suggested in the Genome Research paper.

		*Note*: *CAGETrack* and *CAGELabel* are comma / colon separated lists. We can use multiple CAGE files, if available. Prediction results for individual CAGE files will be reported. Note that, number of tracks and number of labels should be identical.

	6. *EpiTrack*, *EpiLabel* and *EpiBinSize*: Epigenome track lists, their labels, and bin sizes. We used 100 bp for bin size as suggested in the Genome Research paper. To edit the bin sizes (and respective bigwig tracks), please check script *data_download.sh*.

	7. *Span* and *Offset*: *Span* means the complete interval to be considered per iteration (default = 6 Mb) while *Offset* is the middle portion (default = 2 Mb). These parameters are suggested in the Genome Research paper.



Output
=========

1) Check the file *OutDir*/*SampleLabel*/TestModel/contact_mat_FDR_*FDRThr*/*CAGELabel*/Final_Summary_Metrics.csv

It lists the NLL (negative log likelihood), rho and sp (correlation) values for individual sets of validation and test chromosomes (remaining chromosomes are used for training the model).

	*Note* For details of these output parameters, please check the Genome Research paper.

2) Check the file *OutDir*/*SampleLabel*/TestModel/contact_mat_FDR_*FDRThr*/valid_chr_*/df_all_predictions.csv

Lists the true and predicted gene expressions for individual bins, as well as the overlapping genes and number of HiChIP contacts associated.


Queries
===========

For any queries, please e-mail:

Sourya Bhattacharyya (sourya@lji.org)

