EpiExpr / 3CExpr
=====================

Predicting gene expression from epigenetic data and 3D chromatin interactions, and also identifying the cis-regulatory elements (CREs) contributing to the specific gene expression.

	1. EpiExpr does not use DNA sequence based embeddings. Rather, it uses epigenetic tracks and 3D chromatin interactions (optional) to predict gene expression.
	2. Supports user-defined epigenetic and CAGE track resolutions (CAGE track resolution should be a multiple of the epigenetic track resolution). We have experimented with CAGE resolution = 5000 bp (5 Kb) and epigenetic resolution = 100 bp.
	3. Supports curation of training and validation datasets composed of multiple cell types, and each cell type may have different numbers and types of epigenetic tracks.

## Installation

Preferably create a conda environment using Python version >= 3.7 and inside that environment, install the following python packages:

	1. numpy, 2. pandas, 3. pyarrow, 4. scipy, 5. h5py, 6. pyBigWig, 7. intervaltree, 8. pysam, 9. torch, 10. torchvision, 11. snakemake

Also make sure to have GPU access.

The scripts (.sh) files can be edited to execute in SLURM / qsub environments, according to the user's chroice.


## Download reference data and reference genome information

Check the folder "Data" and the reference script "Data/data_download_hg19.sh". 
It contains commands to download the EpiGraphReg paper specific epigenomic datasets.

Couple of important points:
	
	1. Check the lines 48 - 70 of this script, about downloading the GENCODE annotations for the reference genome (hg19). Once the reference annotation is downloaded, a custom awk command is applied on the downloaded annotation file to generate a text (or GTF) file containing the following subset of columns: 
		
		1. "chr" (chromosome) 
		2. "gene_start" (gene start) 
		3. "gene_end" (gene end)
		4. "ens" (Ensemble ID) 
		5. "gene" (gene name) 
		6. "strand" (strand information) 
		7. "type" (protein coding / lncRNA etc)

	2. The downloaded bigwig files (ChIP-seq, ATAC-seq) are applied a RPGC normalization as per the data processing suggested in the EpiGraphReg paper.

	3. Lines 70 - 80 of this script lists the command to download the blacklist file corresponding to the reference genome hg19.

For other reference genome (such as hg38) and corresponding datasets (such as the one provided in Basenji paper), user needs to edit the above mentioned script.

** For downloading chromatin interaction data, check our manuscript and reference literature. **


## EpiExpr-1D: Create training data using snakemake pipeline

Check the folder "EpiExpr_1D_Create_Training_Data" to see the underlying scripts and snakemake pipeline.

	1. User needs to edit the file "script_run_training_data.sh" with the proper location of the configuration file (mentioned below), target epigenomic and CAGE track resolutions, and run.
	2. However, first they need to edit the configuration file.

A sample configuration file (configfile.yaml) and its associated data description text file (Table_Config.txt) is provided inside the folder "config".

Parameters in the configuration file (configfile.yaml):

	1. Genome: reference genome information - hg19 / hg38 / mm9 / mm10 / mm39
	2. ChrSize: chromosome size file corresponding to the reference genome (download from UCSC browser)
	3. TSS: GTF file with 7 fields corresponding to the reference genome (as mentioned above in the data download section)
	4. OutDir: Base output directory to store the results
	5. BlackList: blacklist genome information (as mentioned above in the data download section)
	6. Label: string / label corresponding to the current data / run.
	7. CAGEBinSize: CAGE track resolution in bp
	8. EpiBinSize: Epigenomic track resolution in bp
	9. EpiTableFile: Path of the text file (described below) storing the paths of respective epigenomic datasets.
	10. Offset: denotes the middle segment corresponding to the prediction. For 5Kb CAGE resolution, we recommend using 2 Mb. It must be a multiple of the CAGE resolution. The complete chunk for training data is actually 3 times this offset value, so 6 Mb per chunk for 5 Kb CAGE resolution.

Corresponding contents of "EpiTableFile": 

It is a 3 column file with the following headers: CellType,	DataType, Path

	1. CellType: denotes the cell type for the data. User can specify multiple cell types as well, all of which would be included in the generated training (and validation) data.
	2. DataType: Can be either "CAGE" (denotes CAGE tracks) and "EPI" (denoted Epigenomic tracks).
	3. Path: For each data (CAGE or EPI), provide the folder containing the data. 
		** Note: the folder file path is required, not the full path of the underlying bigwig files. **
		** If user has multiple CAGE tracks for the same cell type, they should put them under separate folders and use only one of them for creating corresponding training data. **
		** If user has multiple epigenomic tracks for the same cell type and same histone mark or TF (such as H3K27ac), they should put all of these bigwig files into one folder and just provide the folder path in the configuration text file. That would save users from specifying all the individual file paths. **

** Users are advised to put absolute path names in all scripts / configuration options. **

Once the configuration file and corresponding text files are filled up, user should refer to the correct configuration file path from the script "script_run_training_data.sh" and run the script to create the training (and validation) data.


## EpiExpr-1D: training data using snakemake pipeline - output file / folder description

All the training (and validation) datasets are created inside the folder specified by the "OutDir" parameter of the configfile.yaml. Inside that folder, following subfolders and files are present:

	1. seqs_bed/${RefGenome}/${CAGEResolution}/sequences_chr*.bed: Here *RefGenome* = reference genome, *CAGEResolution* = CAGE resolution. Chromosome specific intervals with respect to the target CAGE resolution.
	2. TSS/${RefGenome}/${CAGEResolution}/TSS_Info_chr*.csv: For the specific CAGE resolution and reference genome, stores the protein coding TSS information overlapping individual intervals.
	3. Epigenome_Tracks/${RefGenome}/EPI_${EPIResolution}bp_CAGE_${CAGEResolution}bp/*/seq_cov_chr*.h5: For the input epigenomic and CAGE resolutions, stores the information of individual input epigenetic tracks into .h5 format.
	4. CAGE_Tracks/${RefGenome}/${CAGEResolution}bp/*/seq_cov_chr*.h5: For the input CAGE resolution,  stores the information of CAGE tracks into .h5 format.
	5. OUT_MODEL_EPI_${EPIResolution}bp_CAGE_${CAGEResolution}bp_1D_CNN/TrainingData/Offset_${Offset}/${Label}/*/chr*/*.pkl: For the input epigenomic and CAGE resolutions, Offset (specified in the "Offset" parameter of the configuration file), and Label (specified in the "Label" parameter of the configuration file), stores the training and validation datasets per chromosome in .pkl format.


## EpiExpr-1D: Prediction model

With the above generated training data and configuration files, we'll now perform the prediction for EpiExpr-1D with respect to a selected set of validation and test chromosomes (remaining chromosomes are used for training).

** Note: Currently, all training and validation are performed on the autosomal chromosomes. **

First check the folder "EpiExpr_1D_Prediction_Model" containing python codes and two scripts.

Script 1: script_Train.sh

	Script to train the model. Requires GPU access, otherwise the training would be too slow.

	1. Edit the parameters "RefGenome", "EPISIZE", "CAGESIZE", "Offset" and "Label" according to the configuration file.
	2. validchrlist: underscore separated list of chromosomes (numbers) used for validation. For example, "1_11" means chromosomes 1 and 11 would be used for validation.
	3. testchrlist: underscore separated list of chromosomes (numbers) used for testing. For example, "2_12" means chromosomes 2 and 12 would be used for testing.
	4. Trainmodel: Can be 1 or 2. 1 means simple CNN model whereas 2 (recommended) means proposed residual net based model.
	5. ProjChannel: An integer used to define the channel dimension for projecting input data. Users can adjust this parameter depending on the input count of epigenetic tracks. 
		** For EpiGraphReg data with 6 epigenomic tracks per cell type, we used ProjChannel = 8 **
		** For Basenji data with ~ 200 epigenomic tracks for GM12878, we used 256 **
	6. TrainingDataDir: Directory containing the training datasets for the current run (as mentioned in the previous section).
	7. TrainingModelDir: Directory to contain the output training model.
		** Note: Inside this directory, the output training model file name would be "Model_valid_chr_"${validchrlist}"_test_chr_"${testchrlist}".pt"


Once the above script is edited, and successfully executed, and the above mentioned training model file is generated, check the following script for testing the model.

Script 2: script_Test.sh

	Script to test the model. Requires GPU access.

	1. Edit the parameters "RefGenome", "EPISIZE", "CAGESIZE", "BlackListFile", "Offset" and "Label" according to the configuration file.
	2. Parameters "validchrlist", "testchrlist", "ProjChannel" and "Trainmodel" should be identical to the training script.
	3. TSSDir: Directory containing TSS information, as described in the output folder description of EpiExpr-1D create training data module.
	4. ModelFileName: Training model file name.
	5. TestOutDir: Directory to contain the output testing results.

Once the above script is edited, and successfully executed, user should see a file "df_all_predictions.txt" within the specified "TestOutDir".
	** This file contains individual genomic intervals (according to the specific CAGE resolution) of the test chromosomes (provided by the parameter testchrlist), their true and predicted CAGE expressions, as well as the number of TSS and TSS information on each interval.


## EpiExpr-3D: Generate FitHiChIP significant loops corresponding to input chromatin interactions (Hi-C, HiChIP, PCHi-C)

The 3D model uses chromatin interactions in addition to the epigenomic datasets.

	==>> Step 1: Calling significant loops from input chromatin interactions

		We process input chromatin interactions (HiChIP, Hi-C, PCHi-C) using our tool FitHiChIP (https://github.com/ay-lab/FitHiChIP). 

		If you are not familiar with FitHiChIP, please read the documentation (https://ay-lab.github.io/FitHiChIP/) to call significant loops from chromatin interaction data.

		** Note: FitHiChIP loop resolution should be identical to the CAGE track resolution **


	==>> Step 2: Converting these loops to EpiExpr-3D format

		Check the output file "*OUTPREFIX*.interactions_FitHiC.bed" generated from FitHiChIP containing all the input chromatin interactions along with their FitHiChIP significance (FDR), without any filtering based on the FDR threshold.

		Now check the folder "FitHiChIP" and follow the scripts "Extract_FitHiChIP_Loops.py" and "Extract_FitHiChIP_Loops.sh" to convert these loops into EpiExpr-3D compatible format loop file "FitHiChIP_Input_EpiExpr.txt"


## EpiExpr-3D: Create training data using snakemake pipeline

Check the folder "EpiExpr_3D_Create_Training_Data".

	1. Script "script_run_training_data.sh" is similar to the EpiExpr-1D model.
	2. Check the configuration file "config/configfile.yaml" - similar to EpiExpr-1D with following additional parameters:
		1. LoopLabel: Label of FitHiChIP loops used - can be used to denote the loop resolution, FDR threshold, lower and upper distance thresholds. 
		** If users are not familiar with loop calling, request to first read the FitHiChIP documentation **
		2. EdgeNorm: Can be 1 or 2. If 1, uses scikit-learn implemented row normalization on the GNN model and underlying graph edges. Otherwise, employ double stochastic normalization.
	3. Check the associated text file "Table_Config.txt" storing the folder paths of the respective tracks.
		1. Similar to the description of EpiExpr-1D. 
		2. Here the track type "Loop" is added. User needs to provide the full path of the file "FitHiChIP_Input_EpiExpr.txt" generated in the previous step (converting FitHiChIP output to EpiExpr-3D compatible format). ** Note: here, not the folder but the full file path to be mentioned. **		

** Once again, users are advised to put absolute path names in all scripts / configuration options. **


## EpiExpr-3D: Prediction model

Similar to EpiExpr-1D, first check the folder "EpiExpr_3D_Prediction_Model" and the underlying python codes and scripts. ** Note: Currently, all training and validation are performed on the autosomal chromosomes. **

Script 1: script_Train.sh

	Script to train the model. Requires GPU access, otherwise the training would be too slow.

	1. Parameters "RefGenome", "EPISIZE", "CAGESIZE", "Offset", "Label", "validchrlist", "testchrlist", "ProjChannel", are similar to EpiExpr-1D and need to be filled up according to the configuration options.
	2. "NumGatLayer": Number of GAT / GT layers. We recommend using 2.
	3. "NumGATHeader": Number of GAT / GT heads. We recommend using 8.
	4. "LoopLabel": FitHiChIP loop label - should be identical to the configuration file.
	5. "EdgeNorm": Graph edge normalization technique - 1 for Scikit-learn implemented row normalization, 2 for Double stochastic normalization.
	6. "ActFn": activation function - we recommend "gelu".
	7. "Modeltype_3D": 1 for GAT and 2 for GT.
	8. "ResidGAT": if 1, uses initial residual connection.
	9. "currTrainingDataFolder": specify the folder containing the training datasets. Check the previous section.
	10. "TrainModelDir": Directory to contain the output training model. 
	11. "trainmodelfile": Output training model file with respect to the current set of validation and test chromosomes.


Once the above script is edited, and successfully executed, and the above mentioned training model file is generated, check the following script for testing the model.

Script 2: script_Test.sh

	Script to test the model. Requires GPU access.
	Similar to EpiExpr-1D description, and the parameters follow the "script_Train.sh" script of the EpiExpr-3D model.

	Once the model runs, user should see a file "df_all_predictions.txt" within the specified test output folder.


## Performance evaluation

The files "df_all_predictions.txt" from both EpiExpr-1D and EpiExpr-3D models are used for performance evaluation. ** Scripts will be uploaded soon. **

## Manuscript

Bhattacharyya S, and Ay F, Deep learning-based prediction of gene expression from epigenomic signals and chromatin interactions (*under preparation*)


## Contact

For any queries, please e-mail:

Sourya Bhattacharyya <souryabhatta.cs@gmail.com>




