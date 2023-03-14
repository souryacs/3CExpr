##===============
## snakemake file
##===============

configfile: "configfile_GM12878.yaml"

## python executable
python3exec='/home/sourya/packages/Anaconda3/anaconda3/envs/3CExpr/bin/python3'

## directory containing the scripts
SCRIPTDIR="/home/sourya/Code"

#========

rule all:
	input:
		testoutdir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/TestModel/contact_mat_FDR_' + str(config['Loop']['FDRThr'])

rule write_seq:
	""" 
	Get bin-specific information from reference chromosome size file
	"""
	input:
		config['General']['OutDir']
	output:
		seqdir=config['General']['OutDir'] + '/seqs_bed'
	shell:
		"{python3exec} {SCRIPTDIR}/write_seq.py -g {config[General][Genome]} -c {config[General][ChrSize]} -O {config[General][OutDir]} -r {config[Loop][resolution]}"

rule TSS_to_Bin:
	""" 
	Get bin-specific information of TSS
	"""
	input:
		seqdir=config['General']['OutDir'] + '/seqs_bed'
	output:
		tssdir=config['General']['OutDir'] + '/TSS'
	shell:
		"{python3exec} {SCRIPTDIR}/TSS_to_Bin.py -g {config[General][Genome]} -c {config[General][ChrSize]} -O {config[General][OutDir]} -r {config[Loop][resolution]} -T {config[General][GTF]}"

rule FitHiChIP_Loop_to_Graph:
	""" 
	Convert FitHiChIP loops to graph
	"""
	input:
		seqdir=config['General']['OutDir'] + '/seqs_bed'
	output:
		matdir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/FitHiChIP_to_mat/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	shell:
		"{python3exec} {SCRIPTDIR}/FitHiChIP_Loop_to_Graph.py -g {config[General][Genome]} -c {config[General][ChrSize]} -O {config[General][OutDir]} -r {config[Loop][resolution]} -f {config[Loop][FDRThr]} -l {config[Loop][LoopFile]} -n {config[Loop][SampleLabel]}"

rule data_read:
	""" 
	Read adjacency matrices (FitHiChIP loops) and epigenomic tracks
	and convert them to .h5 format
	"""
	input:
		seqdir=config['General']['OutDir'] + '/seqs_bed'
	output:
		epitrackdir=config['General']['OutDir'] + '/Epigenome_Tracks'
	shell:
		"{python3exec} {SCRIPTDIR}/data_read.py -g {config[General][Genome]} -c {config[General][ChrSize]} -O {config[General][OutDir]} -r {config[Loop][resolution]} -B {config[General][BlackList]} -C {config[Epigenome][CAGEBinSize]} -E {config[Epigenome][EpiBinSize]} -X {config[Epigenome][CAGETrack]} -Y {config[Epigenome][EpiTrack]} -x {config[Epigenome][CAGELabel]} -y {config[Epigenome][EpiLabel]}"  
	

rule data_write:
	""" 
	Writing the epigenomic and HiC tracks into tensorflow compatible TFRecord format
	"""
	input:
		seqdir=config['General']['OutDir'] + '/seqs_bed',
		tssdir=config['General']['OutDir'] + '/TSS',
		epitrackdir=config['General']['OutDir'] + '/Epigenome_Tracks',
		matdir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/FitHiChIP_to_mat/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	output:
		tfrecorddir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/TFRecord/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	shell:
		"{python3exec} {SCRIPTDIR}/data_write.py --Span {config[Model][Span]} --Offset {config[Model][Offset]} -g {config[General][Genome]} -c {config[General][ChrSize]} -O {config[General][OutDir]} -r {config[Loop][resolution]} -n {config[Loop][SampleLabel]} -f {config[Loop][FDRThr]} --fasta {config[General][Fasta]} -C {config[Epigenome][CAGEBinSize]} -E {config[Epigenome][EpiBinSize]} -X {config[Epigenome][CAGETrack]} -Y {config[Epigenome][EpiTrack]} -x {config[Epigenome][CAGELabel]} -y {config[Epigenome][EpiLabel]}"


rule trainmodel:
	""" 
	Training of the GAT model to predict gene expression from epigenomic data
	"""
	input:
		tfrecorddir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/TFRecord/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	params:
		Model="epi",
		NumGATLayer="2",
		ValidChrList="1,11",
		TestChrList="2,12"	
	output:
		trainoutdir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/TrainingModel/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	shell:
		"{python3exec} {SCRIPTDIR}/Train_Epigenomic_Model.py --Span {config[Model][Span]} --Offset {config[Model][Offset]} -g {config[General][Genome]} -O {config[General][OutDir]} -r {config[Loop][resolution]} -n {config[Loop][SampleLabel]} -f {config[Loop][FDRThr]} -C {config[Epigenome][CAGEBinSize]} -E {config[Epigenome][EpiBinSize]} -X {config[Epigenome][CAGETrack]} -Y {config[Epigenome][EpiTrack]} -x {config[Epigenome][CAGELabel]} -y {config[Epigenome][EpiLabel]} -n {params.NumGATLayer} -v {params.ValidChrList} -t {params.TestChrList} -M {params.Model}"


rule testmodel:
	""" 
	Testing of the GAT model to predict gene expression from epigenomic data
	"""
	input:
		trainoutdir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/TrainingModel/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	params:
		Model="epi",
		ValidChrList="1,11",
		TestChrList="2,12"	
	output:
		testoutdir=config['General']['OutDir'] + '/' + config['Loop']['SampleLabel'] + '/TestModel/contact_mat_FDR_' + str(config['Loop']['FDRThr'])
	shell:
		"{python3exec} {SCRIPTDIR}/Test_Epigenomic_Model.py --Span {config[Model][Span]} --Offset {config[Model][Offset]} -g {config[General][Genome]} -c {config[General][ChrSize]} -O {config[General][OutDir]} -r {config[Loop][resolution]} -n {config[Loop][SampleLabel]} -f {config[Loop][FDRThr]} -C {config[Epigenome][CAGEBinSize]} -E {config[Epigenome][EpiBinSize]} -X {config[Epigenome][CAGETrack]} -Y {config[Epigenome][EpiTrack]} -x {config[Epigenome][CAGELabel]} -y {config[Epigenome][EpiLabel]} -v {params.ValidChrList} -t {params.TestChrList} -M {params.Model}"


