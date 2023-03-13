#!/usr/bin/env python

# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

# This code is copied from https://github.com/calico/basenji and modified.

## Read the epigenomic and CAGE tracks (bigwig files, which are downloaded before) and convert them to .h5 format
# =========================================================================

from optparse import OptionParser
import os
import sys
import collections
import h5py
import numpy as np
import pyBigWig
import intervaltree
# import configparser
import yaml
import re
import pandas as pd

#import basenji.basenji_data.ModelSeq as ModelSeq
ModelSeq = collections.namedtuple('ModelSeq', ['chr', 'start', 'end', 'label'])

def read_blacklist(blacklist_bed, black_buffer=20):
	"""
	Construct interval trees of blacklist
	 regions for each chromosome.
	"""
	black_chr_trees = {}

	if blacklist_bed is not None and os.path.isfile(blacklist_bed):	
		for line in open(blacklist_bed):
			a = line.split()
			chrm = a[0]
			start = max(0, int(a[1]) - black_buffer)
			end = int(a[2]) + black_buffer
			if chrm not in black_chr_trees:
				black_chr_trees[chrm] = intervaltree.IntervalTree()
			black_chr_trees[chrm][start:end] = True

	return black_chr_trees

class CovFace:
	def __init__(self, cov_file):
		self.cov_file = cov_file
		self.bigwig = False

		cov_ext = os.path.splitext(self.cov_file)[1].lower()
		if cov_ext in ['.bw','.bigwig']:
			self.cov_open = pyBigWig.open(self.cov_file, 'r')
			self.bigwig = True
		elif cov_ext in ['.h5', '.hdf5', '.w5', '.wdf5']:
			self.cov_open = h5py.File(self.cov_file, 'r')
		else:
			print('Cannot identify coverage file extension "%s".' % cov_ext, file=sys.stderr)
			exit(1)

	def read(self, chrm, start, end):
		if self.bigwig:
			cov = self.cov_open.values(chrm, start, end, numpy=True).astype('float16')
		else:
			if chrm in self.cov_open:
				cov = self.cov_open[chrm][start:end]
			else:
				print("WARNING: %s doesn't see %s:%d-%d. Setting to all zeros." % (self.cov_file, chrm, start, end), file=sys.stderr)
				cov = np.zeros(end-start, dtype='float16')
		return cov

	def close(self):
		self.cov_open.close()

##================
## process individual bigwig files
##================
def Process_BigWig(genome_cov_file, Inplabel, OutDir, chrsize_df, black_chr_trees, BaseOutDir, refgenome, Resolution, crop_bp, pool_width, sum_stat, clip, soft_clip, scale):

	## open input genome coverage file (bigwig)
	genome_cov_open = CovFace(genome_cov_file)

	for rowidx in range(chrsize_df.shape[0]):      
		currchr = str(chrsize_df.iloc[rowidx, 0])
		print("\n\n rowidx : ", rowidx, "  processing chromosome : ", currchr)

        ## discard any chromosomes other than chr1 to chr22
		if (currchr == "chrX" or currchr == "chrY" or currchr == "chrM" or "un" in currchr or "Un" in currchr or "random" in currchr or "Random" in currchr or "_" in currchr):
			continue

	    ## sequence file for the current chromosome (generated from previous codes)
		filename_seqs = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution) + '/sequences_' + currchr + '.bed'
		print("\n filename_seqs : ", filename_seqs)
		if (os.path.exists(filename_seqs) == False):
			continue

	    ## output sequence coverage file (.h5) format per chromosome
		pattern_to_replace = "_seq_cov_" + currchr + ".h5"
		seqs_cov_file = OutDir + "/" + os.path.basename(genome_cov_file).replace(".bw", pattern_to_replace)
		print("\n seqs_cov_file : ", seqs_cov_file)
		if (os.path.exists(seqs_cov_file) == True):
			continue

		assert(crop_bp >= 0)

		## read model sequences from the input file "filename_seqs"
		## corresponds to the fixed sized intervals 
		model_seqs = []
		for line in open(filename_seqs):
			a = line.split()
			model_seqs.append(ModelSeq(a[0], int(a[1]), int(a[2]), None))

	    # compute dimensions
		num_seqs = len(model_seqs)
		seq_len_nt = model_seqs[0].end - model_seqs[0].start
		seq_len_nt -= 2*crop_bp
		target_length = seq_len_nt // pool_width
		assert(target_length > 0)
		print("\n\n **** num_seqs : ", num_seqs, '  model_seqs[0].end : ', model_seqs[0].end, '  model_seqs[0].start : ', model_seqs[0].start, '  seq_len_nt : ', seq_len_nt, '  pool_width : ', pool_width, ' target_length : ', target_length)

		## initialize output sequences coverage file
		seqs_cov_open = h5py.File(seqs_cov_file, 'w')
		seqs_cov_open.create_dataset('seqs_cov', shape=(num_seqs, target_length), dtype='float16')

		## for each model sequence (interval)
		## read the sequence content from the input "genome_cov_file"
		for si in range(num_seqs):
			mseq = model_seqs[si]
			# print("\n reading seq - si : ", si, " mseq.start : ", mseq.start, " mseq.end : ", mseq.end)

			## error condition - sourya
			## check the last interval and quit
			if ((mseq.end - mseq.start) < seq_len_nt):
				print("\n continue --- mseq.end : ", mseq.end, " mseq.start : ", mseq.start, " seq_len_nt : ", seq_len_nt)
				continue

			# read coverage
			seq_cov_nt = genome_cov_open.read(mseq.chr, mseq.start, mseq.end)

			# determine baseline coverage
			baseline_cov = np.percentile(seq_cov_nt, 10)
			baseline_cov = np.nan_to_num(baseline_cov)

			# set blacklist to baseline
			if mseq.chr in black_chr_trees:
				for black_interval in black_chr_trees[mseq.chr][mseq.start:mseq.end]:
					# adjust for sequence indexes
					black_seq_start = black_interval.begin - mseq.start
					black_seq_end = black_interval.end - mseq.start
					seq_cov_nt[black_seq_start:black_seq_end] = baseline_cov

			# set NaN's to baseline
			nan_mask = np.isnan(seq_cov_nt)
			seq_cov_nt[nan_mask] = baseline_cov

			# crop
			if crop_bp:
				seq_cov_nt = seq_cov_nt[crop_bp:-crop_bp]

			# sum pool
			seq_cov = seq_cov_nt.reshape(target_length, pool_width)
			if sum_stat == 'sum':
				seq_cov = seq_cov.sum(axis=1, dtype='float32')
			elif sum_stat in ['mean', 'avg']:
				seq_cov = seq_cov.mean(axis=1, dtype='float32')
			elif sum_stat == 'median':
				seq_cov = seq_cov.median(axis=1, dtype='float32')
			elif sum_stat == 'max':
				seq_cov = seq_cov.max(axis=1)
			else:
				print('ERROR: Unrecognized summary statistic "%s".' % sum_stat, file=sys.stderr)
				exit(1)

			# clip
			if clip is not None:
				if soft_clip:
					clip_mask = (seq_cov > clip)
					seq_cov[clip_mask] = clip + np.sqrt(seq_cov[clip_mask] - clip)
				else:
					seq_cov = np.clip(seq_cov, 0, clip)

			# scale
			seq_cov = scale * seq_cov

			# write
			seqs_cov_open['seqs_cov'][si,:] = seq_cov.astype('float16')

		# close sequences coverage file
		seqs_cov_open.close()

	# close genome coverage file
	genome_cov_open.close()


##======================
def parse_options():  
	usage = 'usage: %prog [options]'
	parser = OptionParser(usage)

	parser.add_option('-C', dest='configfile', default=None, type='str', help='Configuration file. Mandatory parameter.'),  
	
	parser.add_option('-c', dest='clip', default=50000, type='float', help='Clip values post-summary to a maximum [Default: %default]'),
	parser.add_option('--crop', dest='crop_bp', default=0, type='int', help='Crop bp off each end [Default: %default]'),
	parser.add_option('-s', dest='scale', default=1., type='float', help='Scale values by [Default: %default]'),
	parser.add_option('--soft', dest='soft_clip', default=False, action='store_true', help='Soft clip values, applying sqrt to the execess above the threshold [Default: %default]'),
	parser.add_option('-u', dest='sum_stat', default='max', help='Summary statistic to compute in windows [Default: %default]'),
	# parser.add_option('-w',dest='pool_width', default=5000, type='int', help='Average pooling width [Default: %default]')

	(options, args) = parser.parse_args()
	return options, args

################################################################################
# main
################################################################################
def main():

	options, args = parse_options()

	## read the input configuration file
	# config = configparser.ConfigParser()
	# config.read(options.configfile)

	config_fp = open(options.configfile, "r")
	config = yaml.load(config_fp, Loader=yaml.FullLoader)    

	refgenome = config['General']['Genome']
	BaseOutDir = config['General']['OutDir']
	chrsizefile = config['General']['ChrSize']
	Resolution = int(config['Loop']['resolution'])
	BlackListFile = config['General']['BlackList']

	CAGEBinSize = int(config['Epigenome']['CAGEBinSize'])
	EpiBinSize = int(config['Epigenome']['EpiBinSize'])
	## track list using comma or colon as delimiter
	CAGETrackList = re.split(r':|,', config['Epigenome']['CAGETrack']) 
	## track list using comma or colon as delimiter
	EpiTrackList = re.split(r':|,', config['Epigenome']['EpiTrack']) 
	## label list using comma or colon as delimiter
	CAGELabelList = re.split(r':|,', config['Epigenome']['CAGELabel']) 
	## label list using comma or colon as delimiter
	EpiLabelList = re.split(r':|,', config['Epigenome']['EpiLabel']) 

	## other parameters
	clip = options.clip
	crop_bp = options.crop_bp
	scale = options.scale
	soft_clip = options.soft_clip
	sum_stat = options.sum_stat

	# read blacklist regions
	black_chr_trees = read_blacklist(BlackListFile)

	## read the chromosome size file
	chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])

	## base output directory to store the epigenome track related sequences
	OutDir = BaseOutDir + "/Epigenome_Tracks/" + refgenome
	if not os.path.exists(OutDir):
		os.makedirs(OutDir)

	## process individual CAGE tracks
	for i in range(len(CAGETrackList)):
		currCAGETrackFile = CAGETrackList[i]
		currCAGETrackLabel = CAGELabelList[i]
		print("\n Processing file : ", currCAGETrackFile, "  label : ", currCAGETrackLabel)
		currOutDir = OutDir + "/" + currCAGETrackLabel 
		if not os.path.exists(currOutDir):
			os.makedirs(currOutDir)
		## pool width - CAGEBinSize
		pool_width = CAGEBinSize
		Process_BigWig(currCAGETrackFile, currCAGETrackLabel, currOutDir, chrsize_df, black_chr_trees, BaseOutDir, refgenome, Resolution, crop_bp, pool_width, sum_stat, clip, soft_clip, scale)

	## process other epigenomic tracks
	for i in range(len(EpiTrackList)):
		currEpiTrackFile = EpiTrackList[i]
		currEpiTrackLabel = EpiLabelList[i]
		print("\n Processing file : ", currEpiTrackFile, "  label : ", currEpiTrackLabel)
		currOutDir = OutDir + "/" + currEpiTrackLabel 
		if not os.path.exists(currOutDir):
			os.makedirs(currOutDir)
		## pool width - EpiBinSize
		pool_width = EpiBinSize
		Process_BigWig(currEpiTrackFile, currEpiTrackLabel, currOutDir, chrsize_df, black_chr_trees, BaseOutDir, refgenome, Resolution, crop_bp, pool_width, sum_stat, clip, soft_clip, scale)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
	main()



