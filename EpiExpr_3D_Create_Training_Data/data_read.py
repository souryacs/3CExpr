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

## =========================================================================
## Read the epigenomic and CAGE tracks (bigwig files, which are downloaded before) 
## and convert them to .h5 format
## default Epigenetic track resolution = 128bp (specified in the configuration file)
## default CAGE track resolution = 4096bp (specified in the configuration file)

## adapted from:
## https://github.com/calico/basenji
## https://github.com/karbalayghareh/GraphReg

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
# =========================================================================

from optparse import OptionParser
import os
import sys
import collections
import h5py
import numpy as np
import pyBigWig
import intervaltree
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
## important parameters:
## 1. pool_width: target track resolution - like 5000bp for CAGE, 100bp for epigenomic tracks
## 2. sum_stat: aggregation operation - sum / mean etc.
##================
def Process_BigWig(InputTrackFile, OutTrackFile, filename_seqs, 
				   black_chr_trees, crop_bp, 
				   pool_width, sum_stat, 
				   clip, soft_clip, scale):

	if 0:
		print("\n\n **** within function Process_BigWig - InputTrackFile : ", InputTrackFile, 
		" OutTrackFile : ", OutTrackFile, 
		"crop_bp : ", crop_bp, 
		"  pool_width : ", pool_width, 
		"  sum_stat : ", sum_stat, 
		" clip : ", clip, 
		" soft_clip : ", soft_clip, 
		" scale : ", scale)

	assert(crop_bp >= 0)
	
	## open input genome coverage file (bigwig)
	InpTrackFile_open = CovFace(InputTrackFile)

	## read model sequences from the input file "filename_seqs"
	## corresponds to the fixed sized intervals
	model_seqs = [] 
	
	## old code - commented
	if 0:
		for line in open(filename_seqs):
			a = line.split()
			model_seqs.append(ModelSeq(a[0], int(a[1]), int(a[2]), None))

	## new code - adapts for header information
	seq_dataframe = pd.read_csv(filename_seqs, delimiter='\t')
	for i in range(seq_dataframe.shape[0]):
		model_seqs.append(
			ModelSeq(
				seq_dataframe.iloc[i,0], 
				int(seq_dataframe.iloc[i,1]), 
				int(seq_dataframe.iloc[i,2]), 
				None
			)
		)

	# compute dimensions
	num_seqs = len(model_seqs)
	seq_len_nt = model_seqs[0].end - model_seqs[0].start
	seq_len_nt -= 2*crop_bp
	target_length = seq_len_nt // pool_width
	assert(target_length > 0)
	if 0:
		print("\n\n **** within function Process_BigWig - OutTrackFile : ", OutTrackFile, 
		"num_seqs : ", num_seqs, 
		"  seq_len_nt : ", seq_len_nt, 
		"  pool_width : ", pool_width, 
		" target_length : ", target_length)

	## initialize output sequences coverage file
	seqs_cov_open = h5py.File(OutTrackFile, 'w')
	seqs_cov_open.create_dataset('seqs_cov', shape=(num_seqs, target_length), dtype='float16')

	## for each model sequence (interval)
	## read the sequence content from the input "InputTrackFile"
	for si in range(num_seqs):
		mseq = model_seqs[si]
		if 0:
			print("\n reading seq - si : ", si, 
		 			" mseq.start : ", mseq.start, 
					" mseq.end : ", mseq.end)

		## error condition - sourya
		## check the last interval and quit
		if ((mseq.end - mseq.start) < seq_len_nt):
			print("\n continue --- mseq.end : ", mseq.end, 
				" mseq.start : ", mseq.start, 
				" seq_len_nt : ", seq_len_nt)
			continue

		# read coverage
		seq_cov_nt = InpTrackFile_open.read(mseq.chr, mseq.start, mseq.end)

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

	# close input track file
	InpTrackFile_open.close()


##======================
def parse_options():  
	usage = 'usage: %prog [options]'
	parser = OptionParser(usage)
	
	parser.add_option('-g', 
				   dest='refgenome', 
				   default=None, 
				   type='str', 
				   help='Reference Genome. Mandatory parameter.')
	parser.add_option('-B', 
				   dest='BlackListFile', 
				   default=None, 
				   type='str', 
				   help='Blacklist genome file. Mandatory parameter.')
	parser.add_option('-c', 
				   dest='chr', 
				   default=None, 
				   type='str', 
				   help='Current chromosome. Mandatory parameter.')
	parser.add_option('-O', 
				   dest='BaseOutDir', 
				   default=None, 
				   type='str', 
				   help='Base output directory. Mandatory parameter.')
	parser.add_option('-C', 
				   dest='CAGEBinSize', 
				   default=4096, 
				   type='int', 
				   help='CAGE bin size. Default 4096 (4 Kb)')
	parser.add_option('-E', 
				   dest='EpiBinSize', 
				   default=128, 
				   type='int', 
				   help='Epigenomic track bin size. Default 128 bp')
	parser.add_option('-l', 
				   dest='TrackLabel', 
				   default=None, 
				   type='str', 
				   help='Current epigenomic track label (as mentioned in the table). Mandatory parameter.')
	parser.add_option('-t', 
				   dest='TrackType', 
				   default=None, 
				   type='str', 
				   help='Current epigenomic track type (as mentioned in the table). Mandatory parameter.')
	parser.add_option('-F', 
				   dest='TrackFilePath', 
				   default=None, 
				   type='str', 
				   help='Full path of the epigenomic track file (as mentioned in the table). Mandatory parameter.')
	
	## additional options - default values are used
	parser.add_option('--clip', 
				   dest='clip', 
				   default=50000,
				   type='float', 
				   help='Clip values post-summary to a maximum [Default: %default]'),
	parser.add_option('--crop', 
				   dest='crop_bp', 
				   default=0, 
				   type='int', 
				   help='Crop bp off each end [Default: %default]'),
	parser.add_option('--scale', 
				   dest='scale', 
				   default=1., 
				   type='float', 
				   help='Scale values by [Default: %default]'),
	parser.add_option('--soft', 
				   dest='soft_clip', 
				   default=False, 
				   action='store_true', 
				   help='Soft clip values, applying sqrt to the execess above the threshold [Default: %default]'),
	parser.add_option('--sum', 
				   dest='sum_stat', 
				   default='max', 
				   help='Summary statistic to compute in windows [Default: %default]'),
	
	# parser.add_option('--width', dest='pool_width', default=5000, type='int', help='Average pooling width [Default: %default]')

	(options, args) = parser.parse_args()
	return options, args

################################################################################
# main
################################################################################
def main():

	options, args = parse_options()

	BaseOutDir = options.BaseOutDir
	refgenome = options.refgenome
	currchr = options.chr	
	BlackListFile = options.BlackListFile
	CAGEBinSize = int(options.CAGEBinSize)
	EpiBinSize = int(options.EpiBinSize)

	## other parameters
	clip = options.clip
	crop_bp = options.crop_bp
	scale = options.scale
	soft_clip = options.soft_clip
	sum_stat = options.sum_stat

	# read blacklist regions
	black_chr_trees = read_blacklist(BlackListFile)

	print("\n\n ===>> Function data_read.py ---- Input parameters --- BaseOutDir : ", BaseOutDir, 
	   "\n refgenome : ", refgenome, 
	   "\n currchr : ", currchr, 
	   "\n CAGEBinSize : ", CAGEBinSize, 
	   "\n EpiBinSize : ", EpiBinSize, 
	   "\n TrackLabel : ", options.TrackLabel, 
	   "\n chromosome : ", options.chr, 
	   "\n TrackType : ", options.TrackType, 
	   "\n TrackFilePath : ", options.TrackFilePath)
	
	print("\n\n ===>> Function data_read.py ---- Additional options --- clip : ", clip, 
	   "\n crop_bp : ", crop_bp, 
	   "\n scale : ", scale, 
	   "\n soft_clip : ", soft_clip, 
	   "\n sum_stat : ", sum_stat)	

	## sequence file for the current chromosome (generated from previous codes)
	filename_seqs = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(CAGEBinSize) + '/sequences_' + currchr + '.bed'

 	## output directory to store the track coverages	
	if options.TrackType == "CAGE":		
		currOutDir = BaseOutDir + "/CAGE_Tracks/" + refgenome + '/' + str(CAGEBinSize) + "bp/" + options.TrackLabel
		pool_width = CAGEBinSize
	else:		
		currOutDir = BaseOutDir + "/Epigenome_Tracks/" + refgenome + '/EPI_' + str(EpiBinSize) + "bp_CAGE_" + str(CAGEBinSize) + "bp/" + options.TrackLabel
		pool_width = EpiBinSize

	os.makedirs(currOutDir, exist_ok = True)
	
	print("\n filename_seqs (for the current chromosome) : ", filename_seqs)
	
	## output track file for the current chromosome	
	OutTrackFile = currOutDir + '/seq_cov_' + currchr + '.h5'

	Process_BigWig(options.TrackFilePath, OutTrackFile, 
				filename_seqs, black_chr_trees, crop_bp, 
				pool_width, sum_stat, clip, soft_clip, scale)


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
	main()



