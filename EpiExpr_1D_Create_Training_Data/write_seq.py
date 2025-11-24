
from optparse import OptionParser
import pandas as pd
import os

##============
## Generate reference genome and resolution specific interval files, 
## separately for each ** autosomal ** chromosomes.
## Note: output CAGE resolution is similar to FitHiChIP track resolution: default 4096bp

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##============

##======================
def parse_options():  
   usage = 'usage: %prog [options]'
   parser = OptionParser(usage)

   parser.add_option('-g', 
                     dest='refgenome', 
                     default=None, 
                     type='str', 
                     help='Reference Genome. Mandatory parameter.')
   
   parser.add_option('-s', 
                     dest='chrsizefile', 
                     default=None, 
                     type='str', 
                     help='Chromosome size file for the reference genome. Mandatory parameter.')
   
   parser.add_option('-c', 
                     dest='currchr', 
                     default=None, 
                     type='str', 
                     help='Target chromosome. Mandatory parameter.')
   
   parser.add_option('-O', 
                     dest='BaseOutDir', 
                     default=None, 
                     type='str', 
                     help='Base output directory. Mandatory parameter.')
   
   parser.add_option('-r', 
                     dest='Resolution', 
                     default=4096, 
                     type='int', 
                     help='Loop / CAGE resolution. Default 4096bp (4 Kb)')
   
   (options, args) = parser.parse_args()
   return options, args

##==============
## main routine
##==============
def main():
   options, args = parse_options()

   refgenome = options.refgenome
   currchr = options.currchr
   BaseOutDir = options.BaseOutDir
   Resolution = int(options.Resolution)
   chrsizefile = options.chrsizefile
   
   print("\n\n ===>> Function write_seq ---- Input parameters <<==== \n\n")
   print("\n refgenome : ", refgenome)
   print("\n Target Chromosome : ", currchr)
   print("\n BaseOutDir : ", BaseOutDir)
   print("\n Resolution : ", Resolution)
   print("\n chrsizefile : ", chrsizefile)

   OutDir = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution)      
   os.makedirs(OutDir, exist_ok = True)

   ## infer intervals from the chromosome size information file: use bedtools 
   temp_chrsizefile_currchr = OutDir + '/temp_chrsizefile_' + str(currchr) + '.txt'      
   os.system("awk \'$1==\"" + str(currchr) + "\"\' " + str(chrsizefile) + " > " + str(temp_chrsizefile_currchr))
   
   filename_seqs = OutDir + '/sequences_' + currchr + '.bed'
   temp_filename_seqs = OutDir + '/temp_sequences_' + currchr + '.bed'
   os.system("bedtools makewindows -g " + str(temp_chrsizefile_currchr) + " -w " + str(Resolution) + " > " + str(temp_filename_seqs))
   
   seq_dataframe = pd.read_csv(temp_filename_seqs, delimiter='\t', header=None)
   seq_dataframe.columns = ["chr", "start", "end"]
   ## write the header information as well, and maintain tab separator
   seq_dataframe.to_csv(filename_seqs, sep="\t", index=False)   

   ## remove the temporary file
   os.system("rm " + str(temp_chrsizefile_currchr))
   os.system("rm " + str(temp_filename_seqs))


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
   main()


