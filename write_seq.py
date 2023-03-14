
from optparse import OptionParser
import numpy as np
import pandas as pd
import os

##============
## Generate reference genome and resolution specific interval files, separately for each chromosomes.
## Note: currently we have considered only autosomal chromosomes.
##============

##======================
def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)

    parser.add_option('-g', dest='refgenome', default=None, type='str', help='Reference Genome. Mandatory parameter.')
    parser.add_option('-c', dest='chrsizefile', default=None, type='str', help='Reference Genome specific chromosome size file. Mandatory parameter.')
    parser.add_option('-O', dest='BaseOutDir', default=None, type='str', help='Base output directory. Mandatory parameter.')
    parser.add_option('-r', dest='Resolution', default=5000, type='int', help='Loop resolution. Default 5000 (5 Kb)')

    (options, args) = parser.parse_args()
    return options, args

##==============
## main routine
##==============
def main():
   options, args = parse_options()

   # ##===============
   # ## old code
   # ##===============
   # if 0:

   #    refgenome = config['General']['Genome']
   #    chrsizefile = config['General']['ChrSize']
   #    BaseOutDir = config['General']['OutDir']
   #    Resolution = int(config['Loop']['resolution'])
   #    Span = int(config['Model']['Span'])

   #    print("\n\n ===>> Input parameters <<==== \n\n")
   #    print("\n refgenome : ", refgenome)
   #    print("\n chrsizefile : ", chrsizefile)
   #    print("\n BaseOutDir : ", BaseOutDir)
   #    print("\n Resolution : ", Resolution)
   #    print("\n Span : ", Span)

   #    OutDir = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution)
   #    sys_cmd = "mkdir -p " + str(OutDir)
   #    os.system(sys_cmd)

   #    ## read the chromosome size file
   #    chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])

   #    ## check individual chromosomes   
   #    for rowidx in range(chrsize_df.shape[0]):
   #       print("\n\n rowidx : ", rowidx)
   #       currchr = str(chrsize_df.iloc[rowidx, 0])
   #       print("  processing chromosome : ", currchr, "  chrsize_df.iloc[rowidx, 0] : ", chrsize_df.iloc[rowidx, 0])
         
   #       ## discard any chromosomes other than chr1 to chr22
   #       if (currchr == "chrX" or currchr == "chrY" or currchr == "chrM" or "un" in currchr or "Un" in currchr or "random" in currchr or "Random" in currchr or "_" in currchr):
   #          continue

   #       filename_seqs = OutDir + '/sequences_' + currchr + '.bed'
   #       if (os.path.exists(filename_seqs)):
   #          continue

   #       currchrsize = int(chrsize_df.iloc[rowidx, 1])
   #       print("  its size : ", currchrsize)

   #       ## get the maximum start bin coordinate
   #       max_start_pos = ((currchrsize // Resolution) - 1) * Resolution      
   #       max_end_pos = max_start_pos + Resolution
   #       print("\n max_start_pos : ", max_start_pos, "  max_end_pos : ", max_end_pos)

   #       nodes_list = []
   #       for i in range(0, max_end_pos, Resolution):
   #         nodes_list.append(i)
   #       nodes_list = np.array(nodes_list)

   #       TT = (Span // Resolution) // 2
   #       left_padding = np.zeros(TT).astype(int)
   #       right_padding = np.zeros(TT).astype(int)

   #       nodes_list = np.append(left_padding, nodes_list)
   #       nodes_list = np.append(nodes_list, right_padding)

   #       out_seq_df = pd.DataFrame(data = [], columns = ["chr", "start", "end"])
   #       out_seq_df['start'] = nodes_list
   #       out_seq_df['end'] = nodes_list + Resolution      
   #       out_seq_df['chr'] = currchr

   #       # output sequence file for the current chromosome      
   #       out_seq_df.to_csv(filename_seqs, index = False, header = False, sep = '\t')

   ##===============
   ## new code
   ##===============
   if 1:

      refgenome = options.refgenome
      chrsizefile = options.chrsizefile
      BaseOutDir = options.BaseOutDir
      Resolution = int(options.Resolution)

      print("\n\n ===>> Input parameters <<==== \n\n")
      print("\n refgenome : ", refgenome)
      print("\n chrsizefile : ", chrsizefile)
      print("\n BaseOutDir : ", BaseOutDir)
      print("\n Resolution : ", Resolution)

      OutDir = BaseOutDir + '/seqs_bed/' + refgenome + '/' + str(Resolution)      
      os.system("mkdir -p " + str(OutDir))

      ## use bedtools routine to create the bin interval file
      temp_bin_interval_file = OutDir + '/temp_bin.txt'      
      if (os.path.exists(temp_bin_interval_file) == False):
         os.system("bedtools makewindows -g " + str(chrsizefile) + " -w " + str(Resolution) + " > " + str(temp_bin_interval_file))

      ## read the chromosome size file
      chrsize_df = pd.read_csv(chrsizefile, sep="\t", header=None, names=["chr", "size"])

      ## check individual chromosomes   
      for rowidx in range(chrsize_df.shape[0]):
         print("\n\n rowidx : ", rowidx)
         currchr = str(chrsize_df.iloc[rowidx, 0])
         print("  processing chromosome : ", currchr)
         
         ## discard any chromosomes other than chr1 to chr22
         if (currchr == "chrX" or currchr == "chrY" or currchr == "chrM" or "un" in currchr or "Un" in currchr or "random" in currchr or "Random" in currchr or "_" in currchr):
            continue

         filename_seqs = OutDir + '/sequences_' + currchr + '.bed'         
         if (os.path.exists(filename_seqs) == False):
            os.system("awk \'($1 == \"" + str(currchr) + "\")\' " + str(temp_bin_interval_file) + " > " + str(filename_seqs))

      ## remove temporary files      
      if (os.path.exists(temp_bin_interval_file) == True):
         os.system("rm " + str(temp_bin_interval_file))

   ## close the configuration file
   config_fp.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()


