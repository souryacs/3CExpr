from optparse import OptionParser
import numpy as np
import pandas as pd
import os

##=======================
## script to extract FitHiChIP loops into a common format
## first 5 fields: interacting bins
## field 6: contact count
## subsequent fields: different features

## Sourya Bhattacharyya
## La Jolla Institute for Immunology, La Jolla, CA 92037
##=======================

def parse_options():  
    usage = 'usage: %prog [options]'
    parser = OptionParser(usage)  

    parser.add_option('-I', 
                      dest='InpFile', 
                      default=None, 
                      type='str', 
                      help='Input FitHiChIP interaction file. Mandatory parameter.')

    parser.add_option('-f', 
                      dest='FDRThr', 
                      default=0.1, 
                      type='float', 
                      help='FDR threshold of Loops. Default 0.1')
    
    parser.add_option('--cccol', 
                      dest='cccol', 
                      default=7, 
                      type='int', 
                      help='column containing the contact counts')

    parser.add_option('--pvalcol', 
                      dest='pvalcol', 
                      default=25, 
                      type='int', 
                      help='column containing the p-values')
    
    parser.add_option('--qvalcol', 
                      dest='qvalcol', 
                      default=26, 
                      type='int', 
                      help='column containing the q-values')
    
    parser.add_option('--expcccol', 
                      dest='expcccol', 
                      default=22, 
                      type='int', 
                      help='column containing the expected contact counts. Available for FitHiChIP loops. Default = 22')

    parser.add_option('--bias1col', 
                     dest='Bias1Col', 
                     default=10, 
                     type='int', 
                     help='Column containing bias values in the bias file. Default = 6')   

    parser.add_option('--bias2col', 
                     dest='Bias2Col', 
                     default=16, 
                     type='int', 
                     help='Column containing bias values in the bias file. Default = 6')   

    parser.add_option('-O', 
                      dest='Outfile', 
                      default=None, 
                      type='str', 
                      help='Output loop file with different features. Mandatory parameter.')
    
    (options, args) = parser.parse_args()
    return options, args

##==============
## main routine
##==============
def main():
    options, args = parse_options()

    InpFile = options.InpFile
    Outfile = options.Outfile    
    FDRThr = float(options.FDRThr)    

    qvalcol = int(options.qvalcol)
    cccol = int(options.cccol)
    pvalcol = int(options.pvalcol)
    expcccol = int(options.expcccol)

    Bias1Col = int(options.Bias1Col)
    Bias2Col = int(options.Bias2Col)

    ## extract the contacts, according to the specified FDR threshold
    sys_cmd = "awk \'function max(x,y) {return x>y?x:y}; {if ((NR>1) && ($" + str(qvalcol) + " < " + str(FDRThr) + ")) {if ($" + str(pvalcol) + "==0) {P=500} else {P=-log($" + str(pvalcol) + ")/log(10)}; if ($" + str(qvalcol) + "==0) {Q=500} else {Q=-log($" + str(qvalcol) + ")/log(10)}; E=$" + str(expcccol) + "; B1=$" + str(Bias1Col) + "; B2=$" + str(Bias2Col) + "; if (E==0) {E=1.0}; if (B1==0) {B1=1.0}; if (B2==0) {B2=1.0}; X=$" + str(cccol) + "/(B1 * B2); Y = $" + str(cccol) + "/E; print $1\"\t\"$2\"\t\"$3\"\t\"$5\"\t\"$6\"\t\"$" + str(cccol) + "\"\t\"P\"\t\"Q\"\t\"Y\"\t\"X }}\' " + str(InpFile) + " > " + str(Outfile)
    os.system(sys_cmd)

    ## insert header line 
    sys_cmd = "sed -i -e \'1i chr\tstart_i\tend_i\tstart_j\tend_j\tRawCC\tMinuslog10pval\tMinuslog10qval\tObsByExpCC\tBiasNormCC' " + Outfile
    os.system(sys_cmd)

    ## also replace infinity entries with 500
    ## (to adjust for any nonzero but very low value which is producing infinity via -log10 transformation)
    sys_cmd = "sed -i \'s/inf/500/g\' " + Outfile
    os.system(sys_cmd)
 
################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()

