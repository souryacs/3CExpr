##===================
## various utility functions and class definitions
## 1) class definition of chunks - related to training and testing the model
## 2) plotting the correlation between the predicted and observed expression
##===================

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re
import math
import pickle

torch.set_printoptions(precision=10)

PLOT_FONTSIZE = 12
PLOT_LABELSIZE = 10

# Custom sorting function
def sort_key(filename):
    # Extract the numeric part using a regular expression
    match = re.match(r"(\d+).pkl", filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number for non-matching filenames

##==============
## class definition of data
## Epigenomic data and CAGE label
class TorchDataClass_Epigenome:
    def __init__(self, 
                 X_1d, 
                 bin_idx, 
                 Y):
        ## epigenomic 1D track
        self.X_1d = X_1d
        ## bin indices - used to obtain the middle sliding window
        self.bin_idx = bin_idx
        ## output CAGE label
        self.Y = Y

##==============
## loss function using true and predicted labels
def poisson_loss(y_true, mu_pred):
    if 0:
        nll = torch.mean(torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred))
    ## modified - sourya
    nll = torch.mean(torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred + 1))
    return nll

def poisson_loss_individual(y_true, mu_pred):
    if 0:
        nll = torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred)
    ## modified - sourya
    nll = torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred + 1)
    return nll

##============
## plot functions
def set_axis_style(ax, labels, positions_tick):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_tick_params(labelsize=15)
    ax.set_xticks(positions_tick)
    ax.set_xticklabels(labels, fontsize=20)

def add_label(violin, labels, label):
    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))

##=========
## correlation of true and predicted CAGE expression (log2 scale)
##=========
def Plot_Corr_log2(plotfile, true_cage, pred_cage, plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(np.log2(true_cage+1),np.log2(pred_cage+1), s=100, cmap=cm, alpha=.7)  #, edgecolors='')
    # plt.xlim((-.5,15))
    # plt.ylim((-.5,15))
    plt.title(plotlabel, fontsize=PLOT_FONTSIZE)
    plt.xlabel("log2 (true + 1)", fontsize=PLOT_FONTSIZE)
    plt.ylabel("log2 (pred + 1)", fontsize=PLOT_FONTSIZE)
    plt.tick_params(axis='x', labelsize=PLOT_LABELSIZE)
    plt.tick_params(axis='y', labelsize=PLOT_LABELSIZE)
    plt.grid(alpha=.5)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0,(np.max(np.log2(pred_cage+1))+1), 
             'R= ' + "{:5.3f}".format(rho_gat) + '  SP= ' + "{:5.3f}".format(sp_gat) + '  NLL= '+str(np.float16(NLL_gat)), 
             horizontalalignment='left', 
             verticalalignment='top', 
             bbox=props, 
             fontsize=PLOT_FONTSIZE)
    cbar = plt.colorbar(sc)
    cbar.set_label(label='log2 (n + 1)', size=PLOT_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_LABELSIZE)
    #plt.show()
    plt.tight_layout()
    plt.savefig(plotfile)

def Plot_Corr(plotfile, true_cage, pred_cage, plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(true_cage, pred_cage, s=100, cmap=cm, alpha=.7)  #, edgecolors='')
    # plt.xlim((-.5,15))
    # plt.ylim((-.5,15))
    plt.title(plotlabel, fontsize=PLOT_FONTSIZE)
    plt.xlabel("true", fontsize=PLOT_FONTSIZE)
    plt.ylabel("pred", fontsize=PLOT_FONTSIZE)
    plt.tick_params(axis='x', labelsize=PLOT_LABELSIZE)
    plt.tick_params(axis='y', labelsize=PLOT_LABELSIZE)
    plt.grid(alpha=.5)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0,(np.max(pred_cage)+1), 
             'R= ' + "{:5.3f}".format(rho_gat) + '  SP= ' + "{:5.3f}".format(sp_gat) + '  NLL= '+str(np.float16(NLL_gat)), 
             horizontalalignment='left', 
             verticalalignment='top', 
             bbox=props, 
             fontsize=PLOT_FONTSIZE)
    cbar = plt.colorbar(sc)
    cbar.set_label(label='n', size=PLOT_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_LABELSIZE)
    #plt.show()
    plt.tight_layout()
    plt.savefig(plotfile)


##=========
## correlation of true and predicted CAGE expression (log2 scale)
## also plotting the number of contacts as a color scale
##=========
def Plot_Corr_log2_with_contact(plotfile, true_cage, pred_cage, n_contacts, 
                                plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(np.log2(true_cage+1),np.log2(pred_cage+1), 
                     c=np.log2(n_contacts+1), 
                     s=100, cmap=cm, alpha=.7)  #, edgecolors='')
    # plt.xlim((-.5,15))
    # plt.ylim((-.5,15))
    plt.title(plotlabel, fontsize=PLOT_FONTSIZE)
    plt.xlabel("log2 (true + 1)", fontsize=PLOT_FONTSIZE)
    plt.ylabel("log2 (pred + 1)", fontsize=PLOT_FONTSIZE)
    plt.tick_params(axis='x', labelsize=PLOT_LABELSIZE)
    plt.tick_params(axis='y', labelsize=PLOT_LABELSIZE)
    plt.grid(alpha=.5)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0,(np.max(np.log2(pred_cage+1))+1), 
             'R= ' + "{:5.3f}".format(rho_gat) + '  SP= ' + "{:5.3f}".format(sp_gat) + '  NLL= '+str(np.float16(NLL_gat)), 
             horizontalalignment='left', 
             verticalalignment='top', 
             bbox=props, 
             fontsize=PLOT_FONTSIZE)
    cbar = plt.colorbar(sc)
    cbar.set_label(label='log2 (n + 1)', size=PLOT_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_LABELSIZE)
    #plt.show()
    plt.tight_layout()
    plt.savefig(plotfile)

def Plot_Corr_with_contact(plotfile, true_cage, pred_cage, n_contacts, 
                           plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(true_cage, pred_cage, 
                     c=n_contacts, s=100, cmap=cm, alpha=.7)  #, edgecolors='')
    # plt.xlim((-.5,15))
    # plt.ylim((-.5,15))
    plt.title(plotlabel, fontsize=PLOT_FONTSIZE)
    plt.xlabel("true", fontsize=PLOT_FONTSIZE)
    plt.ylabel("pred", fontsize=PLOT_FONTSIZE)
    plt.tick_params(axis='x', labelsize=PLOT_LABELSIZE)
    plt.tick_params(axis='y', labelsize=PLOT_LABELSIZE)
    plt.grid(alpha=.5)
    props = dict(boxstyle='round', facecolor='white', alpha=1)
    plt.text(0,(np.max(pred_cage)+1), 
             'R= ' + "{:5.3f}".format(rho_gat) + '  SP= ' + "{:5.3f}".format(sp_gat) + '  NLL= '+str(np.float16(NLL_gat)), 
             horizontalalignment='left', 
             verticalalignment='top', 
             bbox=props, 
             fontsize=PLOT_FONTSIZE)
    cbar = plt.colorbar(sc)
    cbar.set_label(label='n', size=PLOT_FONTSIZE)
    cbar.ax.tick_params(labelsize=PLOT_LABELSIZE)
    #plt.show()
    plt.tight_layout()
    plt.savefig(plotfile)


##=====================
## finds the prime factor sequence to reach from A to B (A < B)
##=====================
def prime_factors_sequence(A, B):
    if B % A != 0:
        raise ValueError("B is not a multiple of A")
    
    factors = []
    current = A
    
    # Helper function to check if a number is prime
    def is_prime(n):
        if n <= 1:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    # Helper function to get the next prime factor
    def next_prime_factor(n, start=2):
        for i in range(start, int(math.sqrt(n)) + 1):
            if n % i == 0 and is_prime(i):
                return i
        return n if is_prime(n) else None
    
    # Calculate the prime factors sequence
    while current != B:
        factor = next_prime_factor(B // current)
        if factor is None:
            raise ValueError("Could not find a suitable prime factor")
        
        factors.append(factor)
        current *= factor
    
    return factors


