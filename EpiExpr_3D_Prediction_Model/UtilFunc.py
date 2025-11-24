##===================
## various utility functions and class definitions
## 1) class definition of chunks - related to training and testing the model
## 2) plotting the correlation between the predicted and observed expression
##===================

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pickle
import os
import re
import seaborn as sns
import networkx as nx
from scipy.stats import entropy
import string
from sklearn.preprocessing import normalize
import math
from torchmetrics.functional import pearson_corrcoef
torch.set_printoptions(precision=10)

PLOT_FONTSIZE = 12
PLOT_LABELSIZE = 10

##===============
## doubly stochastic normalization of a numpy matrix
## employing Sinkhorn-Knopp algorithm
def sinkhorn_knopp(M, num_iters=100, epsilon=1e-6):
    for _ in range(num_iters):
        if 0:
            # Normalize rows
            row_sums = M.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            M = M / row_sums[:, np.newaxis]
            
            # Normalize columns
            col_sums = M.sum(axis=0)
            col_sums[col_sums == 0] = 1  # Avoid division by zero
            M = M / col_sums[np.newaxis, :]

        # Normalize rows
        M = normalize(M, norm='l1', axis=1)
        # Normalize columns
        M = normalize(M.T, norm='l1', axis=1).T
        # Check for convergence
        row_sums = M.sum(axis=1)
        col_sums = M.sum(axis=0)
        
        # Check for convergence
        if np.all(np.abs(row_sums - 1) < epsilon) and np.all(np.abs(col_sums - 1) < epsilon):
            break
    
    return M

##===============
# Custom sorting function
def sort_key(filename):
    # Extract the numeric part using a regular expression
    match = re.match(r"(\d+).pkl", filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Return a large number for non-matching filenames

##==============
## class definition of data
## Epigenomic data, chromatin contact and CAGE label
class TorchDataClass_Epigenome_Contact:
    def __init__(self, 
                 X_1d, 
                 bin_idx, 
                 Y,
                 edge_idx, 
                 edge_feat):
        
        ## epigenomic 1D track
        self.X_1d = X_1d
        ## bin indices - used to obtain the middle sliding window
        self.bin_idx = bin_idx
        ## output CAGE label
        self.Y = Y
        ## edge index (for GAT)
        self.edge_idx = edge_idx
        ## edge features (for GAT)
        self.edge_feat = edge_feat

##==============
## loss function using true and predicted labels
def poisson_loss(y_true, mu_pred):
    if 0:
        nll = torch.mean(torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred))
    else:
        ## modified - sourya
        nll = torch.mean(torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred + 1))
    return nll

def poisson_loss_individual(y_true, mu_pred):
    if 0:
        nll = torch.lgamma(y_true + 1) + mu_pred - y_true * torch.log(mu_pred)
    else:
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
def Plot_Corr_log2_with_contact(plotfile, true_cage, pred_cage, n_contacts, plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(np.log2(true_cage+1),np.log2(pred_cage+1), c=np.log2(n_contacts+1), s=100, cmap=cm, alpha=.7)  #, edgecolors='')
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

def Plot_Corr_with_contact(plotfile, true_cage, pred_cage, n_contacts, plotlabel, NLL_gat, rho_gat, sp_gat):

    plt.figure(figsize=(9,8))
    cm = plt.cm.get_cmap('viridis_r')
    sc = plt.scatter(true_cage, pred_cage, c=n_contacts, s=100, cmap=cm, alpha=.7)  #, edgecolors='')
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


##=========
## Plot attention score heatmap from GAT output
##=========
def Plot_Att_Heatmap(plotfile, attn_scores):
    ## attn_scores: 2nd output from the GATv2conv routine
    attn_scores_np = attn_scores[1].detach().clone().cpu().numpy()      ## first convert it to numpy
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_scores_np, annot=True, fmt=".2f", cmap='viridis')
    plt.title("Attention Scores Heatmap")
    plt.xlabel("Heads")
    plt.ylabel("Edges")
    plt.tight_layout()
    plt.savefig(plotfile)

##=========
## Dump attention scores from GAT output
##=========
def Dump_Att_Score(textfile, attn_scores):
    ## attn_scores: 2nd output from the GATv2conv routine
    ## tuple: (edge_index, attention_weights)
    attn_scores_np = attn_scores[1].detach().clone().cpu().numpy()     
    edge_idx = attn_scores[0].detach().clone().cpu().numpy()
    print(" Dump attention scores - text file : ", str(textfile), 
          " attn_scores_np shape : ", str(attn_scores_np.shape), 
          " edge_idx shape : ", str(edge_idx.shape))
    # Create a pandas DataFrame
    attn_df = pd.DataFrame(attn_scores_np, columns=[f'Head_{i+1}' for i in range(attn_scores_np.shape[1])])
    # Add edge information (optional)
    attn_df['Edge'] = [f'{edge_idx[0][i]}-{edge_idx[1][i]}' for i in range(edge_idx.shape[1])]
    # Save the DataFrame to a CSV file
    attn_df.to_csv(textfile, index=False)

##============
## compute pearson correlation coefficient between two torch tensors
##============
def compute_pearson_corr(x, y):
    x = x.flatten()
    y = y.flatten()
    if torch.std(x) == 0 or torch.std(y) == 0:
        return 0    #torch.tensor(float('nan'))  # or 0, if you prefer
    return pearson_corrcoef(x, y).numpy()

##============
## compute pearson correlation coefficient between two torch tensors, each with batched input
##============
def batch_pearson_corrcoef(preds, targets):
    # preds and targets shape: [B, 1, N]
    B = preds.shape[0]
    preds = preds.squeeze(1)  # [B, N]
    targets = targets.squeeze(1)  # [B, N]

    corrs = []
    for i in range(B):
        if torch.std(preds[i]) == 0 or torch.std(targets[i]) == 0:
            corr = torch.tensor(0)
        else:
            corr = pearson_corrcoef(preds[i], targets[i])
        corrs.append(corr)

    return torch.stack(corrs).mean()  # return average Pearson over batch

##=========
## Plot attention entropy histogram distribution from GAT output
##=========

# # Draws the entropy histogram.
# def draw_entropy_histogram(curraxis, entropy_array, title, color='blue', uniform_distribution=False, num_bins=30):
#     max_value = np.max(entropy_array)
#     bar_width = (max_value / num_bins) * (1.0 if uniform_distribution else 0.75)
#     histogram_values, histogram_bins = np.histogram(entropy_array, bins=num_bins, range=(0.0, max_value))
#     curraxis.plt.bar(histogram_bins[:num_bins], histogram_values[:num_bins], width=bar_width, color=color)
#     curraxis.plt.xlabel(f'entropy bins')
#     curraxis.plt.ylabel(f'# of node neighborhoods')
#     plt.title(title)

# def Plot_Att_Entropy(plotfile, attn_scores, gatmodel, num_of_nodes):
#     ## use the defined GAT structure in "Model_Epigenomic_Contact.py"
#     num_heads_per_layer = gatmodel.num_head_GAT

#     ## attn_scores: 2nd output from the GATv2conv routine
#     ## tuple: (edge_index, attention_weights)
#     attn_scores_np = attn_scores[1].detach().clone().cpu().numpy()     
#     edge_idx = attn_scores[0].detach().clone().cpu().numpy()
        
#     target_node_ids = edge_idx[1].cpu().numpy()

#     fig = plt.figure()

#     for head_id in range(num_heads_per_layer):
#         attn_score_curr_head = attn_scores[:, head_id]


    












#         # attention shape = (N, NH, 1) -> (N, NH) - we just squeeze the last dim it's superfluous
#         all_attention_weights = gatmodel.GATModel.GATLayers[layer_id].attention_weights.squeeze(dim=-1).cpu().numpy()

# #             uniform_dist_entropy_list = [] 
# #             neighborhood_entropy_list = []

# #             for target_node_id in range(num_of_nodes):  # find every the neighborhood for every node in the graph
# #                 # These attention weights sum up to 1 by GAT design so we can treat it as a probability distribution
# #                 neigborhood_attention = all_attention_weights[target_node_ids == target_node_id].flatten()
# #                 # Reference uniform distribution of the same length
# #                 ideal_uniform_attention = np.ones(len(neigborhood_attention))/len(neigborhood_attention)
# #                 # Calculate the entropy
# #                 neighborhood_entropy_list.append(entropy(neigborhood_attention, base=2))
# #                 uniform_dist_entropy_list.append(entropy(ideal_uniform_attention, base=2))

# #             title = f'entropy histogram layer={layer_id}, attention head={head_id}'
# #             draw_entropy_histogram(ax[layer_id, head_id], uniform_dist_entropy_list, 
# #                                    title, color='orange', uniform_distribution=True)
# #             draw_entropy_histogram(ax[layer_id, head_id], neighborhood_entropy_list, title, color='dodgerblue')

# #     fig.savefig(plotfile)


# def Plot_Attention_Weight_Graph(plotfile, attn_scores, num_of_nodes):
#     ## attn_scores: 2nd output from the GATv2conv routine
#     ## tuple: (edge_index, attention_weights)
#     attn_scores_np = attn_scores[1].detach().clone().cpu().numpy()     
#     edge_idx = attn_scores[0].detach().clone().cpu().numpy()

#     target_node_ids = range(0, num_of_nodes)
    
#     ## plot attention scores for individual heads
#     fig = plt.figure(); 
#     # plt.clf()
#     # fig, ax = plt.subplots(attn_scores_np.shape[1], 1, num=1)
#     # print("\n =>> attn_scores_np.shape[1] : ", str(attn_scores_np.shape[1]))

#     ## for each head, print the attention scores in a graph
#     for j in range(attn_scores_np.shape[1]):
#         targetstr = "_head_" + str(j) + "."
#         plotfile1 = plotfile.replace(".", targetstr)
#         ## create a networkx graph with the specified nodes, edges, and attention weights as the edge weights
#         G = nx.Graph()
#         G.add_nodes_from(target_node_ids)
#         for i in range(edge_idx.shape[1]):
#             ## do not add the self loops 
#             if edge_idx[0, i] != edge_idx[1, i]:
#                 G.add_edge(edge_idx[0, i], edge_idx[1, i], weight = attn_scores_np[i, j])
#         ## extract edge weights to use as the scores
#         edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
#         nx.draw_networkx(G, pos = nx.kamada_kawai_layout(G), with_labels = True, node_color='skyblue', 
#                          edgelist = edges, edge_color = weights, edge_vmin=0, edge_vmax=1,
#                          node_size = 0.5, width = 0.1, font_size = 0.5)        
#         fig.savefig(plotfile1)

# ##=========
# ## Filter edge indices such that the columns with at least one entry
# ## from the specific values, will only be retained
# ##=========
# def Filter_Edge_Entries(edge_idx, allowed_val):
#     # Step 1: Create a mask to identify columns with at least one entry in V
#     mask = torch.any(torch.isin(edge_idx, allowed_val.unsqueeze(1)), dim=0)
#     # Step 2: Use the mask to filter the columns of T
#     edge_idx = edge_idx[:, mask]
#     return edge_idx




