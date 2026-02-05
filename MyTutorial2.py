# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 09:37:16 2025

@author: Administrator
"""

import argparse
import random
import numpy as np 
import scipy.sparse as sp
import os
from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix

from peptide_uncertainty_estimation.multi_task_heteroscedastic_regression_loss import *
from peptide_uncertainty_estimation.multi_task_heteroscedastic_regression_model import*
from peptide_uncertainty_estimation.peptide_uncertainty_train import*
from peptide_uncertainty_estimation.peptide_uncertainty_utils import*
from utils import *
#from model import*
from model2 import*
from ChebnetII_pro import*
from prototype_loss import*
from train_stage1 import*
from train_stage2 import*
#from visualization import*

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
#from scprotein import *
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score,adjusted_rand_score,normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
import warnings


parser = argparse.ArgumentParser()
parser.add_argument("--stage1", type=bool, default=True, help='if scPROTEIN starts from stage1')
parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate')
parser.add_argument("--num_hidden", type=int, default=400, help='hidden dimension') 
parser.add_argument("--num_proj_hidden", type=int, default=256, help='dimension of projection head')
parser.add_argument("--activation", type=str, default='prelu', help='activation function') 
parser.add_argument("--num_layers", type=int, default=6, help='num of GCN layers')
parser.add_argument("--num_protos", type=int, default=2, help='num of prototypes')
parser.add_argument("--num_changed_edges", type=int, default=50, help='num of added/removed edges')
parser.add_argument("--topology_denoising", type=bool, default=False, help='if scPROTEIN uses topology denoising')
parser.add_argument("--drop_edge_rate_1", type=float, default=0.2, help='dropedge rate for view1')
parser.add_argument("--drop_edge_rate_2", type=float, default=0.4, help='dropedge rate for view2')
parser.add_argument("--drop_feature_rate_1", type=float, default=0.4, help='mask_feature rate for view1')
parser.add_argument("--drop_feature_rate_2", type=float, default=0.2, help='mask_feature rate for view2')
parser.add_argument("--alpha", type=float, default=0.05, help='balance factor')
parser.add_argument("--tau", type=float, default=0.4, help='temperature coefficient')
parser.add_argument("--weight_decay", type=float, default=0.00001, help='weight_decay')
parser.add_argument("--num_epochs", type=int, default=200, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=39788, help='Random seed.') 
parser.add_argument("--threshold", type=float, default=0.15, help='threshold of graph construct')
parser.add_argument("--feature_preprocess", type=bool, default=True, help='feature preprocess')
args =parser.parse_known_args()[0]   
setup_seed(args.seed)
activation = nn.PReLU() if args.activation == 'prelu' else F.relu


protein_list, cell_list, features = load_sc_proteomic_features(args.stage1)  
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = graph_generation(features, args.threshold, args.feature_preprocess).to(device)
# #encoder = ChebEncoder(data.num_features, args.num_hidden, activation, k=args.num_layers, K=10).to(device)
# edge_index = data.edge_index
# num_nodes = edge_index.max().item() + 1
# E = edge_index.size(1)
# edge_weight = torch.ones(E, dtype=torch.float)

# adj_sparse = torch.sparse_coo_tensor(
#     indices=edge_index,
#     values=edge_weight,
#     size=(num_nodes, num_nodes)
# )

# # Coalesce to remove duplicates and sort indices
# adj_sparse = adj_sparse.coalesce()

# adj = adj_sparse @ adj_sparse
# adj = adj.coalesce()
# edge_index = adj.indices()

# data.edge_index = edge_index

#encoder = Encoder(data.num_features, args.num_hidden, activation, k=args.num_layers).to(device)




# encoder = ChebnetII_Encoder(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,
#     activation=F.relu,
#     base_model=lambda: GuidedChebnet_prop(K=4),  # wrap in lambda to pass params
#     k=2,
#     hidden_dim=64,
#     dropout=0.5,
#     dprate=0.2,
#     residual=True 
# )

# encoder = APPNP_Encoder(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,
#     activation=F.relu,
#     hidden_dim=64,
#     dropout=0.5,
#     K=3,         # number of propagation steps
#     alpha=0.1     # teleport probability
# )

# encoder = GPRGNN_Encoder(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,
#     hidden_dim=64,
#     dropout=0.5,
#     K=5,
#     alpha=0.1,
#     Init='gcn'
# ).to(device)

# encoder = GCN_Encoder(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,
#     activation=F.relu,
#     base_model=GCNConv,
#     k=3,
#     hidden_dim=64,
#     dropout=0.5
# )
#encoder = GPRGNN_Encoder(data.num_features, 64, 64, K=5, alpha_decay=0.7).to(device)


encoder = HeatKernel_Encoder(
    in_channels=data.num_features,
    out_channels=args.num_hidden,  # embedding dimension
    activation=F.relu,
    K=3,            # number of layers
    hidden_dim=400,  # hidden dimension
    dropout=0.5,
    heat_K=6,       # truncation depth for heat kernel
    t=2.0,           # diffusion time
    method = 'RW', # Propagation technique
    kernel = 'RWR_T'
)

#encoder = Encoder(data.num_features, args.num_hidden, activation, k=args.num_layers).to(device)

# encoder = BetaKernel_Encoder(in_channels=data.num_features,
#                            out_channels=args.num_hidden,
#                            hidden_dim=64,
#                            K=4,
#                            num_layers=3)

# encoder = BetaKernelEncoderAPPNP(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,
#     hidden_dim=64,
#     K=3,
#     dropout=0.5
# ).to(device)



# encoder = BetaKernel_Encoder(in_channels=data.num_features,
#                            out_channels=args.num_hidden,
#                            hidden_dim=64,
#                            K=5,
#                            num_layers=3)

# encoder = Mono_GEncoder(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,  # embedding dimension
#     activation=F.relu,
#     K=3,            # number of layers
#     hidden_dim=64,  # hidden dimension
#     dropout=0.5,
#     heat_K=5,       # truncation depth for heat kernel
#     t=2.0,           # diffusion time
#     Init = 'Chebyshev',  # Equadistance, Cheb or 
#     lower = -0.9, 
#     upper = 0.9, 
#     nameFunc = 'g_low_pass'
# )



# encoder = GPRGNN_Encoder(
#     in_channels=data.num_features,
#     out_channels=args.num_hidden,
#     activation=F.relu,
#     K=5,              # propagation steps
#     hidden_dim=64,
#     dropout=0.5,
#     alpha=0.1,         # teleport probability
#     Init='PPR'         # init type for gamma
# )

model = Model(encoder, args.num_hidden, args.num_proj_hidden, args.tau).to(device)
scPROTEIN = scPROTEIN_learning(model,device, data, args.drop_feature_rate_1,args.drop_feature_rate_2,args.drop_edge_rate_1,args.drop_edge_rate_2,
                 args.learning_rate, args.weight_decay, args.num_protos, args.topology_denoising, args.num_epochs, args.alpha, args.num_changed_edges,args.seed)

scPROTEIN.train()

embedding = scPROTEIN.embedding_generation()
embedding.shape


def purity_score(y_true, y_pred):
    contingency_matrix1 = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix1, axis=0)) / np.sum(contingency_matrix1) 


def dimension_reduce(embedding):
    X_trans_PCA = PCA(n_components=30, random_state=seed).fit_transform(embedding)  
    X_trans = TSNE(n_components=2,random_state=seed).fit_transform(X_trans_PCA)
    return X_trans

seed = 1
Y_cell_type_label = load_cell_type_labels()
label_dict = {'sc_m0':0, 'sc_u':1}
target_names = ['Macrophage','Monocyte']
Y_label = np.array(itemgetter(*list(Y_cell_type_label))(label_dict))

k_means = KMeans(n_clusters=len(target_names))
y_predict = k_means.fit_predict(embedding)
df_result = pd.DataFrame()
df_result['ARI'] = [np.round(adjusted_rand_score(Y_label,y_predict),3)]
df_result['ASW'] = [np.round(silhouette_score(embedding,y_predict),3)]
df_result['NMI'] = [np.round(normalized_mutual_info_score(Y_label,y_predict),3)]
df_result['PS'] = [np.round(purity_score(Y_label,y_predict),3)]
print('cell clustering result:')
print(df_result)

X_trans_learned = dimension_reduce(embedding)

# --- Improved Plot ---
import matplotlib.pyplot as plt
import seaborn as sns

# Use a clean style
sns.set(style="whitegrid", context="talk", font_scale=1.2)

# Choose distinct, soft colors
palette = sns.color_palette("Set2", len(target_names))

fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

# Scatter each cluster
for i, name in enumerate(target_names):
    ax.scatter(
        X_trans_learned[Y_label == i, 0],
        X_trans_learned[Y_label == i, 1],
        s=80,
        alpha=0.8,
        color=palette[i],
        edgecolors="black",
        linewidth=0.6,
        label=name
    )

# Aesthetic improvements
ax.set_xlabel("t-SNE 1", fontsize=13, fontweight='bold')
ax.set_ylabel("t-SNE 2", fontsize=13, fontweight='bold')
ax.set_title("scPROTEIN Cell Clustering  RWR - Interpolated ( α = 0.95)", fontsize=15, fontweight='bold', pad=15)

# Remove axis ticks for a cleaner look
ax.set_xticks([])
ax.set_yticks([])

# Add grid and legend
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend(frameon=True, loc='best', fontsize=11, title="Cell Type", title_fontsize=12)

# Add subtle background frame
sns.despine(trim=True)

plt.tight_layout()
#plt.savefig("TSEArnoldiRWR_95.eps", format="eps", bbox_inches="tight")
plt.show()






# # plot
# colors = [plt.cm.Set2(2), plt.cm.Set2(1)]
# fig = plt.figure(figsize=(5,5))
# for i in range(len(target_names)):
#     plt.scatter(X_trans_learned[Y_label == i, 0]  
#                 , X_trans_learned[Y_label == i, 1] 
#                 , s = 10  
#                 , color=colors[i]  
#                 , label=target_names[i] 
#                 )
# plt.xlabel('TSNE 1')
# plt.ylabel('TSNE 2')
# plt.xticks([])
# plt.yticks([])
# plt.title('scPROTEIN') 

