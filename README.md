# scProfiterole
scProfiterole: Clustering of Single-Cell Proteomic Data Using Graph Contrastive Learning via Spectral Filters 

Download repo from

https://github.com/TencentAILabHealthcare/scPROTEIN

Then replace model.py in the folder the model.py in this github repo.

Run MyTutorial2.py



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj

import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv, EdgeWeightNorm, ChebConv, GATConv, HeteroGraphConv

import sklearn
from sklearn.cluster import KMeans
import scipy
import numpy as np
import sympy
import math

# Local imports
from prototype_loss import *
from utils import *

# --- Reproducibility Header ---
def print_environment_info():
    print("="*40)
    print("ENVIRONMENT INFO FOR REPRODUCIBILITY")
    print("="*40)
    info = {
        "PyTorch": torch.__version__,
        "PyG (Torch Geometric)": torch_geometric.__version__,
        "DGL": dgl.__version__,
        "NumPy": np.__version__,
        "SciPy": scipy.__version__,
        "Scikit-Learn": sklearn.__version__,
        "SymPy": sympy.__version__,
    }
    for lib, ver in info.items():
        print(f"{lib:<22}: {ver}")
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Running on            : {device}")
    if torch.cuda.is_available():
        print(f"GPU Device            : {torch.cuda.get_device_name(0)}")
    print("="*40 + "\n")

print_environment_info()



PyTorch               : 1.11.0+cpu
PyG (Torch Geometric) : 2.0.4
DGL                   : 0.8.2
NumPy                 : 1.23.5
SciPy                 : 1.15.3
Scikit-Learn          : 1.7.2
SymPy                 : 1.14.0
Running on            : CPU
========================================**



# --- Colab Reproducibility ---

Please refer to
https://github.com/mustafaCoskunAgu/scProfiterole/blob/main/scProfiterole.ipynb

You may change the Encoder via "Kernel" in line 154 of MyTutorial2.py 


Note that colab results are obtained with following default hyper-parameters:


parser = argparse.ArgumentParser()
parser.add_argument("--stage1", type=bool, default=True, help='if scPROTEIN starts from stage1')
parser.add_argument("--learning_rate", type=float, default=1e-3, help='learning rate')
parser.add_argument("--num_hidden", type=int, default=64, help='hidden dimension') 
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
parser.add_argument("--threshold", type=float, default=0.3, help='threshold of graph construct')
parser.add_argument("--feature_preprocess", type=bool, default=True, help='feature preprocess')
args =parser.parse_known_args()[0]   
setup_seed(args.seed)
activation = nn.PReLU() if args.activation == 'prelu' else F.relu


encoder = HeatKernel_Encoder(
    
    in_channels=data.num_features,
    
    out_channels=args.num_hidden,  # embedding dimension
    
    activation=F.relu,
    
    K=3,            # number of layers
    
    hidden_dim=64,  # hidden dimension
    
    dropout=0.5,
    
    heat_K=12,       # truncation depth for heat kernel
    
    t=3.0,           # diffusion time
    
    method = 'RW', # Propagation technique
    
    kernel = 'Heat_A'

)
