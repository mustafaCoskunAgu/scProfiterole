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
