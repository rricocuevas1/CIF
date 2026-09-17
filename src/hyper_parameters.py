import os
from torch.nn import ReLU
from torch_geometric.nn import global_add_pool #global_mean_pool

# Trainer hyperparameters
trainer_hparams = {
    'patience' : 64, # Patience
    'patience_hparam_tun': 32, # Patience during h param tunning
    'batch_size' : { # Batch Size
        "SPMotif_b_05": 64,
        "SPMotif_b_07": 64,
        "SPMotif_b_09": 64,
        "Graph_SST2": 64,
        "Molhiv": 64,
        "MNIST_75sp": 64,
        "SYN_binary": 64,
        "SYN_multi": 64
    },
    'epochs_hparam_tun': 100, # Maximum number of epochs during h param tunning
    'epochs' : 200, # Maximum number of epochs
    'max_norm' : 1, # Maximum norm for gradient clipping
    'train_frac': 0.60, # Train split
    'val_frac' : 0.20, # Validation split
    'test_frac' : 0.20 # Test split
}
# Model hyperparameters
DIM_EMBEDDING_SPACE = 32
HIDDEN_CHANNELS_MLP = 16
HIDDEN_CHANNELS_GNN = 16
# Sharpness of MC approximation
N_SAMPLES = int(os.environ.get("N_SAMPLES", 16))
SIGMA = 0.01
DROPOUT = 0.1
N_HEADS = 4
N_LAYERS = 2
CAT_OR_ADD = 'cat'
POOLING = global_add_pool # We use add_pool as default following related work (CAL, ICL)
# Hyper-parameters of the graph encoders
# MPNNs (GCN, GAT, GIN)
MPNN_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'global_node_att': False,
    'global_edge_att': False
}
# GTs
# GraphGPS
GraphGPS_encoder_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'rwse_walk_length' : 16, # Random walk length for the SE
    'pe_dim' : 8, # PE dim
    'num_layers': N_LAYERS, # Number of GPS layers
    'attn_type' : 'performer', # Linear Global attention mechanisim
    'attn_heads': N_HEADS, # Number of attention heads
    'attn_kwargs' : {'dropout': DROPOUT},
    'global_node_att': True,
    'global_edge_att': False
}
# GrokFormer
GrokFormer_encoder_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'num_layers': N_LAYERS, # Number of GrokFormer layers
    'k': 8, # Number of eigenvalues/eigenvectors to keep from the Laplacian decomposition
    'nheads': N_HEADS, # Number of attention heads
    'sine_dim': 16, # Eigenvalue encoding dim
    'tran_dropout': DROPOUT,
    'feat_dropout': DROPOUT,
    'prop_dropout': DROPOUT,
    'global_node_att': True,
    'global_edge_att': False
}
# DualFormer
DualFormer_encoder_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_gnn': HIDDEN_CHANNELS_GNN, # Hidden dimension of the GNNs
    'activation': ReLU(),
    'num_gnns': N_LAYERS, 
    'num_sa': 1,
    'num_heads': N_HEADS,
    'dropout': DROPOUT,
    'dropout_sa': DROPOUT,
    'alpha': 0.1,
    'lammda': 0.1,
    'GraphConv': 'sgc',
    'use_bn': True,
    'global_node_att': True,
    'global_edge_att': False
}
# Graph Neural Network encoders hyperparameters        
GNN_encoders_hparams = {
    'GCN_encoder': MPNN_hparams,
    'GAT_encoder': MPNN_hparams,
    'GIN_encoder': MPNN_hparams,
    'GraphGPS_encoder': GraphGPS_encoder_hparams,
    'GrokFormer_encoder': GrokFormer_encoder_hparams,
    'DualFormer_encoder': DualFormer_encoder_hparams
}
#  GNNs 
GNN_hparams = {
    'hidden_channels_mlp': HIDDEN_CHANNELS_MLP, # Hidden dimension of MLPs
}
# CGNNs
CGNN_hparams = {
    'gnn_out_channels': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'cat_or_add': CAT_OR_ADD,
    'with_random': True,
    'eval_random': False,
    'dropout': DROPOUT,
    'hidden_channels_mlp': HIDDEN_CHANNELS_MLP, # Hidden dimension of MLPs
}
# DIR
DIR_hparams = {
    'c_i_f': False
}
# CAL
CAL_hparams = {
    's': 0.5,
    'c': 1.0,
    'combined': 0.5,
    'layers': N_LAYERS,
    'c_i_f': False
}
# ICL
ICL_hparams = {
    'mgda_model': 'loss+',  # gradient normalization: 'loss+', 'loss', 'l2', 'none'
    'c_i_f': False
}
# ACE
ACE_hparams = {
    "c_weight": 0.7,
    "o_weight": 1.0,
    "co_weight": 0.5,
    'c_i_f': False
}
# CIF
CIF_hparams = {
    'causal_guidance': True, # Causal guidance
    'n_samples_c': N_SAMPLES, # Number of causal samples per graph
    'n_samples_s': N_SAMPLES, # Number of spurious samples per causal sample
    'n_samples_z': N_SAMPLES,  # Number of samples in z (in the embedding space)
    'sigma': 0.01, # The standard deviation in N(.,.) for sampling noise,
    'out_channels_causal': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'out_channels_spurious': DIM_EMBEDDING_SPACE, # Dimension of the embedding space
    'hidden_channels_mlp': HIDDEN_CHANNELS_MLP, # Hidden dimension of the MLPs
    'cat_or_add': CAT_OR_ADD, # Summed or concatenated representations
    'c_i_f': True # Causal Information Flow flag
}
# Models 
model_class_hparams = {
    "GNN": GNN_hparams,
    "CIF": CIF_hparams,
    "CIF_NoJ_MC": CIF_hparams,
    "CIF_J_NoMC": CIF_hparams,
    "DIR": DIR_hparams,
    "CAL": CAL_hparams,
    "ICL": ICL_hparams,
    "ACE": ACE_hparams,
}