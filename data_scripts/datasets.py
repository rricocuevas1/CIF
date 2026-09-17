# Read this file and download manually the datasets when required, place them on a folder
# called "data" in the parent directory. Then run compute_dataset_stats.py
# to get the info about the datasets along with the download of those datasets
# that have an automatic download. 
# 
# If you see printed in the terminal: 
#       > the total number of graphs, 
#       > the average nodes per graph, 
#       > the average edges per graph,
#       > the number of classes, 
#       > the head of x
#       > the head of edge_attributes/None 
# for all datasets then you have successfully, downloaded and process all datasets as intended.
# They are stored in the "data" folder.

# Imports
import sys
from pathlib import Path
import torch
from torch_geometric.transforms import BaseTransform
# Data imports 
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
# Specific dataset imports
from data_preparation.mnistsp_dataset import MNIST75sp
from data_preparation.spmotif_dataset import SPMotif
from data_preparation.syn_dataset import SYN_dataset
from ogb.graphproppred import PygGraphPropPredDataset
from data_preparation.graphsst2_dataset import get_dataset_sst
# Imports for the PE/SE required by the GTs
import torch_geometric.transforms as T
# Trainer imports
# Add path so hyper_parameters is findable
sys.path.insert(0, str(Path(__file__).parent.parent)) 
from src.hyper_parameters import trainer_hparams, GraphGPS_encoder_hparams
import warnings
warnings.filterwarnings("ignore")


# Transforms and Pre-transforms
# Data pre-transformations required by the GTs
# Pre-transform for the GrokFormer GT
class AddLaplacianEigen(BaseTransform):
    def __init__(self, k=8):
        self.k = k

    def __call__(self, data):
        num_nodes = data.num_nodes
        edge_index = data.edge_index

        # Build adjacency matrix
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj = adj + adj.T
        adj[adj > 0] = 1.0

        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        deg = adj.sum(dim=1)
        deg[deg == 0] = 1.0
        deg_inv_sqrt = deg.pow(-0.5)
        L = torch.eye(num_nodes) - torch.diag(deg_inv_sqrt) @ adj @ torch.diag(deg_inv_sqrt)

        # Eigendecomposition
        e, u = torch.linalg.eigh(L)

        # Keep top-k or pad if graph has fewer than k nodes
        k = min(self.k, num_nodes)
        eigenvalues = torch.zeros(self.k)
        eigenvectors = torch.zeros(num_nodes, self.k)
        eigenvalues[:k] = e[:k]
        eigenvectors[:, :k] = u[:, :k]

        # Store eigenvalues repeated per node (for PyG batching)
        data.eigenvalues = eigenvalues.unsqueeze(0).expand(num_nodes, -1)  # (num_nodes, k)
        data.eigenvectors = eigenvectors  # (num_nodes, k)

        return data

# List of data pre-transformations required by the GTs
pre_transform = T.Compose([
    # For GraphGPS. Attributes: RWSE
    T.AddRandomWalkPE(walk_length=GraphGPS_encoder_hparams['rwse_walk_length'], attr_name='RWSE'),
    # For GrokFormer. Attributes: eigenvalues, eigenvectors.
    AddLaplacianEigen(k=8),
])


# Datasets:

# SYN dataset family (Requires manual generation)
# The dataset can be generated via data_preparation/spmotif_gen/spmotif.ipynb
# bias in [0.1, 0.3, 0.5, 0.7, 0.9]
def get_SYN(bias, binary, batch_size):
    bin_vs_multi = "binary" if binary else "multi"
    bias_str = str(bias).replace(".", "")
    root = f'../data/SYN_{bin_vs_multi}/bias_{bias_str}/'

    train_dataset = SYN_dataset(root=root, mode='train', bias=bias, binary=binary, pre_transform=pre_transform)
    val_dataset = SYN_dataset(root=root, mode='val', bias=bias, binary=binary, pre_transform=pre_transform)
    test_dataset = SYN_dataset(root=root, mode='test', bias=bias, binary=binary, pre_transform=pre_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

    input_channels = {
        'in_channels': train_dataset[0].x.shape[1],
        'in_channels_e': None
    }
    return input_channels, loader


# SPMotif dataset family (Requires manual generation)
# The dataset can be generated via data_preparation/SYN_binary_gen/generate_SYN_binary.py
# bias in [0.5, 0.7, 0.9]
def get_SPMotif(bias, batch_size):
    train_dataset = SPMotif(
        root='../data/'+ f'SPMotif_{bias}/', 
        mode='train',
        pre_transform=pre_transform
    )
    val_dataset = SPMotif(
        root='../data/'+ f'SPMotif_{bias}/', 
        mode='val',
        pre_transform=pre_transform
    )
    test_dataset = SPMotif(
        root='../data/'+ f'SPMotif_{bias}/', 
        mode='test',
        pre_transform=pre_transform
    )

    # Create the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    input_channels = {
        'in_channels': 4,   # 4 node features
        'in_channels_e': 1,  # 1 edge feature
        'rwse_dim': GraphGPS_encoder_hparams['rwse_walk_length']
    }
    return input_channels, loader


# The MNIST-75sp dataset family (Requires manual download)
# The original version of the files is available at:
# https://drive.google.com/drive/folders/1Prc-n9Nr8-5z-xphdRScftKKIxU4Olzh
# We extend this dataset with more noise levels: noise_level in [0.2, 0.4, 0.6, 0.8] 
# The default noise level is 0.4
def get_mnistsp(noise_level, batch_size):
    # Following the original src, the dataset is split as
    # 20K Train - 5K Validation - 10K Test
    n_train_data, n_val_data = 20000, 5000

    # Load the train data
    train_val = MNIST75sp(
        root='../data/MNISTSP/', 
        mode='train',
        pre_transform=pre_transform) # This one contains 60K graphs
    
    # Sample the training and validation data y 
    perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
    train_val = train_val[perm_idx]
    train_dataset, val_dataset = train_val[:n_train_data], train_val[-n_val_data:]

    # Load the test data
    test_dataset = MNIST75sp(
        root='../data/MNISTSP/', 
        mode='test',
        pre_transform=pre_transform)
    
    # Add the color noise to the test split
    # This is a fixed tensor of shape [total_test_nodes, 3], one RGB noise vector per node
    color_noises = torch.load('../data/MNISTSP/raw/mnist_75sp_color_noise.pt').view(-1,3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    noises = []

    n_samples = 0
    for graph in test_loader: 
        noises.append(color_noises[n_samples:n_samples + graph.x.size(0), :] * noise_level)
        n_samples += graph.x.size(0)
    test_dataset.data.x = test_dataset.data.x.clone()
    test_dataset.data.x[:, :3] = test_dataset.data.x[:, :3] + torch.cat(noises)
    
    # Define the loaders and record the in-channels
    # Create the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    input_channels = {
        'in_channels': 5,
        'in_channels_e': None
    }
    return input_channels, loader


# Molhiv dataset (Automatic download)
def get_molhiv(batch_size):
    dataset = PygGraphPropPredDataset(
        name = 'ogbg-molhiv', 
        root='../data',
        pre_transform=pre_transform)
     
    dataset.data.x = dataset.data.x.float()
    dataset.data.edge_attr = dataset.data.edge_attr.float() 
    dataset.data.y = dataset.data.y.view(-1)
    
    # Split dataset
    split_idx = dataset.get_idx_split()
    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]

    # Create the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    input_channels = {
        'in_channels': 9,   # 9 node features
        'in_channels_e': 3  # 3 edge features
    }
    return input_channels, loader


# Graph-SST2 dataset (Requires manual download)
# The original version of the files is available at:
# https://mailustceducn-my.sharepoint.com/personal/yhy12138_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyhy12138%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2Fpaper%5Fwork%2FGNN%20Explainability%20Survey%2FSurvey%5FText2graph&ga=1
# We included the pre_transform argument in get_dataset_sst so we can do the same pre-processing as in the other datasets for the GTs
def get_sst(
        batch_size, 
        train_frac=trainer_hparams['train_frac'], 
        val_frac=trainer_hparams['val_frac'],
        test_frac=trainer_hparams['test_frac']
    ):

    # Load the full dataset
    dataset = get_dataset_sst(
        dataset_dir='../data/', 
        dataset_name='Graph_SST2', 
        task=None,
        pre_transform=pre_transform)
     
    # Split dataset into Train/Validation/Test
    num_graphs = len(dataset)
    num_train = int(train_frac * num_graphs)
    num_val = int(val_frac * num_graphs)
    num_test = num_graphs - num_train - num_val
    
    # Split always in the same way (set the seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [num_train, num_val, num_test],
        generator=torch.Generator().manual_seed(1998)
    )

    # Create the data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    loader = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    input_channels = {
        'in_channels': 768,   # 768 node features
        'in_channels_e': 1    # 1 edge feature
    }
    return input_channels, loader