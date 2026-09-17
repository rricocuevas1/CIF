import torch
import os
import argparse
import featgen
import gengraph
import random
from torch_geometric.utils import from_networkx
from tqdm import tqdm
import numpy as np

# Dataset generation
# This is untouched as in https://github.com/yongduosui/CAL/blob/main/utils.py  
def creat_one_pyg_graph(
        context, 
        shape, 
        label, 
        feature_dim, 
        shape_num, 
        settings_dict, 
        args=None
    ):
    if args is None:
        noise = 0
    else:
        noise = args.noise
    if feature_dim == -1:
        # use degree as feature
        feature = featgen.ConstFeatureGen(None, max_degree=args.max_degree)
    else:
        feature = featgen.ConstFeatureGen(np.random.uniform(0, 1, feature_dim))
    G, node_label = gengraph.generate_graph(
        basis_type=context,
        shape=shape,
        nb_shapes=shape_num,
        width_basis=settings_dict[context]["width_basis"],
        feature_generator=feature,
        m=settings_dict[context]["m"],
        random_edges=noise
    ) 
    pyg_G = from_networkx(G)
    pyg_G.x = pyg_G.feat
    del pyg_G.feat
    pyg_G.y = torch.tensor([label])
    # Per-node ground-truth motif mask: 1 if the node belongs to the
    # class-defining motif (role_id > 0), 0 if it is a base-graph node.
    roles = torch.as_tensor(np.asarray(node_label), dtype=torch.long)  # role_id per node
    pyg_G.node_gt = (roles > 0).long()                                 # 1 = motif node, 0 = base node
    return pyg_G, node_label

def graph_dataset_generate(args):
    # Binary and multi class generation
    # Set up the path
    bin_vs_multi = "binary" if args.binary else "multi"
    save_path = f'../../../data/SYN_{bin_vs_multi}'

    # For the binary case we only generate house and cyles classes
    if args.binary:
        class_list = ["house", "cycle"]
    # For the multi-class case we generate all 4 motifs (this is the original setting as in CAL)
    else:
        class_list = ["house", "cycle", "grid", "diamond"]

    # This is untouched as in https://github.com/yongduosui/CAL/blob/main/utils.py  
    settings_dict = {"ba": {"width_basis": args.node_num ** 2, "m": 2},
                     "tree": {"width_basis":2, "m": args.node_num}}

    feature_dim = args.feature_dim
    shape_num = args.shape_num
    dataset = {}
    dataset['tree'] = {}
    dataset['ba'] = {}

    for label, shape in enumerate(class_list):
        tr_list = []
        ba_list = []
        print("create shape:{}".format(shape))
        for _ in tqdm(range(args.data_num)):
            tr_g, _ = creat_one_pyg_graph(
                context="tree", 
                shape=shape, 
                label=label, 
                feature_dim=feature_dim, 
                shape_num=shape_num, 
                settings_dict=settings_dict, 
                args=args
            )
            ba_g, _ = creat_one_pyg_graph(
                context="ba", 
                shape=shape, 
                label=label, 
                feature_dim=feature_dim, 
                shape_num=shape_num, 
                settings_dict=settings_dict, 
                args=args
            )
            tr_list.append(tr_g)
            ba_list.append(ba_g)
        dataset['tree'][shape] = tr_list
        dataset['ba'][shape] = ba_list

    save_path += f"/syn_dataset_{bin_vs_multi}.pt" 
    torch.save(dataset, save_path)
    print("save at:{}".format(save_path))
    return dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', action='store_true') # Default multi-class generation
    parser.add_argument('--node_num', type=int, default=15)
    parser.add_argument('--feature_dim', type=int, default=-1)
    parser.add_argument('--shape_num', type=int, default=1)
    parser.add_argument('--data_num', type=int, default=2000) # 2000 graphs per class
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--max_degree', type=int, default=10)
    parser.add_argument('--bias', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=666)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(f'../../../data/SYN_{"binary" if args.binary else "multi"}', exist_ok=True)
    graph_dataset_generate(args)
    """
    In the binary case this generates a total of 8000 graphs:
     - 2000 tree-house graphs (class house)
     - 2000 BA-house graphs (class house)
     - 2000 tree-cycle graphs (class cycle)
     - 2000 BA-cycle graphs (class cycle)
    """
    
    

    

