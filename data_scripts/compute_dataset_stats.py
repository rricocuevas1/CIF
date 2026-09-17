from dataset_names import DATASET_NAMES, GET_DATASET
import torch
import warnings
warnings.filterwarnings("ignore")


def compute_dataset_stats(loaders):
    total_graphs = 0
    total_nodes = 0
    total_edges = 0
    labels = []

    for loader in loaders:
        for batch in loader:
            total_graphs += batch.num_graphs
            labels.append(batch.y)  # Collect labels from this batch
            for i in range(batch.num_graphs):
                start = batch.ptr[i]
                end = batch.ptr[i+1]

                num_nodes = end - start
                num_edges = (batch.edge_index[0] >= start).sum().item() - \
                            (batch.edge_index[0] >= end).sum().item()

                total_nodes += num_nodes
                total_edges += num_edges

    avg_nodes = total_nodes / total_graphs
    avg_edges = total_edges / (total_graphs*2)

    labels = torch.cat(labels)
    num_classes = len(labels.unique())

    return total_graphs, avg_nodes, avg_edges, num_classes


# Compute the dataset stats for every dataset
print("DATASET PRE-PROCESSING (Start)")
for dataset_name in DATASET_NAMES:
    # Fetch dataset
    entry = GET_DATASET[dataset_name]
    input_channels, loader = entry() if callable(entry) else entry
    loaders = [loader['train'], loader['val'], loader['test']]
    total_graphs, avg_nodes, avg_edges, num_classes = compute_dataset_stats(loaders)
    print(dataset_name)
    for batch in loaders[0]:
        print(batch)
        #break
        print(dataset_name)
        print("Total graphs:", total_graphs)
        print("Average nodes per graph:", avg_nodes)
        print("Average edges per graph:", avg_edges)
        print("Number of classes: ", num_classes)
        print(batch.y)
        print("Node features", batch.x.shape)
        #print(batch.x[:3]) # Print 3 rows
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            print("Edge features", batch.edge_attr.shape)
            print(batch.edge_attr[:3]) # Print 3 rows
        else:
            print(None)
        # Check the PEs/SEs:
        print("RWSE", batch.RWSE.shape)
        #print(batch.RWSE[:3]) # Print 3 rows
        print("eigenvalues", batch.eigenvalues.shape)
        #print(batch.eigenvalues[:3])
        print("eigenvectors", batch.eigenvectors.shape)
        #print(batch.eigenvectors[:3])
        print("\n")
        break
print("DATASET PRE-PROCESSING (Finished)")