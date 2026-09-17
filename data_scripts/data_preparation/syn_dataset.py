# syn_binary_dataset.py
# PyG InMemoryDataset for the SYN binary dataset (house vs cycle on tree vs BA)
# Caches processed data to disk like standard PyG datasets.
import random
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset

def print_dataset_info(train_set, val_set, test_set, the, binary=False): # Multi-class by default
    class_list = ["house", "cycle"] if binary else ["house", "cycle", "grid", "diamond"]
    print_split_info(train_set, "Train", class_list, the)
    print_split_info(val_set, "Val", class_list, the)
    print_split_info(test_set, "Test", class_list, the)

def print_split_info(dataset, title, class_list, the):
    class_num = len(class_list)
    tr_list = [0] * class_num
    ba_list = [0] * class_num
    for g in dataset:
        if g.num_edges > the:
            ba_list[g.y.item()] += 1
        else:
            tr_list[g.y.item()] += 1

    print("-" * 80)
    print(f"{title} (Total: {sum(tr_list) + sum(ba_list)})")
    for i, shape in enumerate(class_list):
        total = tr_list[i] + ba_list[i]
        bias_pct = 100 * float(tr_list[i]) / total if total > 0 else 0
        print(f"  {shape:>8s}: Tree={tr_list[i]:<5d} BA={ba_list[i]:<5d} Total={total:<5d} TreeBias={bias_pct:.1f}%")
    print("-" * 80)


class SYN_dataset(InMemoryDataset):
    def __init__(
            self, 
            root,  
            mode='train', 
            bias=0.5,
            binary = True, 
            data_num=2000,
            transform=None, 
            pre_transform=None
        ):
        self.splits = ['train', 'val', 'test']
        self.bin_vs_multi = "binary" if binary else "multi"
        raw_data_path = f"../data/SYN_{self.bin_vs_multi}/syn_dataset_{self.bin_vs_multi}.pt"
        assert mode in self.splits
        self.mode = mode
        self.bias = bias
        self.raw_data_path = raw_data_path
        self.data_num = data_num
        super().__init__(root, transform, pre_transform)
        idx = self.splits.index(self.mode)
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def processed_file_names(self):
        file_names = [
            f'SYN_{self.bin_vs_multi}_train.pt', 
            f'SYN_{self.bin_vs_multi}_val.pt', 
            f'SYN_{self.bin_vs_multi}_test.pt'
        ]
        return file_names

    def process(self):
        dataset = torch.load(self.raw_data_path)
        binary = self.bin_vs_multi == "binary"
        train_list, val_list, test_list = self._bias_split(dataset, binary)

        # Verify split
        tree_edges = np.mean([g.num_edges for g in dataset['tree']['house'][:50]])
        ba_edges = np.mean([g.num_edges for g in dataset['ba']['house'][:50]])
        the = (tree_edges + ba_edges) / 2
        print_dataset_info(train_list, val_list, test_list, the, binary)

        # Save the split
        for split, data_list in zip(self.splits, [train_list, val_list, test_list]):
            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]
            data, slices = self.collate(data_list)
            idx = self.splits.index(split)
            torch.save((data, slices), self.processed_paths[idx])

        # Load the one we need
        idx = self.splits.index(self.mode)
        self.data, self.slices = torch.load(self.processed_paths[idx])

    def _bias_split(self, dataset, binary):
        random.seed(666)
        if binary:
            num_classes = 2
            class_list = ["house", "cycle"]
            bias_dict = {"house": self.bias, "cycle": 1 - self.bias}
            total = self.data_num * num_classes
        else:
            num_classes = 4
            class_list = ["house", "cycle", "grid", "diamond"]
            bias_dict = {"house": self.bias, "cycle": 1 - self.bias, "grid": 1 - self.bias, "diamond": 1 - self.bias}
            total = self.data_num * num_classes
        
        ba_dataset = dataset['ba']
        tr_dataset = dataset['tree']

        split=[7, 1, 2]
        train_split, val_split, test_split = float(split[0]) / 10, float(split[1]) / 10, float(split[2]) / 10
        assert train_split + val_split + test_split == 1
        train_num, val_num, test_num = total * train_split, total * val_split, total * test_split

        # blance class
        class_num = num_classes
        train_class_num, val_class_num, test_class_num = train_num / class_num, val_num / class_num, test_num / class_num
        train_list, val_list, test_list  = [], [], []
        
        for shape in class_list:
            bias = bias_dict[shape]
            train_tr_num = int(train_class_num * bias)
            train_ba_num = int(train_class_num * (1 - bias))
            val_tr_num = int(val_class_num * bias)
            val_ba_num = int(val_class_num * (1 - bias))
            test_tr_num = int(test_class_num * 0.5)
            test_ba_num = int(test_class_num * 0.5)
            train_list += tr_dataset[shape][:train_tr_num] + ba_dataset[shape][:train_ba_num]
            val_list += tr_dataset[shape][train_tr_num:train_tr_num + val_tr_num] + ba_dataset[shape][train_ba_num:train_ba_num + val_ba_num]
            test_list += tr_dataset[shape][train_tr_num + val_tr_num:train_tr_num + val_tr_num + test_tr_num] + ba_dataset[shape][train_ba_num + val_ba_num:train_ba_num + val_ba_num + test_ba_num]
        random.shuffle(train_list)
        random.shuffle(val_list)
        random.shuffle(test_list)
        return train_list, val_list, test_list
