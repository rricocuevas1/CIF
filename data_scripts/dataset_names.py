from datasets import (
    get_mnistsp, # [MNIST_75sp] for 4 noise levels
    get_sst,       # ["Graph_SST2"]
    get_SYN,       # ["SYN_binary_b_01", "SYN_binary_b_03", "SYN_binary_b_05", "SYN_binary_b_07", "SYN_binary_b_09"]
                   # ["SYN_multi_b_01", "SYN_multi_b_03", "SYN_multi_b_05", "SYN_multi_b_07", "SYN_multi_b_09"]
    get_SPMotif,   # ["SPMotif_b_05", "SPMotif_b_07", "SPMotif_b_09"]
    get_molhiv,    # ["Molhiv"]
)
from src.hyper_parameters import trainer_hparams


DATASET_NAMES = [
    "Molhiv",
    "Graph_SST2",
    # SYN_multi
    "SYN_multi_b_01", 
    "SYN_multi_b_03", 
    "SYN_multi_b_05", 
    "SYN_multi_b_07", 
    "SYN_multi_b_09",
    # SPMotif
    "SPMotif_b_09",
    "SPMotif_b_07",
    "SPMotif_b_05", 
    # MNIST_75sp
    "MNIST_75sp_n08",
    "MNIST_75sp_n06",
    "MNIST_75sp_n04",
    "MNIST_75sp_n02",
]

GET_DATASET = {
    # SYN_multi
    "SYN_multi_b_01": lambda :  get_SYN(bias=0.1, binary=False, batch_size=trainer_hparams['batch_size']["SYN_multi"]), 
    "SYN_multi_b_03": lambda :  get_SYN(bias=0.3, binary=False, batch_size=trainer_hparams['batch_size']["SYN_multi"]), 
    "SYN_multi_b_05": lambda :  get_SYN(bias=0.5, binary=False, batch_size=trainer_hparams['batch_size']["SYN_multi"]), 
    "SYN_multi_b_07": lambda :  get_SYN(bias=0.7, binary=False, batch_size=trainer_hparams['batch_size']["SYN_multi"]), 
    "SYN_multi_b_09": lambda :  get_SYN(bias=0.9, binary=False, batch_size=trainer_hparams['batch_size']["SYN_multi"]),
    # SYN_binary
    "SYN_binary_b_01": lambda :  get_SYN(bias=0.1, binary=True, batch_size=trainer_hparams['batch_size']["SYN_binary"]), 
    "SYN_binary_b_03": lambda :  get_SYN(bias=0.3, binary=True, batch_size=trainer_hparams['batch_size']["SYN_binary"]), 
    "SYN_binary_b_05": lambda :  get_SYN(bias=0.5, binary=True, batch_size=trainer_hparams['batch_size']["SYN_binary"]), 
    "SYN_binary_b_07": lambda :  get_SYN(bias=0.7, binary=True, batch_size=trainer_hparams['batch_size']["SYN_binary"]), 
    "SYN_binary_b_09": lambda :  get_SYN(bias=0.9, binary=True, batch_size=trainer_hparams['batch_size']["SYN_binary"]),
    # MNIST_75sp
    "MNIST_75sp_n02": lambda: get_mnistsp(noise_level=0.2, batch_size=trainer_hparams['batch_size']["MNIST_75sp"]),
    "MNIST_75sp_n04": lambda: get_mnistsp(noise_level=0.4, batch_size=trainer_hparams['batch_size']["MNIST_75sp"]),
    "MNIST_75sp_n06": lambda: get_mnistsp(noise_level=0.6, batch_size=trainer_hparams['batch_size']["MNIST_75sp"]),
    "MNIST_75sp_n08": lambda: get_mnistsp(noise_level=0.8, batch_size=trainer_hparams['batch_size']["MNIST_75sp"]),
    # SPMotif
    "SPMotif_b_05" : get_SPMotif(bias=0.5, batch_size=trainer_hparams['batch_size']["SPMotif_b_05"]), 
    "SPMotif_b_07" : get_SPMotif(bias=0.7, batch_size=trainer_hparams['batch_size']["SPMotif_b_07"]), 
    "SPMotif_b_09" : get_SPMotif(bias=0.9, batch_size=trainer_hparams['batch_size']["SPMotif_b_09"]),
    "Graph_SST2" : get_sst(batch_size=trainer_hparams['batch_size']["Graph_SST2"]),
    "Molhiv" : get_molhiv(batch_size=trainer_hparams['batch_size']["Molhiv"]),
}

DATASET_N_CLASSES = {
    # SYN_multi
    "SYN_multi_b_01" : 4, 
    "SYN_multi_b_03" : 4, 
    "SYN_multi_b_05" : 4, 
    "SYN_multi_b_07" : 4, 
    "SYN_multi_b_09" : 4,
    # SYN_binary
    "SYN_binary_b_01" : 2, 
    "SYN_binary_b_03" : 2, 
    "SYN_binary_b_05" : 2, 
    "SYN_binary_b_07" : 2, 
    "SYN_binary_b_09" : 2,
    # MNIST_75sp
    "MNIST_75sp_n02" : 10,
    "MNIST_75sp_n04" : 10,
    "MNIST_75sp_n06" : 10,
    "MNIST_75sp_n08" : 10,
    # SPMotif
    "SPMotif_b_05" : 3, 
    "SPMotif_b_07" : 3, 
    "SPMotif_b_09" : 3,
    "Graph_SST2" : 2,
    "Molhiv" : 2,
}
