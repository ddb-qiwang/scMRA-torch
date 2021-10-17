import sys

import torch
from torchdataloader import UnalignedDataLoader
import torch.utils.data
from torchdatasets import Dataset
from preprocess import*
import scanpy as sc

#construct multi-reference dataset
#S refers to the set of source datasets(reference datasets), T refers to the test dataset(unassigned dataset)
#The input rna_mats = [rna_mat1, rna_mat2, ...] different mats should be aligned by gene
# The input labels = [label1, label2, ...], shape of labeli should be (-1, 1)

# read multiple reference datasets
def dataset_read(rna_mats, labels, highly_variable, batch_size, use_filter=False, filter_index=False,
                 open_case=False, use_latent=False, use_cluster=False):
    
    n_data = len(rna_mats)
    S = []
    T = {}
    dataset_size = []
    
    metadata = rna_mats[0]
    for i in range(n_data):
        if i > 0:
            metadata = np.vstack((metadata, rna_mats[i]))
    metadata = sc.AnnData(metadata)
    metadata = normalize(metadata, highly_genes=highly_variable)  
    
    tmp = {}
    start_idx = 0
    for i in range(n_data - 1):
        end_idx = start_idx+len(labels[i])
        tmp['cells'] = metadata.X[start_idx:end_idx]
        tmp['raw'] = metadata.raw.X[start_idx:end_idx]
        tmp['sf'] = metadata.obs['size_factors'][start_idx:end_idx]
        tmp['labels'] = labels[i]
        S.append(tmp)
        dataset_size.append(len(labels[i]))
        start_idx = end_idx
    
    T['cells'] = metadata.X[start_idx:]
    T['raw'] = metadata.raw.X[start_idx:]
    T['sf'] = metadata.obs['size_factors'][start_idx:]
    if len(labels) == n_data:
        T['labels'] = labels[-1]
    else:
        T['labels] = np.zeros((T['cells'].shape[0], 1))
    dataset_size.append(len(labels[-1]))
    input_size = np.shape(T['cells'])[1]
    target_size = len(labels[-1])
    
    if use_filter:
        T['cells'] = T['cells'][filter_index]
        T['raw'] = T['raw'][filter_index]
        T['sf'] = T['sf'][filter_index]
        if len(labels) == n_data:
            T['labels'] = labels[filter_index]
        else:
            T['labels] = np.zeros((T['cells'].shape[0], 1))
        dataset_size.append(len(T['labels']))
        
    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size)
    dataset = train_loader.load_data()
    
    if open_case:
        return torch.Tensor(T['cells']).cuda(), torch.Tensor(T['raw']).cuda(), torch.Tensor(T['sf']).cuda(), torch.Tensor(T['labels']).cuda()
    
    if use_latent:
        return torch.Tensor(S[0]["cells"]).cuda(), torch.Tensor(T["cells"]).cuda()
    
    if use_cluster:
        return torch.Tensor(T["cells"]).cuda(), T['labels']

    return dataset, min(dataset_size), input_size, target_size
