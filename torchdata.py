import sys

import torch
from torchdataloader import UnalignedDataLoader
import torch.utils.data
from torchdatasets import Dataset
from preprocess import*
import scanpy as sc

#construct multi-reference dataset
#S refers to the set of source datasets(reference datasets), T refers to the test dataset(unassigned dataset)

'''
def dataset_read(highly_variable, batch_size, use_filter=False, filter_index=False, open_case=False, use_latent=False, use_cluster=False):
    
    S1 = {}
    S1_test = {}
    
    S = [S1]
    S_test = [S1_test]

    T = {}
    T_test = {}
    dataset_size = list()
      
    X_all, label_all = prepro("Baron_human+Muraro")
    X1, label1 = prepro("Baron_human")
    X2, label2 = prepro("Muraro")
    len_1,len_2 = len(label1),len(label2) 
    label1 = label_all[:-len_2]
    label2 = label_all[-len_2:]
    print(np.unique(label1))
    print(np.unique(label2))
    
    
    index = []
    indexm = []
    
    for i in range(len_1):
        if label_all[i][0] in {2,5,6,7,8,9,10,11}:
            if label_all[i][0] == 2:
                label_all[i][0] == 0
            else:
                label_all[i][0] -= 4
            index.append(i)
    
    for i in range(len_1,len_1+len_2):
        if label_all[i][0] in {2,5,6,7,8,9,10,11,12}:
            if label_all[i][0] == 2:
                label_all[i][0] = 0
            elif label_all[i][0] == 12:
                label_all[i][0] = -1
            else:
                label_all[i][0] -= 4
            index.append(i)
            indexm.append(i)

    target_size = len(indexm)
    #print(len(label_all)==len_baronM_partial+len_muraro_partial)
    X_all, label_all = X_all[index], label_all[index]
    X_all = sc.AnnData(X_all)
    print(len(X_all))
    X_all = normalize(X_all,highly_genes=highly_variable)
    print(len(X_all))
    input_size = X_all.X.shape[1]
    
    S[0]['cells'] = X_all.X[:-target_size]
    S[0]['raw'] = X_all.raw.X[:-target_size]
    S[0]['sf'] = X_all.obs['size_factors'][:-target_size]
    S[0]['labels'] = label_all[:-target_size]
    dataset_size.append(S[0]['cells'].shape[0])
    
    T['cells'] = X_all.X[-target_size:]
    T['raw'] = X_all.raw.X[-target_size:]
    T['sf'] = X_all.obs['size_factors'][-target_size:]
    T['labels']= label_all[-target_size:]
    dataset_size.append(T['cells'].shape[0])
    
    if use_filter:
        T['cells'] = T['cells'][filter_index]
        T['raw'] = T['raw'][filter_index]
        T['sf'] = T['sf'][filter_index]
        T['labels'] = T['labels'][filter_index]
        dataset_size.append(T['cells'].shape[1])

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


def dataset_read(highly_variable, batch_size, use_filter=False, filter_index=False, open_case=False, use_latent=False, use_cluster=False):
    
    S1 = {}
    S2 = {}
    S = [S1, S2]

    T = {}
    dataset_size = list()
    X_all, label_all = prepro('Baron_human+Enge+Lawlor+Muraro+Xin_2016')
    X1, label1 = prepro("Baron_human")
    X2, label2 = prepro("Enge")
    X3, label3 = prepro("Lawlor")
    X4, label4 = prepro("Muraro")
    X5, label5 = prepro("Xin_2016")

    len_1 = len(label1)
    len_2 = len(label2)
    len_3 = len(label3)
    len_4 = len(label4)
    len_5 = len(label5)
    label1 = label_all[:len_1]
    label2 = label_all[len_1:len_1+len_2]
    label3 = label_all[len_1+len_2:-len_4-len_5]
    label4 = label_all[-len_4-len_5:-len_5]
    label5 = label_all[-len_5:]
    
    print(np.unique(label_all))
    print(np.unique(label1))
    print(np.unique(label2))
    print(np.unique(label3))
    print(np.unique(label4))
    print(np.unique(label5))
    
    index = []
    for i in range(len_1, len_1+len_2+len_3):
        if label_all[i][0] in {5,6,8,9,11,12}:
            if label_all[i][0] > 4:
                label_all[i][0] -= 5
            if label_all[i][0] > 2:
                label_all[i][0] -= 1
            if label_all[i][0] > 4:
                label_all[i][0] -= 1
            index.append(i)
    source1_size = len(index) - len_2 
    target_size = len_2
    for i in range(len_1+len_2+len_3,len_1+len_2+len_3+len_4):
        if label_all[i][0] in {5,6,8,9,11,12}:
            if label_all[i][0] > 4:
                label_all[i][0] -= 5
            if label_all[i][0] > 2:
                label_all[i][0] -= 1
            if label_all[i][0] > 4:
                label_all[i][0] -= 1
            index.append(i)
    source2_size = len(index) - source1_size - target_size
    
    X_all, label_all = X_all[index], label_all[index]
    
    if highly_variable != None:
        X_all = sc.AnnData(X_all)
        X_all = normalize(X_all, highly_genes=highly_variable)
        input_size = X_all.X.shape[1]
    
    S[0]['cells'] = X_all.X[target_size:-source2_size]
    S[0]['raw'] = X_all.raw.X[target_size:-source2_size]
    S[0]['sf'] = X_all.obs['size_factors'][target_size:-source2_size]
    S[0]['labels'] = label_all[target_size:-source2_size]
    dataset_size.append(S[0]['cells'].shape[0])
    
    S[1]['cells'] = X_all.X[-source2_size:]
    S[1]['raw'] = X_all.raw.X[-source2_size:]
    S[1]['sf'] = X_all.obs['size_factors'][-source2_size:]
    S[1]['labels'] = label_all[-source2_size:]
    dataset_size.append(S[1]['cells'].shape[0])
    
    T['cells'] = X_all.X[:target_size]
    T['raw'] = X_all.raw.X[:target_size]
    T['sf'] = X_all.obs['size_factors'][:target_size]
    T['labels'] = label_all[:target_size]
    dataset_size.append(T['cells'].shape[0])
    

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size)
    dataset = train_loader.load_data()
    
    if use_latent:
        return torch.Tensor(S[0]["cells"]).cuda(), torch.Tensor(S[1]["cells"]).cuda(), torch.Tensor(T["cells"]).cuda()
    
    if use_cluster:
        return torch.Tensor(T["cells"]).cuda(), T['labels']

    # return dataset, dataset_test, min(dataset_size), max(dataset_size)
    return dataset, min(dataset_size), input_size, target_size


'''
def dataset_read(highly_variable, batch_size,  use_filter=False, filter_index=False, open_case=False, use_latent=False, use_cluster=False):
    
    S1 = {}
    S2 = {}
    S3 = {}
    S = [S1, S2, S3]

    T = {}
    dataset_size = list()
      
    X_all, label_all = prepro('Baron_human+Enge+Lawlor+Muraro+Xin_2016')
    X1, label1 = prepro("Baron_human")
    X2, label2 = prepro("Enge")
    X3, label3 = prepro("Lawlor")
    X4, label4 = prepro("Muraro")
    X5, label5 = prepro("Xin_2016")

    len_1 = len(label1)
    len_2 = len(label2)
    len_3 = len(label3)
    len_4 = len(label4)
    len_5 = len(label5)
    label1 = label_all[:len_1]
    label2 = label_all[len_1:len_1+len_2]
    label3 = label_all[len_1+len_2:-len_4-len_5]
    label4 = label_all[-len_4-len_5:-len_5]
    label5 = label_all[-len_5:]
    
    print(np.unique(label_all))
    print(np.unique(label1))
    print(np.unique(label2))
    print(np.unique(label3))
    print(np.unique(label4))
    print(np.unique(label5))
  
    indexs3 = []
    for i in range(len_1):
        if label_all[i][0] in {5,6,8,9,11,12}:
            if label_all[i][0] > 4:
                label_all[i][0] -= 5
            if label_all[i][0] > 2:
                label_all[i][0] -= 1
            if label_all[i][0] > 4:
                label_all[i][0] -= 1
            indexs3.append(i)
    source3_size = len(indexs3)
    
    indext = []
    for i in range(len_1, len_1+len_2):
        if label_all[i][0] in {5,6,8,9,11,12}:
            if label_all[i][0] > 4:
                label_all[i][0] -= 5
            if label_all[i][0] > 2:
                label_all[i][0] -= 1
            if label_all[i][0] > 4:
                label_all[i][0] -= 1
            indext.append(i)
    target_size = len(indext)
    
    indexs1 = []
    for i in range(len_1+len_2, len_1+len_2+len_3):
        if label_all[i][0] in {5,6,8,9,11,12}:
            if label_all[i][0] > 4:
                label_all[i][0] -= 5
            if label_all[i][0] > 2:
                label_all[i][0] -= 1
            if label_all[i][0] > 4:
                label_all[i][0] -= 1
            indexs1.append(i)
    source1_size = len(indexs1)
    
    indexs2 = []
    for i in range(len_1+len_2+len_3,len_1+len_2+len_3+len_4):
        if label_all[i][0] in {5,6,8,9,11,12}:
            if label_all[i][0] > 4:
                label_all[i][0] -= 5
            if label_all[i][0] > 2:
                label_all[i][0] -= 1
            if label_all[i][0] > 4:
                label_all[i][0] -= 1
            indexs2.append(i)
    source2_size = len(indexs2)
    
    X_all = sc.AnnData(X_all)
    X_all = normalize(X_all, highly_genes=highly_variable)
    input_size = X_all.X.shape[1]
    
    S[0]['cells'] = X_all.X[indexs3]
    S[0]['raw'] = X_all.raw.X[indexs3]
    S[0]['sf'] = X_all.obs['size_factors'][indexs3]
    S[0]['labels'] = label_all[indexs3]
    dataset_size.append(S[0]['cells'].shape[0])
    
    S[1]['cells'] = X_all.X[indexs1]
    S[1]['raw'] = X_all.raw.X[indexs1]
    S[1]['sf'] = X_all.obs['size_factors'][indexs1]
    S[1]['labels'] = label_all[indexs1]
    dataset_size.append(S[1]['cells'].shape[0])
    
    S[2]['cells'] = X_all.X[indexs2]
    S[2]['raw'] = X_all.raw.X[indexs2]
    S[2]['sf'] = X_all.obs['size_factors'][indexs2]
    S[2]['labels'] = label_all[indexs2]
    dataset_size.append(S[1]['cells'].shape[0])
    
    T['cells'] = X_all.X[indext]
    T['raw'] = X_all.raw.X[indext]
    T['sf'] = X_all.obs['size_factors'][indext]
    T['labels'] = label_all[indext]
    dataset_size.append(T['cells'].shape[0])
    

    train_loader = UnalignedDataLoader()
    train_loader.initialize(S, T, batch_size, batch_size)
    dataset = train_loader.load_data()

    if use_latent:
        return torch.Tensor(S[0]["cells"]).cuda(), torch.Tensor(S[1]["cells"]).cuda(), torch.Tensor(S[2]["cells"]).cuda(), torch.Tensor(T["cells"]).cuda()
    
    if use_cluster:
        return torch.Tensor(T["cells"]).cuda(), T['labels']
    
    
    # return dataset, dataset_test, min(dataset_size), max(dataset_size)
    return dataset, min(dataset_size), input_size, target_size
