# scMRA-torch
scMRA: A robust deep learning method to annotate scRNA-seq data with multiple reference datasets

This is a torch implement of scMRA method.

# Data Format
Before we get started, we need to preprocess the scRNA-seq data into rna_mats and labels

    rna_mats = [rna_mat1, rna_mat2, ..., rna_matn]
    
Here rna_mati are scRNA_seq count matrices aligned by gene. rna_matn is the query data (target data), others are reference data (source data).

    labels = [label1, label2, ..., labeln]
    
Where labeli are column vectors. For annotation use, the labeln which refers the targe label is not needed.

# Training
Once you accquire the rna_mats and labels, you can use scMRA to predict the cell type of query samples.

We give an instance that train the scMRA model with test data for 200 epochs.

    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    from preprocess import*
    from torchdata import*
    from loss import*
    from network import*
    import random
    
    random.seed(1111)
    
    solver = Solver(rna_mats=rna_mats, labels=labels)
    
    #train scMRA model for 200 epochs
    for t in range(200):
        print('Epoch: ', t)
        num = solver.train_gcn_adapt(t)
        best_acc = solver.test(t)
    
    ARI = np.around(adjusted_rand_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
    AMI = np.around(adjusted_mutual_info_score(np.array(best_acc[1]),np.array(best_acc[0])),5)
