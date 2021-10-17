# scMRA-torch
scMRA: A robust deep learning method to annotate scRNA-seq data with multiple reference datasets
This is a torch implement of scMRA method.

# Data Format
Before we get started, we need to preprocess the scRNA-seq data into rna_mats and labels
    rna_mats = [rna_mat1, rna_mat2, ..., rna_matn]
Here rna_mati are scRNA_seq count matrices aligned by gene. rna_matn is the query data (target data), others are reference data (source data).
    labels = [label1, label2, ..., labeln]
For annotation use, the labeln which refers the targe label is not needed.

# Training
