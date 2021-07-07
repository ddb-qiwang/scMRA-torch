import torch.utils.data
from builtins import object
import torchvision.transforms as transforms
from torchdatasets import Dataset
import numpy as np

random_seed = [1111,2222,3333,4444,5555,6666,7777,8888,9999]
np.random.seed(random_seed[0])


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B, data_loader_C, data_loader_t, max_dataset_size):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.data_loader_C = data_loader_C
        self.data_loader_t = data_loader_t
     
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.stop_t = False
        self.max_dataset_size = max_dataset_size

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.stop_C = False
        self.stop_t = False

        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.data_loader_C_iter = iter(self.data_loader_C)
        self.data_loader_t_iter = iter(self.data_loader_t)
        self.iter = 0
        return self

    def __next__(self):
        A, A1, A2, A3 = None, None, None, None
        B, B1, B2, B3 = None, None, None, None
        C, C1, C2, C3 = None, None, None, None
        t, t1, t2, t3 = None, None, None, None
        try:
            A, A1, A2, A3 = next(self.data_loader_A_iter)
        except StopIteration:
            if A is None or A1 is None or A2 is None or A3 is None:
                self.stop_A = True
                self.data_loader_A_iter = iter(self.data_loader_A)
                A, A1, A2, A3 = next(self.data_loader_A_iter)

        try:
            B, B1, B2, B3 = next(self.data_loader_B_iter)
        except StopIteration:
            if B is None or B1 is None or B2 is None or B3 is None:
                self.stop_B = True
                self.data_loader_B_iter = iter(self.data_loader_B)
                B, B1, B2, B3 = next(self.data_loader_B_iter)
        
        try:
            C, C1, C2, C3 = next(self.data_loader_C_iter)
        except StopIteration:
            if C is None or C1 is None or C2 is None or C3 is None:
                self.stop_C = True
                self.data_loader_C_iter = iter(self.data_loader_C)
                C, C1, C2, C3 = next(self.data_loader_C_iter)
     
        try:
            t, t1, t2, t3 = next(self.data_loader_t_iter)
        except StopIteration:
            if t is None or t1 is None or t2 is None or t3 is None:
                self.stop_t = True
                self.data_loader_t_iter = iter(self.data_loader_t)
                t, t1, t2, t3 = next(self.data_loader_t_iter)

        if (self.stop_A and self.stop_B and self.stop_t) or self.iter > self.max_dataset_size:
            self.stop_A = False
            self.stop_B = False
            self.stop_C = False
            self.stop_t = False
            raise StopIteration()
        else:
            self.iter += 1
            return {'S1': A, 'S1_raw': A1, 'S1_sf': A2, 'S1_label': A3,
                    'S2': B, 'S2_raw': B1, 'S2_sf': B2, 'S2_label': B3,
                    'S3': C, 'S3_raw': C1, 'S3_sf': C2, 'S3_label': C3,
                    'T': t, 'T_raw': t1, 'T_sf': t2, 'T_label': t3}


class UnalignedDataLoader():
    def initialize(self, source, target, batch_size1, batch_size2):       
        
        dataset_source1 = Dataset(source[0]['cells'],source[0]['raw'],source[0]['sf'],source[0]['labels'])
        data_loader_s1 = torch.utils.data.DataLoader(dataset_source1, batch_size=batch_size1, shuffle=True, num_workers=1)
        self.dataset_s1 = dataset_source1

        dataset_source2 = Dataset(source[1]['cells'],source[1]['raw'],source[1]['sf'],source[1]['labels'])
        data_loader_s2 = torch.utils.data.DataLoader(dataset_source2, batch_size=batch_size1, shuffle=True, num_workers=1)
        self.dataset_s2 = dataset_source2
        
        dataset_source3 = Dataset(source[2]['cells'],source[2]['raw'],source[2]['sf'],source[2]['labels'])
        data_loader_s3 = torch.utils.data.DataLoader(dataset_source3, batch_size=batch_size1, shuffle=True, num_workers=1)
        self.dataset_s3 = dataset_source3

        dataset_target = Dataset(target['cells'],target['raw'],target['sf'],target['labels'])
        data_loader_t = torch.utils.data.DataLoader(dataset_target, batch_size=batch_size2, shuffle=True, num_workers=1)
        

        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s1, data_loader_s2, data_loader_s3, data_loader_t, float("inf"))
       

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return min(max(len(self.dataset_s1),len(self.dataset_s2),len(self.dataset_s3),len(self.dataset_t)), float("inf"))
