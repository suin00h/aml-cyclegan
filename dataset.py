import os
import random
import torch
import cv2
import torch.utils.data as data
from utils.transform import transform

class BaseDataset(data.Dataset):
    def __init__(self, params):
        self.params = params
        self.dataname = params.dataname
        self.datapath = params.datapath

    def get_dataset(self, dir):
        image_paths = [os.path.join(dir,fname)
                  for fname in sorted(os.listdir(dir))]
        return image_paths
    
    
class TrainDataset(BaseDataset):
    def __init__(self, params):
        super().__init__(params)
        # self.BtoA = params.direction == "BtoA"

        self.dir_A = os.path.join(self.datapath, self.params.mode + "A")
        self.dir_B = os.path.join(self.datapath, self.params.mode + "B")

        self.paths_A = self.get_dataset(self.dir_A)
        self.paths_B = self.get_dataset(self.dir_B)

        self.size_A = len(self.paths_A)
        self.size_B = len(self.paths_B)

        # self.input_nc = params.output_nc if self.BtoA else params.input_nc
        # self.output_nc = params.input_nc if self.BtoA else params.output_nc
        
        self.transform = transform(params)

         
    def __getitem__(self, index):
        # if length of A and B is different, cycle again
        # randomize index for domain B        
        path_A = self.paths_A[index % self.size_A]
        path_B = self.paths_B[random.randint(0, self.size_B - 1)]

        image_A = cv2.imread(path_A, cv2.COLOR_BGR2RGB)
        image_B = cv2.imread(path_B, cv2.COLOR_BGR2RGB)

        image_A = torch.from_numpy(image_A).permute(2, 0, 1).float() / 255.0
        image_B = torch.from_numpy(image_B).permute(2, 0, 1).float() / 255.0
        
        A = self.transform(image_A)
        B = self.transform(image_B)

        # Add batch dimension and move to GPU
        A = A.unsqueeze(0).cuda()
        B = B.unsqueeze(0).cuda()

        return A, B

    def __len__(self):
        return max(self.A_size, self.B_size)