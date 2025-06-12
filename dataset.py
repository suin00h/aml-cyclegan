import os
import random
from PIL import Image
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
        self.BtoA = params.direction == "BtoA"

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
        if self.BtoA:
            input_path = self.paths_B[index % self.size_B]
            target_path = self.paths_A[random.randint(0, self.size_A - 1)]
        else:
            input_path = self.paths_A[index % self.size_A]
            target_path = self.paths_B[random.randint(0, self.size_B - 1)]

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        input_tensor = self.transform(input_image).cuda()
        target_tensor = self.transform(target_image).cuda()

        return input_tensor, target_tensor

    def __len__(self):
        return max(self.size_A, self.size_B)
    
class TestDataset(BaseDataset):
    def __init__(self, params):
        super().__init__(params)
        self.params = params
        self.BtoA = params.direction == "BtoA"

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
        if self.BtoA:
            input_path = self.paths_B[index % self.size_B]
            target_path = self.paths_A[index % self.size_A]
        else:
            input_path = self.paths_A[index % self.size_A]
            target_path = self.paths_B[index % self.size_B]

        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        input_tensor = self.transform(input_image).cuda()
        target_tensor = self.transform(target_image).cuda()

        if self.params.dataname == "cityscapes":
            base_filename = os.path.basename(input_path).rsplit(".", 1)[0].replace("_A", "").replace("_B", "")
            return input_tensor, target_tensor, base_filename
        
        return input_tensor, target_tensor

    def __len__(self):
        return max(self.size_A, self.size_B)