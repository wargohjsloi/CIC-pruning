import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random


class INaturalist_moco(Dataset):
    num_classes = 8142
    def __init__(self, root, txt, transform=None, class_balance=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.class_balance=class_balance
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data=[[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y=self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.class_balance:
           label=random.randint(0,self.num_classes-1)
           index=random.choice(self.class_data[label])
           path1 = self.img_path[index]
        else:
           path1 = self.img_path[index]
           label = self.labels[index]

        with open(path1, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            samples = [tr(img) for tr in self.transform]

        return samples, label, np.array(index)
