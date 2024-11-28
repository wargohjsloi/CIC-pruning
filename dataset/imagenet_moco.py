import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random

class ImageNetLT_moco(Dataset):
    num_classes=1000
    def __init__(self, root, txt, transform=None, class_balance=False, cls_sep=True):
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

        self.img_path_back = self.img_path.copy() if cls_sep else None
        self.labels_back = self.labels.copy() if cls_sep else None

        self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_classes)]


    def __len__(self):
        return len(self.labels)

    def reorganize_cls_sqe(self, batch_size):
        self.img_path = []
        self.labels = []
        for cls in range(self.num_classes):
            indices = self.class_data[cls]
            pad = batch_size - len(indices) % batch_size
            pad_idx = np.random.randint(0, len(indices), size=pad)
            for id in pad_idx:
                indices.append(self.class_data[cls][id])
            for idx in indices:
                self.img_path.append(self.img_path_back[idx])
                self.labels.append(self.labels_back[idx])

    def reset_data(self):
        if self.img_path_back is not None:
            self.img_path = self.img_path_back.copy()
            self.labels = self.labels_back.copy()
        else:
            print('No back data for reset!')

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


