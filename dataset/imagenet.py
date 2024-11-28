import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNetLT(Dataset):
    def __init__(self, root, txt, transform=None, cls_sep=True):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.num_classes = 1000
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

        self.class_data = [[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y = self.labels[i]
            self.class_data[y].append(i)

        self.img_path_back = self.img_path.copy() if cls_sep else None
        self.labels_back = self.labels.copy() if cls_sep else None

        self.cls_num_list = [len(self.class_data[i]) for i in range(self.num_classes)]

        # print(self.cls_num_list)

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
            # print('cls {} num batch:{}'.format(cls, len(indices) / batch_size))

    def set_num_type(self, num_type='all'):
        if num_type == 'all':
            self.img_path = self.img_path_back
            self.labels = self.labels_back
            print('set num img_path: ', len(self.img_path))
            print('set num labels: ', len(self.labels))
            return
        elif num_type == 'many':
            indices = np.where(np.array(self.cls_num_list) > 100)[0]
        elif num_type == 'mid':
            indices = np.where((np.array(self.cls_num_list) >= 20) & (np.array(self.cls_num_list) <= 100))[0]
        elif num_type == 'few':
            indices = np.where(np.array(self.cls_num_list) < 20)[0]

        class_data = [self.class_data[i] for i in indices]
        class_data_idx = []
        for idxs in class_data:
            for idx in idxs:
                class_data_idx.append(idx)
        self.img_path = [self.img_path_back[idx] for idx in class_data_idx]
        self.labels = [self.labels_back[idx] for idx in class_data_idx]

    def set_cls_dataset(self, cls):
        if self.img_path_back is None:
            self.img_path_back = self.img_path.copy()
            self.labels_back = self.labels.copy()
        indices = self.class_data[cls]
        self.img_path = [self.img_path_back[idx] for idx in indices]
        self.labels = [self.labels_back[idx] for idx in indices]

    def reset_data(self):
        if self.img_path_back is not None:
            self.img_path = self.img_path_back.copy()
            self.labels = self.labels_back.copy()
        else:
            print('No back data for reset!')

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        # print('path: {} label:{}'.format(path, label))
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label
