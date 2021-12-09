import os
import torch
import math
from PIL import Image
from torch.utils.data import Dataset
class ScoliosisDataset(Dataset):

    def __init__(self, data_dir, transform=None, target_transform=None, train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.image_label = self.get_imageFiles()

    def get_imageFiles(self):
        labels_dict = {'Normal': 0, 'Mild': 1, 'Moderate': 2, 'Serious': 3}
        if self.train:
            image_path = os.path.join(self.data_dir, "train")
        else:
            image_path = os.path.join(self.data_dir, "test")
        labels = sorted(os.listdir(image_path))

        image_list = []
        for label in labels:
            image_lists = os.listdir(os.path.join(image_path, label))
            for image_name in image_lists:
                if image_name:
                    image_list.append((os.path.join(image_path, label, image_name), labels_dict[label]))
        return sorted(image_list)

    def normal_sampling(self,mean, label_k, std=1):
        return math.exp(-(label_k - mean) ** 2 / (2 * std ** 2)) / (math.sqrt(2 * math.pi) * std)

    def __getitem__(self, item):
        image_path, label = self.image_label[item]
        image = Image.open(image_path).convert('RGB')
        if self.train:
            if self.transform is not None:
                image = self.transform(image)
        else:
            if self.target_transform is not None:
                image = self.target_transform(image)
        labels = [self.normal_sampling(int(label), i) for i in range(4)]
        labels = [i if i > 1e-10 else 1e-10 for i in labels]
        labels = torch.Tensor(labels)
        # label = torch.FloatTensor(label)
        # print(labels)
        return image,label#labels,



    def __len__(self):
        return len(self.image_label)
