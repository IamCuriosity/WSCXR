import os
import random
from enum import Enum

import PIL
import torch
from torchvision import transforms
import json
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class MedDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        source,
        imagesize=256,
        split='train',
        normal_only=True,
        max_normal=-1,
        **kwargs,
    ):

        super().__init__()
        self.source = source
        self.split = split
        self.normal_only=normal_only
        self.max_normal=max_normal
        self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize((imagesize,imagesize),Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_img = transforms.Compose(self.transform_img)
        self.imagesize = (3, imagesize, imagesize)


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path=os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB")

        image = self.transform_img(image)

        return {
            "image": image,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)


    def get_image_data(self):
        data_to_iterate = []
        with open(os.path.join(self.source,'samples',"{}.json".format(self.split)), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)

        if self.normal_only:
            data_to_iterate=[data  for data in data_to_iterate if data['label']==0]

        if self.split=='train' and self.max_normal!=-1:
            data_to_iterate=random.sample(
                data_to_iterate,self.max_normal
            )
            # data_to_iterate=data_to_iterate[:self.max_normal]
        return data_to_iterate


class MedAbnormalDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        source,
        abnormal_k,
        imagesize=256,
        split='train',
        train_val_split=1.0,
        **kwargs,
    ):

        super().__init__()
        self.source = source
        self.split = split
        self.train_val_split = train_val_split
        self.abnormal_k = abnormal_k
        self.data_to_iterate = self.get_image_data()

        self.transform_img = [
            transforms.Resize((imagesize,imagesize),Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        self.transform_img = transforms.Compose(self.transform_img)
        self.imagesize = (3, imagesize, imagesize)


    def __getitem__(self, idx):
        info = self.data_to_iterate[idx]
        image_path=os.path.join(self.source,'images',info['filename'])
        image = PIL.Image.open(image_path).convert("RGB")

        image = self.transform_img(image)

        return {
            "image": image,
            "classname": info['clsname'],
            "is_anomaly": info['label'],
            "image_path": image_path,
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        data_to_iterate = []

        with open(os.path.join(self.source,'samples',"{}.json".format(self.split)), "r") as f_r:
            for line in f_r:
                meta = json.loads(line)
                data_to_iterate.append(meta)

        data_to_iterate=[data  for data in data_to_iterate if data['label']==1]

        abnormal_dict={}

        for data in data_to_iterate:
            if data['clsname'] not in abnormal_dict:
                abnormal_dict[data['clsname']]=[]
            abnormal_dict[data['clsname']].append(data)

        data_to_iterate=[]
        for key in abnormal_dict:
            data_to_iterate.extend(random.sample(abnormal_dict[key],self.abnormal_k))

        return data_to_iterate



