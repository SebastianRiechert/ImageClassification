from typing import Iterable
from typing import Tuple

import os

import torch
from torch import nn
import torchvision
from PIL import Image
import pandas as pd


class CSVDataset(object):
    def __init__(self, root, path_to_csv, transforms, dataset_type='binary'):
        self.root = root
        self.transforms = transforms
        df = pd.read_csv(path_to_csv)
        
        self.imgs = list(df.iloc[:, 0])
        
        if dataset_type == 'binary':
            self.labels = list(df[df.columns[1]])
            self.classes = ['non_'+df.columns[1], df.columns[1]]
        elif dataset_type == 'multi_class':
            raise NotImplementedError(f'dataset_type "{dataset_type}" not yet implemented')
        elif dataset_type == 'multi_label':
            raise NotImplementedError(f'dataset_type "{dataset_type}" not yet implemented')
        else:
            raise ValueError(f'invalid value for dataset_type "{dataset_type}". Valid values are "binary", "multi_class" and "multi_label"')

    def __getitem__(self, idx):
        # load images and labels
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path)
        target = self.labels[idx]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

class OptimizerFactory:
    def create_optimizer(self, name, parameters, args) -> torch.optim.Optimizer:
        if name == 'SGD':
            return torch.optim.SGD(parameters,
                                   lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)

        elif name == 'RMSprop':
            return torch.optim.RMSprop(parameters, lr=args.lr,
                                       alpha=args.alpha, eps=args.eps,
                                       weight_decay=args.weight_decay, momentum = args.momentum)

        elif name == 'Adam':
            return torch.optim.Adam(parameters, lr=args.lr,
                                    eps=args.eps, weight_decay=args.weight_decay)

class DatasetFactory:
    def create_dataset(self, name, root, path_to_csv, transforms) -> Iterable[Tuple[Image.Image,int]]:
        if name == 'CSVDataset':
            return CSVDataset(root=root, path_to_csv=path_to_csv, transforms=transforms)

        elif name == 'ImageFolder':
            return torchvision.datasets.ImageFolder(root=root, transform = transforms)

class ModelFactory:
    def create_model(self, name, pretrained) -> nn.Module:
        if name == 'SmallCNN':
            raise NotImplementedError(f'model "{name}" not yet implemented')

        else:
            return torchvision.models.__dict__[name](pretrained=pretrained)
        
class LossFactory:
    def create_loss(self, name) -> nn.Module:
        if name == 'MyLoss':
            raise NotImplementedError(f'loss "{name}" not yet implemented')

        else:
            return nn.modules.loss.__dict__[name]()
        
class LRSchedulerFactory:
    def create_scheduler(self, name, optimizer, args) -> torch.optim.lr_scheduler._LRScheduler:
        if name == 'MyScheduler':
            raise NotImplementedError(f'scheduler "{name}" not yet implemented')

        elif name == 'StepLR':
            return torch.optim.lr_scheduler.__dict__[name](optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        
        elif name == 'ExponentialLR':
            return torch.optim.lr_scheduler.__dict__[name](optimizer, gamma=args.lr_gamma)
        
        else:
            return torch.optim.lr_scheduler.__dict__[name]()
        
class TransformFactory:
    def create_transform(self, name, args) -> torch.optim.Optimizer:
        if name == 'Normalize':
            return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
        
        elif name == 'RandomHorizontalFlip':
            return torchvision.transforms.RandomHorizontalFlip()

        elif name == 'RandomResizedCrop':
            return torchvision.transforms.RandomResizedCrop(tuple(args.input_size))
        
        elif name == 'ToTensor':
            return torchvision.transforms.ToTensor()
        
        elif name == 'Resize':
            return torchvision.transforms.Resize(tuple(args.input_size))
