import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transform
from config.config import Config
import os

cfg = Config()

data_transforms = {'train': transform.Compose([transform.Resize(224),
                                               transform.RandomHorizontalFlip(),
                                               transform.ToTensor(),
                                               transform.Normalize(mean=cfg.mean, std=cfg.std)]),
                   'val': transform.Compose([transform.Resize(256),
                                             transform.CenterCrop(224),
                                             transform.ToTensor(),
                                             transform.Normalize(mean=cfg.mean, std=cfg.std)])}

data_preprocess = transform.Compose([transform.Resize(40),
                                     transform.CenterCrop(32),
                                     transform.ToTensor()])

# datasets = {x: ImageFolder(root=os.path.join(cfg.catdog_root_dir, x), transform=data_transforms[x])
#             for x in ['train', 'val']}
# print(datasets)
# data_loaders = {x: DataLoader(dataset=datasets[x], batch_size=32, shuffle=True,
#                               num_workers=cfg.num_workers) for x in ['train', 'val']}
# print(data_loaders)
# dataset_sizes = {x: len(data_loaders[x]) for x in ['train', 'val']}
# print(dataset_sizes)

datasets = ImageFolder(root=cfg.catdog_train_dir, transform=data_preprocess)


print(datasets.class_to_idx)
print(len(datasets.targets))
print(datasets.classes)
print(datasets.imgs[:30])
print(datasets[0][0], datasets[0][1])

data_loader = DataLoader(dataset=datasets, batch_size=32, num_workers=4)
for index, data in enumerate(data_loader):
    image, label = data
    print(image.size(), label.size())
    break
