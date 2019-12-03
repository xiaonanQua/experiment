"""
实现将pytorch模型转成onnx
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import torchvision.transforms as transforms
from config.cifar10_config import Cifar10Config
import os

cfg = Cifar10Config()
test_data_preprocess = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=cfg.mean,
                                                                std=cfg.std)])

test_loader = cfg.dataset_loader(root=cfg.cifar_10_dir, train=False, shuffle=False,
                                 data_preprocess=test_data_preprocess)

net = resnet50()
in_features = net.fc.in_features
net.fc = nn.Linear(in_features=in_features, out_features=10)

if os.path.exists(cfg.checkpoints):
    net.load_state_dict(torch.load(cfg.checkpoints))
net = net.to(cfg.device)

for index, data in enumerate(test_loader):
    images, labels = data
    images = images.to(cfg.device)
    torch.onnx.export(net, images, 'resnet_cifar10.onnx', verbose=True)
    break

# input = torch.randn(10, 3, 224, 224, device='cuda')
# model = torchvision.models.alexnet(pretrained=True).cuda()
#
    # python3 mo.py --input_model /home/xiaonan/experients/utils/resnet_cifar10.onnx --output_dir /home/xiaonan/test --data_type FP16

# ./ classification_sample_async - i / opt / intel / openvino / deployment_tools / demo / car.png - m / home / xiaonan / test / resnet_cifar10.xml - d
# CPU

