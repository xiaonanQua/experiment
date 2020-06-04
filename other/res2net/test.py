import other.res2net.res2net as res
from torchvision.models import resnet50
import torch

data = torch.randn([1, 3, 224, 224])

net = res.res2net50()
net2 = resnet50()
print(net(data))
# print(net)