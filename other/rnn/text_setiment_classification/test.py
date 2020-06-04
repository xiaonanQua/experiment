import torch

a = torch.zeros(2, 3, 4)
b = torch.ones(2, 3, 4)

print(torch.cat((a[0], b[-1]), -1).size())