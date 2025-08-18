import torch
x = torch.rand(1, 3, 64, 64)
conv = torch.nn.Conv2d(3, 8, 3)
y = conv(x)
print(y.shape)