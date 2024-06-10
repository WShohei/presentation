import torch

x = torch.ones(2, 2, requires_grad=True)
print(x.requires_grad)  # True
with torch.no_grad():
    print((x**2).requires_grad)  # False
