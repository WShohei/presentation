import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z  # tensor([[27., 27.], [27., 27.]])
out.backward(gradient=torch.tensor([[1, 2], [3, 4]]))
print(x.grad)
# tensor([[18., 36.],
#         [54., 72.]])
