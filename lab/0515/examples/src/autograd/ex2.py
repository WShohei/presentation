import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z  # tensor([[27., 27.], [27., 27.]])
out.backward(gradient=torch.ones(2, 2))  # 各要素に対する重みを指定
# d(out)/dx = 6(x + 2) = 18
print(x.grad)
# tensor([[18., 18.],
#         [18., 18.]])
