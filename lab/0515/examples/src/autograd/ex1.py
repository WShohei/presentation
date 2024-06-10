import torch

x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y)
# tensor([[3., 3.],
#         [3., 3.]], grad_fn=<AddBackward0>)
z = y * y * 3  # = 3(x + 2)^2
print(z)
# tensor([[27., 27.],
#         [27., 27.]], grad_fn=<MulBackward0>)
out = z.mean()  # tensor(27.)
print(out)
out.backward()
print(x.grad)  # d(out)/dx = 6(x + 2)/4 = 4.5
# tensor([[4.5000, 4.5000],
#         [4.5000, 4.5000]])
