import torch

w = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
x = torch.tensor([[1.0], [2.0]], requires_grad=True)

y = w @ x
y.backward(torch.tensor([[1.0], [1.0]]))
print(f"x =\n {x}")
print(f"w =\n {w}")
print(f"y = w @ x =\n {y}")
print(f"w.grad =\n {w.grad}")
print(f"x.grad =\n {x.grad}")
