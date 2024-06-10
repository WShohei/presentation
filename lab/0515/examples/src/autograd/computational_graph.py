import torchviz
import torch

x = torch.ones(2, 2, requires_grad=True)
y = x * x * x * x
# print(y)
out = y.mean()
out.backward()
dot = torchviz.make_dot(out, params=dict(x=x, y=y))
dot.render("graph", format="png", view=True, cleanup=True)
