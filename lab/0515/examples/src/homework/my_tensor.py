import numpy as np
from graphviz import Digraph


class Tensor:
    def __init__(self, data, requires_grad=False, name=None):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()
        self._op = ""  # 操作の名前を記録するためのプロパティ
        self.name = name  # 変数名を記録

    def __add__(self, other):
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._op = "+"  # 操作の名前を記録

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __sub__(self, other):
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._op = "-"  # 操作の名前を記録

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad -= out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._op = "*"  # 操作の名前を記録

        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __matmul__(self, other):
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
        )
        out._op = "@"  # 操作の名前を記録

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def backward(self):
        topological_order = []
        visited = set()

        def build_topological(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topological(child)
                topological_order.append(node)

        build_topological(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topological_order):
            node._backward()

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, name={self.name})"

    def draw_graph(self):
        dot = Digraph(format="png", graph_attr={"rankdir": "LR"})

        def add_nodes(tensor):
            if tensor not in dot.node_names:
                node_label = (
                    f"{tensor.name}\n{tensor._op}\n{tensor.data.shape}"
                    if tensor.name
                    else f"{tensor._op}\n{tensor.data.shape}"
                )
                dot.node(str(id(tensor)), label=node_label)
                dot.node_names.add(tensor)
                for child in tensor._prev:
                    dot.edge(str(id(child)), str(id(tensor)))
                    add_nodes(child)

        dot.node_names = set()
        add_nodes(self)

        return dot


# 使用例
a = Tensor(np.array([[1, 2], [3, 4]]), requires_grad=True, name="a")
b = Tensor(np.array([[5, 6], [7, 8]]), requires_grad=True, name="b")
c = a @ b
c.name = "c"
d = a + b
d.name = "d"
e = a * c * d
e.name = "e"

e.backward()

print(a)  # Tensor(data=[[1 2] [3 4]], grad=[[1 1] [1 1]], name=a)
print(b)  # Tensor(data=[[5 6] [7 8]], grad=[[4 4] [6 6]], name=b)
print(c)  # Tensor(data=[[19 22] [43 50]], grad=[[1 1] [1 1]], name=c)
print(d)  # Tensor(data=[[ 6  8] [10 12]], grad=[[1 1] [1 1]], name=d)
print(e)  # Tensor(data=[[114 176] [430 600]], grad=[[1 1] [1 1]], name=e)

# 計算グラフの出力
dot = e.draw_graph()
dot.render("computation_graph", view=False)
