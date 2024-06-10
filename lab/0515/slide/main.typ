#import "templates/template.typ": *
#show: doc => slides(doc)

#show link: l => underline(l)

#slide(
  title: "環境構築(雑)", pad(
    x: 2em,
  )[
  ローカルで動かせなくても聞けるようにはしたつもりですが、\
  自分で実行したい場合は、
  - Poetryがあれば、
  ```bash
  poetry install
  ```
  で必要なライブラリをインストールできます。\
  (versionが合わない⇒次ページ)
  - requirements.txtがあるので、
  ```bash
  pip install -r requirements.txt
  ```
  でも大丈夫だと思います。 \
  (Ryeは使ったことないので分からないです...)
])

#slide(
  title: "環境構築(雑)", pad(
    x: 2em,
  )[
  バージョンが合わない場合は、
  ```bash
  pyenv install 3.9.{いくつでも大丈夫なはず} # 3.9.13で動作確認
  pyenv local 3.9.{}
  ```
  などでPythonのバージョンを変更してください。
  その後、
  ```bash
  poetry install
  pip install -r requirements.txt
  ```
  で必要なライブラリをインストールしてください。
]
)

#slide(
  title: "topics", pad(
    x: 2em,
  )[
  tensorの基本から始めてモデル学習の流れまでを説明します。
  + Tensors
  + AutoGrad
  + Datasets & Dataloaders
  + Neural Networks (Linear, Conv1dまで)
  + Optimizing Models
  + おまけ
  + Homework
])

= Tensors

#slide(
  title: "", pad(
    x: 2em,
  )[
  - PyTorchではtensorというデータ構造が中心的な役割を果たす
  - Numpyのndarrayに似ているが、以下のような特徴がある
    - GPUを使った計算が可能
    - 自動微分が可能
  - tensorの初期化は以下に示すような方法がある
  ],
)

#slide(
  title: "Tensor Initialization", pad(
    x: 2em,
  )[
  ```python
  import torch
  import numpy as np
  ```
  - データから直接tensorを作成する
    ```python
    data = [[1, 2], [3, 4]]
    x_data = torch.tensor(data)
                        ```
  - Numpyのndarrayからtensorを作成する
    ```python
    np_array = np.array(data)
    x_np = torch.from_numpy(np_array)
                        ```
  - 他のtensorから新しいtensorを作成する
    ```python
    x_ones = torch.ones_like(x_data)
    x_rand = torch.rand_like(x_data, dtype=torch.float)
                        ```
  ],
)

#slide(
  title: "Tensor Initialization", pad(
    x: 2em,
  )[
  - ランダムな値で初期化されたtensorを作成する
    ```python
    shape = (2, 3,)
    rand_tensor = torch.rand(shape)
    ones_tensor = torch.ones(shape)
    zeros_tensor = torch.zeros(shape)
                        ```
  - deviceを指定してtensorを作成する 
    ```python
    dtype = torch.float
    device = torch.device('cpu') # or 'cuda' etc.
    tensor = torch.ones(shape, dtype=dtype, device=device)
                        ```
  ],
)

#slide(
  title: "Tensor Operations", pad(
    x: 2em,
  )[
  - tensorはNumpyのndarrayと同様に演算が可能
    ```python
    x = torch.ones(2, 2)
    y = torch.ones(2, 2)
    z = x + y # +, -, *, /, @ などNumpyと同様の演算子が使える
    print(z)
    # tensor([[2., 2.],
    #         [2., 2.]])
    z[:, 1] = 0 # slicingも可能
    print(z)
    # tensor([[2., 0.],
    #         [2., 0.]])
                        ```
  ]
)

#slide(
  title: "Sending Tensors to GPU", pad(
    x: 2em,
  )[
  - tensorは`to`メソッドを使ってGPUに送ることができる
    ```python
    tensor = torch.ones(4, 4)
    if torch.cuda.is_available():
        tensor = tensor.to('cuda') # send to default GPU
        tensor = tensor.to('cuda:0') # send to GPU 0
                        ```
  - tensorは`device`属性を使ってGPUにあるかどうかを確認できる
    ```python
    print(tensor.device)
    # cuda:0
    ```
  ],
)

== AutoGrad

#slide(
  title: "", pad(
    x: 2em,
  )[
  - PyTorchの最も重要な機能の一つは自動微分機能
  - tensorは`requires_grad=True`を指定することで、そのtensorに対する操作を追跡し、
    微分を計算することができる
    ```python
      x = torch.ones(2, 2, requires_grad=True)
    ```
  - 微分を計算するには`backward`メソッドを呼び出す
  - 微分を計算するためには、
    + `backward`メソッドを呼び出すtensorがスカラーであるか、
    + あるいは`backward`メソッドに引数としてweightを表すtensorを渡す必要がある
  ],
)

#slide(
 title: "Example", pad(
   x: 2em,
 )[
 - `backward`メソッドを呼び出す例 (`autograd/ex1.py`)
    ```python
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    # tensor([[3., 3.],
    #         [3., 3.]], grad_fn=<AddBackward0>)
    z = y * y * 3 # = 3(x + 2)^2
    # tensor([[27., 27.],
    #         [27., 27.]], grad_fn=<MulBackward0>)
    out = z.mean() # tensor(27., grad_fn=<MeanBackward0>)
    out.backward()
    print(x.grad) # d(out)/dx = 6(x + 2)/4 = 4.5
    # tensor([[4.5000, 4.5000],
    #         [4.5000, 4.5000]])
    ```
  ]
)

#slide(
  title: "Example", pad(
    x: 2em,
  )[
  - `backward`メソッドに引数を渡す例1 (`autograd/ex2.py`)
    ```python
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z # tensor([[27., 27.], [27., 27.]])
    out.backward(gradient=torch.ones(2, 2)) # 各要素に対する重みを指定 
    # d(out)/dx = 6(x + 2) = 18
    print(x.grad)
    # tensor([[18., 18.],
    #         [18., 18.]])
    ```
  ]
)

#slide(
  title: "Example", pad(
    x: 2em,
  )[
  - `backward`メソッドに引数を渡す例2 (`autograd/ex3.py`)
    ```python
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z # tensor([[27., 27.], [27., 27.]])
    out.backward(gradient=torch.tensor([[1, 2], [3, 4]]))
    print(x.grad)
    # tensor([[18., 36.],
    #         [54., 72.]])
    ```
  ]
)

#slide(
  title: "非スカラーのTensorの微分", pad(
    x: 2em,
  )[
  上述の通り、`backward`メソッドを非スカラーのtensorに対して呼び出す時には、
  `gradient`引数を指定する必要がある\
  数学的な意味での微分と、異なる動作をすることに注意\
])

#slide(
  title: "非スカラーのTensorの微分", pad(
    x: 2em,
  )[
  $bold(x) := vec(x_1, x_2, dots.v, x_n), 
  bold(W) = mat(
    w_(11), w_(12), dots.h, w_(1n) ;
    w_(21), w_(22), dots.h, w_(2n) ;
    dots.v , dots.v , dots.down , dots.v ;
    w_(m 1), w_(m 2), dots.h, w_(m n) ;
  ), 
  bold(y) = bold(W) bold(x) := vec(y_1, y_2, dots.v, y_m)$
  としたとき、\
  $ y_i = sum_(l=1)^(n) w_(i l) x_l $
  $W$ の各要素 $w_(j k)$ による $bold(y)$ の微分を求めると、
  $ (partial y_i)/(partial w_(j k)) = delta_(i j) x_k $
  という3階のテンソルが得られる\
  ]
)

#slide(
  title: "非スカラーのTensorの微分", pad(
    x: 2em,
  )[
  NNのweightを更新したい場合、上述のような3階テンソルではなく、\
  2階(weightの次元に一致)テンソルを得る必要がある\
  実際にautogradで得られるのは、\
  $
  (partial bold(y)) / (partial bold(W)) =
  mat(
    x_1, x_2, dots.h, x_n ;
    x_1, x_2, dots.h, x_n ;
    dots.v , dots.v , dots.down , dots.v ;
    x_1, x_2, dots.h, x_n ;
  )
  $
  となる。(ref. `diff_tensor.py`)
  ]
)

#slide(
  title: "非スカラーのTensorの微分", pad(
    x: 2em,
  )[
  #set text(size: 20pt)
  行列積の微分は次のように計算できる。\
  Lossを$L$ (スカラー)、$bold(Y) = bold(W) bold(X)$ としたとき、\
  $
  (partial L) / (partial bold(W)) & = (partial L) / (partial w_(i j))\
  & = (partial L) / (partial y_(i k)) (partial y_(i k)) / (partial w_(i j)) space (because y_(i k) = w_(i j) x_(j k))\
  & = (partial L) / (partial y_(i k)) x_(j k) space (because (partial y_(i k)) / (partial w_(i j)) = x_(j k))\
  & = (partial L) / (partial bold(Y)) bold(X)^T
  $
  非スカラーのtensorに対する微分を計算する際に指定する`gradient`引数は、\
  ここでの$(partial L) / (partial bold(Y))$に相当する部分である
]
)

#slide(
  title: "Disabling Autograd", pad(
    x: 2em,
  )[
  - `requires_grad`が`True`のtensorに対しては、その計算は追跡されるが、`torch.no_grad`ブロック内では追跡を無効にすることができる(`autograd/no_grad.py`)
    ```python
    x = torch.ones(2, 2, requires_grad=True)
    print(x.requires_grad) # True
    with torch.no_grad():
        print((x ** 2).requires_grad) # False
    ```
  ]
)

#slide(
  title: "Disabling Autograd", pad(
    x: 2em,
  )[
  - `detach()` メソッドを使うことでも追跡を無効にすることができる
    ```python
    x = torch.ones(2, 2, requires_grad=True)
    y = x.detach()
    print(y.requires_grad) # False
    ```

  ユースケースとしては、
  + ファインチューニング時に一部のパラメータを固定する場合
  + `forward`の結果のみが必要な場合の高速化
  などがある
  ]
)

#slide(
 title: "Computational Graph", pad(
   x: 2em,
 )[
 PyTorchではAutoGradのために計算グラフが構築される
  - tensorは`grad_fn`属性を持ち、そのtensorを作成した演算を記録している
  - 各演算の微分があらかじめ定義されており、逆伝播時にはそれを使って微分を計算する
  - `backward`メソッドを呼ぶと各変数tensorの`grad`属性に微分が格納される
  - 計算グラフは動的であり、学習を行いながらモデルの構造を変更することが可能
 ]
)

#split_slide(
  title: "Computational Graph", pad(
    x: 2em,
  )[
  `torchviz`を使って計算グラフを可視化することができる (`autograd/computational_graph.py`)
  ```python
  import torchviz
  import torch
  
  x = torch.ones(2, 2, requires_grad=True)
  y = x * x * x * x
  out = y.mean()
  out.backward()
  dot = torchviz.make_dot(out, params=dict(x=x, y=y))
  dot.render("graph", format="png")
  ```
  ],
  [
  #image("images/graph.png")
  ],
  left_ratio: 2.4fr,
  right_ratio: 1fr,
)

= Datasets & Dataloaders

#slide(title: "Datasets & Dataloaders", [
    - PyTorchでは、データセットとデータローダーを使ってデータを扱う
    - データセットはデータを格納し、データローダーはデータセットからバッチを取得する
    - データセットは`torch.utils.data.Dataset`クラスを継承して作成する
    - データローダーは`torch.utils.data.DataLoader`クラスを使って作成する
])

#slide(title: "Example (MNIST)", [
    1. ライブラリのインポート
    ```python
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    ```])

#slide(title: "Example (MNIST)", [
    #set text(size: 21pt)
    2. データセットのダウンロードと変換\
    ```python
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, # rootにデータがない場合にダウンロードする
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    ```
])

#slide(title: "Example (MNIST)", [
    3. データローダーの作成\
    ```python
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=64, 
        shuffle=False
    )
    ```\
])

#slide(title: "Custom Dataset", [
    - 自前のデータでDatasetを作ることも可能
    - `torch.utils.data.Dataset`クラスを継承して、\
      `__len__`メソッドと`__getitem__`メソッドを実装する
])

#slide(title: "Datasetの作成", [
    ステップ1: CustomDatasetクラスの作成\
    ```python
    class CustomDataset(Dataset):
        def __init__(self, data, labels, transform=None):
            self.data = data
            self.labels = labels
            self.transform = transform
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            if self.transform:
                sample = self.transform(sample) # 前処理
            return sample, label
    ```
])

#slide(title: "DataLoaderの使用", [
    ステップ2: Datasetの初期化\
    ```python
    data = ... # your data
    labels = ... # your labels
    dataset = CustomDataset(data, labels)
    ```
    ステップ3: DataLoaderの作成\
    ```python
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for batch_data, batch_labels in dataloader:
        # training code here
    ```
])

#slide(title: "__len__", [
    `__len__`メソッドは、データセットのサイズを返す。\
    ```python
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
    ```
    データローダーがデータセットの終了を確認するために使用される。\
 ])

#slide(title: "__getitem__", [
    `__getitem__`メソッドは、データセットから\
    特定のインデックスにあるサンプルを取得する。\
    ```python
    class CustomDataset(Dataset):
        def __init__(self, data):
            self.data = data
        
        def __getitem__(self, idx):
            return self.data[idx]
    ```
    このメソッドを実装することで、\
    データセットから特定のデータポイントを取得できる。\
    データローダーが各バッチのデータを読み込む際に使用される。\
])


= Neural Networks

#slide(
  title: "Neural Networks", pad(
    x: 2em,
  )[
  - Pytorchでは`torch.nn`にニューラルネットワークの構築に必要な\
    モジュールが含まれている
  - `torch.nn.Module`クラスを継承して、\
    `__init__`メソッドと`forward`メソッドを実装する
])

#split_slide(
  title: "Example", pad(
    x: 2em,
  )[
  #set text(size: 20pt)
  ```python
  class NeuralNetwork(nn.Module):
      def __init__(self):
          super(NeuralNetwork, self).__init__()
          self.flatten = nn.Flatten()
          self.linear_relu_stack = nn.Sequential(
              nn.Linear(28*28, 512),
              nn.ReLU(),
              nn.Linear(512, 512),
              nn.ReLU(),
              nn.Linear(512, 10)
          )
      def forward(self, x):
          x = self.flatten(x)
          logits = self.linear_relu_stack(x)
          return logits
  ```
  ],
  [
  #set text(size: 20pt)
  - `__init__`メソッドでは、\
    ネットワークの構造を定義する
  - `forward`メソッドでは、\
    データがネットワークを通過するときの処理を定義する
  ],
  left_ratio: 1.7fr,right_ratio: 1.02fr)


#slide(title: "Model Parameters", pad(
  x: 2em,
)[
  nn.Modlueをsubclassに持つモデルのパラメータは、\
  `parameters()`や`named_parameters()`メソッドを使って取得できる\
  (自前のパラメータは`nn.Parameter`を使って作成する必要)
  ```python
  model = NeuralNetwork()
  for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
  # Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0022, -0.0021, ...],
  # ...
  ```
])

//#slide(
//  title: "nn.Flatten", pad(
//    x: 2em,
//  )[
//  ```python
//  torch.nn.Flatten(start_dim=1, end_dim=-1)
//  ```
//  `start_dim`から`end_dim`までの次元を平坦化する
//  ```python
//  x = torch.rand(1, 28, 28)
//  flatten = nn.Flatten()
//  flatten(x).shape
//  # torch.Size([1, 784])
//  ```
//  ]
//)

#slide(title: "nn.Linear", pad(
  x: 2em,
)[
  ```python
  torch.nn.Linear(in_features, out_features, bias=True)
  ```
  入力の線形変換を行う $y = x A^T + b$\
    - `in_features`: 入力のサイズ
    - `out_features`: 出力のサイズ
    - `bias`: バイアス項を含めるかどうか
  ```python
  linear = nn.Linear(20, 30)
  input = torch.randn(128, 20)
  output = linear(input)
  print(output.size())
  # torch.Size([128, 30])
  ```
])

#slide(
  title: "nn.Conv1d", pad(
    x: 2em,
  )[
  ```python
  torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
  ```
  1次元の畳み込みを行う\
    - `in_channels`: 入力のチャンネル数
    - `out_channels`: 出力のチャンネル数
    - `kernel_size`: カーネルのサイズ
    - `stride`: ストライド (default: 1)
    - `padding`: パディング (default: 0)
    - `dilation`: カーネルの間隔 (default: 1)
    - `groups`: グループ数 (default: 1)
    - `bias`: バイアス項を含めるかどうか (default: True)
])

#slide(
  title: "nn.Conv1d", pad(
    x: 2em,
  )[
  `dilation`はカーネルの間隔を指定する\
  通常のカーネル
  ```python
  [1, 2, 3]
  ```
  dilation=2の場合
  ```python
  [1, 0, 2, 0, 3]
  ```
])

#slide(
  title: "nn.Conv1d", pad(
    x: 2em,
  )[
  `groups`は入力と出力のチャンネルをグループに分割する\
  `in_channels=4, out_channels=8, groups=2`の場合\
  ```python
  Group 1: Input channels [0, 1] -> Output channels [0, 1, 2, 3]
  Group 2: Input channels [2, 3] -> Output channels [4, 5, 6, 7]
  ```
])

#slide(
  title: "nn.Conv1d", pad(
    x: 2em,
  )[
  出力サイズは以下のように計算される\
  $
  O = floor((I + 2 times P - D times (K - 1) - 1) / S) + 1 \
  (O: "output size", I: "input size", D: "dilation" \
  P: "padding", K: "kernel size", S: "stride")
  $
])

#slide(
  title: "nn.Conv1d", pad(
    x: 2em,
  )[
  ```python
  # 入力チャンネル数3、出力チャンネル数6、カーネルサイズ5の1次元畳み込みレイヤー
  conv1d_layer 
      = nn.Conv1d(in_channels=3, out_channels=6, kernel_size=5)
  
  # ダミー入力データ (バッチサイズ10, チャンネル数3, シーケンス長50)
  input_data = torch.randn(10, 3, 50)
  
  # 畳み込みレイヤーを通してデータを渡す
  output_data = conv1d_layer(input_data)
  
  print(output_data.shape)  # torch.Size([10, 6, 46])
  ```
])

= Optimizing Models

#slide(
  title: "Optimizing Models", pad(
    x: 2em,
  )[
  - PyTorchでは、`torch.optim`に最適化アルゴリズムが実装されている
  - モデルのパラメータを更新するためには、\
    `torch.optim.Optimizer`クラスを使う
  - `torch.optim`モジュールには、\
    SGD、Adam、RMSpropなどの最適化アルゴリズムが含まれている
])

#slide(
  title: "Example", pad(
    x: 2em,
  )[
  ```python
  model = NeuralNetwork()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  ```
  - `model.parameters()`でモデルのパラメータを取得し、\
    `lr`で学習率を指定して最適化アルゴリズムを初期化する

  ```python
  # Inside the training loop
  optimizer.zero_grad() # 勾配を初期化
  loss_fn = nn.CrossEntropyLoss()
  loss = loss_fn(model(data), target)
  loss.backward() # 勾配を計算
  optimizer.step() # parameterのtensor.gradに基づいてパラメータを更新 
  ```
])

#slide(
  title: "Optimizer", pad(
    x: 2em,
  )[
  全体的な学習の流れは`train/training_ex.py`を参照
]
)

= おまけ (pytorchの内部実装)

#slide(
  title: "", pad(
    x: 2em,
  )[
  #set text(size: 20pt)
  #link("https://github.com/pytorch/pytorch/blob/main/torch/__init__.py#L1514")[`torch/__init__.py`]には次のような部分がある
  ```python
  for name in dir(_C._VariableFunctions):
    if name.startswith('__') or name in PRIVATE_OPS:
        continue
    obj = getattr(_C._VariableFunctions, name)
    obj.__module__ = 'torch'
    # Hide some APIs that should not be public
    if name == "segment_reduce":
        # TODO: Once the undocumented FC window is passed, remove the line bellow
        globals()[name] = obj
        name = "_" + name
    globals()[name] = obj
    if not name.startswith("_"):
        __all__.append(name)
  ```
])

#slide(
  title: "", pad(
    x: 2em,
  )[
  pytorchの実装はほとんどがCUDA C++で書かれており、`_C`の実体は`torch._C.so`でありコンパイル時に生成される\
  コンパイル前のソースの多くは#link("https://github.com/pytorch/pytorch/tree/main/aten/src/ATen")[ここ]にある。

  いろいろ遡っていくと、最終的に非オープンソースのcuDNNの関数が呼ばれているところ(たとえば
  #link("https://github.com/pytorch/pytorch/blob/37596769d8b42beba104e14d149cebe0dfd75d12/aten/src/ATen/native/cudnn/Conv_v7.cpp#L697")[ここ]まで辿り着けます。
])

= Homework

#slide(
  title: "", pad(
    x: 2em,
  )[
  + 演算子オーバーロードを用いて、autogradが可能なTensor classを実装する
  + 1で作ったTensorに対して計算グラフの出力を行えるように`draw_graph`メソッドを実装する (`graphviz`を使うと便利)
  + `nn.Conv2d`の実装
])

= References

#slide(
  title: "References", pad(
    x: 2em,
  )[
  + PyTorch Documentation (https://pytorch.org/docs/stable/index.html)
  + PyTorch Tutorials (https://pytorch.org/tutorials/)
  ]
)
