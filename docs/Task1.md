## Task 1

### 1.1 Numerical Derivatives

实现数值微分, 这里第一次用了python的可变参数列表  

Examples:
```

def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    ...


vals // 直接这样用相当于tuple
f_x = f(*vals) // *解包tuple传入f

```

### 1.2 Scalars

实现Scalar的算子前向传播和运算符重载, 参考例子很容易实现, 主要是看一下各个类的结构  

`ScalarFunction`是各个算子的包装类, 统一执行前向和后向, 算子类继承`ScalarFunction`类
```
class ScalarFunction:
    ...
    @classmethod
    def apply(cls, *vals: "ScalarLike") -> Scalar:
        ...
        c = cls._forward(ctx, *raw_vals)
        ...
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)
```

重载运算符使Scalar用起来像基本类型, 直接与基本类型运算时也能转换为Scalar
```
    def __add__(self, b: ScalarLike) -> Scalar:
        return Add.apply(self, b)
```

### 1.3 Chain Rule

这里还没有实现反向传播, 先实现链式法则我是有点懵的...不同的框架叫法和写法都不一样, 我哪知道你想干什么    

函数签名表示这是传入一个梯度值, 返回一个列表, 列表元素应该就是Scalar+梯度
```
def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
```
看`Tensor`类的`chain_rule`实现猜一猜, 再联想反向传播的过程: 从计算图的末端节点传回梯度值, 对于每一个节点, 更新自己的梯度, 还要把新梯度往所有的前驱节点传递.  
那这里的意思应该就是返回所有前驱节点+新梯度值的组合, 用在反向传播的过程中, 从记录的history里面找到上一个函数, 调用反向即可

### 1.4 Backpropagation

实现算子的反向+整个反向传播  

对于一些需要原输入的算子, 在前向时记录`ctx`, 反向时取出来用  
`lt`, `eq`的梯度是0  

反向传播的实现: 先对计算图做拓扑排序, 各个点的梯度值记录到map里, 调用chain_rule更新梯度值