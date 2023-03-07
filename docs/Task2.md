## Task 2

### 2.1 Tensor Data - Indexing

文档的说法是Scalar对于每单个数字都封装Object, 所以再搞个Tensor  
但其实我们都知道你想干嘛对不对.jpg  

为了深度学习中更常见的矩阵运算, matmul, 转置/permute, reshape, 这些操作的高效, 底层直接是个一维数组, 附加维度信息, 运算时不直接改变底层的存储结构  
考虑二维矩阵坐标[i, j]的点, `i*c+j` = 一维数组的坐标, 三维就是`i*r+j*c+k`, 以此类推. 也就是说n维压一维知道shape就可以  
考虑转置操作, 这时候不只是shape交换了, index也要按转置的顺序做一次交换, 所以需要记录下来, 就是strides  

基于上面对TensorData的理解容易写出index和position的互转, index->position就是strides和index的zipMul, position->index相当于进制转换, 除法取模  

### 2.2 Tensor Broadcasting

实现广播机制的两个函数, 一个是求广播后的shape, 另一个比较难理解, 给出大小两个shape, 求大shape某个index的数据来自小shape的哪个index  

广播的机制是右对齐匹配, 如果相同则匹配上, 不同则必须有一个为1, 为1的dim广播到另一个不为1的大小.  

### 2.3 Tensor Operations 

实现map, zip, reduce算子  
因为整个out都要填充, 所以与index没有关系, 直接遍历storage, 再用2.2的函数找in的index即可  
reduce需要聚合reduce_dim的每一个元素  

实现tensor算子的前向  
和Scalar不同的是需要考虑shape/转置/广播等因素, 所以需要上面的map/zip/reduce包装方法  

### 2.4 Gradients and Autograd

实现tensor算子的后向, 同样要注意map/zip/reduce  
某些Scalar中返回0的梯度这里也要返回tensor的类型, 并且应该保留shape, 不然有矩阵乘法的时候会不匹配  

这里还可能发现前面一个拓扑排序的bug: 注意判断Constant. Scalar和Constant是兼容的所以能过判断, 但是Tensor不能.

另一个比较有难度算子的是permute的后向, 需要把原本的顺序恢复回来  

```
比如(1, 3, 0, 2) -> (2, 0, 3, 1) // new_order[order[i]] = i  

python的简单实现:
reverse_order, _ = zip(*sorted(enumerate(order), key=lambda pair: pair[1]))

```