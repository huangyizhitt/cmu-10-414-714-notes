# 自动微分 （Automatic differentiation）

深度学习训练参数时，使用主流优化器，如SGD、Adam等，均依赖梯度计算，如图1中所使用的小批量随机梯度下降算法（Mini-Batch Gradient Descent）。因此，训练过程中，<span style="color:red">使计算机快速、准确、通用地进行微分计算是构建深度学习系统的一个关键技术</span>

<p align="center">
  <img src="../img/gradient.png" alt="求梯度" width="80%">
</p>
<p align="center"><b>图 1：</b> Mini-Batch Gradient Descent算法更新参数</p>

## 一、计算机系统实现微分的方式
### 1. 数值微分（Numerical differentiation）
#### (1) 基本实现原理
该方法是根据微分/偏微分的基本定义，使用有限差分作近似计算。对于 $f(\vec{ \theta }):R^n \rightarrow R$，梯度 $\nabla f=(\frac{\partial f}{\partial \theta_1}, \frac{\partial f}{\partial \theta_2}, ..., \frac{\partial f}{\partial \theta_n})$。
其中， $\frac{\partial f(\vec{ \theta })}{\partial \theta_i}$ 偏微分定义为：

$$
    \frac{\partial f(\vec{ \theta })}{\partial \theta_i}=\lim_{\epsilon \to 0} \frac{f(\vec{ \theta }+\epsilon e_i)-f(\vec{ \theta })}{\epsilon}   \quad     \epsilon>0
$$

如果， $n=1$，即 $\theta$ 是个标量，那么采用微分公式，即：

$$
\frac{d f(\theta)}{d \theta}=\lim_{\epsilon \to 0} \frac{f(\theta+\epsilon e_i)-f(\theta)}{\epsilon}   \quad     \epsilon>0
$$

#### (2) 代码实现
那么，根据上述公式，可设计求梯度的函数为：
```
    def gradient(f, theta):
        epsilon = 0.0001
        return (f(theta + epsilon)-f(theta)) / epsilon
```

#### (3) 优缺点
按照数值微分的方式实现求梯度具有实现简单、通用的优点，但是具有<span style="color:red">三个明显不适合计算机计算的缺点</span>
+ 计算结果不精确
+ 计算复杂度随着函数 $f$ 变化
+ 计算结果对 $\epsilon$ 的取值要求高

#### (4) 改进方法
使用高阶中心差分法，以二阶为例：

$$
    \frac{\partial f(\vec{ \theta })}{\partial \theta_i}=\frac{f(\vec{ \theta }+\epsilon e_i)-f(\vec{ \theta }-\epsilon e_i)}{2\epsilon} + O(\epsilon^2)   \quad     \epsilon>0
$$

#### (5) 实际用途
数值微分存在的缺点使其难以真正作为梯度计算方法用于训练场景。但是，这并不意味着该方法完全没用。事实上，在深度学习系统中，数值微分通常作为单元测试，去验证自动微分的正确性。

$$
    \delta^T \nabla_{\theta} f(\vec{ \theta })=\frac{f(\vec{ \theta }+\epsilon \delta)-f(\vec{ \theta }-\epsilon \delta)}{2\epsilon} + O(\epsilon^2)   \quad     \epsilon>0
$$

从unitball中取 $\delta$ 来验证自动微分的计算


## 2. 符号微分（Symbolic differentiation）

### (1) 基本原理
利用求导法则直接求出函数导数形式的表达式，
例如，函数加法求导：

$$
    \frac{\partial (f(\theta)+g(\theta))}{\partial \theta} = \frac{\partial f(\theta)}{\partial \theta} + \frac{\partial g(\theta)}{\partial \theta}
$$

例如，函数嵌套求导：

$$
    \frac{\partial f(g(\theta))}{\partial \theta} = \frac{\partial f(g(\theta))}{\partial g(\theta)} \cdot \frac{\partial g(\theta)}{\partial \theta}
$$

### (2) 代码实现
以 $n$ 个数的连成为例：

$$
    f(\theta) = \prod \limits_{i=0}^n \theta_i
$$

根据求导法则，可得：

$$
    \frac{\partial f(\theta)}{\partial \theta_k} = \prod \limits_{j \neq k}^n \theta_j
$$

可得函数实现：

```
    def gradient(f, theta):
        grad_array = []
        for i in range(0, n+1):
            grad_i = 1
            for j in range(0, n+1):
                if i != j:
                    grad_i = grad_i * theta[j]
            grad_array.append(grad_i)
```

### (3) 优缺点
由于是按求导法则算出，符号微分具有计算结果精确的优点，但是存在下面三个严重的缺点，使其难以应用到计算机求导中：
+ 每一个算法都必须准备一个导数表达式，并进行代码实现，非常不利于通用性和扩展。
+ 算式复杂将导致导数表达式变得十分复杂。
+ 求偏微分时，计算复杂度与向量维度正相关，当维度很大时，求解非常耗时。

## 3. 自动微分（Automatic differentiation）
### (1) 基本原理
任意复杂计算均是由有限的基本运算组成，利用导数的链式法则，把基本运算导数的符号微分求出，然后组合成复杂计算的导数。

导数的链式法则：

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u_n} \cdot \frac{\partial u_n}{\partial u_{n-1}} \cdot ... \cdot \frac{\partial u_2}{\partial u_{1}} \cdot \frac{u_1}{x}
$$

按照乘法的结合律，链式法则的计算过程可以按正向过程计算，也可以按反向过程计算。正向计算如下：

$$
\frac{\partial y}{\partial x} = (\frac{\partial y}{\partial u_n} \cdot (\frac{\partial u_n}{\partial u_{n-1}} (\cdot ... \cdot (\frac{\partial u_2}{\partial u_{1}} \cdot \frac{u_1}{x}))))
$$

正向计算过程是从输入端开始计算求导，反向过程则是从输出端开始计算求导，反向计算如下：

$$
\frac{\partial y}{\partial x} = (((\frac{\partial y}{\partial u_n} \cdot \frac{\partial u_n}{\partial u_{n-1}})\cdot ...) \cdot \frac{\partial u_2}{\partial u_{1}}) \cdot \frac{u_1}{x}
$$

### (2) 优缺点
自动微分具有精确、可扩展性和通用性强。实现时只需实现基本运算的符号微分，利用链式法则在进行计算时自动组装，实现非常简单。因此，非常适合用于深度学习训练。

必须注意的是<span style="color:red">两个方向的微分存在不同的适应场景</span>，对于 $f: R^n\rightarrow R^k$，此时采用正向自动微分的整体正向传递次数为 $n$ 次（求 $n$ 次偏导），采用反向自动微分则只需要整体传递 $k$ 次。当 $n>k$ 时，即输入维度大于输出维度，反向微分的整体计算次数会更少，反正则更适合正向微分。因此，对于深度学习训练，通常输入维度高于输出维度，更适合反向自动微分。

# 二、深度学习系统中自动微分的实现

