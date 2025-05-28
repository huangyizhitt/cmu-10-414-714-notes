# 自动微分 （Automatic differentiation）

深度学习训练参数时，使用主流优化器，如SGD、Adam等，均依赖梯度计算，如图1中所使用的小批量随机梯度下降算法（Mini-Batch Gradient Descent）。因此，训练过程中，<span style="color:red">使计算机快速、准确、通用地进行微分计算是构建深度学习系统的一个关键技术</span>

<p align="center">
  <img src="../img/gradient.png" alt="求梯度" width="80%">
</p>
<p align="center"><b>图 1：</b> Mini-Batch Gradient Descent算法更新参数</p>

\usepackage{bm}

## 一、计算机系统实现微分的方式
### 1. 数值微分（Numerical differentiation）
#### (1) 基本实现原理
该方法是根据微分/偏微分的基本定义，使用有限差分作近似计算。对于 $f(\boldsymbol{ \theta }):R^n \rightarrow R$，梯度 $\nabla f=(\frac{\partial f}{\partial \theta_1}, \frac{\partial f}{\partial \theta_2}, ..., \frac{\partial f}{\partial \theta_n})$。
其中， $\frac{\partial f(\bm{ \theta })}{\partial \theta_i}$偏微分定义为：

$$
    \frac{\partial f(\bm{ \theta })}{\partial \theta_i}=\lim_{\epsilon \to 0} \frac{f(\bm{ \theta }+\epsilon e_i)-f(\bm{ \theta })}{\epsilon}   \quad     \epsilon>0
$$

如果，$n=1$，即$\theta$是个标量，那么采用微分公式，即：
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
+ 计算复杂度随着函数$f$变化
+ 计算结果对$\epsilon$的取值要求高

#### (4) 改进方法
使用高阶中心差分法，以二阶为例：
$$
    \frac{\partial f(\bm{ \theta })}{\partial \theta_i}=\frac{f(\bm{ \theta }+\epsilon e_i)-f(\bm{ \theta }-\epsilon e_i)}{2\epsilon} + O(\epsilon^2)   \quad     \epsilon>0
$$

#### (5) 实际用途
数值微分存在的缺点使其难以真正作为梯度计算方法用于训练场景。但是，这并不意味着该方法完全没用。事实上，在深度学习系统中，数值微分通常作为单元测试，去验证自动微分的正确性。
$$
    \delta^T \nabla_{\theta} f(\bm{ \theta })=\frac{f(\bm{ \theta }+\epsilon \delta)-f(\bm{ \theta }-\epsilon \delta)}{2\epsilon} + O(\epsilon^2)   \quad     \epsilon>0
$$
从unitball中取$\delta$来验证自动微分的计算
