## 1a

Because the true empirical distribution $\boldsymbol{y}$ is a one-hot vector, for any word $w=o$, only $y_o$ equals $1$ and others equals $0$. So, cross-entropy loss is the same as the naive-softmax loss.
$$
\begin{aligned}
- \sum_{w\in Vocab}y_w\log(\hat{y}_w) &= - [y_1\log(\hat{y}_1) + \cdots + y_o\log(\hat{y}_o) + \cdots + y_w\log(\hat{y}_w)] \\
& = - y_o\log(\hat{y}_o) \\
& = -\log(\hat{y}_o) \\
& = -\log \mathrm{P}(O = o | C = c) 
\end{aligned}
$$

## 1b

先把loss按照cross entropy展开
$$
\because 
J_\mathrm{naive-softmax} = CE(y, \hat{y}) = -\log(\hat{y}_o) \\
\hat{y}_o = \mathrm{softmax}(\theta) = {\exp(\theta_o) \over \sum_j \exp(\theta_j) }\\
\therefore 
J = -\log  {\exp(\theta_o) \over \sum_j \exp(\theta_j) } = -\theta_o + \log \sum_j \exp(\theta_j)
$$
然后求偏导数
$$
\frac{\partial J}{\partial \theta} = -\frac{\partial \theta_o}{\partial\theta} + \frac{\partial\log\sum_j\exp(\theta_j)}{\partial\theta}
$$
对第一部分，对向量求导即对向量内元素分别求导，维度不变。
$$
\frac{\partial\theta_o}{\partial\theta} =y
$$
其中，$\frac{\partial\theta_o}{\partial\theta_o}=1$，其他情况 $j\neq o$ 时 $\frac{\partial\theta_o}{\partial\theta_j}=1$，即 $y=\{0 \cdots y_o \cdots 0\}$。

对第二部分，应用chain rule，同样对向量内元素分别求导。
$$
\begin{aligned}
\frac{\partial{\log(\sum_j{e^{\theta_j})}}}{\partial{\theta_i}}
    &= \frac{\partial{\log(\sum_j{e^{\theta_j})}}} {\sum_j{e^{\theta_j}}} \cdot \frac{\sum_j{e^{\theta_j}}}{\partial{\theta_i}}  \\
    &= \frac{1}{\sum_j{e^{\theta_j}}} \cdot e^{\theta_i}  \\
    &= \frac{e^{\theta_i}}{\sum_j{e^{\theta_j}}}  \\
    &= \hat{y}_i  
\end{aligned}
$$
把两部分合在一起：
$$
\frac{\partial J}{\partial \theta}=- \frac{\theta_o}{\partial\theta} + \frac{\partial\log\sum_j{e^{\theta_j}} }{\partial\theta} = -y + \hat{y}
$$
然后，loss对$v_c$求偏导，应用chain rule：
$$
\begin{aligned}
\frac{\partial J}{\partial v_c} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial v_c} \\
&= (\hat{y} - y) \frac{\partial U^T v_c}{\partial v_c} \\
&= (\hat{y} - y) U^T
\end{aligned}
$$
其中，$U\in R^{V\times h}$包括所有的outside vector $u_w$。

## 1c

类似上面的求解方法
$$
\begin{aligned} 
\frac{\partial J}{\partial u_w} &= \frac{\partial J}{\partial \theta} \frac{\partial \theta}{\partial U} \\
&= (\hat{y} - y) \frac{\partial U^Tv_c}{\partial U} \\
&= v_c(\hat{y} - y)^T 
\end{aligned}
$$

## 1d

已知 $\exp(x)' = \exp(x)$，应用chain rule，令$f=\exp(-x)$
$$
\begin{aligned}
\frac{d}{dx}\sigma(x) &= \frac{d\sigma(x)}{df} \frac{df}{dx} \\
&=-(1+\exp(-x))^{-2} \cdot -\exp(-x) \\
&=\frac1{1+\exp(-x)}\cdot \frac1{1+exp(-x)} \exp(-x) \\
&=\sigma(x) \cdot \sigma(-x)
\end{aligned}
$$

## 1e

$$
J_\mathrm{neg-sample} = -\log(\sigma(u_o^T v_c)) - \sum_{k=1}^K \log(\sigma(-u_k^T v_c))
$$

$$
\begin{aligned}
\frac{\partial J}{\partial v_c} &= -\frac1{\sigma(u_o^T v_c)}\cdot \frac{\partial \sigma(u_o^T v_c)}{\partial v_c} - \sum_{k=1}^K \frac1{\sigma(-u_k^T v_c)} \cdot \frac{\partial \sigma(-u_k^T v_c)}{\partial v_c} \\
&= -\frac1{\sigma(u_o^T v_c)}\cdot \sigma(u_o^T v_c)\sigma(-u_o^T v_c)\cdot \frac{\partial u_o^T v_c}{\partial v_c} - \sum_{k=1}^K \frac1{\sigma(-u_k^T v_c)} \cdot \sigma(-u_k^T v_c)\sigma(u_k^T v_c) \cdot \frac{\partial -u_k^T v_c}{\partial v_c} \\
&= - \sigma(-u_o^T v_c) u_o^T - \sum_{k=1}^K \sigma(u_k^T v_c)(-u_k^T) \\
&=\sigma(u_o^T v_c)u_o^T - u_o^T + \sum_{k=1}^K \sigma(u_k^T v_c)u_k^T \\
\end{aligned}
$$

