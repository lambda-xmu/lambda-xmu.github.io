---
layout:     post
title:      2019-09-22-Factorization-Machine
subtitle:   "因子分解机"
date:       2019-09-22 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Factorization Machine
- Recommender System
---

Concat: [github: lambda_xmu](https://github.com/lambda-xmu)

## CTR预估综述

**点击率(Click through rate)**是点击特定链接的用户与查看页面，电子邮件或广告的总用户数量之比。 它通常用于衡量某个网站的在线广告活动是否成功，以及电子邮件活动的有效性。 点击率是广告点击次数除以总展示次数（广告投放次数）

$$\mathrm{CTR}=\frac{\text { Number of click-throughs }}{\text { Number of impressions }} \times 100(\%)$$

常用的 `CTR` 预估算法有 `FM`, `FFM`, `DeepFM`。

## [Factorization Machines(FM)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)

`FM` 主要目标是：**解决数据稀疏的情况下，特征怎样组合的问题**

`FM` 有一下三个优点：
1. 可以在非常稀疏的数据中进行合理的参数估计
2. FM 模型的时间复杂度是线性的
3. FM 是一个通用模型，它可以用于任何特征为实值的情况

### `one-hot` 编码带来的问题

FM(Factorization Machine) 主要是为了解决数据稀疏的情况下，特征怎样组合的问题。已一个广告分类的问题为例，根据用户与广告位的一些特征，来预测用户是否会点击广告。数据如下：

![]({{ site.url }}/img/one-hot.png)

clicked 是分类值，表明用户有没有点击该广告。1表示点击，0表示未点击。而 country, day, ad_type 则是对应的特征。对于这种 categorical 特征，一般都是进行 one-hot 编码处理。

将上面的数据进行one-hot编码以后，就变成了下面这样 ：

![]({{ site.url }}/img/one-hot2.png)

因为是 categorical 特征，所以经过 one-hot 编码以后，不可避免的样本的数据就变得**很稀疏**。

### 对特征进行组合

普通的线性模型，都是将各个特征独立考虑的，并没有考虑到特征与特征之间的相互关系。但实际上，大量的特征之间是有关联的。

一般的线性模型为：
$$y=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}$$

从上面的式子看出，一般的线性模型没有考虑特征间的关联。为了表述特征间的相关性，我们采用多项式模型。在多项式模型中，特征 $x_i$ 与 $x_j$ 的组合用 $x_ix_j$ 表示。二阶多项式模型表达式如下：
$$y=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}+\sum_{i=1}^{n-1} \sum_{j=i+1}^{n} w_{i j} x_{i} x_{j}$$

上式中，$n$ 表示样本的特征数量, $x_i$ 表示第 $i$ 个特征。与线性模型相比，FM 的模型就多了后面特征组合的部分。

## 模型方程

### 二阶表达式

二项式的特征组合部分的参数有 $\frac{n(n-1)}{2}$ 个。在数据很稀疏的情况下，满足 $x_i$, $x_j$ 都不为 0 的情况非常少，这样将导致 $w_{ij}$ 无法通过训练得出。并且如果特征非常稀疏且维度很高的话，时间复杂度将大大增加。 为了降低时间复杂度，对每一个特征，引入辅助向量 lantent vector $V_{i}=\left[v_{i 1}, v_{i 2}, \ldots, v_{i k}\right]^{T}$，利用 $V_{i}V_{j}^T$ 对 $w_{ij}$ 进行求解。

$$\mathbf{V}=\left(\begin{array}{cccc}{v_{11}} & {v_{12}} & {\cdots} & {v_{1 k}} \\ {v_{21}} & {v_{22}} & {\cdots} & {v_{2 k}} \\ {\vdots} & {\vdots} & {} & {\vdots} \\ {v_{n 1}} & {v_{n 2}} & {\cdots} & {v_{n k}}\end{array}\right)_{n \times k}=\left(\begin{array}{c}{\mathbf{V}_{1}} \\ {\mathbf{V}_{2}} \\ {\vdots} \\ {\mathbf{V}_{n}}\end{array}\right)$$

那么 $w_{ij}$ 组成的矩阵可以表示为:
$$\widehat{\mathbf{W}}=\mathbf{V} \mathbf{V}^{T}=\left(\begin{array}{c}{\mathbf{V}_{1}} \\ {\mathbf{V}_{2}} \\ {\vdots} \\ {\mathbf{V}_{n}}\end{array}\right)\left(\begin{array}{cccc}{\mathbf{V}_{1}^{T}} & {\mathbf{V}_{2}^{T}} & {\cdots} & {\mathbf{V}_{n}^{T}}\end{array}\right)$$

因为 $\widehat{\mathbf{W}}=\mathbf{V} \mathbf{V}^{T}$ 对应一种矩阵分解，因此称模型方法为 Factorization Machines 方法。

对于单个元素：
$$\widehat{w}_{i j}=\mathbf{V}_{i} \mathbf{V}_{j}^{\top} :=\sum_{l=1}^{k} v_{i l} v_{j l}$$
最终可得：
$$y=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}+\sum_{i=1}^{n} \sum_{j=i+1}^{n}<V_{i}, V_{j}>x_{i} x_{j}$$

以上就是 FM 模型的表达式。$k$ 是超参数，即 lantent vector 的维度，一般取 30 或 40, 也可以取其他数，具体情况具体分析。

这里存在两个问题：为什么需要进行分解？分解后表达能力是否足够？
> 1. 因为数据是非常稀疏的，对于样本中未出现的交互特征分量，不能对相应的参数进行估计。但通过分解 $\widehat{w}_{i j}=\mathbf{V}_{i} \mathbf{V}_{j}^{\top}$，某个属性的辅助向量可以通过所用用户这个属性进行进行训练得到（获得这个属性的向量，有点 Embedding 的意味）。
>2. 当 $k$ 足够大时，对于任意对称正定矩阵 $\widehat{W} \in \mathbb{R}^{n \times n}$，均存在实矩阵 $V \in \mathbb{R}^{n \times k}$，使得 $\widehat{W}=V V^{\top}$ 成立。

### 复杂度分析

FM 模型需要估计的参数包括：

$$w_{0} \in \mathbb{R}, \quad \mathbf{w} \in \mathbb{R}^{n}, \quad \mathbf{V} \in \mathbb{R}^{n \times k}$$

共有 $1+n+nk$ 个参数。其中 $w_{0}$ 为整体的偏置量，$\mathbf{w}$ 是对特征向量的各个分量的强度进行建模，$\mathbf{V}$ 对特征中任意两个分量之间的关系进行了建模。

复杂度如下：

$$\{n+(n-1)\}+\left\{\frac{n(n-1)}{2}[k+(k-1)+2]+\frac{n(n-1)}{2}-1\right\}+2=\mathcal{O}\left(k n^{2}\right)$$

第一花括号对应 $\sum_{i=1}^{n} w_{i} x_{i}$ 的加法和乘法操作；第二个花括号对应 $\sum_{i=1}^{n-1} \sum_{j=i+1}^{n}<V_{i}, V_{j}> x_{i} x_{j}$ 的加法和乘法操作数。

但可以通过如下方式化简。对于 FM 的交叉项:

$$\begin{array}{l}{\sum_{i=1}^{n} \sum_{j=i+1}^{n}<V_{i}, V_{j}>x_{i} x_{j}} \\ {=\frac{1}{2} \sum_{i=1}^{n} \sum_{j=1}^{n}<V_{i}, V_{j}>x_{i} x_{j}-\frac{1}{2} \sum_{i=1}^{n}<V_{i}, V_{i}>x_{i} x_{i}} \\ {=\frac{1}{2}\left(\sum_{i=1}^{n} \sum_{j=1}^{n} \sum_{f=1}^{k} v_{i f} v_{i f} x_{i} x_{j}-\sum_{i=1}^{n} \sum_{f=1}^{k} v_{i f} v_{i f} x_{i} x_{i}\right)} \\ {=\frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i f} x_{i}\right)\left(\sum_{j=1}^{n} v_{j f} x_{j}\right)-\sum_{i=1}^{n} v_{i f}^{2} x_{i}^{2}\right)} \\ {=\frac{1}{2} \sum_{f=1}^{k}\left(\left(\sum_{i=1}^{n} v_{i f} x_{i}\right)^{2}-\sum_{i=1}^{n} v_{i f}^{2} x_{i}^{2}\right) }\end{array}$$

通过对每个特征引入 lantent vector $V_i$, 并对表达式进行化简，可以把时间复杂度降低到：

$$k\{[n+(n-1)+1]+[3 n+(n-1)]+1\}+(k-1)+1=\mathcal{O}(k n)$$

### Other

FM 可以推广到多阶模型，可用于分类、回归和排名。可用 SGD、ALS 和 MCMC 进行训练。详见推荐阅读。

## 推荐阅读：

- [分解机(Factorization Machines)推荐算法原理](https://www.cnblogs.com/pinard/p/6370127.html)
- [深入FFM原理与实践](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)

## 参考

- [CTR预估算法之FM, FFM, DeepFM及实践](https://blog.csdn.net/john_xyz/article/details/78933253)
- [推荐系统遇上深度学习(一)--FM模型理论和实践](https://www.jianshu.com/p/152ae633fb00)
