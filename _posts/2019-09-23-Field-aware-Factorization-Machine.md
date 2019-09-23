---
layout:     post
title:      2019-09-23-Field-aware-Factorization-Machine
subtitle:   "场感知分解机"
date:       2019-09-23 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Field-aware Factorization Machine
- Recommender System
---

Concat: [github: lambda_xmu](https://github.com/lambda-xmu)


# [Field-aware Factorization Machines(FFM)](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

FFM 模型中引入了类别的概念，即 field，将相同性质的特征归于同一个 field。

<table border=5 cellpadding=0 cellspacing=0 width=609 style='border-collapse:
 collapse;table-layout:fixed;width:455pt;orphans: 2;widows: 2;-webkit-text-stroke-width: 0px;
 text-decoration-style: initial;text-decoration-color: initial'>
 <col width=87 span=7 style='width:65pt'>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl63 width=87 style='height:18.0pt;width:65pt'>&nbsp;field</td>
  <td class=xl63 width=87 style='width:65pt'>field1年龄</td>
  <td colspan=3 class=xl63 width=261 style='width:195pt'>field2城市</td>
  <td colspan=2 class=xl63 width=174 style='width:130pt'>field3性别</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl63 style='height:18.0pt'>&nbsp;feature</td>
  <td class=xl63>x1年龄</td>
  <td class=xl63>x2北京</td>
  <td class=xl63>x3上海</td>
  <td class=xl63>x4深圳</td>
  <td class=xl63>x5男</td>
  <td class=xl63>x6女</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl63 style='height:18.0pt'>用户1</td>
  <td class=xl63>23</td>
  <td class=xl63>1</td>
  <td class=xl63>0</td>
  <td class=xl63>0</td>
  <td class=xl63>1</td>
  <td class=xl63>0</td>
 </tr>
 <tr height=24 style='height:18.0pt'>
  <td height=24 class=xl63 style='height:18.0pt'>用户2</td>
  <td class=xl63>31</td>
  <td class=xl63>0</td>
  <td class=xl63>0</td>
  <td class=xl63>1</td>
  <td class=xl63>0</td>
  <td class=xl63>1</td>
 </tr>
</table>

在 FFM 中，每一维特征 $x_i$，针对其它特征的每一种 field $f_j$，都会学习一个隐向量 $v_{i,f_j}$，即 $v_i$ 成了一个二维向量 $v_{𝐹\times 𝐾}$，$F$ 是 Field 的总个数。因此，隐向量不仅与特征相关，也与 field 相关。这也是 FFM 中 “field-aware” 的由来。

假设样本的 $n$ 个特征属于 $f$ 个 field，那么 FFM 的二次项有 $nf$ 个隐向量。而在 FM 模型中，每一维特征的隐向量只有一个。FM 可以看作 FFM 的特例，是把所有特征都归属到一个 field 时的 FFM 模型。根据 FFM 的 field 敏感特性，可以导出其模型方程。

$$y(\mathbf{x})=w_{0}+\sum_{i=1}^{n} w_{i} x_{i}+\sum_{i=1}^{n} \sum_{j=i+1}^{n}\left\langle\mathbf{v}_{i, f_{j}}, \mathbf{v}_{j, f_{i}}\right\rangle x_{i} x_{j}$$

其中，$f_j$ 是第 $j$ 个特征所属的 field。如果隐向量的长度为 $k$，那么FFM的二次参数有 $nfk$ 个，远多于FM模型的 $nk$ 个。此外，由于隐向量与 field 相关，FFM二次项不能够化简，其预测复杂度是 $O(kn^2)$。

上表格的组合特征如下：

$$
\hat{y}=v_{1, f 2} \cdot v_{2, f 1} x_{1} x_{2}+v_{1, f 3} \cdot v_{3, f 1} x_{1} x_{3}+v_{1, f 4} \cdot v_{4, f 1} x_{1} x_{4}+\cdots
$$

由于 $x_2,x_3,x_4$ 属于同一个 Field，所以 $f_2,f_3,f_4$ 可以用同一个变量来代替，比如就用 $f_2$。

$$
\hat{y}=v_{1, f 2} \cdot v_{2, f 1} x_{1} x_{2}+v_{1, f 2} \cdot v_{3, f 1} x_{1} x_{3}+v_{1, f 2} \cdot v_{4, f 1} x_{1} x_{4}+\cdots
$$

在实际预测点击率的项目中通常会在 FMM 模型上套一层 sigmoid 函数，用 $a$ 表示对点击率的预测值

$$
a=\sigma(\hat{y})=\frac{1}{1+e^{-\hat{y}}}
$$

在参数更新时，令 $y=0$ 表示负样本，$y=1$ 表示正样本，$C$ 表示交叉熵损失函数。
$$
C \equiv-[y \ln a+(1-y) \ln (1-a)], a=\sigma(\hat{y})
$$

则：
$$
\frac{\partial C}{\partial \hat{y}}=\frac{\partial C}{\partial a} \sigma^{\prime}(\hat{y})=-\frac{y}{a} \sigma^{\prime}(\hat{y})+\frac{1-y}{1-a} \sigma^{\prime}(\hat{y})=\frac{a-y}{a(1-a)} \sigma^{\prime}(\hat{y})=a-y
$$

所以：
$$\frac{\partial C}{\partial \hat{y}}=a-y=\left\{\begin{array}{ll}{-\frac{1}{1+e^{\hat{y}}}} & {\text { if } y \text { 是正样本 }} \\ {\frac{1}{1+e^{-\hat{y}}}} & {\text { if } y \text { 是负样本 }}\end{array}\right.$$

$$\frac{\partial \boldsymbol{C}}{\partial v_{i, f_j}}=\frac{\partial \boldsymbol{C}}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial v_{i, f_j}}$$

其中：
$$\frac{\partial \hat{y}}{\partial v_{i, f_j}}=v_{j, f_i} x_{i} x_{j}$$

因为，$x_j$ 属于 Field $f_j$，且同一个 Field 里面的其他 $x_m$ 都等于0。

## 参考

- [FFM原理及公式推导](https://www.cnblogs.com/zhangchaoyang/articles/8157893.html)
