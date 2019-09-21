---
layout:     post
title:      2019-09-21-Data-competition-From-0-to-1-Part-III
subtitle:   "Rossmann Store Sales"
date:       2018-09-21 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Data Competition
- Entity Embedding
---

Concat: [github: lambda_xmu](https://github.com/lambda-xmu)

赛题地址：[Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales)

## 赛题描述

Rossmann 在欧洲 7 个国家经营了超过 3,000 个商店。现在 Rossmann 管理者想预测未来 6 个星期的销售量。其中，商店的销售量受很多因素影响：促销、竞争对手、学校放假、全国节假日、季节性和地区性等等。

## 评价指标

Root Mean Square Percentage Error (RMSPE)：

$$\operatorname{RMSPE}=\sqrt{\frac{1}{n} \sum_{i=1}^{n}\left(\frac{y_{i}-\hat{y}_{i}}{y_{i}}\right)^{2}}$$

其中 $y_i$ 是每个商店的销售量，$\hat{y}_{i}$ 是相应的预测量。

### 指标转换

为了方便表示：另 $y_{i}=t, \hat{y}_{i}=p$
$$\operatorname{RMSPE}(t, p)=\sqrt{\frac{1}{n} \sum\left(\frac{t-p}{t}\right)^{2}}$$

另 $P$ 和 $T$ 分别是 $p$ 和 $t$ 的转化，即：
$$T \equiv f(t), P \equiv f(p)$$

$$T \equiv f(t)=f(p+\delta) \approx f(p)+f^{\prime}(p) \delta+\ldots=P+f^{\prime}(p) \delta+\ldots$$

其中：$\delta \equiv t-p$

我们的目标是选择一个函数 $f$，使得 $RMS(T,P)$ 近似等于 $RMSPE(t,p)$:
$$RMS(T, P)=\sqrt{\frac{1}{n} \sum(T-P)^{2}}$$
即：
$$T-P \approx \frac{t-p}{t}\\
\Rightarrow f(p)+f^{\prime}(p) \delta-f(p) \approx \frac{t-p}{t}\\
\Rightarrow f^{\prime}(p)\delta \approx \frac{\delta}{t}\\
\Rightarrow f^{\prime}(p)\approx \frac{1}{t}\\
\Rightarrow f(x)\approx \ln(x)+C$$

因为在计算 $T-P$ 时，会把常数项抵消，因此可以假设 $C=0$。所以对 Sale 取对数后使用RMSE最为损失函数是与线上评价指标一致的。当然，这里有个前提：预测值与真实值相差较小。

### 指标优化
假设 $\text{log}(\tilde{y})$ 服从均值为 $\tilde{y}_0 = \log(y_0)$，方差为 $\sigma$ 的正态分布：
$$\tilde{p}(\tilde{y})=\frac{1}{\sigma \sqrt{2 \pi}} \mathrm{e}^{-\frac{(\tilde{y}-\tilde{y}_0)^{2}}{2 \sigma^{2}}}$$

当预测值为 $y^*$ 时，可以计算RMSPE：
$$E\left(y|y^{*}\right)=\int\left(\frac{y-y^{*}}{y}\right)^{2} p(y) \mathrm{d} y=1-2 y^{*} \int \frac{p(y) \mathrm{d} y}{y}+y^{* 2} \int \frac{p(y) \mathrm{d} y}{y^{2}}$$

为了使损失最小，求导数得：

$$y^*=\frac{\int\frac{p(y)\mathrm{d}y}{y}}{\int\frac{p(y)\mathrm{d}y}{y^2}}$$

现在需要计算：

$$f_{\alpha}=\int \frac{p(y) \mathrm{d} y}{y^{\alpha}}$$

其中：$\alpha=1,2$，将$y=e^{\tilde{y}}$带入：

$$p(y)=\tilde{p}(\tilde{y}) \frac{\mathrm{d} \tilde{y}}{\mathrm{d} y}=\frac{\tilde{p}(\tilde{y})}{y}=\tilde{p}(\tilde{y}) \mathrm{e}^{-\tilde{y}}$$

则：

$$f_{\alpha}=\int \mathrm{e}^{-\alpha \tilde{y}} \tilde{p}(\tilde{y}) \mathrm{d} \tilde{y}$$

展开到二次得到：

$$\log \left(f_{\alpha}\right)=-\alpha \tilde{y}+\frac{1}{2} \alpha^{2} \sigma^{2}$$

则：

$$\log \left(y^{*}\right)=\log \left(f_{1}\right)-\log \left(f_{2}\right)=\tilde{y}_{0}-\frac{3}{2} \sigma^{2}$$

因此可以选择最优 $y^*$：

$$y^{*}=y_{0} \cdot \mathrm{e}^{-\frac{3}{2} \sigma^{2}}$$

对于$\sigma^{2} \ll 1$，假设$\sigma^{2}=0.1$，则：

$$\mathrm{e}^{-\frac{3}{2} \sigma^{2}} \approx 1-\frac{3}{2} \sigma^{2} \approx 0.985$$

**Reference:**
- [A connection between RMSPE and the log transform](https://www.kaggle.com/c/rossmann-store-sales/discussion/17026#96290)
- [Correcting log(Sales) prediction for RMSPE](https://www.kaggle.com/c/rossmann-store-sales/discussion/17601)


## 数据

- Id：在测试集中，代表(Store, Date)一个二元组
- Store：每个商店的 Id
- Sales：销售量，即预测值
- Customers：每天一天的顾客量
- Open：商店是否开门的指示变量—— 0 = 关门，1 = 开门
- StateHoliday：全国节假日的指示变量。除非极个别商店，在全国节假日一般都关门。a = 公共节假日，b = 万圣节，c = 圣诞节，0 = None
- SchoolHoliday：学校节假日。代表 (Store, Date) 的销售量受学校放假的影响
- StoreType：商店类型，分为 a, b, c, d 四种类型
- Assortment：分类级别，a = basic, b = extra, c = extended
- CompetitionDistance：与最近的竞争者的距离
- CompetitionOpenSince[Month/Year]：最近的竞争者开门时间（年/月）
- Promo：是否促销的指示变量
- Promo2：Promo2 是连续的促销活动，0 = 商店没参与，1 = 商店参与
- Promo2Since[Year/Week]：描述商店在哪一年，这一年第几周参加了 Promo2
- PromoInterval：描述 Promo2 的开始时间，例如："Feb,May,Aug,Nov" 意味着每一轮开始于每年的 February, May, August, November。

## EDA
### 经验分布函数 （ECDF: empirical cumulative distribution function）

经验分布函数值表示所有观测样本中小于或等于该值的样本所占的比例。

$$\hat{F}_{n}(t)=\frac{\text { number of elements in the sample } \leq t}{n}=\frac{1}{n} \sum_{i=1}^{n} \mathbf{1}_{x_{i} \leq t}$$

可以通过画 ECDF 图来获得**连续变量**的大致分布。

比如此题的`销售量`、`客流量`、`人均销售量`：
![]({{ site.url }}/img/ecdf.png)

可以很容易看出：有接近20%的天数销售额和客户量是0，并且90%的销售额小于10000。

### Univariate Distribution

对于连续变量，除了可以画ECDF，也可以画出变量的分布图：
![]({{ site.url }}/img/CompetitionDistance_Distribution.png)

这是`与最近的竞争者的距离`的分布图，可以看出是个非常偏斜的分布。

### Factorplot

对于连续变量里有很多分类时，可以使用`factorplot`进行绘制：

![]({{ site.url }}/img/factorplot.png)
![]({{ site.url }}/img/factorplot2.png)

从上两张图中可以很容易看出：销售额和客流量与是否促销和类型有很大关系；且在第12个月都有个较大的上涨。猜测是因为`圣诞节`大家都去促销店购物。类型 b 商店的销售额和客流量相比其他类型商店要高很多。

## Embedding

>近年来，神经网络的应用已显着扩展，从图像分割到自然语言处理再到时间序列预测。深度学习的一种特别成功的方法是`嵌入`，**该方法用于将离散变量表示为连续向量**。该技术已在机器翻译的词嵌入和分类变量的实体嵌入中有了实际应用。

嵌入是离散（类别）变量到连续数字矢量的映射。在神经网络中，嵌入是离散变量的低维、连续矢量表示。嵌入不仅可以降低分类变量的维数，而且在转换后的空间中能有意义地表示类别。

神经网络嵌入具有3个主要目的：
1. 在嵌入空间中找到最近的邻居。这些可用于根据用户兴趣或群集类别提出建议
2. 作为对监督任务的机器学习模型的输入
3. 用于概念和类别之间关系的可视化

### One-Hot Encoding 的局限性

One-Hot Encoding 实际上是简单的嵌入，其中每个类别都映射到 0 和 1 的向量。

一键编码技术有两个主要缺点：
1. 对于高基数变量（具有许多独特类别的变量），转换后的向量的维数非常大。
2. 映射是完全没有任何信息的：“相似”类别在嵌入空间中不会相互靠近。

```python
# One Hot Encoding Categoricals
books = ["War and Peace", "Anna Karenina",
          "The Hitchhiker's Guide to the Galaxy"]
books_encoded = [[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]]
Similarity (dot product) between First and Second = 0
Similarity (dot product) between Second and Third = 0
Similarity (dot product) between First and Third = 0
```

### Embedding

考虑到这两个问题，代表类别变量的理想解决方案是映射后的维度要少于类别的数量，并且将相似的类别放置得更近一些。

```python
# Idealized Representation of Embedding
books = ["War and Peace", "Anna Karenina",
          "The Hitchhiker's Guide to the Galaxy"]
books_encoded_ideal = [[0.53,  0.85],
                       [0.60,  0.80],
                       [-0.78, -0.62]]
Similarity (dot product) between First and Second = 0.99
Similarity (dot product) between Second and Third = -0.94
Similarity (dot product) between First and Third = -0.97
```

### Word2vec

Word2vec 依靠了skip-gram 与Continuous Bag of Word (CBOW) 的方法来实作，核心是一个极为浅层的类神经网路。透过使每个字词与前后字词的向量相近，来训练出含有每个字词语义的字词向量。

![]({{ site.url }}/img/Embedding.png)

透过上图可以发现男-女关系的字词在将向量降维到二维平面的结果，会有相似的值与关联。

**Paper**：
- [Word2Vec Original Paper](https://trello-attachments.s3.amazonaws.com/5bc83ce01f094e7e103586e3/5c42e624eb3ab3866cb7b246/3058acf54034adc8716670a088184049/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- [Hierarechy Softmax, Negative Samples](https://trello-attachments.s3.amazonaws.com/5bc83ce01f094e7e103586e3/5c42e624eb3ab3866cb7b246/aee083fd6b575cdb67e620ef5bfeee89/1301.3781.pdf)

## Model
![]({{ site.url }}/img/model.png)

**Paper**：

[Entity Embeddings of Categorical Variables](https://arxiv.org/abs/1604.06737)
