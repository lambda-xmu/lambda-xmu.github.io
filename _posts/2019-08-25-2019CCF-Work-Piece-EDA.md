---
layout:     post
title:      2019-08-25-2019CCF-Work-Piece-EDA
subtitle:   "离散制造过程中典型工件的质量符合率预测"
date:       2018-08-25 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Data Competition
- EDA
- CCF
- Baseline
---

> Concat: [github: lambda_xmu](https://github.com/lambda-xmu)

赛题地址：[离散制造过程中典型工件的质量符合率预测](https://www.datafountain.cn/competitions/351)

## DATA BACKGROUD
在此任务中，以某典型工件生产过程为例，提供一系列工艺参数，以及在相应工艺参数下所生产工件的质量数据，来预测工件的质量符合率。

## Label Distribution
![]({{ site.url }}/img/work-piece1.png)
可以对标签`excellent`,`good`,`pass`,`fail`依次赋值为1，2，3，4。因为他们之间存在相对关系。因此此赛题既可以当做分类问题来做，也可以当做回归问题来做。

## Correlation
![]({{ site.url }}/img/work-piece2.png)
此相关性图是根据上述将`label`进行转化。

**需注意**：这里做相关性图时未对数据预处理。若对数据进行预处理，相关性则会发生改变。比如此图很多负相关，若对异常值删除，则相关性值会变为正。

## 判断是否是类别变量
![]({{ site.url }}/img/work-piece3.png)

可以发现`Parameter5`, `Parameter6`, `Parameter7`, `Parameter8`, `Parameter9`, `Parameter10`是类别变量，因此提取特征时，可根据类别特征进行提取。但是他们之间还是存在大小关系的。

## `Parameter` 与 `label` 关系
![]({{ site.url }}/img/work-piece4.png)
![]({{ site.url }}/img/work-piece5.png)
![]({{ site.url }}/img/work-piece6.png)
![]({{ site.url }}/img/work-piece7.png)
![]({{ site.url }}/img/work-piece8.png)
![]({{ site.url }}/img/work-piece9.png)
![]({{ site.url }}/img/work-piece10.png)
![]({{ site.url }}/img/work-piece11.png)
![]({{ site.url }}/img/work-piece12.png)
![]({{ site.url }}/img/work-piece13.png)

## 训练集和测试集分布比较
### 类别数量分布
![]({{ site.url }}/img/work-piece14.png)
![]({{ site.url }}/img/work-piece15.png)
![]({{ site.url }}/img/work-piece16.png)
![]({{ site.url }}/img/work-piece17.png)
![]({{ site.url }}/img/work-piece18.png)
![]({{ site.url }}/img/work-piece19.png)

可以看出，类别出现次数分布比较一致，且每个类别是比较其中的。

### 密度分布
![]({{ site.url }}/img/work-piece20.png)
![]({{ site.url }}/img/work-piece21.png)
![]({{ site.url }}/img/work-piece22.png)
![]({{ site.url }}/img/work-piece23.png)
![]({{ site.url }}/img/work-piece24.png)
![]({{ site.url }}/img/work-piece25.png)
![]({{ site.url }}/img/work-piece26.png)
![]({{ site.url }}/img/work-piece27.png)
![]({{ site.url }}/img/work-piece28.png)
![]({{ site.url }}/img/work-piece29.png)

### 散点图（查看异常值）
![]({{ site.url }}/img/work-piece30.png)
![]({{ site.url }}/img/work-piece31.png)
![]({{ site.url }}/img/work-piece32.png)
![]({{ site.url }}/img/work-piece33.png)
![]({{ site.url }}/img/work-piece34.png)
![]({{ site.url }}/img/work-piece35.png)
![]({{ site.url }}/img/work-piece36.png)
![]({{ site.url }}/img/work-piece37.png)
![]({{ site.url }}/img/work-piece38.png)
![]({{ site.url }}/img/work-piece39.png)

可以明显的发现存在些异常值，可以将其删除或其他操作。

### 对数转换
![]({{ site.url }}/img/work-piece40.png)
![]({{ site.url }}/img/work-piece41.png)
![]({{ site.url }}/img/work-piece42.png)
![]({{ site.url }}/img/work-piece43.png)
![]({{ site.url }}/img/work-piece44.png)

通过如下`QQ-Plot`可以发现：对于连续值，通过对数转换转为了正态分布，
![]({{ site.url }}/img/work-piece45.png)

分类变量通过对数转换也更容易发现其特征。

代码详见：[2019CCF Work_Piece EDA](https://github.com/lambda-xmu/2019CCF/blob/master/Work_Piece/2019CCF-work_piece-EDA.ipynb)

# 未完待续
