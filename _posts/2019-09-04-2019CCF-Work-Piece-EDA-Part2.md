---
layout:     post
title:      2019-09-04-2019CCF-Work-Piece-EDA
subtitle:   "离散制造过程中典型工件的质量符合率预测"
date:       2018-09-04 12:00:00
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

[2019CCF Work Piece EDA Part 1](http://lambda-xmu.club/2018/08/25/2019CCF-Work-Piece-EDA/)

在赛题中，只要删除`Parameter1`、`Parameter2`、`Parameter3`、`Parameter4`成绩会提升很多，但是为什么会提升很多，因此来看下不同的`label`在`Parameter`中的分布情况

![]({{ site.url }}/img/Parameter1.png)
![]({{ site.url }}/img/Parameter2.png)
![]({{ site.url }}/img/Parameter3.png)
![]({{ site.url }}/img/Parameter4.png)

可以看出，不同的`label`在`Parameter1`、`Parameter2`、`Parameter3`、`Parameter4`的分布很相似，因此`Parameter1`、`Parameter2`、`Parameter3`、`Parameter4`对预测提供的信息也较少。此外在EDA Part1中也可以看到，训练集和测试集在`Parameter1`、`Parameter2`、`Parameter3`、`Parameter4`分布差异也很大，因此删除较好。

以下是分类变量和`label`的分布情况：
![]({{ site.url }}/img/Parameter5.png)
![]({{ site.url }}/img/Parameter6.png)
![]({{ site.url }}/img/Parameter7.png)
![]({{ site.url }}/img/Parameter8.png)
![]({{ site.url }}/img/Parameter9.png)
![]({{ site.url }}/img/Parameter10.png)

