---
layout:     post
title:      2019-08-27-2019CCF-Car-Sales-EDA
subtitle:   "乘用车细分市场销量预测"
date:       2019-08-27 12:00:00
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

赛题地址：[乘用车细分市场销量预测](https://www.datafountain.cn/competitions/352)

## DATA BACKGROUD
历史销量数据包含32个车型在15个省份，从2016年1月至2017年12月的销量。参赛队伍需要预测接下来4个月（2018年1月至2018年4月），这32个车型在15个省份的销量。
> 注：从数据分析得到，车型共60种，而非32种；总共在22省份销售，而非15省份。

## EDA
### 各省总销售量

![]({{ site.url }}/img/各省总销售量.png)

题中未包含：`新疆`、`西藏`、`青海`、`甘肃`、`宁夏`、`贵州`、`吉林`、`天津`、`海南`、`港澳台`数据。`广州`、`山东`、`江苏`、`河南`、`河北`等地销售量较高。

### 全国汽车月销售量与搜索量

![]({{ site.url }}/img/全国汽车月销售量与搜索量.png)

`2017`年销售量整体是要少于`2016`年的销售量。

### 车型销售量占比

![]({{ site.url }}/img/各品牌汽车销售占比（2016 & 2017）.png)

`2016`年和`2017`年的各品牌汽车销售占比基本一致。

![]({{ site.url }}/img/各类型汽车销售占比（2016 & 2017）.png)

`2016`年和`2017`年的各类型汽车销售占比基本一致。其中`SUV`和`MPV`占绝大部分。

### 各省份销售情况

![]({{ site.url }}/img/汽车销售量.png)
![]({{ site.url }}/img/汽车搜索量.png)
![]({{ site.url }}/img/福建汽车月销售量与搜索量.png)
![]({{ site.url }}/img/山东汽车月销售量与搜索量.png)
![]({{ site.url }}/img/黑龙江汽车月销售量与搜索量.png)
![]({{ site.url }}/img/江西汽车月销售量与搜索量.png)
![]({{ site.url }}/img/湖南汽车月销售量与搜索量.png)
![]({{ site.url }}/img/上海汽车月销售量与搜索量.png)
![]({{ site.url }}/img/河南汽车月销售量与搜索量.png)
![]({{ site.url }}/img/陕西汽车月销售量与搜索量.png)
![]({{ site.url }}/img/湖北汽车月销售量与搜索量.png)
![]({{ site.url }}/img/广西汽车月销售量与搜索量.png)
![]({{ site.url }}/img/山西汽车月销售量与搜索量.png)
![]({{ site.url }}/img/四川汽车月销售量与搜索量.png)
![]({{ site.url }}/img/安徽汽车月销售量与搜索量.png)
![]({{ site.url }}/img/河北汽车月销售量与搜索量.png)
![]({{ site.url }}/img/浙江汽车月销售量与搜索量.png)
![]({{ site.url }}/img/内蒙古汽车月销售量与搜索量.png)
![]({{ site.url }}/img/重庆汽车月销售量与搜索量.png)
![]({{ site.url }}/img/云南汽车月销售量与搜索量.png)
![]({{ site.url }}/img/广东汽车月销售量与搜索量.png)
![]({{ site.url }}/img/辽宁汽车月销售量与搜索量.png)
![]({{ site.url }}/img/北京汽车月销售量与搜索量.png)
![]({{ site.url }}/img/江苏汽车月销售量与搜索量.png)

### 每个省销售量占比情况
![]({{ site.url }}/img/各省销售量占比情况.png)

### 各省销售情况补充汽车类型
![]({{ site.url }}/img/福建汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/山东汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/黑龙江汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/江西汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/湖南汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/上海汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/河南汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/陕西汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/湖北汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/广西汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/山西汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/四川汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/安徽汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/河北汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/浙江汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/内蒙古汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/重庆汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/云南汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/广东汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/辽宁汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/北京汽车月销售量与搜索量stack.png)
![]({{ site.url }}/img/江苏汽车月销售量与搜索量stack.png)

### 每种车类型在各省销售占比比较
![]({{ site.url }}/img/Sedan在各省销售占比.png)
![]({{ site.url }}/img/SUV在各省销售占比.png)
![]({{ site.url }}/img/MPV在各省销售占比.png)
![]({{ site.url }}/img/Hatchback在各省销售占比.png)

### 每种类型车在各省销售占比情况
![]({{ site.url }}/img/各省Sedan类型车销售量占比情况.png)
![]({{ site.url }}/img/各省SUV类型车销售量占比情况.png)
![]({{ site.url }}/img/各省MPV类型车销售量占比情况.png)
![]({{ site.url }}/img/各省Hatchbackle类型车销售量占比情况.png)

## Code
[EDA Cdoe](https://github.com/lambda-xmu/2019CCF/blob/master/Car-Sales/EDA.ipynb)
