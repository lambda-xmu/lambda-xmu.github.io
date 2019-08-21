---
layout:     post
title:      2019-08-22-Data-competition-From-0-to-1-Part-II
subtitle:   "Feature Engineering Techniques"
date:       2018-08-22 12:00:00
author:     "lambda"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
- Data Competition
- Feature Engineering
- Processing
---

Concat: [github: lambda_xmu](https://github.com/lambda-xmu)

## Introduction

Feature engineering, the process creating new input features for machine learning, is one of the most effective ways to improve predictive models.

> "Coming up with features is difficult, time-consuming, requires expert knowledge. “Applied machine learning” is basically feature engineering."    -- Andrew Ng

> “... some machine learning projects succeed and some fail. What makes the difference? Easily the most important factor is the features used.”   -- Pedro Domingos

### What is Feature Engineering?
We define feature engineering as creating new features from your existing ones to improve model performance.

**Typical Enterprise Machine Learning Workflow**
![]({{ site.url }}/img/Machine_Learning_Workflow.png)

### What is Not Feature Engineering?
That means there are certain steps we do not consider to be feature engineering:
- We do not consider **initial data collection** to be feature engineering.
- Similarly, we do not consider **creating the target variable** to be feature engineering.
- We do not consider removing duplicates, handling missing values, or fixing mislabeled classes to be feature engineering. We put these under **data cleaning**.
- We do not consider **scaling or normalization** to be feature engineering because these steps belong inside the cross-validation loop.
- Finally, we do not consider **feature selection or PCA** to be feature engineering. These steps also belong inside your cross-validation loop.

### Feature Engineering cycle
![]({{ site.url }}/img/Feature_Engineering_cycle.png)

- Hypothesis
    - Domain knowledge
    - Prior experience
    - EDA
    - ML model feedback
- Validate hypothesis
    - Cross-validation
    - Measurement of desired metrics
    - Avoid leakage

## Processing
### Imputation
Missing values are one of the most common problems you can encounter when you try to prepare your data for machine learning.

The most simple solution to the missing values is to drop the rows or the entire column. There is not an optimum threshold for dropping but you can use 70% as an example value and try to drop the rows and columns which have missing values with higher than this threshold.

```python
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]
```

#### Numerical Imputation
```python
#Filling all missing values with 0
data = data.fillna(0)
#Filling missing values with medians of the columns
data = data.fillna(data.median())
```

#### Categorical Imputation
```python
#Max fill function for categorical columns
data['column_name'].fillna(data['column_name'].value_counts()
.idxmax(), inplace=True)
# or
data['column_name'].fillna('Other', inplace=True)
```

### Handling Outliers
**Best way to detect the outliers is to demonstrate the data visually.** All other statistical methodologies are open to making mistakes, whereas visualizing the outliers gives a chance to take a decision with high precision.

#### Outlier Detection with Standard Deviation
```python
#Dropping the outlier rows with standard deviation
factor = 3
upper_lim = data['column'].mean() + data['column'].std() * factor
lower_lim = data['column'].mean() - data['column'].std() * factor

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]
```

#### Outlier Detection with Percentiles
```python
#Dropping the outlier rows with Percentiles
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)

data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]

#Capping the outlier rows with Percentiles
upper_lim = data['column'].quantile(.95)
lower_lim = data['column'].quantile(.05)
data.loc[(df[column] > upper_lim),column] = upper_lim
data.loc[(df[column] < lower_lim),column] = lower_lim
```

### Log Transform
The benefits of log transform:
- It helps to handle skewed data and after transformation, the distribution becomes more approximate to normal.
- It also decreases the effect of the outliers, due to the normalization of magnitude differences and the model become more robust.

```python
# Negative Values Handling
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)
```

## Feature Engineering Techniques
### Indicator Variables
The first type of feature engineering involves using indicator variables to isolate key information.
- **Indicator variable from thresholds**: create an indicator variable for `age >= 18` to distinguish subjects who were the adult.
- **Indicator variable from multiple features**: You’re predicting real-estate prices and you have the features `n_bedrooms` and `n_bathrooms`. If houses with 2 beds and 2 baths command a premium as rental properties, you can create an indicator variable to flag them.
- **Indicator variable for special events**: You’re modeling weekly sales for an e-commerce site. You can create two indicator variables for the weeks of `11-11` and `6-18`.
- **Indicator variable for groups of classes**: You’re analyzing APP conversions and your dataset has the categorical feature traffic_source. You could create an indicator variable for paid_traffic by flagging observations with traffic source values of `HUAWEI` or `XIAOMI`.

### Interaction Features
This type of feature engineering involves highlighting interactions between two or more features. Well, some features can be combined to provide more information than they would as individuals. Do you know `gender * father's height` and `gender * mather's height` in first homework?

**Note: We don’t recommend using an automated loop to create interactions for all your features. This leads to `feature explosion`.**
- **Sum of two features**
- **Difference between two features**
- **Product of two features**
- **Quotient of two features**

### Feature Representation
Your data won’t always come in the ideal format. You should consider if you’d gain information by representing the same feature in a different way.
- **Date and time features**: Let’s say you have the feature `purchase_datetime`. It might be more useful to extract `purchase_day_of_week` and `purchase_hour_of_day`. You can also aggregate observations to create features such as `purchases_over_last_7_days`, `purchases_over_last_14_days`, `purchases_day_std`, `purchases_week_mean` etc.

```python
from datetime import date

data = pd.DataFrame({'date':
['01-01-2017',
'04-12-2008',
'23-06-1988',
'25-08-1999',
'20-02-1993',
]})

# Transform string to date
data['date'] = pd.to_datetime(data.date, format="%d-%m-%Y")

# Extracting Year
data['year'] = data['date'].dt.year

# Extracting Month
data['month'] = data['date'].dt.month

# Extracting passed years since the date
data['passed_years'] = date.today().year - data['date'].dt.year

# Extracting passed months since the date
data['passed_months'] = (date.today().year - data['date'].dt.year) * 12 + date.today().month - data['date'].dt.month

# Extracting the weekday name of the date
data['day_name'] = data['date'].dt.day_name()

```

- **Binning (分箱)**: Binning can be applied on both categorical and numerical data. When there is a feature with many classes that have low sample counts. You can try grouping similar classes and then grouping the remaining ones into a single `Other` class [**Grouping sparse classes**].
![]({{ site.url }}/img/binning.png)
The main motivation of binning is to make the model more robust and prevent overfitting, however, it has a cost to the performance. Every time you bin something, you sacrifice information and make your data more regularized.
```python
# Numerical Binning Example
data['bin'] = pd.cut(data['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])

# Categorical Binning Example
conditions = [
    data['Country'].str.contains('Spain'),
    data['Country'].str.contains('Italy'),
    data['Country'].str.contains('Chile'),
    data['Country'].str.contains('Brazil')]

choices = ['Europe', 'Europe', 'South America', 'South America']

data['Continent'] = np.select(conditions, choices, default='Other')  # Grouping sparse classes
```
**For numerical columns, except for some obvious overfitting cases, binning might be redundant for some kind of algorithms, due to its effect on model performance. However, for categorical columns, the labels with low frequencies probably affect the robustness of statistical models negatively.**

- **One-hot encoding**: Transform categories into individual binary (0 or 1) features
![]({{ site.url }}/img/one_hot.png)
```python
encoded_columns = pd.get_dummies(data['column'])
data = data.join(encoded_columns).drop('column', axis=1)
```
- **Labeled Encoding**: Interpret the categories as ordered integers (mostly wrong)
![]({{ site.url }}/img/label_encoding.png)

- **Frequency Encoding**: Encoding of categorical levels of feature to values between 0 and 1 based on their relative frequency.
![]({{ site.url }}/img/frequency_encoding.png)

- **Target mean encoding**:
![]({{ site.url }}/img/target_mean_encoding.png)

#### Supplement
**Weighted Average**
It is better to calculate weighted average of the overall mean of the training set and the mean of the level:

$$\lambda(n) * \operatorname{mean}(\text {level})+(1-\lambda(n)) * \text {mean (dataset)}$$

Where $\lambda(n)=\frac{1}{1+\exp \left(\frac{-(x-k)}{f}\right)}$. $x = \text{frequency}, k = \text{inflection point}, f = \text{steepness}$. We can get graph in [desmos_calculator](desmos.com/calculator).

![]({{ site.url }}/img/target_encoding_weight.png)

**To avoid overfitting we could use leave-one-out schema**
![]({{ site.url }}/img/leave_one_out.png)

**Weight of Evidence**
$$W o E=\ln \left(\frac{\% \text { non - events }}{\% \text { events }}\right)$$

To avoid division by zero:

$$\text { WoE }_{adj}=\ln \left(\frac{(\text { Number of non-events in a group }+0.5 )/ \text { Number of non-events }}{(\text { Number of events in a group }+0.5) / \text { Number of events }}\right)$$

**Information Value**
$$I V=\sum(\% \text { non - events }- \% \text { events }) * W o E$$
![]({{ site.url }}/img/WOE_IV.png)

Information value is one of the most useful technique to select important variables in a predictive model.
- Less than 0.02, then the predictor is not useful for modeling (separating the Goods from the Bads)
- 0.02 to 0.1, then the predictor has only a weak relationship to the Goods/Bads odds ratio
- 0.1 to 0.3, then the predictor has a medium strength relationship to the Goods/Bads odds ratio
- 0.3 to 0.5, then the predictor has a strong relationship to the Goods/Bads odds ratio.
- 0.5, suspicious relationship (Check once)

**Information value increases as bins / groups increases for an independent variable. We have to be careful when there are more than 20 bins as some bins may have a very few number of events and non-events.**

### Textual Data
- Bag-of-Words: extract tokens from text and use their occurrences (or TF/IDF weights) as features
    - `CountVectorizer`, `TfidfTransformer`
- NLP techniques
    - Remove stop words (not always necessary)
    - Convert all words to lower case
    - Stemming for English words
    - pingyin
- Deep Learning for textual data
    - Turn each token into a vector of predefined size
    - Help compute “semantic distance” between tokens/words

### External Data
An underused type of feature engineering is bringing in external data. This can lead to some of the biggest breakthroughs in performance.
- **External API’s**: `BAIDU.DITU` etc
- **Geocoding**: Let’s say you have `street_address`, `city`, and `state`. Well, you can geocode them into `latitude` and `longitude`. This will allow you to calculate features such as `local demographics`, `GDP` etc.
- **Other sources of the same data**

### Error Analysis (Post-Modeling)
The final type of feature engineering we’ll cover falls under a process called error analysis. This is performed after training your first model.

Error analysis is a broad term that refers to analyzing the misclassified or high error observations from your model and deciding on your next steps for improvement.
- **Start with larger errors**
- **Segment by classes**
- **Unsupervised clustering**
- **Ask colleagues or domain experts**

## Conclusion
Good features to engineer…

- Can be computed for future observations.
- Are usually intuitive to explain.
- Are informed by domain knowledge or exploratory analysis.
- Must have the potential to be predictive. Don’t just create features for the sake of it.
- **Never touch the target variable**.（穿越问题）
